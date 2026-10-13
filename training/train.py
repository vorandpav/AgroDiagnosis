"""Hydra entrypoint: one config compose = one MLflow training run.

Run from training/

    uv sync --extra cu124
    uv run --extra cu124 python scripts/prepare_dataset.py
    uv run --extra cu124 python -u train.py
    uv run --extra cu124 python -u train.py data=800per_class model=label_smooth
    uv run --extra cu124 python -u train.py -m data=..., model=...
    uv run --extra cu124 python scripts/promote_champion.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Must import before mlflow: sets env (agent hint / uv scan / progress bars).
from src.mlflow_quiet import configure_mlflow_logging

import hydra
import mlflow
import mlflow.pytorch
import torch
from mlflow.entities.logged_model_input import LoggedModelInput
from omegaconf import DictConfig

from src.dataset import build_split_bundle
from src.models import build_mobilenet_v3_small, resolve_device
from src.seed import set_seed
from src.settings import get_settings, resolve_path
from src.tracking import (
    build_file_manifest_dataset,
    log_confusion_matrix,
    log_hydra_configs,
)
from src.trainer import evaluate, predict_all, train_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

configure_mlflow_logging()

log = logging.getLogger(__name__)

CONF_DIR = resolve_path(get_settings().conf_dir)


def _image_size(cfg: DictConfig) -> tuple[int, int]:
    size = cfg.training.image_size
    return int(size[0]), int(size[1])


def _optional_int(data_cfg: DictConfig, key: str) -> int | None:
    value = data_cfg[key]
    if value is None:
        return None
    return int(value)


def train_and_log(
    *,
    cfg: DictConfig,
    data_dir: Path,
    device: torch.device,
) -> float:
    """Fine-tune one composed config and log to MLflow; return val f1_macro."""
    data_name = str(cfg.data.name)
    model_name = str(cfg.model.name)
    run_name = f"{model_name}__{data_name}"

    image_size = _image_size(cfg)
    train_per_class = _optional_int(cfg.data, "train_per_class")
    val_per_class = _optional_int(cfg.data, "val_per_class")
    batch_size = int(cfg.training.batch_size)
    epochs = int(cfg.training.epochs)
    lr = float(cfg.training.lr)
    weight_decay = float(cfg.training.weight_decay)
    num_workers = int(cfg.training.num_workers)
    random_state = int(cfg.training.random_state)
    dropout = float(cfg.model.dropout)
    label_smoothing = float(cfg.model.label_smoothing)
    set_seed(random_state)

    bundle = build_split_bundle(
        data_dir,
        train_per_class=train_per_class,
        val_per_class=val_per_class,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        random_state=random_state,
        pin_memory=device.type == "cuda",
    )
    log.info(
        "[%s] train_per_class=%s val_per_class=%s: %d classes (train=%d, val=%d, test=%d)",
        data_name,
        train_per_class,
        val_per_class,
        len(bundle.class_names),
        bundle.n_train,
        bundle.n_val,
        bundle.n_test,
    )

    model = build_mobilenet_v3_small(
        num_classes=len(bundle.class_names),
        dropout=dropout,
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_name": model_name,
                "data_name": data_name,
                "backbone": str(cfg.model.backbone),
                "dropout": dropout,
                "label_smoothing": label_smoothing,
                "train_per_class": "all" if train_per_class is None else train_per_class,
                "val_per_class": "all" if val_per_class is None else val_per_class,
                "image_size": f"{image_size[0]}x{image_size[1]}",
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "random_state": random_state,
                "device": str(device),
                "n_train": bundle.n_train,
                "n_val": bundle.n_val,
                "n_test": bundle.n_test,
                "n_classes": len(bundle.class_names),
            }
        )
        log_hydra_configs(cfg)
        dataset = build_file_manifest_dataset(
            paths=bundle.manifest_paths,
            class_names_per_row=bundle.manifest_classes,
            splits=bundle.manifest_splits,
            size_bytes=bundle.manifest_sizes,
            data_name=data_name,
            data_dir=data_dir,
        )

        result = train_model(
            model,
            bundle.train_loader,
            bundle.val_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
        )
        test_metrics = evaluate(model, bundle.test_loader, device)
        test_y_pred, test_y_true = predict_all(model, bundle.test_loader, device)
        overfit_gap = result.train_metrics.f1_macro - result.val_metrics.f1_macro
        metrics = {
            "accuracy": result.val_metrics.accuracy,
            "f1_macro": result.val_metrics.f1_macro,
            "f1_weighted": result.val_metrics.f1_weighted,
            "f1_macro_train": result.train_metrics.f1_macro,
            "overfit_gap": overfit_gap,
            "test_accuracy": test_metrics.accuracy,
            "test_f1_macro": test_metrics.f1_macro,
            "test_f1_weighted": test_metrics.f1_weighted,
        }
        log_confusion_matrix(
            result.val_y_true,
            result.val_y_pred,
            bundle.class_names,
        )
        log_confusion_matrix(
            test_y_true,
            test_y_pred,
            bundle.class_names,
            artifact_file="confusion_matrix_test.png",
            title="Test probe confusion matrix",
        )

        model_to_log = model.cpu()
        example = torch.randn(1, 3, image_size[0], image_size[1])
        model_info = mlflow.pytorch.log_model(
            model_to_log,
            name=model_name,
            input_example=example.numpy(),
            serialization_format="pickle",
            pip_requirements=["torch", "torchvision", "cloudpickle"],
        )
        mlflow.set_tag("model_uri", model_info.model_uri)
        mlflow.log_input(
            dataset,
            context="training",
            model=LoggedModelInput(model_id=model_info.model_id),
        )
        mlflow.log_metrics(
            metrics,
            dataset=dataset,
            model_id=model_info.model_id,
        )
        log.info(
            "%s: val_f1=%.4f test_f1=%.4f train_f1=%.4f gap=%.4f",
            run_name,
            result.val_metrics.f1_macro,
            test_metrics.f1_macro,
            result.train_metrics.f1_macro,
            overfit_gap,
        )
        return result.val_metrics.f1_macro


@hydra.main(version_base=None, config_path=str(CONF_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    """Train one Hydra-composed data/model pair and log the run to MLflow."""
    settings = get_settings()
    data_dir = resolve_path(settings.data_dir)
    device = resolve_device()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(str(cfg.mlflow.experiment_name))

    log.info("Using data_dir=%s", data_dir)
    log.info("Device=%s", device)
    log.info("Compose: data=%s model=%s", cfg.data.name, cfg.model.name)

    train_and_log(cfg=cfg, data_dir=data_dir, device=device)


if __name__ == "__main__":
    main()
