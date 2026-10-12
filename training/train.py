"""Hydra entrypoint: fine-tune and register the champion.

Run from training/

    uv sync --extra cu124
    uv run --extra cu124 python scripts/prepare_dataset.py
    uv run --extra cu124 python -u train.py
    uv run --extra cu124 python -u train.py data_to_run=[200per_class] models_to_run=[baseline]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

# Must import before mlflow: sets env (agent hint / uv scan / progress bars).
from src.mlflow_quiet import configure_mlflow_logging

import hydra
import mlflow
import mlflow.pytorch
import torch
from mlflow.entities.logged_model_input import LoggedModelInput
from omegaconf import DictConfig, OmegaConf

from src.dataset import build_split_bundle
from src.models import build_mobilenet_v3_small, resolve_device
from src.register import register_champion
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

CONF_DIR = resolve_path(get_settings().conf_dir)


def _image_size(cfg: DictConfig) -> tuple[int, int]:
    size = cfg.training.image_size
    return int(size[0]), int(size[1])


def _load_group_cfg(group: str, name: str) -> DictConfig:
    path = CONF_DIR / group / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown {group} config: {path}")
    loaded = OmegaConf.load(path)
    return cast(DictConfig, loaded)


def _optional_int(data_cfg: DictConfig, key: str) -> int | None:
    value = data_cfg[key]
    if value is None:
        return None
    return int(value)


def train_and_log(
    run_name: str,
    model_name: str,
    *,
    cfg: DictConfig,
    data_cfg: DictConfig,
    model_cfg: DictConfig,
    data_dir: Path,
    data_name: str,
    device: torch.device,
) -> tuple[str, float]:
    """Fine-tune one model config, log to MLflow, return model URI and val f1."""
    image_size = _image_size(cfg)
    train_per_class = _optional_int(data_cfg, "train_per_class")
    val_per_class = _optional_int(data_cfg, "val_per_class")
    batch_size = int(cfg.training.batch_size)
    epochs = int(cfg.training.epochs)
    lr = float(cfg.training.lr)
    weight_decay = float(cfg.training.weight_decay)
    num_workers = int(cfg.training.num_workers)
    random_state = int(cfg.training.random_state)
    dropout = float(model_cfg.dropout)
    label_smoothing = float(model_cfg.label_smoothing)

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
    print(
        f"[{data_name}] train_per_class={train_per_class} val_per_class={val_per_class}: "
        f"{len(bundle.class_names)} classes "
        f"(train={bundle.n_train}, val={bundle.n_val}, test={bundle.n_test})"
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
                "backbone": str(model_cfg.backbone),
                "dropout": dropout,
                "label_smoothing": label_smoothing,
                "train_per_class": "all" if train_per_class is None else train_per_class,
                "val_per_class": "all" if val_per_class is None else val_per_class,
                "image_size": f"{image_size[0]}x{image_size[1]}",
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "device": str(device),
                "n_train": bundle.n_train,
                "n_val": bundle.n_val,
                "n_test": bundle.n_test,
                "n_classes": len(bundle.class_names),
            }
        )
        log_hydra_configs(cfg, data_cfg, model_cfg)
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
        print(
            f"{run_name}: val_f1={result.val_metrics.f1_macro:.4f} "
            f"test_f1={test_metrics.f1_macro:.4f} "
            f"train_f1={result.train_metrics.f1_macro:.4f} gap={overfit_gap:.4f}"
        )
        return model_info.model_uri, result.val_metrics.f1_macro


@hydra.main(version_base=None, config_path=str(CONF_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    """Train data-model grid and promote the best val f1_macro to @champion."""
    settings = get_settings()
    data_dir = resolve_path(settings.data_dir)
    device = resolve_device()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(str(cfg.mlflow.experiment_name))

    print(f"Using data_dir={data_dir}")
    print(f"Device={device}")
    print(f"Hydra defaults demo: data={cfg.data.name}, model={cfg.model.name}")

    best_model_uri: str | None = None
    best_f1 = -1.0

    for data_name_raw in cfg.data_to_run:
        data_name = str(data_name_raw)
        data_cfg = _load_group_cfg("data", data_name)

        for model_name_raw in cfg.models_to_run:
            model_name = str(model_name_raw)
            run_name = f"{model_name}__{data_name}"
            model_cfg = _load_group_cfg("model", model_name)
            model_uri, f1_macro = train_and_log(
                run_name,
                model_name,
                cfg=cfg,
                data_cfg=data_cfg,
                model_cfg=model_cfg,
                data_dir=data_dir,
                data_name=data_name,
                device=device,
            )
            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_model_uri = model_uri

    if best_model_uri is None:
        raise RuntimeError("No training runs completed")

    version = register_champion(best_model_uri, str(cfg.mlflow.registered_model_name))
    print(
        f"Champion val_f1_macro={best_f1:.4f} -> "
        f"{cfg.mlflow.registered_model_name} v{version} @champion"
    )


if __name__ == "__main__":
    main()
