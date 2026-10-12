"""MLflow helpers: dataset lineage and Hydra config artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import ConfusionMatrixDisplay


def build_file_manifest_dataset(
    *,
    paths: list[str],
    class_names_per_row: list[str],
    splits: list[str],
    size_bytes: list[int],
    data_name: str,
    data_dir: Path,
) -> PandasDataset:
    """Build an MLflow Dataset from a per-image file manifest.

    One row per image: path / class / split / size_bytes.
    Digest hashes this table (paths + sizes), not image pixels - cheap and
    still changes when files are added, removed, or replaced with different size.
    Source is the local processed ``data_dir``.
    """
    if not (len(paths) == len(class_names_per_row) == len(splits) == len(size_bytes)):
        raise ValueError("Manifest columns must have the same length")

    # size_bytes as float64 avoids MLflow integer-missing-value schema warnings.
    frame = pd.DataFrame(
        {
            "path": paths,
            "class": class_names_per_row,
            "split": splits,
            "size_bytes": [float(size) for size in size_bytes],
        }
    )
    return mlflow.data.from_pandas(
        frame,
        name=f"plant-diseases-{data_name}",
        source=str(data_dir.resolve()),
    )


def log_hydra_configs(
    cfg: DictConfig,
    data_cfg: DictConfig,
    model_cfg: DictConfig,
) -> None:
    """Log resolved Hydra configs as run artifacts."""
    mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "hydra/config.yaml")
    mlflow.log_dict(OmegaConf.to_container(data_cfg, resolve=True), "hydra/data.yaml")
    mlflow.log_dict(OmegaConf.to_container(model_cfg, resolve=True), "hydra/model.yaml")


def log_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    *,
    artifact_file: str = "confusion_matrix.png",
    title: str = "Validation confusion matrix",
) -> None:
    """Log a confusion-matrix figure to the active MLflow run."""
    # Probe test only hits a subset of classes - plot those labels only.
    present = sorted(set(int(v) for v in y_true) | set(int(v) for v in y_pred))
    short_names = [
        class_names[i] if len(class_names[i]) <= 24 else f"{class_names[i][:21]}..."
        for i in present
    ]
    fig_w = max(12.0, 0.35 * len(present))
    fig_h = max(10.0, 0.35 * len(present))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=present,
        display_labels=short_names,
        xticks_rotation=90,
        ax=ax,
        colorbar=True,
        values_format="d",
    )
    ax.set_title(title)
    fig.tight_layout()
    mlflow.log_figure(fig, artifact_file)
    plt.close(fig)
