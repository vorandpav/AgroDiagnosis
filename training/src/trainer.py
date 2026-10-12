"""Train / evaluate a torch classifier for plant-disease runs."""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EvalMetrics:
    """Classification metrics on one loader."""

    accuracy: float
    f1_macro: float
    f1_weighted: float


@dataclass(frozen=True)
class TrainResult:
    """Final train/val metrics plus val predictions for artifacts."""

    train_metrics: EvalMetrics
    val_metrics: EvalMetrics
    val_y_true: np.ndarray
    val_y_pred: np.ndarray


def predict_all(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_pred, y_true)`` for every sample in ``loader``."""
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(inputs)
            batch_pred = logits.argmax(dim=1).cpu().numpy()
            preds.append(batch_pred)
            labels.append(targets.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> EvalMetrics:
    """Compute accuracy and F1 on a full loader."""
    y_pred, y_true = predict_all(model, loader, device)
    return EvalMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
) -> TrainResult:
    """Run AdamW fine-tuning; log per-epoch metrics to the active MLflow run."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    val_metrics = EvalMetrics(0.0, 0.0, 0.0)
    val_y_true = np.array([], dtype=np.int64)
    val_y_pred = np.array([], dtype=np.int64)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = int(targets.size(0))
            running_loss += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            seen += batch_size

        val_y_pred, val_y_true = predict_all(model, val_loader, device)
        val_metrics = EvalMetrics(
            accuracy=float(accuracy_score(val_y_true, val_y_pred)),
            f1_macro=float(f1_score(val_y_true, val_y_pred, average="macro", zero_division=0)),
            f1_weighted=float(
                f1_score(val_y_true, val_y_pred, average="weighted", zero_division=0)
            ),
        )
        mean_loss = running_loss / max(seen, 1)
        train_acc_batch = correct / max(seen, 1)
        mlflow.log_metrics(
            {
                "train_loss": mean_loss,
                "train_accuracy_batch": train_acc_batch,
                "val_accuracy": val_metrics.accuracy,
                "val_f1_macro": val_metrics.f1_macro,
                "val_f1_weighted": val_metrics.f1_weighted,
            },
            step=epoch,
        )
        print(
            f"  epoch {epoch + 1}/{epochs} "
            f"loss={mean_loss:.4f} "
            f"val_f1={val_metrics.f1_macro:.4f} "
            f"train_acc_batch={train_acc_batch:.4f}"
        )

    train_metrics = evaluate(model, train_loader, device)
    mlflow.log_metrics(
        {
            "train_accuracy": train_metrics.accuracy,
            "train_f1_macro": train_metrics.f1_macro,
        }
    )
    return TrainResult(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        val_y_true=val_y_true,
        val_y_pred=val_y_pred,
    )
