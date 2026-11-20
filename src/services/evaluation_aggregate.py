"""Build an evaluation report from worker task results."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath
from uuid import UUID

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.config import Settings
from src.schemas import (
    EvaluationMetrics,
    EvaluationPrediction,
    EvaluationReport,
    InferenceTask,
    TaskResult,
)


def _expected_label(image_path: str) -> str:
    """Derive Plant/Condition from a stored evaluation image path."""
    parts = PurePosixPath(image_path).parts
    if len(parts) < 3:
        raise ValueError(f"Cannot derive label from path: {image_path}")
    return "/".join(parts[-3:-1])


def build_evaluation_report(
    *,
    dataset_name: str,
    dataset_sha256: str,
    model_ref: str,
    model_version: str,
    tasks: list[InferenceTask],
    results: dict[UUID, TaskResult],
    settings: Settings,
) -> EvaluationReport:
    """Aggregate per-image worker predictions into dataset-level metrics."""
    root = settings.task_storage_dir.resolve()
    predictions: list[EvaluationPrediction] = []
    for task in sorted(tasks, key=lambda item: item.image_path):
        result = results.get(task.task_id)
        if result is None:
            raise RuntimeError(f"Missing worker result for task {task.task_id}")
        image_path = (root / task.image_path).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError("Evaluation image path escapes task storage")
        image_bytes = image_path.read_bytes()
        predictions.append(
            EvaluationPrediction(
                path=task.image_path,
                sha256=hashlib.sha256(image_bytes).hexdigest(),
                expected=_expected_label(task.image_path),
                predicted=result.prediction.label,
                confidence=result.prediction.confidence,
            )
        )
    expected = [row.expected for row in predictions]
    predicted = [row.predicted for row in predictions]
    labels = sorted(set(expected) | set(predicted))
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected, predicted, labels=labels, average="macro", zero_division=0
    )
    return EvaluationReport(
        dataset_name=dataset_name,
        dataset_sha256=dataset_sha256,
        model=model_ref,
        model_version=model_version,
        metrics=EvaluationMetrics(
            accuracy=float(accuracy_score(expected, predicted)),
            precision_macro=float(precision),
            recall_macro=float(recall),
            f1_macro=float(f1),
        ),
        labels=labels,
        confusion_matrix=confusion_matrix(expected, predicted, labels=labels).tolist(),
        predictions=predictions,
    )
