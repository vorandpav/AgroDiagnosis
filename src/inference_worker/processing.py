"""Synchronous inference for one queue message (runs in a thread pool)."""

import time

from torch import nn

from src.config import Settings
from src.schemas import InferenceTask, TaskResult
from src.services.inference import predict


def process_task(
    task: InferenceTask,
    settings: Settings,
    model: nn.Module,
    class_names: list[str],
    model_version: str,
) -> TaskResult:
    """Run synchronous processing with a configured minimum duration."""
    started = time.perf_counter()
    root = settings.task_storage_dir.resolve()
    image_path = (root / task.image_path).resolve()
    if not image_path.is_relative_to(root):
        raise ValueError("Image path is outside task storage")
    prediction = predict(
        model,
        class_names,
        image_path.read_bytes(),
        f"{settings.mlflow_model_name}@{settings.mlflow_model_alias}",
    )
    time.sleep(max(0, settings.inference_min_seconds - (time.perf_counter() - started)))
    return TaskResult(
        task_id=task.task_id,
        prediction=prediction,
        model_version=model_version,
        processing_seconds=time.perf_counter() - started,
    )
