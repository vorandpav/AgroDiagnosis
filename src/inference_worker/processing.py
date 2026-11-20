"""Synchronous inference for one queue message (runs in a thread pool)."""

import time

from opentelemetry import trace
from torch import nn

from src.config import Settings
from src.inference_worker.predict import predict
from src.schemas import InferenceTask, TaskResult
from src.telemetry import record_inference_duration


def process_task(
    task: InferenceTask,
    settings: Settings,
    model: nn.Module,
    class_names: list[str],
    model_ref: str,
    model_version: str,
) -> TaskResult:
    """Run synchronous processing with a configured minimum duration."""
    with trace.get_tracer(__name__).start_as_current_span("inference") as span:
        span.set_attribute("alias", settings.mlflow_model_alias)
        span.set_attribute("model_version", model_version)
        started = time.perf_counter()
        root = settings.task_storage_dir.resolve()
        image_path = (root / task.image_path).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError("Image path is outside task storage")
        prediction = predict(model, class_names, image_path.read_bytes())
        time.sleep(max(0, settings.inference_min_seconds - (time.perf_counter() - started)))
        elapsed = time.perf_counter() - started
        record_inference_duration(elapsed)
        return TaskResult(
            task_id=task.task_id,
            prediction=prediction,
            model_ref=model_ref,
            model_version=model_version,
            processing_seconds=elapsed,
        )
