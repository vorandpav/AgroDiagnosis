"""Orchestrate a single inference task and persist its result."""

import time

import asyncpg
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


async def save_result(result: TaskResult, pool: asyncpg.Pool) -> None:
    """Atomically save the first result using the unique task_id."""
    await pool.execute(
        """
        INSERT INTO task_results (task_id, payload)
        VALUES ($1, $2::jsonb)
        ON CONFLICT (task_id) DO NOTHING
        """,
        result.task_id,
        result.model_dump_json(),
    )


async def initialize_results(pool: asyncpg.Pool) -> None:
    """Create the results table for the initial pipeline."""
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS task_results (
            task_id uuid PRIMARY KEY,
            payload jsonb NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
