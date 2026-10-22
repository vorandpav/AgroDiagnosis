"""Accept user uploads into storage and the inference queue."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg
from faststream.rabbit import RabbitBroker

from src.config import Settings
from src.exceptions import QueueFullError
from src.schemas import InferenceTask, TaskAccepted, TaskResult
from src.services.publishing import publish_inference_task

log = logging.getLogger(__name__)

_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _upload_suffix(filename: str | None) -> str:
    suffix = Path(filename or "upload.jpg").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        return ".jpg"
    return suffix


async def submit_uploaded_task(
    image_bytes: bytes,
    filename: str | None,
    settings: Settings,
    broker: RabbitBroker,
) -> TaskAccepted:
    """Store the image, publish metadata, and roll back the file on queue full."""
    if not image_bytes:
        raise ValueError("Uploaded file is empty")
    task_id = uuid4()
    relative = Path("uploads") / f"{task_id}{_upload_suffix(filename)}"
    absolute = (settings.task_storage_dir / relative).resolve()
    root = settings.task_storage_dir.resolve()
    if not absolute.is_relative_to(root):
        raise ValueError("Upload path escapes task storage")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_bytes(image_bytes)
    task = InferenceTask(task_id=task_id, image_path=relative.as_posix())
    try:
        await publish_inference_task(broker, task, settings)
    except QueueFullError:
        absolute.unlink(missing_ok=True)
        raise
    log.info("Accepted task %s (%s)", task_id, task.image_path)
    return TaskAccepted(task_id=task_id)


async def fetch_task_result(task_id: UUID, pool: asyncpg.Pool) -> TaskResult | None:
    """Load a persisted task result, if the worker has written one."""
    row = await pool.fetchrow(
        "SELECT payload FROM task_results WHERE task_id = $1",
        task_id,
    )
    if row is None:
        return None
    return TaskResult.model_validate_json(row["payload"])
