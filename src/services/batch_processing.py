"""Publish task batches to RabbitMQ and wait for worker persistence."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict
from uuid import UUID

import asyncpg
from faststream.rabbit import RabbitBroker

from src.config import Settings
from src.db.task_results import wait_for_task_results
from src.logging_config import task_id_context
from src.messaging.publishing import (
    publish_evaluate_task_with_backoff,
    publish_inference_task_with_backoff,
)
from src.schemas import InferenceTask, PublicationReport, TaskResult

log = logging.getLogger(__name__)


class BatchManifestRecord(TypedDict):
    """One line written by prepare_batch."""

    id: str
    path: str


def read_manifest_records(manifest: Path) -> Iterator[BatchManifestRecord]:
    """Read prepare_batch JSONL one record at a time."""
    with manifest.open(encoding="utf-8") as handle:
        while line := handle.readline(4097):
            if len(line) > 4096:
                raise ValueError("Manifest line exceeds 4096 characters")
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("Manifest line must be a JSON object")
            try:
                record_id = payload["id"]
                path = payload["path"]
            except KeyError as exc:
                raise ValueError("Manifest record requires id and path") from exc
            if not isinstance(record_id, str) or not isinstance(path, str):
                raise ValueError("Manifest id and path must be strings")
            yield BatchManifestRecord(id=record_id, path=path)


def to_inference_task(record: BatchManifestRecord) -> InferenceTask:
    """Adapt a batch fixture record into a queue InferenceTask."""
    return InferenceTask(task_id=UUID(record["id"]), image_path=record["path"])


def read_tasks(manifest: Path) -> Iterator[InferenceTask]:
    """Yield InferenceTask messages adapted from a prepare_batch manifest."""
    for record in read_manifest_records(manifest):
        yield to_inference_task(record)


async def publish_inference_manifest(
    manifest: Path, broker: RabbitBroker, settings: Settings
) -> PublicationReport:
    """Publish manifest tasks to the inference queue with backpressure."""
    started = time.perf_counter()
    task_count = 0
    for task in read_tasks(manifest):
        with task_id_context(str(task.task_id)):
            await publish_inference_task_with_backoff(broker, task, settings)
            log.info("Published inference task")
        task_count += 1
    return PublicationReport(task_count=task_count, elapsed_seconds=time.perf_counter() - started)


async def batch_process_evaluate(
    tasks: list[InferenceTask],
    broker: RabbitBroker,
    pool: asyncpg.Pool,
    settings: Settings,
) -> dict[UUID, TaskResult]:
    """Publish evaluation images and return every stored worker result."""
    deadline = asyncio.get_running_loop().time() + settings.evaluation_timeout_seconds
    for task in tasks:
        with task_id_context(str(task.task_id)):
            await publish_evaluate_task_with_backoff(broker, task, settings, deadline=deadline)
            log.info("Published evaluation task")
    task_ids = [task.task_id for task in tasks]
    return await wait_for_task_results(
        pool,
        task_ids,
        timeout_seconds=settings.evaluation_timeout_seconds,
        deadline=deadline,
    )
