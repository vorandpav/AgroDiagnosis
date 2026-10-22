"""Adapt a prepare_batch JSONL manifest into InferenceTask messages and publish.

Manifest lines (from task-data/scripts/prepare_batch.py):
    {"id": "<uuid>", "path": "demo/images/<uuid>.jpg"}

Usage:
    uv run python -m src.scripts.batch_publisher path/to/tasks.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict
from uuid import UUID

from faststream.rabbit import RabbitBroker

from src.messaging.broker import create_broker, dead_letter_queue, task_queue
from src.config import Settings, get_settings
from src.logging_config import setup_logging
from src.schemas import InferenceTask, PublicationReport
from src.messaging.publishing import publish_inference_task_with_backoff

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


async def publish_manifest(
    manifest: Path, broker: RabbitBroker, settings: Settings
) -> PublicationReport:
    """Publish adapted tasks with backpressure when the bounded queue is full."""
    started = time.perf_counter()
    task_count = 0
    for task in read_tasks(manifest):
        await publish_inference_task_with_backoff(broker, task, settings)
        log.info("Published task %s", task.task_id)
        task_count += 1
    return PublicationReport(task_count=task_count, elapsed_seconds=time.perf_counter() - started)


async def run_batch_publisher(manifest: Path, settings: Settings) -> None:
    """Connect the publisher without loading a model or accessing the database."""
    async with create_broker(settings) as broker:
        await broker.declare_queue(dead_letter_queue(settings))
        await broker.declare_queue(task_queue(settings))
        report = await publish_manifest(manifest, broker, settings)
        log.info(
            "Published %d tasks in %.3fs (%.2f tasks/s)",
            report.task_count,
            report.elapsed_seconds,
            report.tasks_per_second,
        )


def main() -> None:
    """Parse the manifest path and publish the batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)
    asyncio.run(run_batch_publisher(args.manifest, settings))


if __name__ == "__main__":
    main()
