"""Publish a JSONL manifest: uv run python -m src.source path/to/tasks.jsonl."""

import argparse
import asyncio
import logging
import time
from collections.abc import Iterator
from pathlib import Path

from faststream.rabbit import RabbitBroker

from src.broker import create_broker, dead_letter_queue, task_queue
from src.config import Settings, get_settings
from src.logging_config import setup_logging
from src.schemas import InferenceTask, PublicationReport

log = logging.getLogger(__name__)


def read_tasks(manifest: Path) -> Iterator[InferenceTask]:
    """Read one task at a time while limiting manifest line length."""
    with manifest.open(encoding="utf-8") as source:
        while line := source.readline(4097):
            if len(line) > 4096:
                raise ValueError("Manifest line exceeds 4096 characters")
            if line.strip():
                yield InferenceTask.model_validate_json(line)


async def publish_manifest(
    manifest: Path, broker: RabbitBroker, settings: Settings
) -> PublicationReport:
    """Publish tasks sequentially, awaiting confirmation for each task."""
    started = time.perf_counter()
    task_count = 0
    for task in read_tasks(manifest):
        await broker.publish(
            task,
            queue=task_queue(settings),
            message_id=str(task.task_id),
            persist=True,
            mandatory=True,
            timeout=settings.publish_timeout_seconds,
        )
        log.info("Published task %s", task.task_id)
        task_count += 1
    return PublicationReport(task_count=task_count, elapsed_seconds=time.perf_counter() - started)


async def run_source(manifest: Path, settings: Settings) -> None:
    """Connect the source to the broker without loading a model or accessing the database."""
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
    """Parse the manifest path and run the batch source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)
    asyncio.run(run_source(args.manifest, settings))


if __name__ == "__main__":
    main()
