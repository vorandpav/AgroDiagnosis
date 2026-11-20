"""Adapt a prepare_batch JSONL manifest into InferenceTask messages and publish.

Manifest lines (from task-data/scripts/prepare_batch.py):
    {"id": "<uuid>", "path": "demo/images/<uuid>.jpg"}

Usage:
    uv run python -m src.scripts.batch_publisher path/to/tasks.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from src.config import Settings, get_settings
from src.logging_config import setup_logging
from src.messaging.broker import create_broker, dead_letter_queue, task_queue
from src.services.batch_processing import publish_inference_manifest

log = logging.getLogger(__name__)


async def run_batch_publisher(manifest: Path, settings: Settings) -> None:
    """Connect the publisher without loading a model or accessing the database."""
    async with create_broker(settings) as broker:
        await broker.declare_queue(dead_letter_queue(settings))
        await broker.declare_queue(task_queue(settings))
        report = await publish_inference_manifest(manifest, broker, settings)
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
