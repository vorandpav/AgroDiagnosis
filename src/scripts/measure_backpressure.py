"""Measure bounded vs unbounded publish behaviour for the backpressure lab (§7).

Runs against a live RabbitMQ. Prints Markdown tables for README.

Usage:
    uv run python -m src.scripts.measure_backpressure
    uv run python -m src.scripts.measure_backpressure --tasks 80 --soft 20 --hard 30
    uv run python -m src.scripts.measure_backpressure --steady --tasks 40 --work-seconds 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
from uuid import uuid4

import aio_pika
import psutil
from faststream.rabbit import RabbitBroker, RabbitQueue

from src.config import Settings, get_settings
from src.exceptions import QueueFullError
from src.logging_config import setup_logging
from src.messaging.broker import create_broker, queue_depth
from src.messaging.publishing import publish_inference_task, publish_inference_task_with_backoff
from src.schemas import InferenceTask

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScenarioResult:
    """One comparative measurement row."""

    name: str
    published: int
    elapsed_seconds: float
    max_depth: int
    peak_rss_mb: float
    tasks_per_second: float


def _queue(name: str, *, max_length: int | None) -> RabbitQueue:
    """Declare a durable queue, optionally with a hard length limit."""
    if max_length is None:
        return RabbitQueue(name, durable=True)
    return RabbitQueue(
        name,
        durable=True,
        arguments={"x-max-length": max_length, "x-overflow": "reject-publish"},
    )


def _task() -> InferenceTask:
    return InferenceTask(task_id=uuid4(), image_path=f"measure/{uuid4()}.jpg")


async def _publish_burst(
    *,
    name: str,
    broker: RabbitBroker,
    settings: Settings,
    queue_name: str,
    task_count: int,
    soft_limit: int | None,
    process: psutil.Process,
) -> ScenarioResult:
    """Publish up to task_count messages; stop on hard reject or soft stall."""
    local = settings.model_copy(
        update={
            "task_queue": queue_name,
            "task_queue_soft_limit": soft_limit or settings.task_queue_soft_limit,
        }
    )
    max_depth = 0
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    published = 0
    soft_stalls = 0

    while published < task_count:
        if soft_limit is not None and await queue_depth(broker, queue_name) >= soft_limit:
            soft_stalls += 1
            if soft_stalls >= 3:
                break
            await asyncio.sleep(local.publish_retry_delay_seconds)
            continue
        soft_stalls = 0
        try:
            if soft_limit is None:
                await publish_inference_task(broker, _task(), local)
            else:
                await publish_inference_task_with_backoff(broker, _task(), local)
        except QueueFullError:
            break
        published += 1
        max_depth = max(max_depth, await queue_depth(broker, queue_name))
        peak_rss = max(peak_rss, process.memory_info().rss)

    elapsed = max(time.perf_counter() - started, 1e-9)
    return ScenarioResult(
        name=name,
        published=published,
        elapsed_seconds=elapsed,
        max_depth=max_depth,
        peak_rss_mb=peak_rss / (1024 * 1024),
        tasks_per_second=published / elapsed,
    )


async def _slow_consumer(
    connection: aio_pika.RobustConnection,
    queue_name: str,
    work_seconds: float,
    stop: asyncio.Event,
) -> None:
    """Ack messages after a fixed delay (stand-in for inference_min_seconds)."""
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)
    queue = await channel.declare_queue(queue_name, durable=True, passive=True)
    async with queue.iterator() as messages:
        async for message in messages:
            if stop.is_set():
                break
            async with message.process(requeue=True):
                await asyncio.sleep(work_seconds)


async def _measure_steady(
    settings: Settings,
    *,
    task_count: int,
    soft_limit: int,
    hard_limit: int,
    work_seconds: float,
    consumers: int,
) -> ScenarioResult:
    """Publish with soft backoff while N slow consumers drain the queue."""
    process = psutil.Process()
    queue_name = f"measure.steady.{consumers}w.{uuid4().hex[:8]}"
    local = settings.model_copy(
        update={
            "task_queue": queue_name,
            "task_queue_soft_limit": soft_limit,
            "task_queue_max_length": hard_limit,
            "publish_retry_delay_seconds": 0.1,
        }
    )
    stop = asyncio.Event()
    async with await aio_pika.connect_robust(settings.rabbitmq_url) as consumer_conn:
        channel = await consumer_conn.channel()
        await channel.declare_queue(
            queue_name,
            durable=True,
            arguments={"x-max-length": hard_limit, "x-overflow": "reject-publish"},
        )
        worker_tasks = [
            asyncio.create_task(
                _slow_consumer(consumer_conn, queue_name, work_seconds, stop),
                name=f"slow-consumer-{index}",
            )
            for index in range(consumers)
        ]
        try:
            async with create_broker(settings) as broker:
                await broker.declare_queue(_queue(queue_name, max_length=hard_limit))
                max_depth = 0
                peak_rss = process.memory_info().rss
                started = time.perf_counter()
                for _ in range(task_count):
                    await publish_inference_task_with_backoff(broker, _task(), local)
                    max_depth = max(max_depth, await queue_depth(broker, queue_name))
                    peak_rss = max(peak_rss, process.memory_info().rss)
                elapsed = max(time.perf_counter() - started, 1e-9)
                return ScenarioResult(
                    name=(f"steady soft={soft_limit} consumers={consumers} work={work_seconds}s"),
                    published=task_count,
                    elapsed_seconds=elapsed,
                    max_depth=max_depth,
                    peak_rss_mb=peak_rss / (1024 * 1024),
                    tasks_per_second=task_count / elapsed,
                )
        finally:
            stop.set()
            for task in worker_tasks:
                task.cancel()
            await asyncio.gather(*worker_tasks, return_exceptions=True)


async def measure(
    settings: Settings,
    *,
    task_count: int,
    soft_limit: int,
    hard_limit: int,
    steady: bool,
    work_seconds: float,
) -> list[ScenarioResult]:
    """Run comparative publish scenarios, optionally with slow consumers."""
    logging.getLogger("src.messaging.publishing").setLevel(logging.WARNING)
    process = psutil.Process()
    suffix = uuid4().hex[:8]
    unbounded_name = f"measure.unbounded.{suffix}"
    hard_name = f"measure.hard.{suffix}"
    soft_name = f"measure.soft.{suffix}"
    rows: list[ScenarioResult] = []

    async with create_broker(settings) as broker:
        await broker.declare_queue(_queue(unbounded_name, max_length=None))
        await broker.declare_queue(_queue(hard_name, max_length=hard_limit))
        await broker.declare_queue(_queue(soft_name, max_length=hard_limit))
        rows.append(
            await _publish_burst(
                name="unbounded (no limit)",
                broker=broker,
                settings=settings,
                queue_name=unbounded_name,
                task_count=task_count,
                soft_limit=None,
                process=process,
            )
        )
        rows.append(
            await _publish_burst(
                name=f"hard reject-publish (max={hard_limit})",
                broker=broker,
                settings=settings.model_copy(update={"task_queue_max_length": hard_limit}),
                queue_name=hard_name,
                task_count=task_count,
                soft_limit=None,
                process=process,
            )
        )
        rows.append(
            await _publish_burst(
                name=f"soft backoff (soft={soft_limit}, hard={hard_limit})",
                broker=broker,
                settings=settings.model_copy(
                    update={
                        "task_queue_max_length": hard_limit,
                        "task_queue_soft_limit": soft_limit,
                    }
                ),
                queue_name=soft_name,
                task_count=task_count,
                soft_limit=soft_limit,
                process=process,
            )
        )

    if steady:
        steady_tasks = min(task_count, 40)
        for consumers in (1, 2):
            rows.append(
                await _measure_steady(
                    settings,
                    task_count=steady_tasks,
                    soft_limit=soft_limit,
                    hard_limit=hard_limit,
                    work_seconds=work_seconds,
                    consumers=consumers,
                )
            )
    return rows


def _print_markdown(rows: list[ScenarioResult]) -> None:
    """Emit a Markdown table for README."""
    print("| Scenario | Published | Elapsed (s) | Max depth | Peak RSS (MiB) | Rate (tasks/s) |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.name} | {row.published} | {row.elapsed_seconds:.2f} | "
            f"{row.max_depth} | {row.peak_rss_mb:.1f} | {row.tasks_per_second:.2f} |"
        )


def main() -> None:
    """Parse CLI options and print measurement results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=80)
    parser.add_argument("--soft", type=int, default=20)
    parser.add_argument("--hard", type=int, default=30)
    parser.add_argument(
        "--steady",
        action="store_true",
        help="Also measure soft-backed publish with 1 and 2 slow consumers.",
    )
    parser.add_argument(
        "--work-seconds",
        type=float,
        default=0.5,
        help="Per-message consumer delay for --steady (stand-in for inference).",
    )
    args = parser.parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)
    rows = asyncio.run(
        measure(
            settings,
            task_count=args.tasks,
            soft_limit=args.soft,
            hard_limit=args.hard,
            steady=args.steady,
            work_seconds=args.work_seconds,
        )
    )
    _print_markdown(rows)


if __name__ == "__main__":
    main()
