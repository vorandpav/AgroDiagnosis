"""RabbitMQ consumer for inference-worker: manual ack, retries, DLQ."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from functools import partial

from faststream import FastStream
from faststream.middlewares import AckPolicy
from faststream.rabbit import RabbitBroker, RabbitMessage

from src.config import Settings
from src.db import close_db_pool, create_db_pool
from src.inference_worker.health import create_health_server
from src.inference_worker.processing import process_task
from src.messaging.broker import create_broker, dead_letter_queue, task_queue
from src.schemas import InferenceTask, TaskResult
from src.services.inference import load_model
from src.tasks.results import initialize_results, save_result

log = logging.getLogger(__name__)

_ATTEMPT_HEADER = "x-delivery-attempt"


def delivery_attempt(message: RabbitMessage) -> int:
    """Read the current delivery attempt from message headers (0-based)."""
    headers = message.headers or {}
    raw = headers.get(_ATTEMPT_HEADER, 0)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


async def recover_failed_delivery(
    broker: RabbitBroker,
    settings: Settings,
    task: InferenceTask,
    message: RabbitMessage,
    attempt: int,
) -> None:
    """Republish with an incremented attempt or reject to DLQ when exhausted."""
    if attempt + 1 >= settings.task_max_delivery_attempts:
        log.error(
            "Task %s failed permanently after %s attempt(s); sending to DLQ",
            task.task_id,
            attempt + 1,
        )
        await message.reject(requeue=False)
        return
    log.warning(
        "Task %s failed on attempt %s; republishing after delay",
        task.task_id,
        attempt,
    )
    await asyncio.sleep(settings.task_retry_delay_seconds)
    await broker.publish(
        task,
        queue=task_queue(settings),
        message_id=str(task.task_id),
        persist=True,
        headers={_ATTEMPT_HEADER: attempt + 1},
    )
    await message.ack()


def register_handler(
    broker: RabbitBroker,
    settings: Settings,
    execute: Callable[[InferenceTask], Awaitable[TaskResult]],
    save: Callable[[TaskResult], Awaitable[None]],
) -> None:
    """Register a manual-ack handler with limited retries then DLQ."""

    @broker.subscriber(task_queue(settings), ack_policy=AckPolicy.MANUAL)
    async def handle(task: InferenceTask, message: RabbitMessage) -> None:
        """Save the result before ack; republish or DLQ on failure."""
        attempt = delivery_attempt(message)
        try:
            result = await execute(task)
            await save(result)
        except Exception:
            log.exception("Task %s failed during processing", task.task_id)
            await recover_failed_delivery(broker, settings, task, message, attempt)
            return
        await message.ack()
        log.info("Task %s saved (%.3fs)", task.task_id, result.processing_seconds)


async def run_inference_worker(settings: Settings) -> None:
    """Load the model once and release resources after the broker stops."""
    health = create_health_server(settings.worker_health_host, settings.worker_health_port)
    health_task = asyncio.create_task(health.serve(), name="inference-worker-health")
    log.info(
        "Inference worker health on %s:%s/healthz",
        settings.worker_health_host,
        settings.worker_health_port,
    )
    try:
        async with AsyncExitStack() as resources:
            executor = resources.enter_context(
                ThreadPoolExecutor(max_workers=settings.worker_executor_workers)
            )
            loop = asyncio.get_running_loop()
            model, class_names, model_version = await loop.run_in_executor(
                executor, load_model, settings
            )
            log.info(
                "Loaded %s@%s version %s",
                settings.mlflow_model_name,
                settings.mlflow_model_alias,
                model_version,
            )
            pool = await create_db_pool(settings)
            resources.push_async_callback(close_db_pool, pool)
            await initialize_results(pool)
            broker = create_broker(settings)

            async def execute(task: InferenceTask) -> TaskResult:
                """Run file reads, inference, and the delay in a single executor thread."""
                return await loop.run_in_executor(
                    executor,
                    partial(process_task, task, settings, model, class_names, model_version),
                )

            register_handler(broker, settings, execute, partial(save_result, pool=pool))
            app = FastStream(broker)

            @app.on_startup
            async def prepare_queue() -> None:
                """Declare the dead-letter queue before consuming tasks."""
                await broker.connect()
                await broker.declare_queue(dead_letter_queue(settings))

            await app.run()
    finally:
        health.should_exit = True
        await health_task
