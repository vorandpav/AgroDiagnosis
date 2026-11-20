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
from faststream.rabbit import RabbitBroker, RabbitMessage, RabbitQueue
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.propagate import inject
from uvicorn import Server

from src.config import Settings
from src.db.database import create_database
from src.db.task_results import initialize_results, save_result
from src.inference_worker.health import create_health_server
from src.inference_worker.processing import process_task
from src.logging_config import TASK_ID
from src.messaging.broker import (
    create_broker,
    dead_letter_queue,
    evaluate_dead_letter_queue,
    evaluate_task_queue,
    task_queue,
)
from src.messaging.publishing import TASK_ID_HEADER
from src.mlflow_runtime import load_model
from src.schemas import InferenceTask, TaskResult
from src.telemetry import (
    adjust_tasks_in_flight,
    call_in_context,
    record_redelivery,
    record_span_exception,
    record_worker_outcome,
)

log = logging.getLogger(__name__)

_ATTEMPT_HEADER = "x-delivery-attempt"


def message_task_id(message: RabbitMessage) -> str | None:
    """Return correlation_id, or the x-task-id header when it is absent."""
    if message.correlation_id:
        return message.correlation_id
    raw = message.headers.get(TASK_ID_HEADER)
    if isinstance(raw, str) and raw:
        return raw
    return None


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
    queue: RabbitQueue,
) -> None:
    """Republish with an incremented attempt or reject to DLQ when exhausted."""
    if attempt + 1 >= settings.task_max_delivery_attempts:
        log.error(
            "Task failed permanently after %s attempt(s); sending to DLQ",
            attempt + 1,
        )
        record_worker_outcome("dlq")
        await message.reject(requeue=False)
        return
    log.warning(
        "Task failed on attempt %s; republishing after delay",
        attempt,
    )
    record_worker_outcome("error")
    await asyncio.sleep(settings.task_retry_delay_seconds)
    await broker.publish(
        task,
        queue=queue,
        message_id=str(task.task_id),
        correlation_id=str(task.task_id),
        persist=True,
        headers=_republish_headers(task, attempt + 1),
    )
    await message.ack()


def _republish_headers(task: InferenceTask, attempt: int) -> dict[str, str | int]:
    """Return retry headers plus the current W3C traceparent."""
    headers: dict[str, str | int] = {
        _ATTEMPT_HEADER: attempt,
        TASK_ID_HEADER: str(task.task_id),
    }
    inject(headers)
    return headers


def _mark_handler_failure(exc: Exception, attempt: int) -> None:
    """Record a handler failure on the current consumer span."""
    record_span_exception(exc)
    trace.get_current_span().set_attribute(_ATTEMPT_HEADER, attempt)


def _register_queue_handler(
    broker: RabbitBroker,
    settings: Settings,
    queue: RabbitQueue,
    execute: Callable[[InferenceTask], Awaitable[TaskResult]],
    save: Callable[[TaskResult], Awaitable[None]],
) -> None:
    """Register a manual-ack handler with limited retries then DLQ."""

    @broker.subscriber(queue, ack_policy=AckPolicy.MANUAL)
    async def handle(task: InferenceTask, message: RabbitMessage) -> None:
        """Save the result before ack; republish or DLQ on failure."""
        attempt = delivery_attempt(message)
        if attempt > 0:
            record_redelivery()
        adjust_tasks_in_flight(1)
        token = TASK_ID.set(message_task_id(message))
        try:
            try:
                result = await execute(task)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _mark_handler_failure(exc, attempt)
                log.exception("Task failed during inference", extra={"attempt": attempt})
                await recover_failed_delivery(broker, settings, task, message, attempt, queue)
                return
            try:
                with trace.get_tracer(__name__).start_as_current_span("save_result"):
                    await save(result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _mark_handler_failure(exc, attempt)
                log.exception("Task failed while saving result", extra={"attempt": attempt})
                await recover_failed_delivery(broker, settings, task, message, attempt, queue)
                return
            await message.ack()
            record_worker_outcome("ok")
            log.info("Task saved (%.3fs)", result.processing_seconds)
        finally:
            TASK_ID.reset(token)
            adjust_tasks_in_flight(-1)


def register_inference_handler(
    broker: RabbitBroker,
    settings: Settings,
    execute: Callable[[InferenceTask], Awaitable[TaskResult]],
    save: Callable[[TaskResult], Awaitable[None]],
) -> None:
    """Register the inference queue consumer."""
    _register_queue_handler(broker, settings, task_queue(settings), execute, save)


def register_evaluate_handler(
    broker: RabbitBroker,
    settings: Settings,
    execute: Callable[[InferenceTask], Awaitable[TaskResult]],
    save: Callable[[TaskResult], Awaitable[None]],
) -> None:
    """Register the evaluation queue consumer."""
    _register_queue_handler(broker, settings, evaluate_task_queue(settings), execute, save)


async def run_inference_worker(settings: Settings) -> None:
    """Load the model, then serve health and consume until the broker stops."""
    health: Server | None = None
    health_task: asyncio.Task[None] | None = None
    shutdown_started = False

    def log_shutdown_started() -> None:
        """Log the beginning of teardown once."""
        nonlocal shutdown_started
        if shutdown_started:
            return
        shutdown_started = True
        log.info("Graceful shutdown started")

    try:
        async with AsyncExitStack() as resources:
            executor = resources.enter_context(
                ThreadPoolExecutor(max_workers=settings.worker_executor_workers)
            )

            loop = asyncio.get_running_loop()

            model = await loop.run_in_executor(executor, load_model, settings)
            log.info(
                "Loaded model classes=%s",
                ",".join(model.class_names),
                extra={
                    "alias": settings.mlflow_model_alias,
                    "model_version": model.model_version,
                },
            )

            health = create_health_server(settings.worker_health_host, settings.worker_health_port)
            health_task = asyncio.create_task(health.serve(), name="inference-worker-health")
            log.info(
                "Inference worker health on %s:%s/healthz",
                settings.worker_health_host,
                settings.worker_health_port,
            )

            database = await create_database(settings)
            resources.push_async_callback(database.close)
            await database.declare_table(initialize_results)
            pool = database.pool

            broker = create_broker(settings)

            async def execute(task: InferenceTask) -> TaskResult:
                """Run file reads, inference, and the delay in a single executor thread."""
                parent = otel_context.get_current()
                return await loop.run_in_executor(
                    executor,
                    partial(
                        call_in_context,
                        parent,
                        partial(
                            process_task,
                            task,
                            settings,
                            model.model,
                            model.class_names,
                            model.model_ref,
                            model.model_version,
                        ),
                    ),
                )

            save = partial(save_result, pool=pool)
            register_inference_handler(broker, settings, execute, save)
            register_evaluate_handler(broker, settings, execute, save)
            app = FastStream(broker)

            @app.on_startup
            async def prepare_queue() -> None:
                """Declare pipeline queues before consuming tasks."""
                await broker.connect()
                await broker.declare_queue(dead_letter_queue(settings))
                await broker.declare_queue(task_queue(settings))
                await broker.declare_queue(evaluate_dead_letter_queue(settings))
                await broker.declare_queue(evaluate_task_queue(settings))

            @app.on_shutdown
            async def begin_graceful_shutdown() -> None:
                """Log the start of worker teardown."""
                log_shutdown_started()

            await app.run()
    finally:
        log_shutdown_started()
        if health is not None and health_task is not None:
            health.should_exit = True
            await health_task
        log.info("Graceful shutdown finished")
