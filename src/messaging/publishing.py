"""Publish InferenceTask messages with distinct overload policies."""

from __future__ import annotations

import asyncio
import logging

from aio_pika.exceptions import DeliveryError
from faststream.rabbit import RabbitBroker

from src.config import Settings
from src.exceptions import QueueFullError
from src.logging_config import task_id_context
from src.messaging.broker import evaluate_task_queue, queue_depth, task_queue
from src.schemas import InferenceTask
from src.telemetry import (
    one_task_trace,
    queue_wait_span,
    record_publish_nack,
    record_published_task,
    record_queue_wait_seconds,
)

log = logging.getLogger(__name__)

TASK_ID_HEADER = "x-task-id"

# TODO: Unify functions


async def ensure_below_soft_limit(broker: RabbitBroker, settings: Settings) -> None:
    """Block batch publishers when depth reaches the soft watermark."""
    depth = await queue_depth(broker, settings.task_queue)
    if depth >= settings.task_queue_soft_limit:
        raise QueueFullError(
            f"Task queue soft limit reached ({depth}>={settings.task_queue_soft_limit})"
        )


def _task_headers(task: InferenceTask) -> dict[str, str]:
    """Return the correlation header carried with every task publish."""
    return {TASK_ID_HEADER: str(task.task_id)}


async def _wait_out_queue_pressure(settings: Settings) -> None:
    """Log queue pressure and wait before the next batch publish."""
    with queue_wait_span():
        log.info("Queue pressure; retrying in %.3fs", settings.publish_retry_delay_seconds)
        await asyncio.sleep(settings.publish_retry_delay_seconds)
    record_queue_wait_seconds(settings.publish_retry_delay_seconds)


async def publish_inference_task(
    broker: RabbitBroker,
    task: InferenceTask,
    settings: Settings,
) -> None:
    """Publish a user task; only the hard broker limit may reject it."""
    with task_id_context(str(task.task_id)):
        try:
            await broker.publish(
                task,
                queue=task_queue(settings),
                message_id=str(task.task_id),
                correlation_id=str(task.task_id),
                headers=_task_headers(task),
                persist=True,
                mandatory=True,
                timeout=settings.publish_timeout_seconds,
            )
        except DeliveryError as exc:
            log.warning("Task queue hard limit rejected publish")
            raise QueueFullError("Task queue hard limit rejected publish") from exc


async def publish_inference_task_with_backoff(
    broker: RabbitBroker,
    task: InferenceTask,
    settings: Settings,
) -> None:
    """Publish a batch task; wait while soft watermark or hard reject blocks."""
    with task_id_context(str(task.task_id)), one_task_trace():
        while True:
            try:
                await ensure_below_soft_limit(broker, settings)
            except QueueFullError:
                await _wait_out_queue_pressure(settings)
                continue
            try:
                await publish_inference_task(broker, task, settings)
            except QueueFullError:
                record_publish_nack()
                await _wait_out_queue_pressure(settings)
                continue
            record_published_task()
            return


async def ensure_evaluate_below_soft_limit(broker: RabbitBroker, settings: Settings) -> None:
    """Block evaluation batch publishers when depth reaches the soft watermark."""
    depth = await queue_depth(broker, settings.evaluate_task_queue)
    if depth >= settings.task_queue_soft_limit:
        raise QueueFullError(
            f"Evaluate queue soft limit reached ({depth}>={settings.task_queue_soft_limit})"
        )


async def publish_evaluate_task(
    broker: RabbitBroker,
    task: InferenceTask,
    settings: Settings,
) -> None:
    """Publish an evaluation image task; only the hard broker limit may reject it."""
    with task_id_context(str(task.task_id)):
        try:
            await broker.publish(
                task,
                queue=evaluate_task_queue(settings),
                message_id=str(task.task_id),
                correlation_id=str(task.task_id),
                headers=_task_headers(task),
                persist=True,
                mandatory=True,
                timeout=settings.publish_timeout_seconds,
            )
        except DeliveryError as exc:
            log.warning("Evaluate queue hard limit rejected publish")
            raise QueueFullError("Evaluate queue hard limit rejected publish") from exc


def _raise_if_evaluation_deadline_passed(deadline: float, timeout_seconds: float) -> None:
    """Raise TimeoutError once the shared evaluation deadline has passed."""
    if asyncio.get_running_loop().time() >= deadline:
        raise TimeoutError(f"Evaluation timed out after {timeout_seconds:g} seconds")


async def _wait_for_evaluate_capacity(
    task: InferenceTask,
    settings: Settings,
    deadline: float,
) -> None:
    """Sleep until the next evaluate publish attempt or the shared deadline."""
    with queue_wait_span():
        log.info(
            "Evaluate queue pressure; retrying in %.3fs",
            settings.publish_retry_delay_seconds,
        )
        _raise_if_evaluation_deadline_passed(deadline, settings.evaluation_timeout_seconds)
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(
                f"Evaluation timed out after {settings.evaluation_timeout_seconds:g} seconds"
            )
        await asyncio.sleep(min(settings.publish_retry_delay_seconds, remaining))


async def publish_evaluate_task_with_backoff(
    broker: RabbitBroker,
    task: InferenceTask,
    settings: Settings,
    *,
    deadline: float,
) -> None:
    """Publish an evaluation task until the shared evaluation deadline."""
    with task_id_context(str(task.task_id)), one_task_trace():
        while True:
            _raise_if_evaluation_deadline_passed(deadline, settings.evaluation_timeout_seconds)
            try:
                await ensure_evaluate_below_soft_limit(broker, settings)
            except QueueFullError:
                await _wait_for_evaluate_capacity(task, settings, deadline)
                continue
            try:
                await publish_evaluate_task(broker, task, settings)
            except QueueFullError:
                await _wait_for_evaluate_capacity(task, settings, deadline)
                continue
            return
