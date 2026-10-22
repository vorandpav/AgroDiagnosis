"""Publish InferenceTask messages with distinct overload policies."""

from __future__ import annotations

import asyncio
import logging

from aio_pika.exceptions import DeliveryError
from faststream.rabbit import RabbitBroker

from src.config import Settings
from src.exceptions import QueueFullError
from src.messaging.broker import queue_depth, task_queue
from src.schemas import InferenceTask

log = logging.getLogger(__name__)


async def ensure_below_soft_limit(broker: RabbitBroker, settings: Settings) -> None:
    """Block batch publishers when depth reaches the soft watermark."""
    depth = await queue_depth(broker, settings.task_queue)
    if depth >= settings.task_queue_soft_limit:
        raise QueueFullError(
            f"Task queue soft limit reached ({depth}>={settings.task_queue_soft_limit})"
        )


async def publish_inference_task(
    broker: RabbitBroker,
    task: InferenceTask,
    settings: Settings,
) -> None:
    """Publish a user task; only the hard broker limit may reject it."""
    try:
        await broker.publish(
            task,
            queue=task_queue(settings),
            message_id=str(task.task_id),
            persist=True,
            mandatory=True,
            timeout=settings.publish_timeout_seconds,
        )
    except DeliveryError as exc:
        raise QueueFullError("Task queue hard limit rejected publish") from exc


async def publish_inference_task_with_backoff(
    broker: RabbitBroker,
    task: InferenceTask,
    settings: Settings,
) -> None:
    """Publish a batch task; wait while soft watermark or hard reject blocks."""
    while True:
        try:
            await ensure_below_soft_limit(broker, settings)
            await publish_inference_task(broker, task, settings)
            return
        except QueueFullError:
            log.info(
                "Queue pressure for batch task %s; retrying in %.3fs",
                task.task_id,
                settings.publish_retry_delay_seconds,
            )
            await asyncio.sleep(settings.publish_retry_delay_seconds)
