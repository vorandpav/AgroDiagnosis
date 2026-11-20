"""Configure the RabbitMQ client and shared pipeline queue."""

from math import ceil

from faststream.rabbit import Channel, RabbitBroker, RabbitQueue
from faststream.rabbit.opentelemetry import RabbitTelemetryMiddleware

from src.config import Settings


def create_broker(settings: Settings) -> RabbitBroker:
    """Create a broker whose default channel prefetches one task per worker thread.

    Inference and evaluation subscribers share this channel. A full evaluate
    queue cannot occupy slots while the user queue is empty, and a user task
    is not lined up behind the evaluate backlog. Telemetry middleware injects
    and extracts the W3C traceparent on every publish and consume.
    """
    return RabbitBroker(
        settings.rabbitmq_url,
        default_channel=Channel(
            prefetch_count=settings.worker_executor_workers,
            publisher_confirms=True,
            on_return_raises=True,
        ),
        middlewares=(RabbitTelemetryMiddleware(),),
    )


def task_queue(settings: Settings) -> RabbitQueue:
    """Define the bounded durable queue for publishers and the inference worker."""
    return RabbitQueue(
        settings.task_queue,
        durable=True,
        arguments={
            "x-max-length": settings.task_queue_max_length,
            "x-overflow": "reject-publish",
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": dead_letter_queue(settings).name,
        },
    )


def dead_letter_queue(settings: Settings) -> RabbitQueue:
    """Define the dead-letter queue for the initial pipeline."""
    return RabbitQueue(f"{settings.task_queue}.dlq", durable=True)


def evaluate_task_queue(settings: Settings) -> RabbitQueue:
    """Define the bounded durable queue for evaluation batch workers."""
    return RabbitQueue(
        settings.evaluate_task_queue,
        durable=True,
        arguments={
            "x-max-length": settings.task_queue_max_length,
            "x-overflow": "reject-publish",
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": evaluate_dead_letter_queue(settings).name,
        },
    )


def evaluate_dead_letter_queue(settings: Settings) -> RabbitQueue:
    """Define the dead-letter queue for evaluation tasks."""
    return RabbitQueue(f"{settings.evaluate_task_queue}.dlq", durable=True)


def retry_after_seconds(settings: Settings) -> int:
    """Estimate seconds until a queue slot may free under one busy worker."""
    return max(1, ceil(settings.inference_min_seconds))


async def queue_depth(broker: RabbitBroker, queue_name: str) -> int:
    """Return ready-message count via a fresh underlay passive declare."""
    channel = await broker.config.channel_manager.get_channel()
    underlay = await channel.get_underlay_channel()
    declared = await underlay.queue_declare(queue_name, durable=True, passive=True)
    return int(declared.message_count)
