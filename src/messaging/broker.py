"""Configure the RabbitMQ client and shared pipeline queue."""

from math import ceil

from faststream.rabbit import Channel, RabbitBroker, RabbitQueue

from src.config import Settings


def create_broker(settings: Settings) -> RabbitBroker:
    """Create a broker with publisher confirms and prefetch matching the executor."""
    return RabbitBroker(
        settings.rabbitmq_url,
        default_channel=Channel(
            prefetch_count=settings.worker_executor_workers,
            publisher_confirms=True,
            on_return_raises=True,
        ),
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


def retry_after_seconds(settings: Settings) -> int:
    """Estimate seconds until a queue slot may free under one busy worker."""
    return max(1, ceil(settings.inference_min_seconds))


async def queue_depth(broker: RabbitBroker, queue_name: str) -> int:
    """Return ready-message count via a fresh underlay passive declare."""
    channel = await broker.config.channel_manager.get_channel()
    underlay = await channel.get_underlay_channel()
    declared = await underlay.queue_declare(queue_name, durable=True, passive=True)
    return int(declared.message_count)
