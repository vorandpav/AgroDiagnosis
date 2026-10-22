"""Configure the RabbitMQ client and shared pipeline queue."""

from faststream.rabbit import Channel, RabbitBroker, RabbitQueue

from src.config import Settings


def create_broker(settings: Settings) -> RabbitBroker:
    """Create a broker with publisher confirms and one in-flight message."""
    return RabbitBroker(
        settings.rabbitmq_url,
        default_channel=Channel(prefetch_count=1, publisher_confirms=True, on_return_raises=True),
    )


def task_queue(settings: Settings) -> RabbitQueue:
    """Define the same durable queue for publishers and the inference worker."""
    return RabbitQueue(
        settings.task_queue,
        durable=True,
        arguments={
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": dead_letter_queue(settings).name,
        },
    )


def dead_letter_queue(settings: Settings) -> RabbitQueue:
    """Define the dead-letter queue for the initial pipeline."""
    return RabbitQueue(f"{settings.task_queue}.dlq", durable=True)
