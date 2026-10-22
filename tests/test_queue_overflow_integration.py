"""Integration: real RabbitMQ rejects publish when x-max-length is reached."""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("testcontainers")

from testcontainers.community.rabbitmq import RabbitMqContainer

from src.config import Settings
from src.exceptions import QueueFullError
from src.messaging.broker import create_broker, dead_letter_queue, queue_depth, task_queue
from src.messaging.publishing import publish_inference_task
from src.schemas import InferenceTask

pytestmark = pytest.mark.integration

_RABBIT_IMAGE = "rabbitmq:4.1-management"


def _amqp_url(rabbit: RabbitMqContainer) -> str:
    """Build an AMQP URL from the Testcontainers connection params."""
    params = rabbit.get_connection_params()
    credentials = params.credentials
    assert credentials is not None
    return f"amqp://{credentials.username}:{credentials.password}@{params.host}:{params.port}/"


def _overflow_settings(rabbit: RabbitMqContainer, *, max_length: int) -> Settings:
    """Settings pointing at the temporary broker with a tiny hard limit."""
    return Settings(
        rabbitmq_url=_amqp_url(rabbit),
        task_queue=f"test.overflow.{uuid4().hex}",
        task_queue_max_length=max_length,
        task_queue_soft_limit=max_length,
        publish_timeout_seconds=3.0,
    )


def _task() -> InferenceTask:
    return InferenceTask(task_id=uuid4(), image_path=f"uploads/{uuid4()}.jpg")


async def test_hard_limit_rejects_publish_on_real_rabbitmq() -> None:
    """Fill the bounded queue without a consumer; the next confirm must fail."""
    with RabbitMqContainer(_RABBIT_IMAGE) as rabbit:
        settings = _overflow_settings(rabbit, max_length=3)
        async with create_broker(settings) as broker:
            await broker.declare_queue(dead_letter_queue(settings))
            await broker.declare_queue(task_queue(settings))
            for _ in range(settings.task_queue_max_length):
                await publish_inference_task(broker, _task(), settings)
            assert await queue_depth(broker, settings.task_queue) == settings.task_queue_max_length
            with pytest.raises(QueueFullError, match="hard limit"):
                await publish_inference_task(broker, _task(), settings)
