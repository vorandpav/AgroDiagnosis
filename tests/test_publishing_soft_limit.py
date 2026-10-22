"""Soft watermark applies to batch publishing, not interactive HTTP publishes."""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.config import Settings
from src.exceptions import QueueFullError
from src.schemas import InferenceTask
from src.messaging.publishing import (
    publish_inference_task,
    publish_inference_task_with_backoff,
)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        task_queue_soft_limit=3,
        task_queue_max_length=10,
        publish_retry_delay_seconds=0.01,
    )


async def test_user_publish_ignores_soft_limit(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.messaging.publishing.queue_depth", AsyncMock(return_value=3))
    broker = Mock()
    broker.publish = AsyncMock()
    task = InferenceTask(task_id=uuid4(), image_path="uploads/x.jpg")
    await publish_inference_task(broker, task, settings)
    broker.publish.assert_awaited_once()


async def test_batch_publish_waits_at_soft_limit_then_succeeds(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("src.messaging.publishing.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(
        "src.messaging.publishing.queue_depth",
        AsyncMock(side_effect=[3, 0]),
    )
    broker = Mock()
    broker.publish = AsyncMock()
    task = InferenceTask(task_id=uuid4(), image_path="demo/images/x.jpg")
    await publish_inference_task_with_backoff(broker, task, settings)
    assert broker.publish.await_count == 1
    assert sleeps == [settings.publish_retry_delay_seconds]


async def test_batch_publish_raises_soft_limit_before_publish(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.messaging.publishing.queue_depth", AsyncMock(return_value=3))
    monkeypatch.setattr(
        "src.messaging.publishing.asyncio.sleep", AsyncMock(side_effect=QueueFullError("stop"))
    )
    broker = Mock()
    broker.publish = AsyncMock()
    task = InferenceTask(task_id=uuid4(), image_path="demo/images/x.jpg")
    with pytest.raises(QueueFullError, match="stop"):
        await publish_inference_task_with_backoff(broker, task, settings)
    broker.publish.assert_not_awaited()
