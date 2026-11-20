"""Soft watermark applies to batch publishing, not interactive HTTP publishes."""

import asyncio
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.config import Settings
from src.db.task_results import wait_for_task_results
from src.exceptions import QueueFullError
from src.messaging.publishing import (
    publish_evaluate_task_with_backoff,
    publish_inference_task,
    publish_inference_task_with_backoff,
)
from src.schemas import InferenceTask
from src.services.batch_processing import batch_process_evaluate


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


async def test_evaluate_publish_stops_when_deadline_already_passed(settings: Settings) -> None:
    broker = Mock()
    broker.publish = AsyncMock()
    task = InferenceTask(task_id=uuid4(), image_path="demo/images/x.jpg")
    with pytest.raises(TimeoutError, match="Evaluation timed out after 120"):
        await publish_evaluate_task_with_backoff(broker, task, settings, deadline=0)
    broker.publish.assert_not_awaited()


async def test_evaluate_publish_retries_only_until_deadline(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.evaluation_timeout_seconds = 0.15
    settings.publish_retry_delay_seconds = 0.01
    settings.task_queue_soft_limit = 1
    monkeypatch.setattr("src.messaging.publishing.queue_depth", AsyncMock(return_value=1))
    broker = Mock()
    broker.publish = AsyncMock()
    task = InferenceTask(task_id=uuid4(), image_path="demo/images/x.jpg")
    deadline = asyncio.get_running_loop().time() + settings.evaluation_timeout_seconds
    with pytest.raises(TimeoutError, match=r"Evaluation timed out after 0\.15"):
        await asyncio.wait_for(
            publish_evaluate_task_with_backoff(broker, task, settings, deadline=deadline),
            timeout=2,
        )
    broker.publish.assert_not_awaited()


async def test_wait_for_task_results_stops_at_shared_deadline() -> None:
    pool = Mock()
    pool.fetchval = AsyncMock(return_value=0)
    pool.fetch = AsyncMock(return_value=[])
    with pytest.raises(TimeoutError, match="Evaluation timed out after 120"):
        await wait_for_task_results(pool, [uuid4()], timeout_seconds=120, deadline=0)
    pool.fetch.assert_not_awaited()
    pool.fetchval.assert_awaited_once()


async def test_batch_process_evaluate_bounds_result_wait(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.evaluation_timeout_seconds = 0.2
    monkeypatch.setattr("src.messaging.publishing.queue_depth", AsyncMock(return_value=0))
    broker = Mock()
    broker.publish = AsyncMock()
    pool = Mock()
    pool.fetchval = AsyncMock(return_value=0)
    pool.fetch = AsyncMock(return_value=[])
    task = InferenceTask(task_id=uuid4(), image_path="demo/images/x.jpg")
    with pytest.raises(TimeoutError, match=r"Evaluation timed out after 0\.2"):
        await asyncio.wait_for(
            batch_process_evaluate([task], broker, pool, settings),
            timeout=2,
        )
    broker.publish.assert_awaited()
    pool.fetch.assert_not_awaited()
    assert pool.fetchval.await_count >= 1
