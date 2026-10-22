"""Verify batch publisher waits instead of shedding under queue pressure."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from aio_pika.exceptions import DeliveryError

from src.config import Settings
from src.scripts.batch_publisher import publish_manifest


@pytest.fixture
def settings() -> Settings:
    return Settings(
        rabbitmq_url="amqp://guest:guest@127.0.0.1:59998/",
        publish_retry_delay_seconds=0.01,
        task_queue_soft_limit=2,
        task_queue_max_length=5,
    )


async def test_publish_manifest_retries_after_soft_limit_then_succeeds(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_id = uuid4()
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text(
        json.dumps({"id": str(task_id), "path": "demo/images/x.jpg"}) + "\n",
        encoding="utf-8",
    )
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("src.messaging.publishing.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(
        "src.messaging.publishing.queue_depth",
        AsyncMock(side_effect=[2, 0]),
    )
    broker = Mock()
    broker.publish = AsyncMock()
    report = await publish_manifest(manifest, broker, settings)
    assert report.task_count == 1
    assert broker.publish.await_count == 1
    assert sleeps == [settings.publish_retry_delay_seconds]


async def test_publish_manifest_retries_after_hard_reject_then_succeeds(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_id = uuid4()
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text(
        json.dumps({"id": str(task_id), "path": "demo/images/x.jpg"}) + "\n",
        encoding="utf-8",
    )
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("src.messaging.publishing.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("src.messaging.publishing.queue_depth", AsyncMock(return_value=0))
    broker = Mock()
    broker.publish = AsyncMock(side_effect=[DeliveryError(Mock(), None), None])
    report = await publish_manifest(manifest, broker, settings)
    assert report.task_count == 1
    assert broker.publish.await_count == 2
    assert sleeps == [settings.publish_retry_delay_seconds]
