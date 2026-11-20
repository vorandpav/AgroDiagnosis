"""In-memory metrics for the API, batch source, and inference worker."""

from __future__ import annotations

import io
import os
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from aio_pika.exceptions import DeliveryError
from asgi_lifespan import LifespanManager
from faststream.rabbit import TestRabbitBroker
from httpx import ASGITransport, AsyncClient
from opentelemetry.metrics import _internal as metrics_internal
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics._internal.exemplar.exemplar_filter import AlwaysOffExemplarFilter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, MetricsData
from PIL import Image

from src.app import app
from src.config import Settings
from src.inference_worker.consumer import recover_failed_delivery, register_inference_handler
from src.inference_worker.processing import process_task
from src.messaging.broker import create_broker, task_queue
from src.messaging.publishing import publish_inference_task_with_backoff
from src.schemas import InferenceResult, InferenceTask, TaskResult
from src.telemetry import (
    api_request_duration_view,
    inference_duration_boundaries,
    inference_duration_view,
)


def _install_meter_provider(provider: MeterProvider | None) -> None:
    """Replace the process meter provider so each test collects its own series."""
    metrics_internal._METER_PROVIDER_SET_ONCE._done = False
    metrics_internal._METER_PROVIDER = None
    if provider is not None:
        metrics_internal.set_meter_provider(provider)


@contextmanager
def _collecting(inference_min_seconds: float) -> Iterator[InMemoryMetricReader]:
    """Install an in-memory reader with the same histogram views as the app."""
    previous_disabled = os.environ.pop("OTEL_SDK_DISABLED", None)
    reader = InMemoryMetricReader()
    provider = MeterProvider(
        metric_readers=[reader],
        shutdown_on_exit=False,
        exemplar_filter=AlwaysOffExemplarFilter(),
        views=[
            inference_duration_view(inference_min_seconds),
            api_request_duration_view(),
        ],
    )
    _install_meter_provider(provider)
    try:
        yield reader
    finally:
        provider.shutdown()
        _install_meter_provider(None)
        if previous_disabled is None:
            os.environ.pop("OTEL_SDK_DISABLED", None)
        else:
            os.environ["OTEL_SDK_DISABLED"] = previous_disabled


def _metrics_data(reader: InMemoryMetricReader) -> MetricsData:
    data = reader.get_metrics_data()
    assert data is not None
    return data


def _assert_no_identity_attributes(data: MetricsData) -> None:
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                for point in metric.data.data_points:
                    attributes: Mapping[str, object] = point.attributes or {}
                    assert "task_id" not in attributes
                    assert "trace_id" not in attributes


def _counter_values(data: MetricsData, name: str) -> dict[tuple[tuple[str, str], ...], int | float]:
    values: dict[tuple[tuple[str, str], ...], int | float] = {}
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name != name:
                    continue
                for point in metric.data.data_points:
                    attributes = point.attributes or {}
                    key = tuple(sorted((str(k), str(v)) for k, v in attributes.items()))
                    values[key] = point.value
    return values


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


def _task() -> InferenceTask:
    return InferenceTask(task_id=uuid4(), image_path="leaf.jpg")


@pytest.fixture
async def tasks_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[AsyncClient]:
    """App client with a fake broker and temporary task storage."""
    monkeypatch.setenv("TASK_STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr("src.services.health.probe_mlflow_tracking", lambda _uri: None)
    broker = Mock()
    broker.connect = AsyncMock()
    broker.close = AsyncMock()
    broker.declare_queue = AsyncMock()
    broker.publish = AsyncMock()
    monkeypatch.setattr("src.app.create_broker", lambda _settings: broker)
    monkeypatch.setattr("src.messaging.publishing.queue_depth", AsyncMock(return_value=0))
    pool = Mock()
    pool.execute = AsyncMock()
    pool.fetchrow = AsyncMock(return_value=None)
    database = Mock()
    database.pool = pool
    database.declare_table = AsyncMock()
    database.close = AsyncMock()
    monkeypatch.setattr("src.app.create_database", AsyncMock(return_value=database))
    async with LifespanManager(app):
        app.state.broker = broker
        app.state.db_pool = pool
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


async def test_request_counter_counts_202_and_503(tasks_client: AsyncClient) -> None:
    with _collecting(1.0) as reader:
        accepted = await tasks_client.post(
            "/api/v1/tasks",
            files={"file": ("leaf.png", _png_bytes(), "image/png")},
        )
        app.state.broker.publish = AsyncMock(side_effect=DeliveryError(Mock(), None))
        rejected = await tasks_client.post(
            "/api/v1/tasks",
            files={"file": ("leaf.png", _png_bytes(), "image/png")},
        )
        data = _metrics_data(reader)
    assert accepted.status_code == 202
    assert rejected.status_code == 503
    values = _counter_values(data, "api_requests")
    assert values[(("route", "/api/v1/tasks"), ("status", "202"))] == 1
    assert values[(("route", "/api/v1/tasks"), ("status", "503"))] == 1
    _assert_no_identity_attributes(data)


def test_inference_histogram_uses_multiples_of_min_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    minimum = 2.0
    settings = Settings(task_storage_dir=tmp_path, inference_min_seconds=minimum)
    (tmp_path / "leaf.jpg").write_bytes(b"image")
    monkeypatch.setattr(
        "src.inference_worker.processing.predict",
        lambda _model, _names, _data: InferenceResult(label="Potato/Early_blight", confidence=0.9),
    )
    monkeypatch.setattr("src.inference_worker.processing.time.sleep", lambda _seconds: None)
    with _collecting(minimum) as reader:
        process_task(_task(), settings, Mock(), ["Potato/Early_blight"], "models:/plants/1", "1")
        data = _metrics_data(reader)
    point = next(
        point
        for resource_metrics in data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
        if metric.name == "worker_inference_duration_seconds"
        for point in metric.data.data_points
    )
    assert list(point.explicit_bounds) == list(inference_duration_boundaries(minimum))
    assert point.count == 1
    _assert_no_identity_attributes(data)


async def test_worker_outcomes_are_only_ok_error_or_dlq(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(task_max_delivery_attempts=2, task_retry_delay_seconds=0.01)
    monkeypatch.setattr("src.inference_worker.consumer.asyncio.sleep", AsyncMock())
    task = _task()
    result = TaskResult(
        task_id=task.task_id,
        prediction=InferenceResult(label="Potato/Early_blight", confidence=0.9),
        model_ref="models:/plants/1",
        model_version="1",
        processing_seconds=0.1,
    )
    broker = create_broker(settings)
    register_inference_handler(broker, settings, AsyncMock(return_value=result), AsyncMock())
    failed = Mock()
    failed.publish = AsyncMock()
    message = Mock()
    message.ack = AsyncMock()
    message.reject = AsyncMock()
    with _collecting(settings.inference_min_seconds) as reader:
        async with TestRabbitBroker(broker):
            await broker.publish(task, queue=task_queue(settings))
        await recover_failed_delivery(
            failed, settings, task, message, attempt=0, queue=task_queue(settings)
        )
        await recover_failed_delivery(
            failed, settings, task, message, attempt=1, queue=task_queue(settings)
        )
        data = _metrics_data(reader)
    values = _counter_values(data, "worker_task_outcomes")
    assert set(values) == {
        (("result", "ok"),),
        (("result", "error"),),
        (("result", "dlq"),),
    }
    assert all(value == 1 for value in values.values())
    _assert_no_identity_attributes(data)


async def test_published_batch_wait_adds_wait_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        task_queue_soft_limit=1,
        task_queue_max_length=5,
        publish_retry_delay_seconds=0.2,
    )
    monkeypatch.setattr("src.messaging.publishing.asyncio.sleep", AsyncMock())
    monkeypatch.setattr(
        "src.messaging.publishing.queue_depth",
        AsyncMock(side_effect=[1, 0]),
    )
    broker = Mock()
    broker.publish = AsyncMock()
    with _collecting(settings.inference_min_seconds) as reader:
        await publish_inference_task_with_backoff(broker, _task(), settings)
        data = _metrics_data(reader)
    values = _counter_values(data, "source_queue_wait_seconds")
    assert values[()] == pytest.approx(settings.publish_retry_delay_seconds)
    _assert_no_identity_attributes(data)
