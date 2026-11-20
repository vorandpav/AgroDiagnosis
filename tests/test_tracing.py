"""In-memory distributed tracing: one task continues across publish, consume, and threads."""

from __future__ import annotations

import io
import json
import logging
import os
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from aio_pika.exceptions import DeliveryError
from fastapi import UploadFile
from faststream.rabbit import RabbitMessage, TestRabbitBroker
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ReadableSpan,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import ProxyTracerProvider, StatusCode

from src.api.tasks import create_task
from src.config import Settings
from src.inference_worker.consumer import recover_failed_delivery, register_inference_handler
from src.inference_worker.processing import process_task
from src.logging_config import setup_logging
from src.messaging.broker import create_broker, task_queue
from src.messaging.publishing import publish_inference_task_with_backoff
from src.schemas import InferenceResult, InferenceTask, TaskResult
from src.telemetry import call_in_context, setup_telemetry


class _FailingExporter(SpanExporter):
    """Raise on export the way a down collector does."""

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        raise OSError("collector unavailable")

    def shutdown(self) -> None:
        return None


def _install_provider(provider: trace.TracerProvider | None) -> None:
    """Install a provider, or clear the global one back to the default proxy."""
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    if provider is None or isinstance(provider, ProxyTracerProvider):
        trace._TRACER_PROVIDER = None
        return
    trace.set_tracer_provider(provider)


@pytest.fixture
def memory_exporter() -> Iterator[InMemorySpanExporter]:
    """Install a span exporter for one test and put the global provider back."""
    previous_disabled = os.environ.pop("OTEL_SDK_DISABLED", None)
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous = trace.get_tracer_provider()
    _install_provider(provider)
    try:
        yield exporter
    finally:
        provider.shutdown()
        _install_provider(previous)
        if previous_disabled is None:
            os.environ.pop("OTEL_SDK_DISABLED", None)
        else:
            os.environ["OTEL_SDK_DISABLED"] = previous_disabled


def _trace_id(span: ReadableSpan) -> str:
    return f"{span.context.trace_id:032x}"


def _png_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


def _task() -> InferenceTask:
    return InferenceTask(task_id=uuid4(), image_path="leaf.jpg")


def _result(task: InferenceTask) -> TaskResult:
    return TaskResult(
        task_id=task.task_id,
        prediction=InferenceResult(label="healthy", confidence=0.9),
        model_ref="models:/plants/7",
        model_version="7",
        processing_seconds=0.1,
    )


def _header(headers: object, name: str) -> str:
    if not hasattr(headers, "get"):
        raise AssertionError(headers)
    raw = headers.get(name)
    if isinstance(raw, bytes):
        return raw.decode()
    if isinstance(raw, str):
        return raw
    raise AssertionError(headers)


async def test_publish_inside_current_span_puts_matching_traceparent(
    memory_exporter: InMemorySpanExporter,
) -> None:
    settings = Settings()
    broker = create_broker(settings)
    seen: dict[str, str] = {}
    task = _task()

    @broker.subscriber(task_queue(settings))
    async def capture(message: RabbitMessage) -> None:
        seen["traceparent"] = _header(message.headers, "traceparent")

    tracer = trace.get_tracer("tests.tracing")
    async with TestRabbitBroker(broker):
        with tracer.start_as_current_span("parent") as parent:
            trace_id = f"{parent.get_span_context().trace_id:032x}"
            await broker.publish(task, queue=task_queue(settings))
    assert seen["traceparent"].split("-")[1] == trace_id
    assert memory_exporter.get_finished_spans()


async def test_consumer_span_continues_traceparent_and_is_not_a_new_root(
    memory_exporter: InMemorySpanExporter,
) -> None:
    settings = Settings()
    broker = create_broker(settings)
    task = _task()

    @broker.subscriber(task_queue(settings))
    async def capture(message: RabbitMessage) -> None:
        return None

    tracer = trace.get_tracer("tests.tracing")
    async with TestRabbitBroker(broker):
        with tracer.start_as_current_span("parent") as parent:
            trace_id = f"{parent.get_span_context().trace_id:032x}"
            await broker.publish(task, queue=task_queue(settings))
    process = next(
        span for span in memory_exporter.get_finished_spans() if span.name.endswith(" process")
    )
    assert _trace_id(process) == trace_id
    assert process.parent is not None
    assert process.parent.trace_id == process.context.trace_id
    assert process.parent.is_valid


def test_inference_span_in_worker_thread_is_child_of_process_span(
    memory_exporter: InMemorySpanExporter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        task_storage_dir=tmp_path,
        inference_min_seconds=0,
        mlflow_model_alias="champion",
    )
    (tmp_path / "leaf.jpg").write_bytes(b"image")
    task = _task()
    monkeypatch.setattr(
        "src.inference_worker.processing.predict",
        lambda _model, _names, _data: InferenceResult(label="healthy", confidence=0.9),
    )
    monkeypatch.setattr("src.inference_worker.processing.time.sleep", lambda _seconds: None)
    tracer = trace.get_tracer("tests.tracing")
    with tracer.start_as_current_span("process") as process:
        parent = otel_context.get_current()

        def run() -> None:
            call_in_context(
                parent,
                lambda: process_task(task, settings, Mock(), ["healthy"], "models:/plants/7", "7"),
            )

        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        process_context = process.get_span_context()
    inference = next(
        span for span in memory_exporter.get_finished_spans() if span.name == "inference"
    )
    assert inference.context.trace_id == process_context.trace_id
    assert inference.parent is not None
    assert inference.parent.span_id == process_context.span_id
    assert inference.attributes["alias"] == "champion"
    assert inference.attributes["model_version"] == "7"


async def test_save_result_span_is_child_of_process_span(
    memory_exporter: InMemorySpanExporter,
) -> None:
    settings = Settings()
    task = _task()
    broker = create_broker(settings)
    register_inference_handler(broker, settings, AsyncMock(return_value=_result(task)), AsyncMock())
    async with TestRabbitBroker(broker):
        await broker.publish(task, queue=task_queue(settings))
    spans = memory_exporter.get_finished_spans()
    process = next(span for span in spans if span.name.endswith(" process"))
    saved = next(span for span in spans if span.name == "save_result")
    assert saved.context.trace_id == process.context.trace_id
    assert saved.parent is not None
    assert saved.parent.span_id == process.context.span_id


async def test_queue_full_response_sets_current_span_error(
    memory_exporter: InMemorySpanExporter,
    tmp_path: Path,
) -> None:
    settings = Settings(task_storage_dir=tmp_path)
    broker = Mock()
    broker.publish = AsyncMock(side_effect=DeliveryError(Mock(), None))
    upload = UploadFile(io.BytesIO(_png_bytes()), filename="leaf.png")
    tracer = trace.get_tracer("tests.tracing")
    with tracer.start_as_current_span("POST /api/v1/tasks") as span:
        response = await create_task(upload, settings, broker)
    assert response.status_code == 503
    assert span.status.status_code is StatusCode.ERROR
    assert any(event.name == "exception" for event in span.events)
    assert memory_exporter.get_finished_spans()


def test_log_record_inside_span_contains_trace_and_span_ids(
    memory_exporter: InMemorySpanExporter,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = logging.getLogger()
    previous_handlers = root.handlers[:]
    previous_level = root.level
    setup_logging("INFO")
    try:
        tracer = trace.get_tracer("tests.tracing")
        with tracer.start_as_current_span("work") as span:
            logging.getLogger("tests.tracing.logs").info("inside-span")
            trace_id = f"{span.get_span_context().trace_id:032x}"
            span_id = f"{span.get_span_context().span_id:016x}"
        payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert payload["trace_id"] == trace_id
        assert payload["span_id"] == span_id
        assert payload["message"] == "inside-span"
    finally:
        root.handlers = previous_handlers
        root.setLevel(previous_level)
    assert memory_exporter.get_finished_spans()


async def test_queue_wait_span_is_child_of_that_task_trace(
    memory_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(task_queue_soft_limit=1, publish_retry_delay_seconds=0.01)
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("src.messaging.publishing.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(
        "src.messaging.publishing.queue_depth",
        AsyncMock(side_effect=[1, 0]),
    )
    broker = Mock()
    broker.publish = AsyncMock()
    await publish_inference_task_with_backoff(broker, _task(), settings)
    spans = memory_exporter.get_finished_spans()
    root = next(span for span in spans if span.name == "publish_task")
    waited = next(span for span in spans if span.name == "queue_wait")
    assert waited.context.trace_id == root.context.trace_id
    assert waited.parent is not None
    assert waited.parent.span_id == root.context.span_id
    assert any(event.name == "queue_reject" for event in waited.events)
    assert sleeps == [settings.publish_retry_delay_seconds]


async def test_republish_copies_traceparent_and_keeps_task_headers(
    memory_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(task_max_delivery_attempts=3)
    monkeypatch.setattr("src.inference_worker.consumer.asyncio.sleep", AsyncMock())
    task = _task()
    broker = Mock()
    broker.publish = AsyncMock()
    message = Mock()
    message.ack = AsyncMock()
    message.reject = AsyncMock()
    tracer = trace.get_tracer("tests.tracing")
    with tracer.start_as_current_span("process") as span:
        await recover_failed_delivery(
            broker, settings, task, message, attempt=0, queue=task_queue(settings)
        )
        trace_id = f"{span.get_span_context().trace_id:032x}"
    headers = broker.publish.await_args.kwargs["headers"]
    assert headers["x-delivery-attempt"] == 1
    assert headers["x-task-id"] == str(task.task_id)
    assert headers["traceparent"].split("-")[1] == trace_id
    assert broker.publish.await_args.kwargs["correlation_id"] == str(task.task_id)
    assert memory_exporter.get_finished_spans()


def test_setup_telemetry_disabled_installs_no_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    previous = trace.get_tracer_provider()
    _install_provider(None)
    try:
        setup_telemetry()
        assert not isinstance(trace.get_tracer_provider(), TracerProvider)
    finally:
        _install_provider(previous)


def test_export_failure_does_not_raise_from_the_request_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(BatchSpanProcessor(_FailingExporter()))
    tracer = provider.get_tracer("tests.tracing")
    with tracer.start_as_current_span("request"):
        pass
    assert provider.force_flush(timeout_millis=1000) is not None
    provider.shutdown()
