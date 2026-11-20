"""Process-wide OpenTelemetry setup driven only by environment variables."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal

from opentelemetry import context as otel_context
from opentelemetry import metrics, trace
from opentelemetry.context.context import Context
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

_INFERENCE_DURATION_FACTORS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
_API_DURATION_BOUNDARIES: tuple[float, ...] = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
_PROCESS_MEMORY_CONFIG: dict[str, list[str] | None] = {
    "process.runtime.memory": ["rss", "vms"],
}
_IN_FLIGHT = 0

WorkerOutcome = Literal["ok", "error", "dlq"]


def inference_duration_boundaries(inference_min_seconds: float) -> tuple[float, ...]:
    """Return histogram edges at fixed multiples of the inference floor."""
    return tuple(factor * inference_min_seconds for factor in _INFERENCE_DURATION_FACTORS)


def inference_duration_view(inference_min_seconds: float) -> View:
    """Bucket ``worker_inference_duration_seconds`` from the inference floor."""
    return View(
        instrument_name="worker_inference_duration_seconds",
        aggregation=ExplicitBucketHistogramAggregation(
            boundaries=inference_duration_boundaries(inference_min_seconds),
        ),
    )


def api_request_duration_view() -> View:
    """Bucket API latency from tens of milliseconds through a few seconds."""
    return View(
        instrument_name="api_request_duration_seconds",
        aggregation=ExplicitBucketHistogramAggregation(boundaries=_API_DURATION_BOUNDARIES),
    )


def setup_telemetry(inference_min_seconds: float = 1.0) -> None:
    """Install batching OTLP trace and metric providers unless the SDK is disabled.

    ``OTEL_SDK_DISABLED`` is read once here. Exporter endpoint, resource,
    sampler, and batch timeouts come from the standard ``OTEL_*`` variables.

    Args:
        inference_min_seconds: Floor used to place inference histogram buckets.
    """
    if _sdk_disabled():
        return
    _setup_traces()
    _setup_metrics(inference_min_seconds)
    _instrument_asyncpg()
    _instrument_process_memory()


def force_flush_telemetry() -> None:
    """Flush pending spans and metrics before a short-lived process exits."""
    meter_provider = metrics.get_meter_provider()
    if isinstance(meter_provider, MeterProvider):
        meter_provider.force_flush()
    tracer_provider = trace.get_tracer_provider()
    if isinstance(tracer_provider, TracerProvider):
        tracer_provider.force_flush()


def record_api_request(route: str, status_code: int, duration_seconds: float) -> None:
    """Count one HTTP response and record its duration."""
    meter = metrics.get_meter(__name__)
    meter.create_counter(
        "api_requests",
        description="HTTP requests handled by the API",
        unit="1",
    ).add(1, {"route": route, "status": str(status_code)})
    meter.create_histogram(
        "api_request_duration_seconds",
        description="HTTP request duration",
        unit="s",
    ).record(duration_seconds, {"route": route})


def record_published_task() -> None:
    """Count one task the batch publisher accepted into the broker."""
    metrics.get_meter(__name__).create_counter(
        "source_tasks_published",
        description="Tasks published by the batch source",
        unit="1",
    ).add(1)


def record_publish_nack() -> None:
    """Count one broker reject-publish nack of a batch task."""
    metrics.get_meter(__name__).create_counter(
        "source_publish_nacks",
        description="Batch publishes rejected by the broker",
        unit="1",
    ).add(1)


def record_queue_wait_seconds(seconds: float) -> None:
    """Add seconds the batch publisher spent waiting on queue pressure."""
    metrics.get_meter(__name__).create_counter(
        "source_queue_wait_seconds",
        description="Seconds spent waiting for queue capacity",
        unit="s",
    ).add(seconds)


def record_worker_outcome(result: WorkerOutcome) -> None:
    """Count one worker outcome of ok, error, or dlq."""
    metrics.get_meter(__name__).create_counter(
        "worker_task_outcomes",
        description="Inference handler outcomes",
        unit="1",
    ).add(1, {"result": result})


def record_inference_duration(seconds: float) -> None:
    """Record one inference duration, including the configured floor."""
    metrics.get_meter(__name__).create_histogram(
        "worker_inference_duration_seconds",
        description="Time spent in one inference including the configured floor",
        unit="s",
    ).record(seconds)


def record_redelivery() -> None:
    """Count one task delivery after a previous failed attempt."""
    metrics.get_meter(__name__).create_counter(
        "worker_redeliveries",
        description="Tasks delivered again after a failed attempt",
        unit="1",
    ).add(1)


def adjust_tasks_in_flight(delta: int) -> None:
    """Move the in-flight gauge by delta and publish the new value."""
    global _IN_FLIGHT
    _IN_FLIGHT += delta
    metrics.get_meter(__name__).create_gauge(
        "worker_tasks_in_flight",
        description="Tasks currently handled by this worker",
        unit="1",
    ).set(_IN_FLIGHT)


def call_in_context[T](parent: Context, work: Callable[[], T]) -> T:
    """Run work with parent attached so child spans stay on that trace."""
    token = otel_context.attach(parent)
    try:
        return work()
    finally:
        otel_context.detach(token)


def record_span_exception(exc: BaseException) -> None:
    """Mark the current span failed and attach the exception event."""
    span = trace.get_current_span()
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


@contextmanager
def one_task_trace() -> Iterator[None]:
    """Open a root span so this task is not folded into a batch trace."""
    token = otel_context.attach(Context())
    try:
        with trace.get_tracer(__name__).start_as_current_span("publish_task"):
            yield
    finally:
        otel_context.detach(token)


@contextmanager
def queue_wait_span() -> Iterator[None]:
    """Record one broker reject and the following sleep on this task."""
    with trace.get_tracer(__name__).start_as_current_span("queue_wait") as span:
        span.add_event("queue_reject")
        yield


def _sdk_disabled() -> bool:
    """Return whether the OpenTelemetry SDK is switched off."""
    return os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true"


def _setup_traces() -> None:
    """Install the OTLP tracer provider once for this process."""
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        return
    provider = TracerProvider(resource=Resource.create())
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)


def _setup_metrics(inference_min_seconds: float) -> None:
    """Install the OTLP meter provider and explicit histogram views."""
    if isinstance(metrics.get_meter_provider(), MeterProvider):
        return
    provider = MeterProvider(
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
        resource=Resource.create(),
        views=[
            inference_duration_view(inference_min_seconds),
            api_request_duration_view(),
        ],
    )
    metrics.set_meter_provider(provider)


def _instrument_process_memory() -> None:
    """Export process memory via the system-metrics instrumentor."""
    instrumentor = SystemMetricsInstrumentor(config=_PROCESS_MEMORY_CONFIG)
    if instrumentor.is_instrumented_by_opentelemetry:
        return
    try:
        instrumentor.instrument()
    except Exception as exc:
        raise RuntimeError("Failed to instrument process memory") from exc


def _instrument_asyncpg() -> None:
    """Instrument asyncpg once for this process."""
    instrumentor = AsyncPGInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        return
    try:
        instrumentor.instrument()
    except Exception as exc:
        raise RuntimeError("Failed to instrument asyncpg") from exc
