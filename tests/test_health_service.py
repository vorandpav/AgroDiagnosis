"""Unit tests for health service status calculation logic."""

import urllib.error
from unittest.mock import AsyncMock, Mock

import pytest
from aio_pika.exceptions import AMQPError, ChannelNotFoundEntity

from src.config import Settings
from src.schemas import HealthStatus, ReportStatus
from src.services import health as health_service


class _DeadPool:
    """Mock connection pool simulating database unavailability."""

    async def fetchval(self, query: str) -> None:
        """Simulates connection failure upon query execution."""
        raise OSError("Connection refused")


class _LivePool:
    """Mock connection pool simulating a successful PostgreSQL execution."""

    async def fetchval(self, query: str) -> str:
        """Returns a dummy PostgreSQL version string."""
        return "PostgreSQL 16.2, compiled by Visual C++"


def test_check_application_healthy() -> None:
    """Verifies that application check returns healthy status with environment details."""
    settings = Settings(app_name="agrodiagnosis", version="9.9.9", environment="test")
    dependency = health_service.check_application(settings)
    assert dependency.status is HealthStatus.healthy
    assert "9.9.9" in (dependency.detail or "")


async def test_check_postgres_unavailable_on_error() -> None:
    """Verifies that connection failures are caught and marked as unavailable."""
    dependency = await health_service.check_postgres(_DeadPool())  # type: ignore[arg-type]
    assert dependency.status is HealthStatus.unavailable
    assert "Postgres is unavailable" in (dependency.detail or "")


async def test_check_postgres_healthy_on_success() -> None:
    """Verifies that successful query execution returns healthy status with version."""
    dependency = await health_service.check_postgres(_LivePool())  # type: ignore[arg-type]
    assert dependency.status is HealthStatus.healthy
    assert "PostgreSQL 16.2" in (dependency.detail or "")


async def test_check_mlflow_healthy_when_tracking_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies MLflow is healthy when Tracking responds."""
    monkeypatch.setattr(health_service, "probe_mlflow_tracking", lambda _uri: None)
    settings = Settings(mlflow_tracking_uri="http://localhost:5000")
    dependency = await health_service.check_mlflow(settings)
    assert dependency.status is HealthStatus.healthy
    assert "http://localhost:5000" in (dependency.detail or "")


async def test_check_mlflow_unavailable_when_tracking_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies MLflow is unavailable when the Tracking Server probe fails."""

    def _raise(_uri: str) -> None:
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(health_service, "probe_mlflow_tracking", _raise)
    dependency = await health_service.check_mlflow(Settings())
    assert dependency.status is HealthStatus.unavailable
    assert "MLflow Tracking Server is unavailable" in (dependency.detail or "")


async def test_check_rabbitmq_healthy_when_both_queues_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth = AsyncMock(side_effect=[3, 4])
    monkeypatch.setattr(health_service, "queue_depth", depth)
    settings = Settings()
    broker = Mock()
    dependency = await health_service.check_rabbitmq(broker, settings)
    assert dependency.status is HealthStatus.healthy
    detail = dependency.detail or ""
    assert f"queue {settings.task_queue} depth=3" in detail
    assert f"queue {settings.evaluate_task_queue} depth=4" in detail
    assert depth.await_args_list[0].args == (broker, settings.task_queue)
    assert depth.await_args_list[1].args == (broker, settings.evaluate_task_queue)


async def test_check_rabbitmq_unavailable_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        health_service,
        "queue_depth",
        AsyncMock(side_effect=ConnectionError("broker down")),
    )
    dependency = await health_service.check_rabbitmq(Mock(), Settings())
    assert dependency.status is HealthStatus.unavailable
    assert "RabbitMQ is unavailable" in (dependency.detail or "")


async def test_check_rabbitmq_unavailable_on_amqp_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        health_service,
        "queue_depth",
        AsyncMock(side_effect=AMQPError("connection closed")),
    )
    dependency = await health_service.check_rabbitmq(Mock(), Settings())
    assert dependency.status is HealthStatus.unavailable
    assert dependency.detail == "RabbitMQ is unavailable"


async def test_check_rabbitmq_unavailable_when_evaluate_queue_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth = AsyncMock(side_effect=[2, ChannelNotFoundEntity("missing evaluate queue")])
    monkeypatch.setattr(health_service, "queue_depth", depth)
    settings = Settings()
    dependency = await health_service.check_rabbitmq(Mock(), settings)
    assert dependency.status is HealthStatus.unavailable
    assert dependency.detail == "RabbitMQ is unavailable"
    assert depth.await_count == 2


async def test_check_worker_healthy_when_healthz_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health_service, "probe_worker_health", lambda _url: None)
    settings = Settings(worker_health_url="http://127.0.0.1:8001/healthz")
    dependency = await health_service.check_worker(settings)
    assert dependency.status is HealthStatus.healthy
    assert "8001/healthz" in (dependency.detail or "")


async def test_check_worker_unavailable_when_healthz_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(_url: str) -> None:
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(health_service, "probe_worker_health", _raise)
    dependency = await health_service.check_worker(Settings())
    assert dependency.status is HealthStatus.unavailable
    assert "Inference worker is unavailable" in (dependency.detail or "")


async def test_build_report_degraded_when_dependency_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that an unhealthy dependency sets the aggregate status to degraded."""
    monkeypatch.setattr(health_service, "probe_mlflow_tracking", lambda _uri: None)
    monkeypatch.setattr(health_service, "queue_depth", AsyncMock(return_value=0))
    monkeypatch.setattr(health_service, "probe_worker_health", lambda _url: None)
    report = await health_service.build_report(
        _DeadPool(),  # type: ignore[arg-type]
        Settings(),
        Mock(),
    )
    assert report.status is ReportStatus.degraded
    assert report.dependencies["application"].status is HealthStatus.healthy
    assert report.dependencies["postgres"].status is HealthStatus.unavailable
    assert report.dependencies["mlflow"].status is HealthStatus.healthy
    assert report.dependencies["rabbitmq"].status is HealthStatus.healthy
    assert report.dependencies["worker"].status is HealthStatus.healthy
