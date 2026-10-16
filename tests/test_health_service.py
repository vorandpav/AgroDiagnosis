"""Unit tests for health service status calculation logic."""

import urllib.error

import pytest

from src.config import Settings
from src.schemas import HealthStatus, ReportStatus
from src.services import health as health_service


class _DeadPool:
    """Mock connection pool simulating database unavailability."""

    async def fetchval(self, query: str) -> None:
        """Simulates connection failure upon query execution."""
        raise OSError("Connection refused")


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


async def test_build_report_degraded_when_dependency_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that an unhealthy dependency sets the aggregate status to degraded."""
    monkeypatch.setattr(health_service, "probe_mlflow_tracking", lambda _uri: None)
    report = await health_service.build_report(
        _DeadPool(),  # type: ignore[arg-type]
        Settings(),
        model_loaded=True,
    )
    assert report.status is ReportStatus.degraded
    assert report.dependencies["application"].status is HealthStatus.healthy
    assert report.dependencies["postgres"].status is HealthStatus.unavailable
    assert report.dependencies["mlflow"].status is HealthStatus.healthy
    assert report.dependencies["model"].status is HealthStatus.healthy


class _LivePool:
    """Mock connection pool simulating a successful PostgreSQL execution."""

    async def fetchval(self, query: str) -> str:
        """Returns a dummy PostgreSQL version string."""
        return "PostgreSQL 16.2, compiled by Visual C++"


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


def test_check_model_healthy_when_loaded() -> None:
    """Verifies model dependency is healthy when the model is in memory."""
    settings = Settings(
        mlflow_model_name="agrodiagnosis-plant-disease",
        mlflow_model_alias="champion",
    )
    dependency = health_service.check_model(settings, model_loaded=True)
    assert dependency.status is HealthStatus.healthy
    assert dependency.detail == "agrodiagnosis-plant-disease@champion loaded"


def test_check_model_unavailable_when_missing() -> None:
    """Verifies model dependency is unavailable when nothing was loaded."""
    settings = Settings(
        mlflow_model_name="agrodiagnosis-plant-disease",
        mlflow_model_alias="champion",
    )
    dependency = health_service.check_model(settings, model_loaded=False)
    assert dependency.status is HealthStatus.unavailable
    assert "agrodiagnosis-plant-disease@champion is not loaded" in (dependency.detail or "")
