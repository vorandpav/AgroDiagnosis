"""Unit tests for health service status calculation logic."""

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
    assert "OSError" in (dependency.detail or "")


async def test_build_report_degraded_when_dependency_down() -> None:
    """Verifies that an unhealthy dependency sets the aggregate status to degraded."""
    report = await health_service.build_report(_DeadPool(), Settings())  # type: ignore[arg-type]
    assert report.status is ReportStatus.degraded
    assert report.dependencies["application"].status is HealthStatus.healthy
    assert report.dependencies["postgres"].status is HealthStatus.unavailable


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
