"""Health status evaluation logic for application dependencies."""

import logging

from asyncpg import Pool, PostgresError

from src.config import Settings
from src.schemas import DependencyHealth, HealthReport, HealthStatus, ReportStatus

log = logging.getLogger(__name__)


def check_application(settings: Settings) -> DependencyHealth:
    """Evaluates the application health state.

    Args:
        settings: Application settings instance containing metadata.

    Returns:
        DependencyHealth: Health details for the core application process.
    """
    detail = f"{settings.app_name} {settings.version} ({settings.environment})"
    return DependencyHealth(status=HealthStatus.healthy, detail=detail)


async def check_postgres(pool: Pool) -> DependencyHealth:
    """Evaluates PostgreSQL connection health by querying the server version.

    Captures execution or network errors and transforms them into an
    `unavailable` health status rather than raising an unhandled exception.

    Args:
        pool: Active asyncpg connection pool.

    Returns:
        DependencyHealth: Health status and version details (or error description).
    """
    try:
        version = await pool.fetchval("SELECT version()")
    except (OSError, PostgresError, TimeoutError) as exc:
        detail = f"{type(exc).__name__}: {exc}"
        return DependencyHealth(status=HealthStatus.unavailable, detail=detail)
    return DependencyHealth(status=HealthStatus.healthy, detail=str(version).split(",")[0])


async def build_report(pool: Pool, settings: Settings) -> HealthReport:
    """Aggregates health checks across all system dependencies into a single report.

    Args:
        pool: Active asyncpg connection pool.
        settings: Application configuration settings instance.

    Returns:
        HealthReport: Summary report containing statuses for each dependency.
    """
    dependencies = {
        "application": check_application(settings),
        "postgres": await check_postgres(pool),
    }
    healthy = all(dep.status is HealthStatus.healthy for dep in dependencies.values())
    return HealthReport(
        status=ReportStatus.ok if healthy else ReportStatus.degraded,
        dependencies=dependencies,
    )
