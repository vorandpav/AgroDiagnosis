"""Health status evaluation logic for application dependencies."""

import asyncio
import logging
import urllib.error
import urllib.request

from asyncpg import Pool, PostgresError

from src.config import Settings
from src.schemas import DependencyHealth, HealthReport, HealthStatus, ReportStatus

log = logging.getLogger(__name__)

_MLFLOW_PROBE_TIMEOUT_SEC = 3.0


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
        DependencyHealth: Health status and version or error details.
    """
    try:
        version = await pool.fetchval("SELECT version()")
    except (OSError, PostgresError, TimeoutError) as exc:
        log.warning("Postgres health check failed: %s", exc)
        return DependencyHealth(status=HealthStatus.unavailable, detail="Postgres is unavailable")
    return DependencyHealth(status=HealthStatus.healthy, detail=str(version).split(",")[0])


def probe_mlflow_tracking(tracking_uri: str) -> None:
    """Probe MLflow Tracking Server liveness endpoint.

    Args:
        tracking_uri: Base Tracking Server URI (e.g. ``http://localhost:5000``).

    Raises:
        OSError: Network failure while contacting the server.
        TimeoutError: Probe exceeded the timeout.
        urllib.error.URLError: HTTP client failure.
    """
    url = f"{tracking_uri.rstrip('/')}/health"
    with urllib.request.urlopen(url, timeout=_MLFLOW_PROBE_TIMEOUT_SEC) as response:
        if response.status != 200:
            raise OSError(f"MLflow /health returned HTTP {response.status}")


async def check_mlflow(settings: Settings) -> DependencyHealth:
    """Evaluate reachability of the MLflow Tracking Server.

    Args:
        settings: Application settings with Tracking URI.

    Returns:
        DependencyHealth: Healthy when Tracking ``/health`` responds with HTTP 200.
    """
    tracking_uri = settings.mlflow_tracking_uri
    try:
        await asyncio.to_thread(probe_mlflow_tracking, tracking_uri)
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        log.warning("MLflow health check failed: %s", exc)
        return DependencyHealth(
            status=HealthStatus.unavailable,
            detail="MLflow Tracking Server is unavailable",
        )
    return DependencyHealth(
        status=HealthStatus.healthy,
        detail=f"MLflow Tracking Server reachable at {tracking_uri}",
    )


def check_model(settings: Settings, *, model_loaded: bool) -> DependencyHealth:
    """Evaluate whether the inference model is loaded in application memory.

    Args:
        settings: Application settings with model Registry identity.
        model_loaded: Whether the Registry model was loaded at startup.

    Returns:
        DependencyHealth: Healthy when the model is present in ``app.state``.
    """
    model_ref = f"{settings.mlflow_model_name}@{settings.mlflow_model_alias}"
    if not model_loaded:
        return DependencyHealth(
            status=HealthStatus.unavailable,
            detail=f"Model {model_ref} is not loaded",
        )
    return DependencyHealth(status=HealthStatus.healthy, detail=f"{model_ref} loaded")


async def build_report(
    pool: Pool,
    settings: Settings,
    *,
    model_loaded: bool,
) -> HealthReport:
    """Aggregates health checks across all system dependencies into a single report.

    Args:
        pool: Active asyncpg connection pool.
        settings: Application configuration settings instance.
        model_loaded: Whether the inference model is present in application state.

    Returns:
        HealthReport: Summary report containing statuses for each dependency.
    """
    dependencies = {
        "application": check_application(settings),
        "postgres": await check_postgres(pool),
        "mlflow": await check_mlflow(settings),
        "model": check_model(settings, model_loaded=model_loaded),
    }
    healthy = all(dep.status is HealthStatus.healthy for dep in dependencies.values())
    return HealthReport(
        status=ReportStatus.ok if healthy else ReportStatus.degraded,
        dependencies=dependencies,
    )
