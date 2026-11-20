"""Health status evaluation logic for application dependencies."""

import asyncio
import logging
import urllib.error
import urllib.request

from aio_pika.exceptions import AMQPError
from asyncpg import Pool, PostgresError
from faststream.rabbit import RabbitBroker

from src.config import Settings
from src.messaging.broker import queue_depth
from src.mlflow_runtime import probe_mlflow_tracking
from src.schemas import DependencyHealth, HealthReport, HealthStatus, ReportStatus

log = logging.getLogger(__name__)

_WORKER_PROBE_TIMEOUT_SEC = 3.0
_RABBITMQ_PROBE_ERRORS = (OSError, TimeoutError, RuntimeError, ConnectionError, AMQPError)


def check_application(settings: Settings) -> DependencyHealth:
    """Evaluate the application process metadata as always healthy."""
    detail = f"{settings.app_name} {settings.version} ({settings.environment})"
    return DependencyHealth(status=HealthStatus.healthy, detail=detail)


async def check_postgres(pool: Pool) -> DependencyHealth:
    """Probe the task-results Postgres via the shared pool."""
    try:
        version = await pool.fetchval("SELECT version()")
    except (OSError, PostgresError, TimeoutError) as exc:
        log.warning("Postgres health check failed: %s", exc)
        return DependencyHealth(status=HealthStatus.unavailable, detail="Postgres is unavailable")
    return DependencyHealth(status=HealthStatus.healthy, detail=str(version).split(",")[0])


async def check_mlflow(settings: Settings) -> DependencyHealth:
    """Probe MLflow Tracking Server liveness."""
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


async def check_rabbitmq(broker: RabbitBroker, settings: Settings) -> DependencyHealth:
    """Probe RabbitMQ by passively declaring the inference and evaluation queues."""
    queue_names = (settings.task_queue, settings.evaluate_task_queue)
    depths: list[int] = []
    for queue_name in queue_names:
        try:
            depth = await queue_depth(broker, queue_name)
        except _RABBITMQ_PROBE_ERRORS as exc:
            log.warning("RabbitMQ health check failed: %s", exc)
            return DependencyHealth(
                status=HealthStatus.unavailable,
                detail="RabbitMQ is unavailable",
            )
        depths.append(depth)
    detail = "; ".join(
        f"queue {queue_name} depth={depth}"
        for queue_name, depth in zip(queue_names, depths, strict=True)
    )
    return DependencyHealth(status=HealthStatus.healthy, detail=detail)


def probe_worker_health(health_url: str) -> None:
    """GET the worker /healthz endpoint."""
    with urllib.request.urlopen(health_url, timeout=_WORKER_PROBE_TIMEOUT_SEC):
        pass


async def check_worker(settings: Settings) -> DependencyHealth:
    """Probe inference-worker liveness HTTP endpoint."""
    health_url = settings.worker_health_url
    try:
        await asyncio.to_thread(probe_worker_health, health_url)
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        log.warning("Worker health check failed: %s", exc)
        return DependencyHealth(
            status=HealthStatus.unavailable,
            detail="Inference worker is unavailable",
        )
    return DependencyHealth(
        status=HealthStatus.healthy,
        detail=f"Inference worker reachable at {health_url}",
    )


async def build_report(
    pool: Pool,
    settings: Settings,
    broker: RabbitBroker,
) -> HealthReport:
    """Aggregate shallow dependency checks into one report."""
    dependencies = {
        "application": check_application(settings),
        "postgres": await check_postgres(pool),
        "mlflow": await check_mlflow(settings),
        "rabbitmq": await check_rabbitmq(broker, settings),
        "worker": await check_worker(settings),
    }
    healthy = all(dep.status is HealthStatus.healthy for dep in dependencies.values())
    return HealthReport(
        status=ReportStatus.ok if healthy else ReportStatus.degraded,
        dependencies=dependencies,
    )
