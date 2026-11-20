"""Application health and status endpoints."""

import logging

from fastapi import APIRouter, Response, status

from src.api.deps import BrokerDep, PoolDep, SettingsDep
from src.schemas import HealthReport, LivenessResponse, ReportStatus, VersionResponse
from src.services import health as health_service

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/healthz", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    """Return process liveness without dependency checks."""
    return LivenessResponse(status="ok")


@router.get("/api/v1/version", response_model=VersionResponse)
async def get_version(settings: SettingsDep) -> VersionResponse:
    """Return current application version metadata."""
    return VersionResponse(version=settings.version)


@router.get("/api/v1/health", response_model=HealthReport)
async def health(
    pool: PoolDep,
    settings: SettingsDep,
    broker: BrokerDep,
    response: Response,
) -> HealthReport:
    """Return a health report; set 503 when any dependency is degraded."""
    report = await health_service.build_report(pool, settings, broker)
    if report.status is not ReportStatus.ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        log.warning("Health report degraded: %s", report.status.value)
    return report
