"""Application health and status endpoints."""

import logging

from fastapi import APIRouter, Request, Response, status

from src.schemas import HealthReport, LivenessResponse, ReportStatus, VersionResponse
from src.services import health as health_service

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/healthz", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    """Returns application process liveness status.

    Fast response without external dependency checks. Used by container
    orchestrators and health checks to ensure the application process is running.

    Returns:
        LivenessResponse: Status payload indicating the process is alive.
    """
    return LivenessResponse(status="ok")


@router.get("/api/v1/version", response_model=VersionResponse)
async def get_version(request: Request) -> VersionResponse:
    """Returns current application version metadata.

    Dynamically retrieves version information from application settings.

    Args:
        request: FastAPI HTTP request instance.

    Returns:
        VersionResponse: The application version string.
    """
    return VersionResponse(version=request.app.state.settings.version)


@router.get("/api/v1/health", response_model=HealthReport)
async def health(request: Request, response: Response) -> HealthReport:
    """Returns a comprehensive health report for the application and its dependencies.

    Sets HTTP status to 503 Service Unavailable if any required dependency
    is degraded or unreachable.

    Args:
        request: FastAPI HTTP request instance.
        response: FastAPI HTTP response instance.

    Returns:
        HealthReport: Aggregated health report containing all dependency statuses.
    """
    report = await health_service.build_report(
        request.app.state.db_pool, request.app.state.settings
    )
    if report.status is not ReportStatus.ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        log.warning("Health report degraded: %s", report.status.value)
    return report
