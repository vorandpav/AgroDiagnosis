"""Application API Pydantic contracts and response schemas."""

from enum import StrEnum

from pydantic import BaseModel, Field


class HelloResponse(BaseModel):
    """Response schema for the root endpoint."""

    message: str = Field(default="Hello, world!", description="Welcome greeting message.")
    app: str = Field(description="Name of the application.")
    version: str = Field(description="Application semantic version.")


class LivenessResponse(BaseModel):
    """Response model for the lightweight liveness probe."""

    status: str = Field(default="ok", description="Process liveness indicator.")


class VersionResponse(BaseModel):
    """Response model for the application version endpoint."""

    version: str = Field(description="Application version.")


class HealthStatus(StrEnum):
    """Health status of an individual application dependency."""

    healthy = "healthy"
    unavailable = "unavailable"


class ReportStatus(StrEnum):
    """Overall status of the application in the health report."""

    ok = "ok"
    degraded = "degraded"


class DependencyHealth(BaseModel):
    """Health status and details of a single dependency."""

    status: HealthStatus
    detail: str | None = Field(default=None, description="Status detail or error message.")


class HealthReport(BaseModel):
    """Summary health report encompassing all application dependencies."""

    status: ReportStatus
    dependencies: dict[str, DependencyHealth]
