"""Root entrypoint endpoint providing general application info."""

import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.config import get_settings

log = logging.getLogger(__name__)

router = APIRouter()


class HelloResponse(BaseModel):
    """Response schema for the root endpoint."""

    message: str = Field(default="Hello, world!", description="Welcome greeting message.")
    app: str = Field(description="Name of the application.")
    version: str = Field(description="Application semantic version.")


@router.get("/", response_model=HelloResponse)
async def hello() -> HelloResponse:
    """Returns application greeting and basic metadata.

    Returns:
        HelloResponse: Payload containing welcome message, app name, and version.
    """
    settings = get_settings()
    log.info("Root URL requested.")
    return HelloResponse(
        message="Hello, world!",
        app=settings.app_name,
        version=settings.version,
    )
