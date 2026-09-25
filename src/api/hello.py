"""Root entrypoint endpoint providing general application info."""

import logging

from fastapi import APIRouter

from src.config import get_settings
from src.schemas import HelloResponse

log = logging.getLogger(__name__)

router = APIRouter()


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
