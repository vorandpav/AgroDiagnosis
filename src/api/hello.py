"""Root entrypoint endpoint providing general application info."""

import logging

from fastapi import APIRouter

from src.api.deps import SettingsDep
from src.schemas import HelloResponse

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=HelloResponse)
async def hello(settings: SettingsDep) -> HelloResponse:
    """Return application greeting and basic metadata."""
    log.info("Root URL requested.")
    return HelloResponse(
        message="Hello, world!",
        app=settings.app_name,
        version=settings.version,
    )
