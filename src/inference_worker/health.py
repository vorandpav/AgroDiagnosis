"""Lightweight HTTP liveness for the inference worker process."""

from __future__ import annotations

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route


async def healthz(_request: Request) -> PlainTextResponse:
    """Return ok while the inference worker event loop is responsive."""
    return PlainTextResponse("ok")


def build_health_app() -> Starlette:
    """Build a minimal ASGI app used only for liveness under load."""
    return Starlette(routes=[Route("/healthz", healthz)])


def create_health_server(host: str, port: int) -> uvicorn.Server:
    """Create a uvicorn server for the inference worker health app."""
    config = uvicorn.Config(
        build_health_app(),
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    return uvicorn.Server(config)
