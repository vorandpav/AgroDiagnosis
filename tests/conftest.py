"""Global pytest fixtures and test environment configuration."""

import os

# Ensure deterministic test execution: point Database URL to an unreachable port
# before importing application modules, as Settings reads ENV on instantiation.
os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@127.0.0.1:59999/agrodiagnosis"

from collections.abc import AsyncIterator

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from src.app import app


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """Provides an asynchronous HTTP test client wrapping the FastAPI application.

    Yields:
        AsyncClient: Configured HTTP client for request simulation.
    """
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            yield http
