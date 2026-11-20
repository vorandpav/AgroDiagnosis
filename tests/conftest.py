"""Global pytest fixtures and test environment configuration."""

import os
from unittest.mock import AsyncMock, Mock

# Ensure deterministic test execution: point Database URL to an unreachable port
# before importing application modules, as Settings reads ENV on instantiation.
os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@127.0.0.1:59999/agrodiagnosis"

from collections.abc import AsyncIterator

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from src.app import app


def _patch_broker_and_results(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Avoid real RabbitMQ and results DDL during app lifespan in tests."""
    broker = Mock()
    broker.connect = AsyncMock()
    broker.close = AsyncMock()
    broker.declare_queue = AsyncMock()
    broker.publish = AsyncMock()
    monkeypatch.setattr("src.app.create_broker", lambda _settings: broker)
    monkeypatch.setattr("src.db.database.Database.declare_table", AsyncMock())
    return broker


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    """Provides an asynchronous HTTP test client wrapping the FastAPI application.

    Yields:
        AsyncClient: Configured HTTP client for request simulation.
    """

    monkeypatch.setattr("src.services.health.probe_mlflow_tracking", lambda _uri: None)
    monkeypatch.setattr("src.services.health.queue_depth", AsyncMock(return_value=0))
    monkeypatch.setattr("src.services.health.probe_worker_health", lambda _url: None)
    _patch_broker_and_results(monkeypatch)

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            yield http
