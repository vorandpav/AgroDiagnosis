"""Health behaviour when the inference model failed to load at startup."""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from src.app import app


@pytest.fixture
async def client_without_model(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    """HTTP client for an app that started without an inference model."""

    def _fail_load_model(settings: object) -> tuple[object, list[str], str]:
        del settings
        raise RuntimeError("Failed to load model from models:/x@champion")

    monkeypatch.setattr("src.services.inference.load_model", _fail_load_model)
    monkeypatch.setattr("src.services.health.probe_mlflow_tracking", lambda _uri: None)
    broker = Mock()
    broker.connect = AsyncMock()
    broker.close = AsyncMock()
    broker.declare_queue = AsyncMock()
    monkeypatch.setattr("src.app.create_broker", lambda _settings: broker)
    monkeypatch.setattr("src.app.initialize_results", AsyncMock())

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


async def test_health_reports_model_unavailable_when_model_missing(
    client_without_model: AsyncClient,
) -> None:
    """Verifies that /api/v1/health marks model unavailable without a loaded model."""
    response = await client_without_model.get("/api/v1/health")
    assert response.status_code == 503
    body = response.json()
    assert body["dependencies"]["mlflow"]["status"] == "healthy"
    assert body["dependencies"]["model"]["status"] == "unavailable"
    assert "not loaded" in body["dependencies"]["model"]["detail"]
