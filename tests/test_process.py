"""Integration tests for the inference endpoint."""

import io
from collections.abc import AsyncIterator

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from PIL import Image

from src.app import app


async def test_process_returns_prediction(client: AsyncClient) -> None:
    """Verifies that POST /process returns a label from the loaded model."""
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(34, 139, 34)).save(buffer, format="PNG")
    buffer.seek(0)

    response = await client.post(
        "/process",
        files={"file": ("leaf.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "class_0"
    assert body["confidence"] > 0.9
    assert body["model"] == "agrodiagnosis-plant-disease@champion"


@pytest.fixture
async def client_without_model(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    """HTTP client for an app that started without an inference model."""

    def _fail_load_model(settings: object) -> tuple[object, list[str]]:
        del settings
        raise RuntimeError("Failed to load model from models:/x@champion")

    monkeypatch.setattr("src.services.inference.load_model", _fail_load_model)
    monkeypatch.setattr("src.services.health.probe_mlflow_tracking", lambda _uri: None)

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


async def test_process_returns_503_when_model_unavailable(
    client_without_model: AsyncClient,
) -> None:
    """Verifies that POST /process returns HTTP 503 when no model was loaded."""
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(34, 139, 34)).save(buffer, format="PNG")
    buffer.seek(0)

    response = await client_without_model.post(
        "/process",
        files={"file": ("leaf.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Inference model is unavailable"


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
