"""Global pytest fixtures and test environment configuration."""

import os

# Ensure deterministic test execution: point Database URL to an unreachable port
# before importing application modules, as Settings reads ENV on instantiation.
os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@127.0.0.1:59999/agrodiagnosis"

from collections.abc import AsyncIterator

import pytest
import torch
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from torch import nn

from src.app import app


class _FakeModel(nn.Module):
    """Minimal stand-in so tests never contact MLflow or download weights."""

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(batch.shape[0], 38)
        logits[:, 0] = 10.0
        return logits


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    """Provides an asynchronous HTTP test client wrapping the FastAPI application.

    Yields:
        AsyncClient: Configured HTTP client for request simulation.
    """

    def _fake_load_model(settings: object) -> tuple[nn.Module, list[str], str]:
        del settings
        return _FakeModel().eval(), [f"class_{index}" for index in range(38)], "1"

    monkeypatch.setattr("src.services.inference.load_model", _fake_load_model)
    monkeypatch.setattr("src.services.health.probe_mlflow_tracking", lambda _uri: None)

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            yield http
