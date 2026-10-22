"""HTTP task submission and result lookup."""

import io
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from aio_pika.exceptions import DeliveryError
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from PIL import Image

from src.app import app
from src.schemas import InferenceResult, TaskResult


@pytest.fixture
async def tasks_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[AsyncClient]:
    """App client with fake broker and temporary task storage."""
    monkeypatch.setenv("TASK_STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(
        "src.services.inference.load_model",
        lambda _settings: (Mock(), [], "1"),
    )
    monkeypatch.setattr("src.services.health.probe_mlflow_tracking", lambda _uri: None)

    broker = Mock()
    broker.connect = AsyncMock()
    broker.close = AsyncMock()
    broker.declare_queue = AsyncMock()
    broker.publish = AsyncMock()
    monkeypatch.setattr("src.app.create_broker", lambda _settings: broker)
    monkeypatch.setattr("src.messaging.publishing.queue_depth", AsyncMock(return_value=0))

    pool = Mock()
    pool.execute = AsyncMock()
    pool.fetchrow = AsyncMock(return_value=None)
    monkeypatch.setattr("src.app.db.create_db_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr("src.app.db.close_db_pool", AsyncMock())
    monkeypatch.setattr("src.app.initialize_results", AsyncMock())

    async with LifespanManager(app):
        app.state.broker = broker
        app.state.db_pool = pool
        app.state.settings = app.state.settings.model_copy(
            update={
                "task_storage_dir": tmp_path,
                "task_queue_soft_limit": 2,
                "task_queue_max_length": 5,
            }
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


async def test_post_tasks_returns_202_and_persists_upload(
    tasks_client: AsyncClient, tmp_path: Path
) -> None:
    payload = _png_bytes()
    response = await tasks_client.post(
        "/api/v1/tasks",
        files={"file": ("leaf.png", payload, "image/png")},
    )
    assert response.status_code == 202
    task_id = response.json()["task_id"]
    saved = list(tmp_path.joinpath("uploads").glob(f"{task_id}.*"))
    assert len(saved) == 1
    assert saved[0].read_bytes() == payload
    app.state.broker.publish.assert_awaited_once()


async def test_post_tasks_rejects_unsupported_suffix(tasks_client: AsyncClient) -> None:
    response = await tasks_client.post(
        "/api/v1/tasks",
        files={"file": ("leaf.gif", _png_bytes(), "image/gif")},
    )
    assert response.status_code == 400
    assert "jpg" in response.json()["detail"].lower()
    app.state.broker.publish.assert_not_awaited()


async def test_post_tasks_succeeds_when_soft_limit_reached_but_hard_has_headroom(
    tasks_client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interactive users may use the reserved band above the batch soft watermark."""
    monkeypatch.setattr("src.messaging.publishing.queue_depth", AsyncMock(return_value=2))
    response = await tasks_client.post(
        "/api/v1/tasks",
        files={"file": ("leaf.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 202
    app.state.broker.publish.assert_awaited_once()


async def test_post_tasks_returns_503_without_pending_when_hard_limit_rejects(
    tasks_client: AsyncClient, tmp_path: Path
) -> None:
    app.state.broker.publish = AsyncMock(side_effect=DeliveryError(Mock(), None))
    response = await tasks_client.post(
        "/api/v1/tasks",
        files={"file": ("leaf.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 503
    assert response.headers["retry-after"]
    uploads = tmp_path / "uploads"
    assert not uploads.exists() or not any(uploads.iterdir())


async def test_get_tasks_returns_404_until_result_exists(tasks_client: AsyncClient) -> None:
    missing = await tasks_client.get(f"/api/v1/tasks/{uuid4()}")
    assert missing.status_code == 404

    result = TaskResult(
        task_id=uuid4(),
        prediction=InferenceResult(label="healthy", confidence=0.9, model="m@champion"),
        model_version="1",
        processing_seconds=0.1,
    )
    app.state.db_pool.fetchrow = AsyncMock(return_value={"payload": result.model_dump_json()})
    found = await tasks_client.get(f"/api/v1/tasks/{result.task_id}")
    assert found.status_code == 200
    assert found.json()["task_id"] == str(result.task_id)
    assert found.json()["prediction"]["label"] == "healthy"
