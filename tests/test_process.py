"""Integration tests for the inference endpoint."""

import io

from httpx import AsyncClient
from PIL import Image


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
