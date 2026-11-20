"""Health behaviour when inference runs only in workers."""

from httpx import AsyncClient


async def test_health_report_excludes_api_model_dependency(client: AsyncClient) -> None:
    """Verifies that /api/v1/health does not probe an API-side inference model."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 503
    body = response.json()
    assert body["dependencies"]["mlflow"]["status"] == "healthy"
    assert body["dependencies"]["rabbitmq"]["status"] == "healthy"
    assert body["dependencies"]["worker"]["status"] == "healthy"
    assert "model" not in body["dependencies"]
