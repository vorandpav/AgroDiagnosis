"""Integration tests for application health check HTTP endpoints."""

from httpx import AsyncClient


async def test_liveness_returns_ok(client: AsyncClient) -> None:
    """Verifies that GET /healthz responds with HTTP 200 without database access."""
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_get_version_returns_current_version(client: AsyncClient) -> None:
    """Verifies that GET /api/v1/version dynamically returns app version."""
    response = await client.get("/api/v1/version")
    assert response.status_code == 200
    assert "version" in response.json()


async def test_health_report_degraded_when_postgres_unavailable(client: AsyncClient) -> None:
    """Verifies that GET /api/v1/health returns HTTP 503 when Postgres is down."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["dependencies"]["postgres"]["status"] == "unavailable"
    assert body["dependencies"]["application"]["status"] == "healthy"
    assert "agrodiagnosis" in body["dependencies"]["application"]["detail"]
