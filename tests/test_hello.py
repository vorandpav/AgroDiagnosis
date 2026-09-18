"""Integration tests for the root entrypoint endpoint."""

from httpx import AsyncClient


async def test_hello_returns_message(client: AsyncClient) -> None:
    """Verifies that GET / responds with HTTP 200 and valid application metadata."""
    response = await client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Hello, world!"
    assert body["app"] == "agrodiagnosis"
    assert "version" in body


async def test_hello_response_has_request_id_header(
    client: AsyncClient,
) -> None:
    """Verifies that the HTTP logging middleware injects the X-Request-ID header."""
    response = await client.get("/")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) == 8
