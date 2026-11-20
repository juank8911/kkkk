"""Pruebas de humo del baseline S0-04 (pytest-asyncio + fakeredis + httpx)."""

from __future__ import annotations

import fakeredis.aioredis
import httpx


async def test_health_smoke_harness(health_harness_client: httpx.AsyncClient) -> None:
    """Valida que el harness de pruebas responde 200 en /health respaldado por fakeredis."""
    response = await health_harness_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["redis"] == "ok"


async def test_openclaw_app_health_smoke(api_client: httpx.AsyncClient) -> None:
    """Criterio S0-04: prueba de humo asíncrona con httpx validando /health de la app real."""
    response = await api_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


async def test_fakeredis_streams_smoke(fake_redis: fakeredis.aioredis.FakeRedis) -> None:
    """Verifica que fakeredis soporta las operaciones básicas de Streams usadas por OpenClaw."""
    msg_id = await fake_redis.xadd("openclaw_smoke_stream", {"payload": '{"ping":"pong"}'})
    assert msg_id is not None
    entries = await fake_redis.xrange("openclaw_smoke_stream", "-", "+")
    assert len(entries) == 1
    assert entries[0][1][b"payload"] == b'{"ping":"pong"}'


async def test_openclaw_app_openapi_smoke(api_client: httpx.AsyncClient) -> None:
    """Verifica que la aplicación FastAPI levanta y publica su esquema OpenAPI."""
    response = await api_client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "OpenClaw API Gateway"
