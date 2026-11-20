"""Baseline compartido de pruebas: TestClient, pytest-asyncio, fakeredis y httpx (S0-04)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import fakeredis.aioredis
import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.main import app as openclaw_app
from app.core.config import settings


class MutableClock:
    """Reloj determinista para pruebas de ventana deslizante sin sleeps reales."""

    def __init__(self, initial: float = 1_700_000_000.0) -> None:
        self._now = initial

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


@pytest.fixture
def fake_clock() -> MutableClock:
    return MutableClock()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Cliente síncrono FastAPI para pruebas de rutas (Dev A) en entorno test."""
    monkeypatch.setattr(settings, "APP_ENV", "test")
    with TestClient(openclaw_app) as test_client:
        yield test_client


@pytest_asyncio.fixture
async def fake_redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client_redis = fakeredis.aioredis.FakeRedis()
    try:
        yield client_redis
    finally:
        await client_redis.aclose()


@pytest_asyncio.fixture
async def api_client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[httpx.AsyncClient]:
    """Cliente asíncrono httpx contra la aplicación principal OpenClaw."""
    monkeypatch.setattr(settings, "APP_ENV", "test")
    transport = httpx.ASGITransport(app=openclaw_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client


@pytest_asyncio.fixture
async def health_harness_client(
    fake_redis: fakeredis.aioredis.FakeRedis,
) -> AsyncIterator[httpx.AsyncClient]:
    """App FastAPI de humo con /health sobre fakeredis para validar el harness S0-04."""
    harness = FastAPI(title="OpenClaw Health Smoke Harness")

    @harness.get("/health")
    async def health() -> dict[str, Any]:
        pong = await fake_redis.ping()
        return {"status": "ok", "redis": "ok" if pong else "degraded"}

    transport = httpx.ASGITransport(app=harness)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client
