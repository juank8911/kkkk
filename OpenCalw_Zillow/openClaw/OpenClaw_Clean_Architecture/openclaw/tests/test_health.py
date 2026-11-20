"""Pruebas de liveness y readiness (Redis + tablas críticas de PostgreSQL)."""

from __future__ import annotations

from collections.abc import Iterable

from app.api.routes import health as health_route
from app.core.config import settings
from app.core.redis_client import redis_client


class _FakeResult:
    def __init__(self, tablas: Iterable[str]) -> None:
        self._tablas = [(t,) for t in tablas]

    def fetchall(self) -> list[tuple[str]]:
        return list(self._tablas)


class _FakeConnection:
    def __init__(self, tablas: Iterable[str] | None = None, error: Exception | None = None) -> None:
        self._tablas = list(tablas or [])
        self._error = error

    async def __aenter__(self) -> _FakeConnection:
        if self._error is not None:
            raise self._error
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def execute(self, _stmt: object) -> _FakeResult:
        return _FakeResult(self._tablas)


class _FakeEngine:
    def __init__(self, tablas: Iterable[str] | None = None, error: Exception | None = None) -> None:
        self._tablas = tablas
        self._error = error

    def connect(self) -> _FakeConnection:
        return _FakeConnection(tablas=self._tablas, error=self._error)


def test_health_responde_ok(client) -> None:
    respuesta = client.get("/health")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["status"] == "ok"
    assert cuerpo["env"] == "test"
    assert cuerpo["version"]


def test_readyz_ok_cuando_redis_y_tablas_postgres_responden(client, monkeypatch) -> None:
    async def ping_ok() -> bool:
        return True

    monkeypatch.setattr(redis_client, "ping", ping_ok)
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
    monkeypatch.setattr(
        health_route,
        "obtener_engine",
        lambda: _FakeEngine(tablas=health_route.TABLAS_CRITICAS_DB),
    )

    respuesta = client.get("/readyz")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["status"] == "ok"
    assert cuerpo["checks"]["redis"] == "ok"
    assert cuerpo["checks"]["postgres"] == "ok"


def test_readyz_503_cuando_redis_falla(client, monkeypatch) -> None:
    async def ping_falla() -> bool:
        raise ConnectionError("sin conexion")

    monkeypatch.setattr(redis_client, "ping", ping_falla)
    monkeypatch.setattr(settings, "DATABASE_URL", "")

    respuesta = client.get("/readyz")

    assert respuesta.status_code == 503
    assert respuesta.json()["status"] == "degradado"
    assert respuesta.json()["checks"]["redis"].startswith("error:")


def test_readyz_503_cuando_faltan_tablas_criticas_en_postgres(client, monkeypatch) -> None:
    async def ping_ok() -> bool:
        return True

    monkeypatch.setattr(redis_client, "ping", ping_ok)
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
    monkeypatch.setattr(
        health_route,
        "obtener_engine",
        lambda: _FakeEngine(tablas=("jobs", "plataformas", "propiedades")),
    )

    respuesta = client.get("/readyz")

    assert respuesta.status_code == 503
    cuerpo = respuesta.json()
    assert cuerpo["status"] == "degradado"
    assert cuerpo["checks"]["postgres"].startswith("faltan_tablas:")
    assert "ofertas" in cuerpo["checks"]["postgres"]
    assert "decisiones" in cuerpo["checks"]["postgres"]


def test_readyz_503_cuando_postgres_no_responde(client, monkeypatch) -> None:
    async def ping_ok() -> bool:
        return True

    monkeypatch.setattr(redis_client, "ping", ping_ok)
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
    monkeypatch.setattr(
        health_route,
        "obtener_engine",
        lambda: _FakeEngine(error=ConnectionError("postgres caido")),
    )

    respuesta = client.get("/readyz")

    assert respuesta.status_code == 503
    cuerpo = respuesta.json()
    assert cuerpo["status"] == "degradado"
    assert cuerpo["checks"]["postgres"] == "error:ConnectionError"
