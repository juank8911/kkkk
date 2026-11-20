"""Pruebas de geocodificacion con cache en Redis (S3-06)."""

from __future__ import annotations

from typing import Any

import fakeredis.aioredis
import httpx
import pytest

from app.services import geocoding
from app.services.geocoding import Coordenadas, clave_geo, geocodificar

DIRECCION = "3505 W 25th St, Cleveland, OH"


class RespuestaFalsa:
    def __init__(self, datos: Any, error: Exception | None = None) -> None:
        self._datos = datos
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def json(self) -> Any:
        return self._datos


class ClienteFalso:
    def __init__(self, datos: Any = None, error: Exception | None = None) -> None:
        self.datos = datos if datos is not None else [{"lat": "41.4993", "lon": "-81.6944"}]
        self.error = error
        self.llamadas: list[dict[str, Any]] = []

    async def get(self, url: str, params: dict[str, Any] | None = None) -> RespuestaFalsa:
        self.llamadas.append(params or {})
        return RespuestaFalsa(self.datos, self.error)

    async def aclose(self) -> None:
        return None


@pytest.fixture(autouse=True)
def redis_falso(
    monkeypatch: pytest.MonkeyPatch, fake_redis: fakeredis.aioredis.FakeRedis
) -> fakeredis.aioredis.FakeRedis:
    monkeypatch.setattr(geocoding, "redis_client", fake_redis)
    return fake_redis


def test_clave_normalizada() -> None:
    assert clave_geo("  3505  W 25th St ") == clave_geo("3505 w 25th st")
    assert clave_geo(DIRECCION).startswith(geocoding.PREFIJO)


async def test_direccion_vacia_no_consulta() -> None:
    cliente = ClienteFalso()
    assert await geocodificar("   ", cliente=cliente) is None
    assert cliente.llamadas == []


async def test_primera_consulta_va_al_proveedor_y_cachea() -> None:
    cliente = ClienteFalso()
    resultado = await geocodificar(DIRECCION, cliente=cliente)

    assert isinstance(resultado, Coordenadas)
    assert resultado.lat == pytest.approx(41.4993)
    assert resultado.lon == pytest.approx(-81.6944)
    assert resultado.fuente == "nominatim"
    assert len(cliente.llamadas) == 1

    segunda = await geocodificar(DIRECCION, cliente=cliente)
    assert segunda is not None
    assert segunda.fuente == "cache"
    assert len(cliente.llamadas) == 1, "la segunda consulta debe salir de cache"


async def test_sin_resultados_cachea_negativo() -> None:
    cliente = ClienteFalso(datos=[])
    assert await geocodificar("Calle Inventada 123", cliente=cliente) is None
    assert await geocodificar("Calle Inventada 123", cliente=cliente) is None
    assert len(cliente.llamadas) == 1


async def test_error_http_devuelve_none_sin_cachear() -> None:
    cliente = ClienteFalso(error=httpx.ConnectError("sin red"))
    assert await geocodificar(DIRECCION, cliente=cliente) is None
    assert await geocodificar(DIRECCION, cliente=cliente) is None
    assert len(cliente.llamadas) == 2, "un fallo transitorio no debe quedar cacheado"


async def test_respuesta_invalida_devuelve_none() -> None:
    cliente = ClienteFalso(datos=[{"lat": "no-numerico", "lon": "0"}])
    assert await geocodificar(DIRECCION, cliente=cliente) is None
