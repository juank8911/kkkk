"""Geocodificacion de direcciones con cache en Redis (S3-06).

Los portales inmobiliarios devuelven direcciones sin coordenadas utiles para
comparar propiedades. Geocodificar cada anuncio contra Nominatim es caro y esta
limitado a una peticion por segundo, asi que:

1. se consulta primero la cache de Redis (``openclaw:geo:<sha256>``);
2. solo en fallo de cache se llama al proveedor;
3. el resultado (incluido el negativo) se guarda para no repetir consultas
   inutiles.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings
from app.core.redis_client import redis_client

logger = logging.getLogger("OpenClaw.Geocoding")

PREFIJO = "openclaw:geo:"
SIN_RESULTADO = "null"


@dataclass(frozen=True, slots=True)
class Coordenadas:
    """Punto geografico devuelto por el proveedor."""

    lat: float
    lon: float
    fuente: str = "nominatim"


def clave_geo(direccion: str) -> str:
    """Clave de cache estable e independiente de mayusculas y espacios."""
    normalizada = " ".join(direccion.lower().split())
    digest = hashlib.sha256(normalizada.encode("utf-8")).hexdigest()
    return f"{PREFIJO}{digest}"


async def _leer_cache(clave: str) -> tuple[bool, Coordenadas | None]:
    crudo = await redis_client.get(clave)
    if crudo is None:
        return False, None
    texto = crudo.decode("utf-8") if isinstance(crudo, bytes) else str(crudo)
    if texto == SIN_RESULTADO:
        return True, None
    datos: dict[str, Any] = json.loads(texto)
    return True, Coordenadas(lat=float(datos["lat"]), lon=float(datos["lon"]), fuente="cache")


async def _escribir_cache(clave: str, coordenadas: Coordenadas | None) -> None:
    if coordenadas is None:
        valor = SIN_RESULTADO
    else:
        valor = json.dumps({"lat": coordenadas.lat, "lon": coordenadas.lon})
    await redis_client.set(clave, valor, ex=settings.GEOCODING_TTL_SEGUNDOS)


async def geocodificar(
    direccion: str,
    *,
    cliente: httpx.AsyncClient | None = None,
) -> Coordenadas | None:
    """Devuelve coordenadas de la direccion o ``None`` si no se pudo resolver."""
    if not direccion.strip():
        return None

    clave = clave_geo(direccion)
    en_cache, cacheado = await _leer_cache(clave)
    if en_cache:
        logger.debug("geocoding_cache_hit", extra={"clave": clave})
        return cacheado

    propio = cliente is None
    if cliente is None:
        cliente_http = httpx.AsyncClient(
            timeout=10.0,
            headers={"User-Agent": settings.GEOCODING_USER_AGENT},
        )
    else:
        cliente_http = cliente

    try:
        respuesta = await cliente_http.get(
            settings.GEOCODING_URL,
            params={"q": direccion, "format": "json", "limit": 1},
        )
        respuesta.raise_for_status()
        resultados: list[dict[str, Any]] = respuesta.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("geocoding_error", extra={"error": str(exc)})
        return None
    finally:
        if propio:
            await cliente_http.aclose()

    if not resultados:
        await _escribir_cache(clave, None)
        logger.info("geocoding_sin_resultado", extra={"direccion": direccion[:120]})
        return None

    primero = resultados[0]
    try:
        coordenadas = Coordenadas(lat=float(primero["lat"]), lon=float(primero["lon"]))
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("geocoding_respuesta_invalida", extra={"error": str(exc)})
        return None

    await _escribir_cache(clave, coordenadas)
    return coordenadas
