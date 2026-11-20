"""Endpoints de salud.

``/health`` es liveness: responde mientras el proceso viva.
``/readyz`` es readiness: comprueba Redis, PostgreSQL y la existencia de las
tablas críticas del esquema; devuelve 503 si alguna dependencia falla.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.core.config import settings
from app.core.redis_client import redis_client
from app.db.base import obtener_engine

logger = logging.getLogger("OpenClaw.Salud")

router = APIRouter(tags=["salud"])

TABLAS_CRITICAS_DB: tuple[str, ...] = (
    "jobs",
    "plataformas",
    "propiedades",
    "decisiones",
    "eventos",
    "ofertas",
    "oferta_eventos",
)

DB_CHECK_TIMEOUT_SEGUNDOS = 3.0


async def verificar_tablas_postgres() -> tuple[bool, str]:
    """Comprueba conexión a PostgreSQL y existencia de todas las tablas críticas."""
    if not settings.DATABASE_URL.strip():
        if settings.APP_ENV == "test":
            return True, "omitido:sin_database_url"
        return False, "error:database_url_vacia"

    try:
        engine = obtener_engine()

        async def _consultar_tablas() -> set[str]:
            async with engine.connect() as conn:
                resultado = await conn.execute(
                    text("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public'")
                )
                return {str(fila[0]) for fila in resultado.fetchall()}

        existentes = await asyncio.wait_for(_consultar_tablas(), timeout=DB_CHECK_TIMEOUT_SEGUNDOS)
    except Exception as exc:  # noqa: BLE001 - cualquier fallo degrada la preparación
        logger.warning("postgres_no_disponible", extra={"error": str(exc)})
        return False, f"error:{type(exc).__name__}"

    faltantes = sorted(set(TABLAS_CRITICAS_DB) - existentes)
    if faltantes:
        logger.warning("postgres_tablas_faltantes", extra={"faltantes": faltantes})
        return False, f"faltan_tablas:{','.join(faltantes)}"

    return True, "ok"


@router.get("/health", summary="Liveness del proceso")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env": settings.APP_ENV,
    }


@router.get("/readyz", summary="Readiness de las dependencias")
async def readyz(response: Response) -> dict[str, Any]:
    checks: dict[str, str] = {}
    redis_ok = True

    try:
        await redis_client.ping()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001 - cualquier fallo degrada el servicio
        redis_ok = False
        checks["redis"] = f"error:{type(exc).__name__}"
        logger.warning("redis_no_disponible", extra={"error": str(exc)})

    postgres_ok, estado_postgres = await verificar_tablas_postgres()
    checks["postgres"] = estado_postgres

    faltantes = settings.faltantes()
    checks["configuracion"] = "ok" if not faltantes else f"faltan:{','.join(faltantes)}"

    todo_ok = redis_ok and postgres_ok
    if not todo_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if todo_ok else "degradado",
        "checks": checks,
    }
