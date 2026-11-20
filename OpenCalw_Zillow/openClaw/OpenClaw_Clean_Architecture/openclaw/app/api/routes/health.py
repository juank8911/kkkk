"""Endpoints de salud.

``/health`` es liveness: responde mientras el proceso viva.
``/readyz`` es readiness: comprueba dependencias y devuelve 503 si Redis no responde.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Response, status

from app.core.config import settings
from app.core.redis_client import redis_client

logger = logging.getLogger("OpenClaw.Salud")

router = APIRouter(tags=["salud"])


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

    try:
        await redis_client.ping()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001 - cualquier fallo degrada el servicio
        checks["redis"] = f"error:{type(exc).__name__}"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning("redis_no_disponible", extra={"error": str(exc)})

    faltantes = settings.faltantes()
    checks["configuracion"] = "ok" if not faltantes else f"faltan:{','.join(faltantes)}"

    return {
        "status": "ok" if response.status_code == status.HTTP_200_OK else "degradado",
        "checks": checks,
    }
