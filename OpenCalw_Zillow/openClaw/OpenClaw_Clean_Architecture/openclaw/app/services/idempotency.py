"""Idempotencia de trabajo por ``job_id`` y plataforma (S3-02).

El worker reclama una clave antes de raspar una plataforma. Si la clave ya
existia, otro intento (o un mensaje reprocesado por ``XAUTOCLAIM``) ya hizo ese
trabajo y se omite: reprocesar el mismo mensaje no duplica propiedades ni llamadas
al proveedor.

La reclamacion se hace con ``SET NX`` sobre Redis, asi que es atomica entre
consumidores.
"""

from __future__ import annotations

import logging
from typing import Final
from uuid import UUID

from app.core.config import settings
from app.core.redis_client import redis_client

logger = logging.getLogger("OpenClaw.Idempotencia")

PREFIJO: Final = "openclaw:idem:"


def clave(job_id: UUID | str, plataforma: str) -> str:
    """Clave estable por job y plataforma."""
    return f"{PREFIJO}{job_id}:{plataforma}"


async def reclamar(job_id: UUID | str, plataforma: str) -> bool:
    """True si este consumidor es el primero en procesar esa plataforma."""
    creada = await redis_client.set(
        clave(job_id, plataforma),
        "1",
        nx=True,
        ex=settings.IDEMPOTENCIA_TTL_SEGUNDOS,
    )
    if not creada:
        logger.info(
            "trabajo_ya_realizado",
            extra={"job_id": str(job_id), "plataforma": plataforma},
        )
        return False
    return True


async def liberar(job_id: UUID | str, plataforma: str) -> None:
    """Suelta la reclamacion para permitir un reintento legitimo."""
    await redis_client.delete(clave(job_id, plataforma))


async def ya_realizado(job_id: UUID | str, plataforma: str) -> bool:
    """Consulta sin reclamar."""
    return bool(await redis_client.exists(clave(job_id, plataforma)))
