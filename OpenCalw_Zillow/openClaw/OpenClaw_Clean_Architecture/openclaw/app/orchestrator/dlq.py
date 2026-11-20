"""Dead-letter de decisiones fallidas (S4-03).

Cuando un nodo del grafo agota sus reintentos no se puede dejar la propiedad a
medio evaluar: se anota el fallo, se manda el caso a
``openclaw_decisiones_dlq`` con el motivo y el numero de intentos, y la
propiedad queda con veredicto ``error`` para que un humano la revise.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from app.core.config import settings
from app.core.redis_client import redis_client

logger = logging.getLogger("OpenClaw.DecisionesDLQ")

MAX_MOTIVO = 500


def construir_registro_fallo(
    *,
    nodo: str,
    motivo: str,
    intentos: int,
    propiedad_id: UUID | str | None = None,
    job_id: UUID | str | None = None,
    dedup_key: str | None = None,
) -> dict[str, str]:
    """Registro plano listo para ``XADD`` (Redis Streams solo admite strings)."""
    return {
        "nodo": nodo,
        "motivo": motivo[:MAX_MOTIVO],
        "intentos": str(intentos),
        "propiedad_id": str(propiedad_id) if propiedad_id else "",
        "job_id": str(job_id) if job_id else "",
        "dedup_key": dedup_key or "",
    }


async def enviar_a_dlq(registro: dict[str, str]) -> str | None:
    """Escribe en la DLQ; un fallo de Redis no debe tumbar el grafo."""
    try:
        return await redis_client.xadd(
            settings.REDIS_STREAM_DLQ_DECISIONES,
            registro,
            maxlen=10000,
            approximate=True,
        )
    except Exception as exc:  # noqa: BLE001 - la DLQ es best-effort
        logger.error("dlq_no_disponible", extra={"error": str(exc), "nodo": registro.get("nodo")})
        return None


async def registrar_fallo_nodo(
    *,
    nodo: str,
    state: dict[str, Any],
    motivo: str,
    intentos: int,
) -> str | None:
    """Anota el fallo definitivo de un nodo en la DLQ de decisiones."""
    registro = construir_registro_fallo(
        nodo=nodo,
        motivo=motivo,
        intentos=intentos,
        propiedad_id=state.get("property_id"),
        job_id=state.get("job_id"),
        dedup_key=state.get("dedup_key"),
    )
    logger.error(
        "nodo_a_dlq",
        extra={
            "nodo": nodo,
            "intentos": intentos,
            "propiedad_id": registro["propiedad_id"],
            "motivo": registro["motivo"][:200],
        },
    )
    return await enviar_a_dlq(registro)
