"""Maquina de estados del job de barrido (S3-03).

Un job solo puede avanzar por transiciones declaradas aqui. Cualquier intento de
salto ilegal lanza ``TransicionInvalida`` y deja el estado intacto, de modo que un
mensaje reprocesado no puede devolver un job terminado a ``queued``.
"""

from __future__ import annotations

import logging
from typing import Final
from uuid import UUID

from app.models.contracts import JobResponse, JobStatus
from app.services.job_store import job_store

logger = logging.getLogger("OpenClaw.Estados")

TRANSICIONES: Final[dict[JobStatus, frozenset[JobStatus]]] = {
    JobStatus.QUEUED: frozenset({JobStatus.RUNNING, JobStatus.FAILED}),
    JobStatus.RUNNING: frozenset(
        {JobStatus.COMPLETED, JobStatus.PARTIAL, JobStatus.FAILED}
    ),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.PARTIAL: frozenset({JobStatus.RUNNING}),
    JobStatus.FAILED: frozenset({JobStatus.QUEUED, JobStatus.RUNNING}),
}

ESTADOS_TERMINALES: Final[frozenset[JobStatus]] = frozenset(
    {JobStatus.COMPLETED, JobStatus.PARTIAL, JobStatus.FAILED}
)


class TransicionInvalida(RuntimeError):
    """Se intento un cambio de estado no permitido."""

    def __init__(self, origen: JobStatus, destino: JobStatus) -> None:
        self.origen = origen
        self.destino = destino
        super().__init__(f"Transicion no permitida: {origen.value} -> {destino.value}")


def transicion_valida(origen: JobStatus, destino: JobStatus) -> bool:
    """Indica si el cambio de estado esta declarado en la maquina."""
    return destino in TRANSICIONES.get(origen, frozenset())


def es_terminal(estado: JobStatus) -> bool:
    return estado in ESTADOS_TERMINALES


async def aplicar_transicion(
    job_id: UUID,
    destino: JobStatus,
    *,
    progress: dict[str, str] | None = None,
    error: str | None = None,
) -> JobResponse | None:
    """Valida y persiste la transicion; si es ilegal, no toca el almacen."""
    job = await job_store.obtener(job_id)
    if job is None:
        return None

    origen = job.status
    if not transicion_valida(origen, destino):
        logger.warning(
            "transicion_rechazada",
            extra={
                "job_id": str(job_id),
                "origen": origen.value,
                "destino": destino.value,
            },
        )
        raise TransicionInvalida(origen, destino)

    await job_store.marcar(job_id, destino, progress=progress, error=error)
    logger.info(
        "transicion_aplicada",
        extra={"job_id": str(job_id), "origen": origen.value, "destino": destino.value},
    )
    return await job_store.obtener(job_id)
