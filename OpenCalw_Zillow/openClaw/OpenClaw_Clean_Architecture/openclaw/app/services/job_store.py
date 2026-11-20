"""Almacen de jobs de barrido.

Implementacion interina sobre Redis mientras llega PostgreSQL (S2-02). La
interfaz se mantiene cuando cambie el motor, para no tocar las rutas.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from app.core.redis_client import redis_client
from app.models.contracts import JobResponse, JobStatus

TTL_SEGUNDOS = 7 * 24 * 3600


def _clave(job_id: UUID) -> str:
    return f"openclaw:job:{job_id}"


class JobStore:
    """Persistencia minima del ciclo de vida de un job."""

    async def guardar(self, job: JobResponse) -> None:
        await redis_client.set(_clave(job.job_id), job.model_dump_json(), ex=TTL_SEGUNDOS)

    async def obtener(self, job_id: UUID) -> JobResponse | None:
        crudo = await redis_client.get(_clave(job_id))
        if crudo is None:
            return None
        if isinstance(crudo, bytes):
            crudo = crudo.decode("utf-8")
        return JobResponse.model_validate_json(crudo)

    async def marcar(
        self,
        job_id: UUID,
        estado: JobStatus,
        *,
        progress: dict[str, str] | None = None,
        error: str | None = None,
    ) -> None:
        job = await self.obtener(job_id)
        if job is None:
            return

        cambios: dict[str, object] = {
            "status": estado,
            "updated_at": datetime.now(timezone.utc),
            "error": error,
        }
        if progress is not None:
            cambios["progress"] = progress

        await self.guardar(job.model_copy(update=cambios))


job_store = JobStore()
