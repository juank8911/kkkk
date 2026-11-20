"""Repositorios de persistencia sobre PostgreSQL (S2-03).

Cada repositorio encapsula las consultas de una tabla y no gestiona
transacciones: eso corresponde a la unidad de trabajo. Asi el worker puede
decidir el alcance de cada transaccion (una por propiedad) sin duplicar logica.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Decision, Evento, Job, Propiedad


class RepositorioJobs:
    """Lectura y escritura de la tabla ``jobs``."""

    def __init__(self, sesion: AsyncSession) -> None:
        self._sesion = sesion

    async def crear(self, job: Job) -> Job:
        self._sesion.add(job)
        await self._sesion.flush()
        return job

    async def obtener(self, job_id: uuid.UUID) -> Job | None:
        return await self._sesion.get(Job, job_id)

    async def actualizar_estado(
        self,
        job_id: uuid.UUID,
        estado: str,
        *,
        error: str | None = None,
    ) -> bool:
        resultado = await self._sesion.execute(
            update(Job)
            .where(Job.job_id == job_id)
            .values(status=estado, error=error, updated_at=datetime.now(UTC))
        )
        return bool(resultado.rowcount)

    async def listar(self, *, limite: int = 50, desplazamiento: int = 0) -> Sequence[Job]:
        consulta = (
            select(Job)
            .order_by(Job.created_at.desc())
            .limit(limite)
            .offset(desplazamiento)
        )
        return (await self._sesion.execute(consulta)).scalars().all()


class RepositorioPropiedades:
    """Persistencia de anuncios normalizados."""

    def __init__(self, sesion: AsyncSession) -> None:
        self._sesion = sesion

    async def guardar(self, propiedad: Propiedad) -> Propiedad:
        self._sesion.add(propiedad)
        await self._sesion.flush()
        return propiedad

    async def guardar_muchas(self, propiedades: Iterable[Propiedad]) -> int:
        lote = list(propiedades)
        self._sesion.add_all(lote)
        await self._sesion.flush()
        return len(lote)

    async def listar_por_job(self, job_id: uuid.UUID) -> Sequence[Propiedad]:
        consulta = select(Propiedad).where(Propiedad.job_id == job_id)
        return (await self._sesion.execute(consulta)).scalars().all()

    async def buscar_por_huella(self, huella: str) -> Sequence[Propiedad]:
        consulta = select(Propiedad).where(Propiedad.huella_dedupe == huella)
        return (await self._sesion.execute(consulta)).scalars().all()

    async def existe_url(self, job_id: uuid.UUID, url: str) -> bool:
        consulta = select(Propiedad.id).where(
            Propiedad.job_id == job_id, Propiedad.url_origen == url
        )
        return (await self._sesion.execute(consulta)).first() is not None


class RepositorioDecisiones:
    """Decisiones del grafo de due diligence."""

    def __init__(self, sesion: AsyncSession) -> None:
        self._sesion = sesion

    async def registrar(self, decision: Decision) -> Decision:
        self._sesion.add(decision)
        await self._sesion.flush()
        return decision

    async def listar_por_propiedad(self, propiedad_id: uuid.UUID) -> Sequence[Decision]:
        consulta = (
            select(Decision)
            .where(Decision.propiedad_id == propiedad_id)
            .order_by(Decision.created_at.desc())
        )
        return (await self._sesion.execute(consulta)).scalars().all()


class RepositorioEventos:
    """Traza append-only de lo ocurrido."""

    def __init__(self, sesion: AsyncSession) -> None:
        self._sesion = sesion

    async def registrar(
        self,
        tipo: str,
        mensaje: str,
        *,
        job_id: uuid.UUID | None = None,
        nivel: str = "info",
        payload: dict[str, object] | None = None,
    ) -> Evento:
        evento = Evento(
            job_id=job_id,
            tipo=tipo,
            nivel=nivel,
            mensaje=mensaje,
            payload=payload or {},
        )
        self._sesion.add(evento)
        await self._sesion.flush()
        return evento

    async def listar_recientes(self, *, limite: int = 100) -> Sequence[Evento]:
        consulta = select(Evento).order_by(Evento.ocurrido_en.desc()).limit(limite)
        return (await self._sesion.execute(consulta)).scalars().all()
