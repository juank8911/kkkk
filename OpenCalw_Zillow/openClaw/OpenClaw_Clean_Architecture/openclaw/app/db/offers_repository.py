"""Repositorio de ofertas y eventos de la Fase 95."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Oferta, OfertaEvento  # pyright: ignore[reportAttributeAccessIssue]
from app.models.offers import OfertaEstado
from app.services.offers import ahora_si_cerrada, validar_transicion_oferta


class RepositorioOfertas:
    def __init__(self, sesion: AsyncSession) -> None:
        self._sesion = sesion

    async def obtener(self, oferta_id: uuid.UUID) -> Oferta | None:
        return await self._sesion.get(Oferta, oferta_id)

    async def listar_por_propiedad(self, propiedad_id: uuid.UUID) -> list[Oferta]:
        resultado = await self._sesion.execute(
            select(Oferta).where(Oferta.propiedad_id == propiedad_id).order_by(Oferta.creada_en.desc())
        )
        return list(resultado.scalars().all())

    async def crear(self, oferta: Oferta, *, notas_evento: str | None = None) -> Oferta:
        self._sesion.add(oferta)
        await self._sesion.flush()
        self._sesion.add(
            OfertaEvento(
                oferta_id=oferta.id,
                estado_anterior=None,
                estado_nuevo=oferta.estado,
                actor=oferta.actor_ultima_actualizacion,
                notas=notas_evento,
            )
        )
        await self._sesion.flush()
        return oferta

    async def cambiar_estado(
        self,
        oferta: Oferta,
        nuevo: OfertaEstado,
        *,
        actor: str,
        notas: str | None = None,
        importe_contraoferta_usd: float | None = None,
    ) -> Oferta:
        validar_transicion_oferta(oferta.estado, nuevo)
        anterior = oferta.estado
        oferta.estado = nuevo.value
        oferta.actor_ultima_actualizacion = actor
        oferta.actualizada_en = datetime.now(UTC)
        oferta.cerrada_en = ahora_si_cerrada(nuevo)
        if importe_contraoferta_usd is not None:
            oferta.importe_contraoferta_usd = importe_contraoferta_usd
        if notas is not None:
            oferta.notas = notas
        self._sesion.add(
            OfertaEvento(
                oferta_id=oferta.id,
                estado_anterior=anterior,
                estado_nuevo=nuevo.value,
                actor=actor,
                notas=notas,
            )
        )
        await self._sesion.flush()
        return oferta
