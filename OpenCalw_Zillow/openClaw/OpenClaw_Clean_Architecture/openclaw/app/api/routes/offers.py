"""API de seguimiento post-oferta (Fase 95).

Los endpoints registran el pipeline comercial; no envian comunicaciones ni
aceptan ofertas automaticamente.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import requiere_api_key
from app.db.base import get_session
from app.db.models import Oferta  # pyright: ignore[reportAttributeAccessIssue]
from app.db.offers_repository import RepositorioOfertas
from app.models.offers import OfertaCrear, OfertaEstadoActualizar, OfertaResponse
from app.services.offers import TransicionOfertaInvalida

router = APIRouter(
    prefix="/v1/openclaw/propiedades/{propiedad_id}/ofertas",
    tags=["ofertas"],
    dependencies=[Depends(requiere_api_key)],
)


@router.post("", response_model=OfertaResponse, status_code=status.HTTP_201_CREATED)
async def crear_oferta(
    propiedad_id: UUID,
    payload: OfertaCrear,
    sesion: AsyncSession = Depends(get_session),
) -> Oferta:
    oferta = Oferta(
        id=uuid4(),
        propiedad_id=propiedad_id,
        estado="propuesta",
        importe_usd=payload.importe_usd,
        notas=payload.notas,
        actor_ultima_actualizacion=payload.actor,
    )
    repositorio = RepositorioOfertas(sesion)
    try:
        await repositorio.crear(oferta, notas_evento=payload.notas)
        await sesion.commit()
    except Exception as exc:
        await sesion.rollback()
        raise HTTPException(status_code=404, detail="La propiedad no existe") from exc
    return oferta


@router.get("", response_model=list[OfertaResponse])
async def listar_ofertas(
    propiedad_id: UUID,
    sesion: AsyncSession = Depends(get_session),
) -> list[Oferta]:
    return await RepositorioOfertas(sesion).listar_por_propiedad(propiedad_id)


@router.patch("/{oferta_id}/estado", response_model=OfertaResponse)
async def actualizar_estado_oferta(
    propiedad_id: UUID,
    oferta_id: UUID,
    payload: OfertaEstadoActualizar,
    sesion: AsyncSession = Depends(get_session),
) -> Oferta:
    repositorio = RepositorioOfertas(sesion)
    oferta = await repositorio.obtener(oferta_id)
    if oferta is None or oferta.propiedad_id != propiedad_id:
        raise HTTPException(status_code=404, detail="Oferta no encontrada")
    try:
        await repositorio.cambiar_estado(
            oferta,
            payload.estado,
            actor=payload.actor,
            notas=payload.notas,
            importe_contraoferta_usd=payload.importe_contraoferta_usd,
        )
        await sesion.commit()
    except TransicionOfertaInvalida as exc:
        await sesion.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return oferta
