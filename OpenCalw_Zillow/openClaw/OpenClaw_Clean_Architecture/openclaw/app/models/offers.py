"""Contratos de la Fase 95: seguimiento post-oferta."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class OfertaEstado(StrEnum):
    PROPUESTA = "propuesta"
    CONTRAOFERTA = "contraoferta"
    ACEPTADA = "aceptada"
    RECHAZADA = "rechazada"
    CERRADA = "cerrada"
    CANCELADA = "cancelada"


class OfertaCrear(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    importe_usd: float = Field(..., gt=0, le=100_000_000)
    notas: str | None = Field(default=None, max_length=4000)
    actor: str = Field(default="usuario", min_length=1, max_length=120)


class OfertaEstadoActualizar(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    estado: OfertaEstado
    importe_contraoferta_usd: float | None = Field(default=None, gt=0, le=100_000_000)
    notas: str | None = Field(default=None, max_length=4000)
    actor: str = Field(default="usuario", min_length=1, max_length=120)


class OfertaResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    propiedad_id: UUID
    estado: OfertaEstado
    importe_usd: float
    importe_contraoferta_usd: float | None = None
    notas: str | None = None
    actor_ultima_actualizacion: str
    creada_en: datetime
    actualizada_en: datetime
    cerrada_en: datetime | None = None


class OfertaEventoResponse(BaseModel):
    tipo: str
    estado_anterior: OfertaEstado | None = None
    estado_nuevo: OfertaEstado
    actor: str
    ocurrido_en: datetime = Field(default_factory=lambda: datetime.now(UTC))
