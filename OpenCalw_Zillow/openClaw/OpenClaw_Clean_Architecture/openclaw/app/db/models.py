"""Modelos ORM del esquema inicial (S2-02).

Cuatro tablas cubren el ciclo completo: el barrido pedido (``jobs``), lo
extraido (``propiedades``), lo decidido (``decisiones``) y la traza de lo que
ocurrio (``eventos``).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Job(Base):
    """Barrido solicitado por un cliente."""

    __tablename__ = "jobs"

    job_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    query: Mapped[str] = mapped_column(String(200), nullable=False)
    plataformas: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    propiedades: Mapped[list["Propiedad"]] = relationship(back_populates="job")

    __table_args__ = (
        CheckConstraint(
            "status IN ('queued','running','completed','partial','failed')",
            name="ck_jobs_status_valido",
        ),
        Index("ix_jobs_created_at", "created_at"),
    )


class Propiedad(Base):
    """Anuncio normalizado extraido durante un barrido."""

    __tablename__ = "propiedades"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False
    )
    plataforma: Mapped[str] = mapped_column(String(40), nullable=False)
    url_origen: Mapped[str] = mapped_column(Text, nullable=False)
    ubicacion: Mapped[str] = mapped_column(Text, nullable=False)
    condado: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    precio_usd: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    ingreso_anual_estimado: Mapped[float] = mapped_column(
        Numeric(14, 2), nullable=False, default=0
    )
    imagenes: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    latitud: Mapped[float | None] = mapped_column(Numeric(9, 6), nullable=True)
    longitud: Mapped[float | None] = mapped_column(Numeric(9, 6), nullable=True)
    descripcion_cruda: Mapped[str] = mapped_column(Text, nullable=False)
    huella_dedupe: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    job: Mapped[Job] = relationship(back_populates="propiedades")
    decisiones: Mapped[list["Decision"]] = relationship(back_populates="propiedad")

    __table_args__ = (
        UniqueConstraint("job_id", "url_origen", name="uq_propiedades_job_url"),
        CheckConstraint("precio_usd >= 0", name="ck_propiedades_precio_positivo"),
    )


class Oferta(Base):
    """Oferta comercial con pipeline de contraoferta y cierre (Fase 95)."""

    __tablename__ = "ofertas"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    propiedad_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("propiedades.id", ondelete="CASCADE"), nullable=False
    )
    estado: Mapped[str] = mapped_column(String(20), nullable=False, default="propuesta", index=True)
    importe_usd: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    importe_contraoferta_usd: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    notas: Mapped[str | None] = mapped_column(Text, nullable=True)
    actor_ultima_actualizacion: Mapped[str] = mapped_column(String(120), nullable=False)
    creada_en: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    actualizada_en: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    cerrada_en: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    propiedad: Mapped[Propiedad] = relationship()
    eventos: Mapped[list["OfertaEvento"]] = relationship(back_populates="oferta")

    __table_args__ = (
        CheckConstraint(
            "estado IN ('propuesta','contraoferta','aceptada','rechazada','cerrada','cancelada')",
            name="ck_ofertas_estado_valido",
        ),
        CheckConstraint("importe_usd > 0", name="ck_ofertas_importe_positivo"),
    )


class OfertaEvento(Base):
    """Historial append-only de transiciones de una oferta."""

    __tablename__ = "oferta_eventos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    oferta_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("ofertas.id", ondelete="CASCADE"), nullable=False, index=True
    )
    estado_anterior: Mapped[str | None] = mapped_column(String(20), nullable=True)
    estado_nuevo: Mapped[str] = mapped_column(String(20), nullable=False)
    actor: Mapped[str] = mapped_column(String(120), nullable=False)
    notas: Mapped[str | None] = mapped_column(Text, nullable=True)
    ocurrido_en: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    oferta: Mapped[Oferta] = relationship(back_populates="eventos")


class Decision(Base):
    """Resultado del grafo de due diligence para una propiedad."""

    __tablename__ = "decisiones"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    propiedad_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("propiedades.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    veredicto: Mapped[str] = mapped_column(String(30), nullable=False)
    cap_rate: Mapped[float | None] = mapped_column(Numeric(6, 4), nullable=True)
    roi_estimado: Mapped[float | None] = mapped_column(Numeric(6, 4), nullable=True)
    capex_estimado: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    motivo: Mapped[str | None] = mapped_column(Text, nullable=True)
    detalle: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    propiedad: Mapped[Propiedad] = relationship(back_populates="decisiones")

    __table_args__ = (
        CheckConstraint(
            "veredicto IN ('comprar','revisar','descartar','error')",
            name="ck_decisiones_veredicto_valido",
        ),
    )


class Evento(Base):
    """Traza append-only de lo ocurrido en el sistema."""

    __tablename__ = "eventos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("jobs.job_id", ondelete="SET NULL"), nullable=True
    )
    tipo: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    nivel: Mapped[str] = mapped_column(String(10), nullable=False, default="info")
    mensaje: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False, default=dict)
    ocurrido_en: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    __table_args__ = (
        CheckConstraint(
            "nivel IN ('debug','info','warning','error','critical')",
            name="ck_eventos_nivel_valido",
        ),
    )


class Plataforma(Base):
    """Catalogo cerrado de plataformas soportadas (S2-04 / ADR-01)."""

    __tablename__ = "plataformas"

    clave: Mapped[str] = mapped_column(String(40), primary_key=True)
    nombre: Mapped[str] = mapped_column(String(80), nullable=False)
    dominio: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    plantilla_url: Mapped[str] = mapped_column(Text, nullable=False)
    activa: Mapped[bool] = mapped_column(nullable=False, default=True)
    creada_en: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
