"""Mapeadores de dominio Dev 2 -> modelos ORM PostgreSQL Dev 1 (S2-02 / S2-07).

Desacopla los modelos Pydantic del dominio (`DatosPropiedadNormalizados`,
`BuyAuditEvent` y el estado del orquestador) de las tablas SQLAlchemy definidas
por Dev 1 en `app/db/models.py` (`Propiedad`, `Decision`, `Evento`), listos para
conectarse a la unidad de trabajo de `S2-03`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Decision, Evento, Propiedad
from app.models.buy import BuyAuditEvent, BuyOrderStatus
from app.models.property import DatosPropiedadNormalizados
from app.services.deduplicator import generar_clave_deduplicacion

_VALID_VERDICTS = frozenset({"comprar", "revisar", "descartar", "error"})
_VALID_LEVELS = frozenset({"debug", "info", "warning", "error", "critical"})


def map_normalized_property_to_orm(
    *,
    job_id: UUID,
    plataforma: str,
    propiedad: DatosPropiedadNormalizados,
    huella_dedupe: str | None = None,
) -> Propiedad:
    """Convierte `DatosPropiedadNormalizados` en una entidad ORM `Propiedad`."""
    dedupe = (
        huella_dedupe
        if huella_dedupe is not None
        else generar_clave_deduplicacion(propiedad.ubicacion, propiedad.condado)
    )
    return Propiedad(
        job_id=job_id,
        plataforma=plataforma.strip().lower(),
        url_origen=propiedad.id_plataforma_origen,
        ubicacion=propiedad.ubicacion,
        direccion_normalizada=propiedad.direccion_normalizada,
        ciudad=propiedad.ciudad,
        codigo_postal=propiedad.codigo_postal,
        condado=propiedad.condado,
        fuente_ubicacion=propiedad.fuente_ubicacion,
        confianza_ubicacion=propiedad.confianza_ubicacion,
        precio_usd=propiedad.precio_usd,
        ingreso_anual_estimado=propiedad.ingreso_anual_estimado,
        imagenes=list(propiedad.imagenes),
        latitud=propiedad.latitud,
        longitud=propiedad.longitud,
        descripcion_cruda=propiedad.descripcion_cruda,
        huella_dedupe=dedupe,
        created_at=propiedad.fecha_consulta,
    )


def derive_decision_veredicto(state: Mapping[str, Any]) -> str:
    """Mapea el estado final del orquestador al dominio cerrado `ck_decisiones_veredicto_valido`."""
    estado = str(state.get("estado") or "").upper()
    requiere_revision = bool(state.get("requiere_revision_humana", False))
    vision_disponible = state.get("vision_disponible", True)

    if estado in ("ALERTA_ERROR", "ERROR"):
        return "error"
    if estado in ("RECHAZADO", "DESCARTADO_PRECIO"):
        return "descartar"
    if requiere_revision or estado == "VISION_NO_DISPONIBLE" or vision_disponible is False:
        return "revisar"
    if estado in ("ALERTA_ENVIADA", "ALERTA_DUPLICADA_OMITIDA", "INSPECCION_VISUAL_OK"):
        return "comprar"
    return "revisar"


def map_pipeline_state_to_decision_orm(
    *,
    propiedad_id: UUID,
    state: Mapping[str, Any],
) -> Decision:
    """Convierte la salida del grafo de evaluación en una fila ORM `Decision`."""
    veredicto = derive_decision_veredicto(state)
    assert veredicto in _VALID_VERDICTS

    cap_rate_raw = state.get("cap_rate_ajustado")
    if cap_rate_raw is None:
        cap_rate_raw = state.get("cap_rate")

    capex_raw = state.get("capex")
    motivo = (
        state.get("vision_motivo")
        or state.get("motivo_financiero")
        or (state.get("auditoria_legal") or {}).get("resumen")
        or f"Estado final: {state.get('estado', 'desconocido')}"
    )

    detalle: dict[str, object] = {
        "estado": state.get("estado"),
        "noi": state.get("noi"),
        "vision_disponible": state.get("vision_disponible"),
        "vision_confianza": state.get("vision_confianza"),
        "estado_legal": state.get("estado_legal"),
        "riesgo_legal": state.get("riesgo_legal"),
        "requiere_revision_humana": bool(state.get("requiere_revision_humana", False)),
        "senales_legales": state.get("senales_legales") or [],
        "ficha_url": state.get("ficha_url"),
        "reglas_financieras_version": state.get("reglas_financieras_version"),
    }

    return Decision(
        propiedad_id=propiedad_id,
        veredicto=veredicto,
        cap_rate=round(float(cap_rate_raw), 4) if cap_rate_raw is not None else None,
        roi_estimado=None,
        capex_estimado=round(float(capex_raw), 2) if capex_raw is not None else None,
        motivo=str(motivo),
        detalle=detalle,
    )


def _audit_event_level(status: BuyOrderStatus) -> str:
    if status in (
        BuyOrderStatus.REJECTED_DISABLED,
        BuyOrderStatus.REJECTED_DESTINATION,
        BuyOrderStatus.REJECTED_LIMIT,
        BuyOrderStatus.REJECTED_CONFIRMATION,
        BuyOrderStatus.REJECTED_CONFLICT,
    ):
        return "warning"
    if status == BuyOrderStatus.UPSTREAM_QUEUED_RETRY:
        return "error"
    return "info"


def map_buy_audit_event_to_orm(
    event: BuyAuditEvent,
    *,
    job_id: UUID | None = None,
) -> Evento:
    """Convierte `BuyAuditEvent` en una fila append-only ORM `Evento`."""
    nivel = _audit_event_level(event.status)
    assert nivel in _VALID_LEVELS

    tipo = f"buy.{event.action}"[:60]
    mensaje = (
        f"/buy [{event.mode.value}] {event.status.value} "
        f"prop={event.property_id} amount={event.amount} {event.asset} "
        f"dest={event.destination} idem={event.idempotency_key}"
    )
    return Evento(
        job_id=job_id,
        tipo=tipo,
        nivel=nivel,
        mensaje=mensaje,
        payload=event.model_dump(mode="json"),
        ocurrido_en=event.timestamp,
    )


class SqlAlchemyBuyAuditStore:
    """Implementación de `BuyAuditStorePort` persistiendo en la tabla `eventos` (S2-02)."""

    def __init__(self, session: AsyncSession, *, job_id: UUID | None = None) -> None:
        self._session = session
        self._job_id = job_id

    async def append(self, event: BuyAuditEvent) -> None:
        row = map_buy_audit_event_to_orm(event, job_id=self._job_id)
        self._session.add(row)
        await self._session.flush()

    async def list_events(self, *, idempotency_key: str | None = None) -> list[BuyAuditEvent]:
        stmt = select(Evento).where(Evento.tipo.like("buy.%")).order_by(Evento.ocurrido_en.asc())
        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        events = [BuyAuditEvent.model_validate(row.payload) for row in rows]
        if idempotency_key is None:
            return events
        return [ev for ev in events if ev.idempotency_key == idempotency_key]
