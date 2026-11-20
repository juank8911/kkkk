"""Consulta y clasificación de propiedades por barrido (job) para el panel web.

Permite inspeccionar para cualquier barrido cuáles propiedades ya fueron
analizadas y enviadas a aprobación, cuáles fueron rechazadas directamente,
cuáles tuvieron error y cuáles siguen pendientes de análisis.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.config import settings
from app.db.base import obtener_sessionmaker
from app.db.models import Decision, Propiedad
from app.orchestrator.nodes.telegram_alert import extraer_contacto_vendedor

EstadoAnalisisPropiedad = Literal[
    "enviada_aprobacion",
    "rechazada_directamente",
    "error_analisis",
    "pendiente_analisis",
]

ETIQUETAS_ESTADO: dict[EstadoAnalisisPropiedad, str] = {
    "enviada_aprobacion": "Enviada a aprobación",
    "rechazada_directamente": "Rechazada directamente",
    "error_analisis": "Error en análisis",
    "pendiente_analisis": "Pendiente de análisis",
}


class PropiedadBarridoItem(BaseModel):
    """Vista resumida y auditable de una propiedad dentro de un barrido."""

    propiedad_id: UUID
    plataforma: str
    ubicacion: str
    ciudad: str | None = None
    codigo_postal: str | None = None
    precio_usd: float = Field(..., ge=0.0)
    ingreso_anual_estimado: float = Field(default=0.0, ge=0.0)
    url_origen: str
    imagenes: list[str] = Field(default_factory=list)
    contacto_vendedor: str | None = None
    analizada: bool
    estado_analisis: EstadoAnalisisPropiedad
    etiqueta_estado: str
    veredicto: str | None = None
    enviada_a_aprobacion: bool = False
    alerta_telegram_enviada: bool = False
    cap_rate_pct: float | None = None
    capex_estimado: float | None = None
    motivo: str | None = None
    ficha_url: str | None = None
    created_at: datetime


class ResumenPropiedadesBarrido(BaseModel):
    """Contadores agregados del estado de análisis de las propiedades del barrido."""

    total: int = 0
    analizadas: int = 0
    enviadas_aprobacion: int = 0
    rechazadas_directamente: int = 0
    errores_analisis: int = 0
    pendientes_analisis: int = 0


class PropiedadesBarridoResponse(BaseModel):
    """Respuesta completa para el visor de propiedades por barrido del panel."""

    job_id: UUID
    resumen: ResumenPropiedadesBarrido
    propiedades: list[PropiedadBarridoItem]


def _cap_rate_a_porcentaje(valor: Any) -> float | None:
    if valor is None:
        return None
    try:
        num = float(valor)
    except (TypeError, ValueError):
        return None
    if num <= 1.0:
        num *= 100.0
    return round(num, 2)


def _extraer_alerta_enviada(detalle: dict[str, Any]) -> bool:
    alerta = detalle.get("alerta")
    if isinstance(alerta, dict) and (
        bool(alerta.get("sent")) or str(alerta.get("status") or "").lower() == "sent"
    ):
        return True
    estado = str(detalle.get("estado") or "").upper()
    return estado == "ALERTA_ENVIADA"


def clasificar_propiedad_con_decision(
    propiedad: Propiedad,
    decision: Decision | None,
) -> PropiedadBarridoItem:
    """Clasifica una propiedad según su última decisión del orquestador."""
    fotos = [
        str(u).strip()
        for u in (propiedad.imagenes or [])
        if str(u).strip().startswith(("https://", "http://"))
    ][:2]
    contacto = extraer_contacto_vendedor(propiedad.descripcion_cruda)

    if decision is None:
        estado: EstadoAnalisisPropiedad = "pendiente_analisis"
        return PropiedadBarridoItem(
            propiedad_id=propiedad.id,
            plataforma=propiedad.plataforma,
            ubicacion=propiedad.ubicacion,
            ciudad=propiedad.ciudad,
            codigo_postal=propiedad.codigo_postal,
            precio_usd=float(propiedad.precio_usd),
            ingreso_anual_estimado=float(propiedad.ingreso_anual_estimado or 0.0),
            url_origen=propiedad.url_origen,
            imagenes=fotos,
            contacto_vendedor=contacto,
            analizada=False,
            estado_analisis=estado,
            etiqueta_estado=ETIQUETAS_ESTADO[estado],
            veredicto=None,
            enviada_a_aprobacion=False,
            alerta_telegram_enviada=False,
            created_at=propiedad.created_at,
        )

    detalle = decision.detalle if isinstance(decision.detalle, dict) else {}
    alerta_enviada = _extraer_alerta_enviada(detalle)
    veredicto = (decision.veredicto or "").strip().lower()

    if veredicto == "descartar":
        estado = "rechazada_directamente"
        enviada_aprobacion = False
    elif veredicto == "error":
        estado = "error_analisis"
        enviada_aprobacion = False
    else:
        estado = "enviada_aprobacion"
        enviada_aprobacion = True

    ficha_raw = detalle.get("ficha_url")
    ficha_url = str(ficha_raw) if isinstance(ficha_raw, str) and ficha_raw.strip() else None

    return PropiedadBarridoItem(
        propiedad_id=propiedad.id,
        plataforma=propiedad.plataforma,
        ubicacion=propiedad.ubicacion,
        ciudad=propiedad.ciudad,
        codigo_postal=propiedad.codigo_postal,
        precio_usd=float(propiedad.precio_usd),
        ingreso_anual_estimado=float(propiedad.ingreso_anual_estimado or 0.0),
        url_origen=propiedad.url_origen,
        imagenes=fotos,
        contacto_vendedor=contacto,
        analizada=True,
        estado_analisis=estado,
        etiqueta_estado=ETIQUETAS_ESTADO[estado],
        veredicto=veredicto or None,
        enviada_a_aprobacion=enviada_aprobacion,
        alerta_telegram_enviada=alerta_enviada,
        cap_rate_pct=_cap_rate_a_porcentaje(decision.cap_rate),
        capex_estimado=(
            float(decision.capex_estimado) if decision.capex_estimado is not None else None
        ),
        motivo=decision.motivo,
        ficha_url=ficha_url,
        created_at=propiedad.created_at,
    )


def construir_respuesta_propiedades_barrido(
    job_id: UUID,
    filas: Sequence[tuple[Propiedad, Decision | None]],
) -> PropiedadesBarridoResponse:
    """Construye la respuesta agregada y ordenada de propiedades del barrido."""
    orden_prioridad: dict[EstadoAnalisisPropiedad, int] = {
        "enviada_aprobacion": 0,
        "rechazada_directamente": 1,
        "error_analisis": 2,
        "pendiente_analisis": 3,
    }
    items = [clasificar_propiedad_con_decision(prop, dec) for prop, dec in filas]
    items.sort(key=lambda item: (orden_prioridad[item.estado_analisis], item.precio_usd))

    enviadas = sum(1 for item in items if item.estado_analisis == "enviada_aprobacion")
    rechazadas = sum(1 for item in items if item.estado_analisis == "rechazada_directamente")
    errores = sum(1 for item in items if item.estado_analisis == "error_analisis")
    pendientes = sum(1 for item in items if item.estado_analisis == "pendiente_analisis")

    return PropiedadesBarridoResponse(
        job_id=job_id,
        resumen=ResumenPropiedadesBarrido(
            total=len(items),
            analizadas=enviadas + rechazadas + errores,
            enviadas_aprobacion=enviadas,
            rechazadas_directamente=rechazadas,
            errores_analisis=errores,
            pendientes_analisis=pendientes,
        ),
        propiedades=items,
    )


async def obtener_propiedades_por_barrido(job_id: UUID) -> PropiedadesBarridoResponse:
    """Lee de PostgreSQL las propiedades del barrido y su decisión más reciente."""
    if not settings.DATABASE_URL.strip():
        raise RuntimeError("DATABASE_URL no está configurada")

    fabrica = obtener_sessionmaker()
    async with fabrica() as sesion:
        props_result = await sesion.execute(
            select(Propiedad)
            .where(Propiedad.job_id == job_id)
            .order_by(Propiedad.created_at.desc())
        )
        propiedades = list(props_result.scalars().all())
        if not propiedades:
            return construir_respuesta_propiedades_barrido(job_id, [])

        prop_ids = [p.id for p in propiedades]
        dec_result = await sesion.execute(
            select(Decision)
            .where(Decision.propiedad_id.in_(prop_ids))
            .order_by(Decision.created_at.desc())
        )
        ultima_decision: dict[UUID, Decision] = {}
        for decision in dec_result.scalars().all():
            ultima_decision.setdefault(decision.propiedad_id, decision)

    filas = [(prop, ultima_decision.get(prop.id)) for prop in propiedades]
    return construir_respuesta_propiedades_barrido(job_id, filas)
