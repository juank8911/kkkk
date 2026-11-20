"""Persistencia de la decision del grafo por propiedad (S4-02).

El grafo devuelve un estado con el veredicto, el cap rate, el CAPEX y las
alertas legales. Aqui se traduce ese estado a una fila de ``decisiones`` y se
registra el evento correspondiente dentro de la misma transaccion, de modo que
cada propiedad evaluada deja rastro auditable.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from app.db.models import Decision, Propiedad
from app.db.uow import UnidadDeTrabajo
from app.orchestrator.graph import ESTADO_FALLO, evaluar_propiedad

logger = logging.getLogger("OpenClaw.Decisiones")

VEREDICTOS = frozenset({"comprar", "revisar", "descartar", "error"})
MAX_MOTIVO = 500


def _fraccion(valor: Any) -> float | None:
    """Normaliza un porcentaje a fraccion para ``Numeric(6,4)``."""
    if valor is None:
        return None
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    if numero > 1.0:
        numero = numero / 100.0
    return round(numero, 4)


def veredicto_desde_estado(state: dict[str, Any]) -> str:
    """Traduce el estado final del grafo a un veredicto de la tabla."""
    estado = str(state.get("estado") or "").upper()

    if estado == ESTADO_FALLO or estado.startswith("FALLO"):
        return "error"
    if "DESCART" in estado or "RECHAZ" in estado:
        return "descartar"
    if state.get("requiere_revision_humana"):
        return "revisar"
    if estado == "ALERTA_ENVIADA":
        return "comprar"
    return "revisar"


def construir_decision(state: dict[str, Any], *, propiedad_id: UUID) -> Decision:
    """Construye la fila de decision a partir del estado final del grafo."""
    veredicto = veredicto_desde_estado(state)
    cap_rate = state.get("cap_rate_ajustado")
    if cap_rate is None:
        cap_rate = state.get("cap_rate")
    capex = state.get("capex")
    motivo = str(
        state.get("motivo_fallo") or state.get("motivo_financiero") or state.get("estado") or ""
    )[:MAX_MOTIVO]

    detalle: dict[str, object] = {
        "estado": state.get("estado"),
        "ruta": state.get("ruta"),
        "nodo_fallido": state.get("nodo_fallido"),
        "ficha_url": state.get("ficha_url"),
        "vision": {
            "disponible": state.get("vision_disponible"),
            "confianza": state.get("vision_confianza"),
            "motivo": state.get("vision_motivo"),
        },
        "legal": {
            "estado": state.get("estado_legal"),
            "riesgo": state.get("riesgo_legal"),
            "requiere_revision_humana": state.get("requiere_revision_humana"),
            "senales": state.get("senales_legales") or [],
        },
        "alerta": state.get("alerta"),
    }

    return Decision(
        propiedad_id=propiedad_id,
        veredicto=veredicto,
        cap_rate=_fraccion(cap_rate),
        capex_estimado=float(capex) if capex is not None else None,
        motivo=motivo,
        detalle=detalle,
    )


def estado_inicial_desde_orm(propiedad: Propiedad) -> dict[str, Any]:
    """Adapta una fila de ``propiedades`` al estado que consume el grafo."""
    return {
        "property_id": str(propiedad.id),
        "job_id": str(propiedad.job_id),
        "dedup_key": propiedad.huella_dedupe or propiedad.url_origen,
        "ubicacion": propiedad.ubicacion,
        "condado": propiedad.condado or "N/A",
        "precio_usd": float(propiedad.precio_usd),
        "ingreso_anual_estimado": float(propiedad.ingreso_anual_estimado),
        "hoa_mensual_usd": 0.0,
        "imagenes": list(propiedad.imagenes or []),
        "descripcion_cruda": propiedad.descripcion_cruda,
        "notas_legales": propiedad.descripcion_cruda,
    }


async def evaluar_y_registrar_decision(
    unidad: UnidadDeTrabajo,
    *,
    propiedad_id: UUID,
    estado_inicial: dict[str, Any],
    job_id: UUID | None = None,
    thread_id: str | None = None,
    **kwargs: Any,
) -> Decision:
    """Ejecuta el grafo y persiste la decision con su evento de auditoria."""
    resultado = await evaluar_propiedad(
        estado_inicial,
        thread_id=thread_id or f"propiedad:{propiedad_id}",
        **kwargs,
    )

    decision = construir_decision(resultado, propiedad_id=propiedad_id)
    guardada = await unidad.decisiones.registrar(decision)

    if job_id is not None:
        await unidad.eventos.registrar(
            "decision.registrada",
            f"Veredicto {guardada.veredicto} para la propiedad {propiedad_id}",
            job_id=job_id,
            nivel="info" if guardada.veredicto != "error" else "error",
            payload={
                "propiedad_id": str(propiedad_id),
                "veredicto": guardada.veredicto,
                "cap_rate": str(guardada.cap_rate),
                "motivo": guardada.motivo,
            },
        )

    await unidad.confirmar()
    logger.info(
        "decision_registrada",
        extra={"propiedad_id": str(propiedad_id), "veredicto": guardada.veredicto},
    )
    return guardada
