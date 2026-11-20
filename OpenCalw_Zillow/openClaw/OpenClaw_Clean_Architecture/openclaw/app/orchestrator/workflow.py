"""Enrutador de due diligence, mapa de nodos LangGraph y persistencia de decisiones (Sprint 4)."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict
from uuid import UUID

from app.db.models import Decision
from app.db.uow import UnidadDeTrabajo
from app.orchestrator.config import AlertConfig, FinancialConfig, VisionConfig
from app.orchestrator.nodes.financial import nodo_analisis_financiero
from app.orchestrator.nodes.legal_audit import nodo_auditoria_legal
from app.orchestrator.nodes.telegram_alert import (
    AlertDedupeStorePort,
    NotifierPort,
    nodo_alerta_telegram,
)
from app.orchestrator.nodes.vision import ValuationPort, nodo_vision_artificial
from app.services.db_mappers import map_pipeline_state_to_decision_orm
from app.services.enrichment import enriquecer_propiedad_desde_markdown

NodeCallable = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class PropertyEvaluationState(TypedDict, total=False):
    """Estado tipado compartido por los nodos de evaluación (compatible con LangGraph S4-01)."""

    property_id: str
    dedup_key: str
    ubicacion: str
    condado: str
    precio_usd: float
    ingreso_anual_estimado: float
    hoa_mensual_usd: float
    imagenes: list[str]
    descripcion_cruda: str
    notas_legales: str
    estado: str
    noi: float
    cap_rate: float
    cap_rate_ajustado: float
    capex: float | None
    vision_disponible: bool
    vision_confianza: float
    vision_motivo: str
    estado_legal: str
    riesgo_legal: str
    requiere_revision_humana: bool
    senales_legales: list[dict[str, Any]]
    ficha_url: str
    alerta: dict[str, Any]
    traza_nodos: list[dict[str, Any]]


def enrutador_due_diligence(
    state: dict[str, Any],
    *,
    config: FinancialConfig | None = None,
) -> str:
    """Enruta la propiedad hacia el nodo financiero o descarte inmediato según precio."""
    max_price = config.max_price_usd if config is not None else 150_000.0
    precio = float(state.get("precio_usd") or 0.0)
    if precio <= 0.0 or precio > max_price:
        return "descarte_inmediato"
    return "agente_financiero"


def construir_mapa_nodos(
    *,
    valuator: ValuationPort | None = None,
    notifier: NotifierPort | None = None,
    dedupe_store: AlertDedupeStorePort | None = None,
    financial_config: FinancialConfig | None = None,
    vision_config: VisionConfig | None = None,
    alert_config: AlertConfig | None = None,
    chat_id: str | None = None,
) -> dict[str, NodeCallable]:
    """Devuelve los nodos asíncronos listos para registrarse en `StateGraph` (S4-01)."""

    async def _financiero(state: dict[str, Any]) -> dict[str, Any]:
        return nodo_analisis_financiero(state, config=financial_config)

    async def _vision(state: dict[str, Any]) -> dict[str, Any]:
        return await nodo_vision_artificial(
            state,
            valuator=valuator,
            vision_config=vision_config,
            financial_config=financial_config,
        )

    async def _legal(state: dict[str, Any]) -> dict[str, Any]:
        return nodo_auditoria_legal(state)

    async def _alerta(state: dict[str, Any]) -> dict[str, Any]:
        return await nodo_alerta_telegram(
            state,
            notifier=notifier,
            dedupe_store=dedupe_store,
            alert_config=alert_config,
            chat_id=chat_id,
        )

    return {
        "agente_financiero": _financiero,
        "vision": _vision,
        "auditoria_legal": _legal,
        "alerta": _alerta,
    }


def _registrar_traza(
    traza: list[dict[str, Any]],
    nodo: str,
    started: float,
    estado_salida: str,
) -> None:
    traza.append(
        {
            "nodo": nodo,
            "duracion_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "estado": estado_salida,
        }
    )


async def ejecutar_pipeline_propiedad(
    initial_state: dict[str, Any],
    *,
    valuator: ValuationPort | None = None,
    notifier: NotifierPort | None = None,
    dedupe_store: AlertDedupeStorePort | None = None,
    financial_config: FinancialConfig | None = None,
    vision_config: VisionConfig | None = None,
    alert_config: AlertConfig | None = None,
    chat_id: str | None = None,
) -> dict[str, Any]:
    """Ejecuta el pipeline completo de evaluación con trazabilidad por nodo."""
    state: dict[str, Any] = dict(initial_state)
    traza: list[dict[str, Any]] = list(state.get("traza_nodos") or [])

    # Enriquecimiento automático de HOA si viene en la descripción y no se pasó explícito
    if "hoa_mensual_usd" not in state and state.get("descripcion_cruda"):
        enr = enriquecer_propiedad_desde_markdown(str(state["descripcion_cruda"]))
        if enr.hoa_mensual_usd is not None:
            state["hoa_mensual_usd"] = enr.hoa_mensual_usd

    nodos = construir_mapa_nodos(
        valuator=valuator,
        notifier=notifier,
        dedupe_store=dedupe_store,
        financial_config=financial_config,
        vision_config=vision_config,
        alert_config=alert_config,
        chat_id=chat_id,
    )

    t0 = time.perf_counter()
    ruta = enrutador_due_diligence(state, config=financial_config)
    _registrar_traza(traza, "enrutador", t0, ruta)
    if ruta == "descarte_inmediato":
        state["estado"] = "DESCARTADO_PRECIO"
        state["traza_nodos"] = traza
        return state

    t0 = time.perf_counter()
    state.update(await nodos["agente_financiero"](state))
    _registrar_traza(traza, "agente_financiero", t0, str(state.get("estado")))
    if state.get("estado") != "ANALISIS_FINANCIERO_OK":
        state["traza_nodos"] = traza
        return state

    t0 = time.perf_counter()
    state.update(await nodos["vision"](state))
    _registrar_traza(traza, "vision", t0, str(state.get("estado")))
    if state.get("estado") == "RECHAZADO":
        state["traza_nodos"] = traza
        return state

    t0 = time.perf_counter()
    state.update(await nodos["auditoria_legal"](state))
    _registrar_traza(traza, "auditoria_legal", t0, str(state.get("estado_legal")))

    t0 = time.perf_counter()
    state.update(await nodos["alerta"](state))
    _registrar_traza(traza, "alerta", t0, str(state.get("estado")))

    state["traza_nodos"] = traza
    return state


async def evaluar_y_persistir_propiedad_con_uow(
    unidad: UnidadDeTrabajo,
    *,
    propiedad_id: UUID,
    job_id: UUID | None,
    initial_state: dict[str, Any],
    valuator: ValuationPort | None = None,
    notifier: NotifierPort | None = None,
    dedupe_store: AlertDedupeStorePort | None = None,
    financial_config: FinancialConfig | None = None,
    vision_config: VisionConfig | None = None,
    alert_config: AlertConfig | None = None,
    chat_id: str | None = None,
) -> tuple[dict[str, Any], Decision]:
    """Evalúa una propiedad a través del pipeline S4 y registra su `Decision` + `Evento` en UoW."""
    final_state = await ejecutar_pipeline_propiedad(
        {"property_id": str(propiedad_id), **initial_state},
        valuator=valuator,
        notifier=notifier,
        dedupe_store=dedupe_store,
        financial_config=financial_config,
        vision_config=vision_config,
        alert_config=alert_config,
        chat_id=chat_id,
    )
    decision_orm = map_pipeline_state_to_decision_orm(
        propiedad_id=propiedad_id,
        state=final_state,
    )
    saved_decision = await unidad.decisiones.registrar(decision_orm)
    await unidad.eventos.registrar(
        "orquestador.decision",
        f"Decisión '{saved_decision.veredicto}' registrada para propiedad {propiedad_id}",
        job_id=job_id,
        nivel="warning" if final_state.get("requiere_revision_humana") else "info",
        payload={
            "propiedad_id": str(propiedad_id),
            "veredicto": saved_decision.veredicto,
            "estado": final_state.get("estado"),
            "traza_nodos": final_state.get("traza_nodos", []),
        },
    )
    return final_state, saved_decision
