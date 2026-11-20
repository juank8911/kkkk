"""Enrutador de due diligence y pipeline composable de nodos de dominio (Sprint 4)."""

from __future__ import annotations

from typing import Any, TypedDict

from app.orchestrator.config import AlertConfig, FinancialConfig, VisionConfig
from app.orchestrator.nodes.financial import nodo_analisis_financiero
from app.orchestrator.nodes.legal_audit import nodo_auditoria_legal
from app.orchestrator.nodes.telegram_alert import (
    AlertDedupeStorePort,
    NotifierPort,
    nodo_alerta_telegram,
)
from app.orchestrator.nodes.vision import ValuationPort, nodo_vision_artificial


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
    """Ejecuta el pipeline completo de evaluación: Router -> Financiero -> Visión -> Legal -> Alerta."""
    state: dict[str, Any] = dict(initial_state)

    ruta = enrutador_due_diligence(state, config=financial_config)
    if ruta == "descarte_inmediato":
        state["estado"] = "DESCARTADO_PRECIO"
        return state

    state.update(nodo_analisis_financiero(state, config=financial_config))
    if state.get("estado") != "ANALISIS_FINANCIERO_OK":
        return state

    state.update(
        await nodo_vision_artificial(
            state,
            valuator=valuator,
            vision_config=vision_config,
            financial_config=financial_config,
        )
    )
    if state.get("estado") == "RECHAZADO":
        return state

    state.update(nodo_auditoria_legal(state))

    state.update(
        await nodo_alerta_telegram(
            state,
            notifier=notifier,
            dedupe_store=dedupe_store,
            alert_config=alert_config,
            chat_id=chat_id,
        )
    )
    return state
