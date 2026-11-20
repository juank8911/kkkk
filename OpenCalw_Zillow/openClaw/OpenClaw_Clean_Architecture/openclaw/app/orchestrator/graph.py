"""Grafo de due diligence con aristas condicionales y checkpointer (S4-01, S4-03).

El recorrido es:

    clasificar -> financiero -> vision -> legal -> alerta

con dos salidas tempranas: el precio fuera de rango descarta sin llamar al
proveedor, y un nodo que agota sus reintentos termina el grafo en ``FALLO_NODO``
dejando el caso en la DLQ de decisiones.

Si `langgraph` no esta instalado (entornos minimos de pruebas), se usa el
pipeline secuencial de `app.orchestrator.workflow` para no romper el worker.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

from app.core.config import settings
from app.orchestrator.config import AlertConfig, FinancialConfig, VisionConfig
from app.orchestrator.dlq import registrar_fallo_nodo
from app.orchestrator.nodes.financial import nodo_analisis_financiero
from app.orchestrator.nodes.legal_audit import nodo_auditoria_legal
from app.orchestrator.nodes.telegram_alert import (
    AlertDedupeStorePort,
    NotifierPort,
    nodo_alerta_telegram,
)
from app.orchestrator.nodes.vision import ValuationPort, nodo_vision_artificial
from app.orchestrator.workflow import (
    PropertyEvaluationState,
    ejecutar_pipeline_propiedad,
    enrutador_due_diligence,
)

logger = logging.getLogger("OpenClaw.Grafo")

try:  # pragma: no cover - depende del entorno
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph

    LANGGRAPH_DISPONIBLE = True
except (ImportError, TypeError):  # pragma: no cover - incompatibilidad pydantic.v1/Py3.12+
    MemorySaver = None  # type: ignore[assignment]
    END = "__end__"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]
    LANGGRAPH_DISPONIBLE = False

FIN = "__fin__"

NODO_CLASIFICAR = "clasificar"
NODO_FINANCIERO = "financiero"
NODO_VISION = "vision"
NODO_LEGAL = "legal"
NODO_ALERTA = "alerta"

ESTADO_FALLO = "FALLO_NODO"

NodoFn = Callable[[dict[str, Any]], Any]


class EstadoGrafo(PropertyEvaluationState, total=False):
    """Estado del grafo: el de los nodos mas lo que necesita el enrutado."""

    job_id: str
    ruta: str
    nodo_fallido: str
    motivo_fallo: str


def langgraph_disponible() -> bool:
    return LANGGRAPH_DISPONIBLE


async def _invocar(fn: NodoFn, state: dict[str, Any]) -> dict[str, Any]:
    resultado = fn(state)
    if inspect.isawaitable(resultado):
        resultado = await resultado
    return dict(resultado or {})


def con_reintentos(
    nombre: str,
    fn: NodoFn,
    *,
    intentos: int | None = None,
) -> NodoFn:
    """Envuelve un nodo con reintentos y DLQ al agotarlos (S4-03)."""
    limite = settings.DECISION_MAX_INTENTOS_NODO if intentos is None else max(1, intentos)

    async def envuelto(state: dict[str, Any]) -> dict[str, Any]:
        ultimo: Exception | None = None
        for intento in range(1, limite + 1):
            try:
                return await _invocar(fn, state)
            except Exception as exc:  # noqa: BLE001 - se reintenta cualquier fallo del nodo
                ultimo = exc
                logger.warning(
                    "nodo_fallo",
                    extra={"nodo": nombre, "intento": intento, "limite": limite, "error": str(exc)},
                )
        motivo = f"{type(ultimo).__name__}: {ultimo}"
        await registrar_fallo_nodo(nodo=nombre, state=state, motivo=motivo, intentos=limite)
        return {"estado": ESTADO_FALLO, "nodo_fallido": nombre, "motivo_fallo": motivo}

    return envuelto


def _clasificar(state: dict[str, Any], config: FinancialConfig | None) -> dict[str, Any]:
    ruta = enrutador_due_diligence(state, config=config)
    if ruta == "descarte_inmediato":
        return {"ruta": ruta, "estado": "DESCARTADO_PRECIO"}
    return {"ruta": ruta}


def _tras_clasificar(state: dict[str, Any]) -> str:
    return FIN if state.get("ruta") == "descarte_inmediato" else NODO_FINANCIERO


def _tras_financiero(state: dict[str, Any]) -> str:
    if state.get("estado") in {ESTADO_FALLO, "RECHAZADO"}:
        return FIN
    return NODO_VISION


def _tras_vision(state: dict[str, Any]) -> str:
    if state.get("estado") in {ESTADO_FALLO, "RECHAZADO"}:
        return FIN
    return NODO_LEGAL


def _tras_legal(state: dict[str, Any]) -> str:
    return FIN if state.get("estado") == ESTADO_FALLO else NODO_ALERTA


def construir_grafo(
    *,
    valuator: ValuationPort | None = None,
    notifier: NotifierPort | None = None,
    dedupe_store: AlertDedupeStorePort | None = None,
    financial_config: FinancialConfig | None = None,
    vision_config: VisionConfig | None = None,
    alert_config: AlertConfig | None = None,
    chat_id: str | None = None,
    checkpointer: Any | None = None,
    intentos_por_nodo: int | None = None,
) -> Any:
    """Compila el grafo con checkpointer en memoria por defecto."""
    if not LANGGRAPH_DISPONIBLE or StateGraph is None:
        raise RuntimeError("langgraph no esta instalado: use evaluar_propiedad() como fachada")

    async def financiero(state: dict[str, Any]) -> dict[str, Any]:
        return await _invocar(lambda s: nodo_analisis_financiero(s, config=financial_config), state)

    async def vision(state: dict[str, Any]) -> dict[str, Any]:
        return await _invocar(
            lambda s: nodo_vision_artificial(
                s,
                valuator=valuator,
                vision_config=vision_config,
                financial_config=financial_config,
            ),
            state,
        )

    async def legal(state: dict[str, Any]) -> dict[str, Any]:
        return await _invocar(nodo_auditoria_legal, state)

    async def alerta(state: dict[str, Any]) -> dict[str, Any]:
        return await _invocar(
            lambda s: nodo_alerta_telegram(
                s,
                notifier=notifier,
                dedupe_store=dedupe_store,
                alert_config=alert_config,
                chat_id=chat_id,
            ),
            state,
        )

    grafo = StateGraph(EstadoGrafo)
    grafo.add_node(NODO_CLASIFICAR, con_reintentos(NODO_CLASIFICAR, lambda s: _clasificar(s, financial_config), intentos=1))
    grafo.add_node(NODO_FINANCIERO, con_reintentos(NODO_FINANCIERO, financiero, intentos=intentos_por_nodo))
    grafo.add_node(NODO_VISION, con_reintentos(NODO_VISION, vision, intentos=intentos_por_nodo))
    grafo.add_node(NODO_LEGAL, con_reintentos(NODO_LEGAL, legal, intentos=intentos_por_nodo))
    grafo.add_node(NODO_ALERTA, con_reintentos(NODO_ALERTA, alerta, intentos=intentos_por_nodo))

    grafo.set_entry_point(NODO_CLASIFICAR)
    grafo.add_conditional_edges(
        NODO_CLASIFICAR, _tras_clasificar, {NODO_FINANCIERO: NODO_FINANCIERO, FIN: END}
    )
    grafo.add_conditional_edges(
        NODO_FINANCIERO, _tras_financiero, {NODO_VISION: NODO_VISION, FIN: END}
    )
    grafo.add_conditional_edges(NODO_VISION, _tras_vision, {NODO_LEGAL: NODO_LEGAL, FIN: END})
    grafo.add_conditional_edges(NODO_LEGAL, _tras_legal, {NODO_ALERTA: NODO_ALERTA, FIN: END})
    grafo.add_edge(NODO_ALERTA, END)

    if checkpointer is None and MemorySaver is not None:
        checkpointer = MemorySaver()
    return grafo.compile(checkpointer=checkpointer)


async def evaluar_propiedad(
    initial_state: dict[str, Any],
    *,
    thread_id: str | None = None,
    valuator: ValuationPort | None = None,
    notifier: NotifierPort | None = None,
    dedupe_store: AlertDedupeStorePort | None = None,
    financial_config: FinancialConfig | None = None,
    vision_config: VisionConfig | None = None,
    alert_config: AlertConfig | None = None,
    chat_id: str | None = None,
    checkpointer: Any | None = None,
    intentos_por_nodo: int | None = None,
) -> dict[str, Any]:
    """Ejecuta el grafo para una propiedad y devuelve el estado final."""
    estado = dict(initial_state)

    if not LANGGRAPH_DISPONIBLE:
        logger.warning("langgraph_ausente_pipeline_secuencial")
        return await ejecutar_pipeline_propiedad(
            estado,
            valuator=valuator,
            notifier=notifier,
            dedupe_store=dedupe_store,
            financial_config=financial_config,
            vision_config=vision_config,
            alert_config=alert_config,
            chat_id=chat_id,
        )

    grafo = construir_grafo(
        valuator=valuator,
        notifier=notifier,
        dedupe_store=dedupe_store,
        financial_config=financial_config,
        vision_config=vision_config,
        alert_config=alert_config,
        chat_id=chat_id,
        checkpointer=checkpointer,
        intentos_por_nodo=intentos_por_nodo,
    )
    configuracion: dict[str, Any] = {}
    if thread_id:
        configuracion["configurable"] = {"thread_id": thread_id}

    resultado = await grafo.ainvoke(estado, config=configuracion or None)
    return dict(resultado)
