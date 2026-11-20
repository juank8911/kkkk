"""Pruebas del grafo de due diligence, la DLQ y el registro de decisiones (S4)."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import fakeredis.aioredis
import pytest

from app.core.config import settings
from app.orchestrator import decisiones, dlq, graph
from app.orchestrator.decisiones import (
    _fraccion,
    construir_decision,
    estado_inicial_desde_orm,
    evaluar_y_registrar_decision,
    veredicto_desde_estado,
)

PROPIEDAD_ID = uuid4()
JOB_ID = uuid4()


@pytest.fixture
def redis_falso(
    monkeypatch: pytest.MonkeyPatch, fake_redis: fakeredis.aioredis.FakeRedis
) -> fakeredis.aioredis.FakeRedis:
    monkeypatch.setattr(dlq, "redis_client", fake_redis)
    return fake_redis


# --- veredictos -------------------------------------------------------------


def test_veredicto_error_por_fallo_de_nodo() -> None:
    assert veredicto_desde_estado({"estado": "FALLO_NODO"}) == "error"


def test_veredicto_descartar_por_precio() -> None:
    assert veredicto_desde_estado({"estado": "DESCARTADO_PRECIO"}) == "descartar"
    assert veredicto_desde_estado({"estado": "RECHAZADO"}) == "descartar"


def test_veredicto_revisar_por_riesgo_legal() -> None:
    assert (
        veredicto_desde_estado({"estado": "AUDITORIA_LEGAL_ALERTA", "requiere_revision_humana": True})
        == "revisar"
    )


def test_veredicto_comprar_solo_si_se_alerto() -> None:
    assert veredicto_desde_estado({"estado": "ALERTA_ENVIADA"}) == "comprar"
    assert veredicto_desde_estado({"estado": "ALERTA_OMITIDA"}) == "revisar"


def test_fraccion_normaliza_porcentajes() -> None:
    assert _fraccion(9.5) == pytest.approx(0.095)
    assert _fraccion(0.095) == pytest.approx(0.095)
    assert _fraccion(None) is None
    assert _fraccion("no-numerico") is None


def test_construir_decision_mapea_campos() -> None:
    decision = construir_decision(
        {
            "estado": "ALERTA_ENVIADA",
            "cap_rate_ajustado": 9.5,
            "capex": 12000.0,
            "ficha_url": "https://openclaw.internal/propiedades/1",
            "vision_disponible": True,
            "vision_confianza": 0.82,
            "estado_legal": "AUDITORIA_LEGAL_OK",
        },
        propiedad_id=PROPIEDAD_ID,
    )

    assert decision.propiedad_id == PROPIEDAD_ID
    assert decision.veredicto == "comprar"
    assert decision.cap_rate == pytest.approx(0.095)
    assert decision.capex_estimado == pytest.approx(12000.0)
    assert decision.detalle["vision"]["confianza"] == pytest.approx(0.82)
    assert decision.detalle["ficha_url"].endswith("/1")


def test_estado_inicial_desde_orm() -> None:
    class FilaFalsa:
        id = PROPIEDAD_ID
        job_id = JOB_ID
        huella_dedupe = "abc123"
        url_origen = "https://zillow.com/h/1"
        ubicacion = "Cleveland, OH"
        condado = None
        precio_usd = 95000
        ingreso_anual_estimado = 11400
        imagenes = ["a.jpg"]
        descripcion_cruda = "Casa de 3 habitaciones"

    estado = estado_inicial_desde_orm(FilaFalsa())  # type: ignore[arg-type]

    assert estado["property_id"] == str(PROPIEDAD_ID)
    assert estado["dedup_key"] == "abc123"
    assert estado["condado"] == "N/A"
    assert estado["precio_usd"] == pytest.approx(95000.0)
    assert estado["notas_legales"] == "Casa de 3 habitaciones"


# --- reintentos y DLQ -------------------------------------------------------


async def test_nodo_que_falla_dos_veces_va_a_dlq(
    redis_falso: fakeredis.aioredis.FakeRedis,
) -> None:
    llamadas: list[int] = []

    async def nodo_roto(state: dict[str, Any]) -> dict[str, Any]:
        llamadas.append(1)
        raise RuntimeError("parser roto")

    envuelto = graph.con_reintentos("legal", nodo_roto, intentos=2)
    resultado = await envuelto({"property_id": str(PROPIEDAD_ID), "job_id": str(JOB_ID)})

    assert len(llamadas) == 2
    assert resultado["estado"] == graph.ESTADO_FALLO
    assert resultado["nodo_fallido"] == "legal"

    entradas = await redis_falso.xrange(settings.REDIS_STREAM_DLQ_DECISIONES, "-", "+")
    assert len(entradas) == 1
    campos = entradas[0][1]
    assert campos[b"nodo"] == b"legal"
    assert campos[b"intentos"] == b"2"
    assert b"parser roto" in campos[b"motivo"]
    assert campos[b"propiedad_id"] == str(PROPIEDAD_ID).encode()


async def test_nodo_que_falla_una_vez_se_recupera(
    redis_falso: fakeredis.aioredis.FakeRedis,
) -> None:
    intentos: list[int] = []

    async def nodo_inestable(state: dict[str, Any]) -> dict[str, Any]:
        intentos.append(1)
        if len(intentos) == 1:
            raise RuntimeError("timeout")
        return {"estado": "OK"}

    envuelto = graph.con_reintentos("vision", nodo_inestable, intentos=2)
    resultado = await envuelto({})

    assert resultado == {"estado": "OK"}
    entradas = await redis_falso.xrange(settings.REDIS_STREAM_DLQ_DECISIONES, "-", "+")
    assert entradas == []


async def test_dlq_no_rompe_el_grafo_si_redis_falla(monkeypatch: pytest.MonkeyPatch) -> None:
    class RedisCaido:
        async def xadd(self, *args: Any, **kwargs: Any) -> None:
            raise ConnectionError("redis caido")

    monkeypatch.setattr(dlq, "redis_client", RedisCaido())
    registro = dlq.construir_registro_fallo(nodo="alerta", motivo="x", intentos=2)
    assert await dlq.enviar_a_dlq(registro) is None


# --- registro de decisiones -------------------------------------------------


class DecisionesFalsas:
    def __init__(self) -> None:
        self.registradas: list[Any] = []

    async def registrar(self, decision: Any) -> Any:
        decision.id = uuid4()
        self.registradas.append(decision)
        return decision


class EventosFalsos:
    def __init__(self) -> None:
        self.eventos: list[tuple[str, str]] = []

    async def registrar(self, tipo: str, mensaje: str, **kwargs: Any) -> None:
        self.eventos.append((tipo, mensaje))


class UnidadFalsa:
    def __init__(self) -> None:
        self.decisiones = DecisionesFalsas()
        self.eventos = EventosFalsos()
        self.confirmaciones = 0

    async def confirmar(self) -> None:
        self.confirmaciones += 1


async def test_evaluar_y_registrar_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def grafo_falso(estado: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return {**estado, "estado": "ALERTA_ENVIADA", "cap_rate": 10.0, "capex": 5000.0}

    monkeypatch.setattr(decisiones, "evaluar_propiedad", grafo_falso)
    unidad = UnidadFalsa()

    decision = await evaluar_y_registrar_decision(
        unidad,  # type: ignore[arg-type]
        propiedad_id=PROPIEDAD_ID,
        estado_inicial={"precio_usd": 90000.0},
        job_id=JOB_ID,
    )

    assert decision.veredicto == "comprar"
    assert decision.cap_rate == pytest.approx(0.10)
    assert unidad.decisiones.registradas == [decision]
    assert unidad.eventos.eventos[0][0] == "decision.registrada"
    assert unidad.confirmaciones == 1


# --- grafo real -------------------------------------------------------------


def test_grafo_descarta_por_precio_fuera_de_rango() -> None:
    pytest.importorskip("langgraph")
    import asyncio

    resultado = asyncio.run(graph.evaluar_propiedad({"precio_usd": 0.0}))

    assert resultado["estado"] == "DESCARTADO_PRECIO"
    assert resultado["ruta"] == "descarte_inmediato"


def test_grafo_expone_aristas_condicionales() -> None:
    pytest.importorskip("langgraph")

    assert graph.langgraph_disponible() is True
    assert graph._tras_clasificar({"ruta": "descarte_inmediato"}) == graph.FIN
    assert graph._tras_clasificar({"ruta": "agente_financiero"}) == graph.NODO_FINANCIERO
    assert graph._tras_financiero({"estado": "RECHAZADO"}) == graph.FIN
    assert graph._tras_financiero({"estado": "ANALISIS_FINANCIERO_OK"}) == graph.NODO_VISION
    assert graph._tras_vision({"estado": "RECHAZADO"}) == graph.FIN
    assert graph._tras_legal({"estado": graph.ESTADO_FALLO}) == graph.FIN
    assert graph._tras_legal({"estado": "AUDITORIA_LEGAL_OK"}) == graph.NODO_ALERTA
