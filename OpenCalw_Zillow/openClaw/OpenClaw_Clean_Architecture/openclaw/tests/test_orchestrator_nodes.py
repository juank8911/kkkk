"""Pruebas de nodos de orquestación Sprint 4: financiero, visión, legal y alerta."""

from __future__ import annotations

from typing import Any

import fakeredis.aioredis
import httpx
import pytest

from app.orchestrator.config import AlertConfig, FinancialConfig, VisionConfig
from app.orchestrator.nodes.financial import nodo_analisis_financiero
from app.orchestrator.nodes.legal_audit import auditar_riesgo_legal, nodo_auditoria_legal
from app.orchestrator.nodes.telegram_alert import (
    InMemoryAlertDedupeStore,
    RedisAlertDedupeStore,
    TelegramNotifierAdapter,
    nodo_alerta_telegram,
)
from app.orchestrator.nodes.vision import VisionAssessment, nodo_vision_artificial
from app.orchestrator.workflow import ejecutar_pipeline_propiedad


class FakeMultimodalValuator:
    def __init__(self, assessment: VisionAssessment | Exception) -> None:
        self._assessment = assessment

    async def evaluate_images(
        self,
        image_urls: list[str],
        *,
        context: dict[str, Any],
    ) -> VisionAssessment:
        if isinstance(self._assessment, Exception):
            raise self._assessment
        return self._assessment.model_copy(update={"imagenes_evaluadas": len(image_urls)})


class FakeNotifier:
    def __init__(self) -> None:
        self.sent_messages: list[tuple[str, str]] = []

    async def send_message(self, chat_id: str, text: str) -> tuple[bool, str | None]:
        self.sent_messages.append((chat_id, text))
        return True, None


def test_s4_04_financial_node_changes_cap_rate_from_config_without_code_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Criterio S4-04: cambiar tasas en config altera el cap rate sin tocar código."""
    state = {"precio_usd": 100_000.0, "ingreso_anual_estimado": 18_000.0}

    cfg_base = FinancialConfig(
        vacancy_rate=0.08,
        maintenance_rate=0.10,
        property_tax_rate=0.02,
        insurance_annual_usd=1200.0,
        min_cap_rate_pct=8.0,
    )
    res_base = nodo_analisis_financiero(state, config=cfg_base)
    assert res_base["estado"] == "ANALISIS_FINANCIERO_OK"
    assert res_base["cap_rate"] == pytest.approx(11.56, abs=0.01)

    # Cambiar tasas vía variables de entorno altera el cap rate sin tocar código
    monkeypatch.setenv("OPENCLAW_FIN_VACANCY_RATE", "0.20")
    monkeypatch.setenv("OPENCLAW_FIN_MAINTENANCE_RATE", "0.20")
    monkeypatch.setenv("OPENCLAW_FIN_PROPERTY_TAX_RATE", "0.03")
    monkeypatch.setenv("OPENCLAW_FIN_INSURANCE_ANNUAL_USD", "2000")
    res_strict = nodo_analisis_financiero(state)
    assert res_strict["cap_rate"] < res_base["cap_rate"]
    assert res_strict["estado"] == "RECHAZADO"


async def test_s4_05_vision_node_returns_structured_capex_and_degrades_without_images() -> None:
    """Criterio S4-05: devuelve CAPEX, confianza y motivo; sin imágenes marca no disponible."""
    base_state = {
        "ubicacion": "1420 Maple Ave, Cleveland, OH 44102",
        "precio_usd": 100_000.0,
        "noi": 12_000.0,
        "imagenes": ["https://photos.zillowstatic.com/1001.jpg"],
    }

    valuator_ok = FakeMultimodalValuator(
        VisionAssessment(
            capex_usd=12_000.0,
            confianza=0.87,
            motivo="Techo en buen estado; requiere pintura interior y renovación menor de cocina.",
            severidad="baja",
        )
    )
    res_ok = await nodo_vision_artificial(base_state, valuator=valuator_ok)
    assert res_ok["estado"] == "INSPECCION_VISUAL_OK"
    assert res_ok["vision_disponible"] is True
    assert res_ok["capex"] == 12_000.0
    assert res_ok["vision_confianza"] == pytest.approx(0.87)
    assert "pintura" in res_ok["vision_motivo"]
    assert res_ok["cap_rate_ajustado"] == pytest.approx(10.7143, abs=0.01)

    # Sin imágenes -> no disponible
    no_img_state = {**base_state, "imagenes": []}
    res_no_img = await nodo_vision_artificial(no_img_state, valuator=valuator_ok)
    assert res_no_img["estado"] == "VISION_NO_DISPONIBLE"
    assert res_no_img["vision_disponible"] is False
    assert res_no_img["capex"] is None
    assert "no disponible" in res_no_img["vision_evaluacion"]["estado_vision"].replace("_", " ")

    # Baja confianza -> degrada a no disponible
    valuator_low = FakeMultimodalValuator(
        VisionAssessment(
            capex_usd=20_000.0,
            confianza=0.25,
            motivo="Imágenes borrosas e insuficientes.",
            severidad="desconocida",
        )
    )
    res_low = await nodo_vision_artificial(
        base_state, valuator=valuator_low, vision_config=VisionConfig(min_confidence=0.6)
    )
    assert res_low["estado"] == "VISION_NO_DISPONIBLE"
    assert res_low["vision_disponible"] is False

    # Error del proveedor -> degrada limpiamente sin romper el flujo
    res_err = await nodo_vision_artificial(
        base_state, valuator=FakeMultimodalValuator(RuntimeError("LLM timeout"))
    )
    assert res_err["estado"] == "VISION_NO_DISPONIBLE"


def test_s4_06_legal_audit_detects_liens_and_back_taxes_requiring_human_review() -> None:
    """Criterio S4-06: detecta señales de liens y back taxes; toda alerta exige revisión humana."""
    risky_state = {
        "descripcion_cruda": (
            "As-is cash sale. Property carries delinquent taxes from 2023-2025 and "
            "an active municipal tax lien plus an open HOA lien. Buyer assumes code violations."
        )
    }
    res_risky = nodo_auditoria_legal(risky_state)
    assert res_risky["estado_legal"] == "AUDITORIA_LEGAL_ALERTA"
    assert res_risky["riesgo_legal"] == "alto"
    assert res_risky["requiere_revision_humana"] is True

    categorias = {s["categoria"] for s in res_risky["senales_legales"]}
    assert "tax_lien" in categorias
    assert "hoa_lien" in categorias
    assert "back_taxes" in categorias
    assert "code_violation" in categorias

    clean = auditar_riesgo_legal("Turnkey renovated brick colonial with clear marketable title.")
    assert clean.estado == "AUDITORIA_LEGAL_OK"
    assert clean.riesgo == "bajo"
    assert clean.requiere_revision_humana is False


async def test_s4_07_alert_node_template_dedupe_and_ficha_link(
    fake_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    """Criterio S4-07: una oportunidad genera una sola alerta y apunta a su ficha."""
    opportunity = {
        "property_id": "prop-cle-7788",
        "ubicacion": "1420 Maple Ave, Cleveland, OH 44102",
        "condado": "Cuyahoga",
        "precio_usd": 110_000.0,
        "noi": 13_200.0,
        "cap_rate_ajustado": 10.8,
        "capex": 12_000.0,
        "vision_confianza": 0.85,
        "requiere_revision_humana": True,
        "riesgo_legal": "alto",
        "estado": "INSPECCION_VISUAL_OK",
    }

    notifier = FakeNotifier()
    dedupe = RedisAlertDedupeStore(fake_redis)
    cfg = AlertConfig(ficha_base_url="https://panel.openclaw.local/propiedades")

    first = await nodo_alerta_telegram(
        opportunity, notifier=notifier, dedupe_store=dedupe, alert_config=cfg, chat_id="chat-1"
    )
    second = await nodo_alerta_telegram(
        opportunity, notifier=notifier, dedupe_store=dedupe, alert_config=cfg, chat_id="chat-1"
    )

    assert first["estado"] == "ALERTA_ENVIADA"
    assert second["estado"] == "ALERTA_DUPLICADA_OMITIDA"
    assert len(notifier.sent_messages) == 1
    sent_chat, sent_text = notifier.sent_messages[0]
    assert sent_chat == "chat-1"
    assert "https://panel.openclaw.local/propiedades/prop-cle-7788" in sent_text
    assert "Revisión humana obligatoria" in sent_text


async def test_telegram_notifier_adapter_verifies_http_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 42}})

    adapter = TelegramNotifierAdapter(
        bot_token="test-bot-token",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    ok, err = await adapter.send_message("chat-1", "hola")
    assert ok is True
    assert err is None


async def test_full_orchestrator_property_pipeline_end_to_end() -> None:
    valuator = FakeMultimodalValuator(
        VisionAssessment(
            capex_usd=8_000.0,
            confianza=0.9,
            motivo="Buen estado estructural.",
            severidad="baja",
        )
    )
    notifier = FakeNotifier()
    dedupe = InMemoryAlertDedupeStore()

    final_state = await ejecutar_pipeline_propiedad(
        {
            "property_id": "prop-e2e-1",
            "ubicacion": "1420 Maple Ave, Cleveland, OH 44102",
            "condado": "Cuyahoga",
            "precio_usd": 100_000.0,
            "ingreso_anual_estimado": 18_000.0,
            "imagenes": ["https://photos.zillowstatic.com/1001.jpg"],
            "descripcion_cruda": "Solid brick rental with back taxes reported in 2024.",
        },
        valuator=valuator,
        notifier=notifier,
        dedupe_store=dedupe,
        chat_id="chat-authorized",
    )

    assert final_state["estado"] == "ALERTA_ENVIADA"
    assert final_state["requiere_revision_humana"] is True
    assert final_state["capex"] == 8_000.0
    assert len(notifier.sent_messages) == 1
