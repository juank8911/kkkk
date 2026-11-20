"""Pruebas de estrategia Seller Financing en inglés, monitoreo de vendedor y reporte diario 11 AM."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import fakeredis.aioredis

from app.core.config import settings
from app.db.models import Oferta, OfertaEvento, Propiedad
from app.services import negotiation_reporter, telegram_workflow
from app.services.negotiation_reporter import (
    construir_mensaje_ingles_vendedor,
    construir_respuesta_ingles_vendedor,
    enviar_reporte_diario_si_corresponde,
    generar_reporte_negociaciones,
    resolver_zona_horaria,
)
from app.services.telegram_workflow import procesar_interaccion_telegram


class _SpyNotifier:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send_message(
        self,
        chat_id: str,
        text: str,
        *,
        reply_markup: dict[str, Any] | None = None,
        photo_urls: list[str] | None = None,
    ) -> tuple[bool, str | None]:
        self.sent.append((chat_id, text))
        return True, None

    async def answer_callback_query(
        self, callback_query_id: str, text: str = ""
    ) -> tuple[bool, str | None]:
        return True, None


def test_mensaje_inicial_vendedor_en_ingles_con_seller_financing_y_bono_6_meses() -> None:
    msg = construir_mensaje_ingles_vendedor(
        ubicacion="2055 Burnham Ave, Memphis, TN 38127",
        url_origen="https://www.zillow.com/homedetails/2055-Burnham-Ave-Memphis-TN-38127/12345_zpid/",
        precio_publicado_usd=54000.0,
        precio_oferta_usd=54000.0,
        contacto_vendedor="Brokered by Keller Williams Realty",
    )
    assert "Hello Brokered by Keller Williams Realty," in msg
    assert "Seller Financing" in msg
    assert "$5,000 to $7,000 USD" in msg
    assert "6 extra monthly payments" in msg
    assert "after the full payoff" in msg
    assert "Non-binding inquiry draft" in msg
    assert "2055 Burnham Ave, Memphis, TN 38127" in msg


def test_respuesta_al_vendedor_en_ingles_y_proteccion_fair_housing() -> None:
    respuesta, analisis = construir_respuesta_ingles_vendedor(
        ubicacion="904 Birch St, Memphis, TN 38108",
        mensaje_vendedor=(
            "We could consider owner financing at $78,000 with $12,000 down and 25 days closing. "
            "Family status mentioned should be ignored."
        ),
        precio_oferta_actual_usd=73500.0,
    )
    assert analisis["precio_solicitado_usd"] == 78000.0
    assert analisis["dias_cierre_solicitados"] == 25
    assert analisis["menciona_seller_financing"] is True
    assert analisis["menciona_down_payment"] is True
    assert "familial_status" in analisis["atributos_protegidos_ignorados"]  # type: ignore[operator]
    assert "$5,000 and $7,000 USD" in respuesta
    assert "6 additional monthly payments" in respuesta
    assert "25-day closing timeline" in respuesta


def test_generar_reporte_negociaciones_resume_activas_y_comunicaciones() -> None:
    prop = Propiedad(
        id=uuid4(),
        job_id=uuid4(),
        plataforma="zillow",
        url_origen="https://www.zillow.com/homedetails/2055-Burnham-Ave/1_zpid/",
        ubicacion="2055 Burnham Ave, Memphis, TN 38127",
        condado="Shelby",
        precio_usd=54000.0,
        ingreso_anual_estimado=15600.0,
        descripcion_cruda="Seller Contact: Crye-Leike Realty",
        created_at=datetime.now(UTC),
    )
    oferta = Oferta(
        id=uuid4(),
        propiedad_id=prop.id,
        estado="contraoferta",
        importe_usd=54000.0,
        importe_contraoferta_usd=56000.0,
        actor_ultima_actualizacion="vendedor",
    )
    ev = OfertaEvento(
        id=1,
        oferta_id=oferta.id,
        estado_anterior="propuesta",
        estado_nuevo="contraoferta",
        actor="vendedor",
        notas="mensaje_vendedor: Open to seller financing with $9k down",
    )
    tz = resolver_zona_horaria("America/New_York")
    reporte = generar_reporte_negociaciones(
        [(oferta, prop, [ev])],
        fecha_local=datetime(2026, 11, 20, 11, 0, tzinfo=tz),
    )
    assert "REPORTE DIARIO DE NEGOCIACIONES" in reporte
    assert "1 activas" in reporte
    assert "1 comunicaciones del vendedor" in reporte
    assert "2055 Burnham Ave, Memphis, TN 38127" in reporte
    assert "$56,000" in reporte
    assert str(oferta.id)[:8] in reporte


async def test_scheduler_reporte_diario_dispara_a_las_11_am_una_sola_vez_por_dia(
    monkeypatch,
) -> None:
    fake_redis = fakeredis.aioredis.FakeRedis()
    monkeypatch.setattr(negotiation_reporter, "redis_client", fake_redis)
    monkeypatch.setattr(settings, "TG_CHAT_ID", "8503241862")
    monkeypatch.setattr(settings, "NEGOTIATION_REPORT_HOUR", 11)
    monkeypatch.setattr(settings, "NEGOTIATION_REPORT_MINUTE", 0)
    monkeypatch.setattr(settings, "NEGOTIATION_REPORT_TZ", "America/New_York")

    async def fake_reporte(*, fecha_local=None) -> str:
        return f"REPORTE OK {fecha_local}"

    monkeypatch.setattr(
        negotiation_reporter,
        "consultar_negociaciones_y_generar_reporte",
        fake_reporte,
    )
    spy = _SpyNotifier()

    # 10:59 AM hora local (15:59 UTC en EST UTC-5) -> NO dispara aún
    antes = datetime(2026, 11, 20, 15, 59, tzinfo=UTC)
    assert await enviar_reporte_diario_si_corresponde(now_utc=antes, notifier=spy) is False  # type: ignore[arg-type]
    assert spy.sent == []

    # 11:00 AM hora local (16:00 UTC en EST UTC-5) -> dispara una vez
    las_11 = datetime(2026, 11, 20, 16, 0, tzinfo=UTC)
    assert await enviar_reporte_diario_si_corresponde(now_utc=las_11, notifier=spy) is True  # type: ignore[arg-type]
    assert len(spy.sent) == 1

    # 11:05 AM mismo día -> deduplicado, no vuelve a enviar
    despues = datetime(2026, 11, 20, 16, 5, tzinfo=UTC)
    assert await enviar_reporte_diario_si_corresponde(now_utc=despues, notifier=spy) is False  # type: ignore[arg-type]
    assert len(spy.sent) == 1

    # Día siguiente a las 11:00 AM -> vuelve a disparar
    manana = datetime(2026, 11, 21, 16, 0, tzinfo=UTC)
    assert await enviar_reporte_diario_si_corresponde(now_utc=manana, notifier=spy) is True  # type: ignore[arg-type]
    assert len(spy.sent) == 2

    await fake_redis.aclose()


async def test_comandos_telegram_msg_vendedor_y_reporte(monkeypatch) -> None:
    prop = Propiedad(
        id=uuid4(),
        job_id=uuid4(),
        plataforma="realtor",
        url_origen="https://www.realtor.com/realestateandhomes-detail/test",
        ubicacion="962 Biggs St, Memphis, TN 38108",
        condado="Shelby",
        precio_usd=89900.0,
        ingreso_anual_estimado=15600.0,
        descripcion_cruda="Seller Contact: Local Broker",
        created_at=datetime.now(UTC),
    )
    oferta = Oferta(
        id=uuid4(),
        propiedad_id=prop.id,
        estado="contraoferta",
        importe_usd=89900.0,
        importe_contraoferta_usd=92000.0,
        actor_ultima_actualizacion="vendedor",
    )

    async def fake_reg(ref: str, msg: str, *, chat_id: str):
        borr, an = construir_respuesta_ingles_vendedor(
            ubicacion=prop.ubicacion,
            mensaje_vendedor=msg,
            precio_oferta_actual_usd=89900.0,
        )
        return oferta, prop, borr, an

    async def fake_rep(*, fecha_local=None) -> str:
        return "📊 REPORTE DIARIO DE NEGOCIACIONES — OK"

    monkeypatch.setattr(telegram_workflow, "registrar_comunicacion_vendedor", fake_reg)
    monkeypatch.setattr(telegram_workflow, "consultar_negociaciones_y_generar_reporte", fake_rep)

    spy = _SpyNotifier()
    res_msg = await procesar_interaccion_telegram(
        {
            "message": {
                "chat": {"id": 8503241862},
                "text": f"/msg {str(oferta.id)[:8]} Seller wants $92,000 and $10,000 down.",
            }
        },
        notifier=spy,  # type: ignore[arg-type]
    )
    assert res_msg["status"] == "comunicacion_vendedor_registrada"
    assert "$5,000 and $7,000 USD" in spy.sent[-1][1]
    assert "6 additional monthly payments" in spy.sent[-1][1]

    res_rep = await procesar_interaccion_telegram(
        {"message": {"chat": {"id": 8503241862}, "text": "/reporte"}},
        notifier=spy,  # type: ignore[arg-type]
    )
    assert res_rep["status"] == "reporte_enviado"
    assert "REPORTE DIARIO DE NEGOCIACIONES" in spy.sent[-1][1]
