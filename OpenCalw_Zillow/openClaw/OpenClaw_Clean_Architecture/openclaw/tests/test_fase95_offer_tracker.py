"""Pruebas de Fase 95 (Dev 2): OfertaTrackerPort, señales de mercado, ambigüedad y locks."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import fakeredis.aioredis
import pytest

from app.integrations.firecrawl_adapter import FirecrawlAdapter
from app.models.offers import OfertaEstado, OfertaResponse
from app.services.offer_tracker import (
    FirecrawlOfertaTrackerAdapter,
    ObservacionTrackerOferta,
    OfertaTrackerPort,
    SenalMercadoOferta,
    detectar_senales_mercado,
    evaluar_observacion_oferta,
    rastrear_oferta_con_lock,
)
from app.services.rate_limiter import SlidingWindowRateLimiter
from tests.test_scraper_contracts import CassetteFirecrawlClient


def _make_oferta(estado: OfertaEstado = OfertaEstado.PROPUESTA) -> OfertaResponse:
    now = datetime.now(UTC)
    return OfertaResponse(
        id=uuid4(),
        propiedad_id=uuid4(),
        estado=estado,
        importe_usd=112_000.0,
        actor_ultima_actualizacion="operador-1",
        creada_en=now,
        actualizada_en=now,
    )


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("Status: Pending Sale - 1420 Maple Ave", SenalMercadoOferta.PENDING),
        (
            "Listing is currently Under Contract in Cuyahoga County",
            SenalMercadoOferta.UNDER_CONTRACT,
        ),
        ("Recently Sold on Nov 18 for $115,000", SenalMercadoOferta.SOLD),
        ("Off Market - Listing Withdrawn by seller", SenalMercadoOferta.WITHDRAWN),
        ("Active Listing - For Sale $115,000", SenalMercadoOferta.ACTIVE),
        ("For Sale today! Also Sold last year!", SenalMercadoOferta.AMBIGUA),
        ("No status markers present in page text", SenalMercadoOferta.AMBIGUA),
    ],
)
def test_fase95_signal_detection_maps_and_flags_ambiguity(
    markdown: str,
    expected: SenalMercadoOferta,
) -> None:
    senal, _ = detectar_senales_mercado(markdown)
    assert senal == expected


def test_fase95_ambiguous_signal_never_closes_or_cancels_offer() -> None:
    oferta = _make_oferta(OfertaEstado.PROPUESTA)
    obs = ObservacionTrackerOferta(
        oferta_id=oferta.id,
        plataforma="zillow",
        url_fuente="https://www.zillow.com/homedetails/1420-maple",
        markdown="For Sale - Sold banner conflict in cached page",
        disponible=True,
    )
    ev = evaluar_observacion_oferta(oferta, obs)
    assert ev.senal_detectada == SenalMercadoOferta.AMBIGUA
    assert ev.estado_sugerido is None
    assert ev.transicion_aplicable is False
    assert ev.requiere_revision_humana is True
    assert ev.evento.estado_anterior == OfertaEstado.PROPUESTA
    assert ev.evento.estado_nuevo == OfertaEstado.PROPUESTA
    assert ev.accion_comercial_automatica is False
    assert ev.aviso_telegram is not None
    assert (
        "No se ha enviado ninguna comunicación ni acción comercial vinculante" in ev.aviso_telegram
    )


def test_fase95_sold_and_withdrawn_produce_valid_suggested_transitions_with_human_review() -> None:
    oferta_aceptada = _make_oferta(OfertaEstado.ACEPTADA)
    ev_sold = evaluar_observacion_oferta(
        oferta_aceptada,
        ObservacionTrackerOferta(
            oferta_id=oferta_aceptada.id,
            plataforma="realtor",
            markdown="Property marked Recently Sold.",
        ),
    )
    assert ev_sold.senal_detectada == SenalMercadoOferta.SOLD
    assert ev_sold.estado_sugerido == OfertaEstado.CERRADA
    assert ev_sold.transicion_aplicable is True
    assert ev_sold.requiere_revision_humana is True

    oferta_prop = _make_oferta(OfertaEstado.PROPUESTA)
    ev_withdrawn = evaluar_observacion_oferta(
        oferta_prop,
        ObservacionTrackerOferta(
            oferta_id=oferta_prop.id,
            plataforma="zillow",
            markdown="Listing removed / Off Market.",
        ),
    )
    assert ev_withdrawn.senal_detectada == SenalMercadoOferta.WITHDRAWN
    assert ev_withdrawn.estado_sugerido == OfertaEstado.CANCELADA
    assert ev_withdrawn.transicion_aplicable is True
    assert ev_withdrawn.requiere_revision_humana is True


async def test_fase95_firecrawl_tracker_adapter_lock_and_period_control(
    fake_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    cassette = {
        "behavior": "response",
        "response": {
            "data": {
                "markdown": "# 1420 Maple Ave\nStatus: Under Contract\n",
                "metadata": {"statusCode": 200},
            }
        },
    }
    client = CassetteFirecrawlClient([cassette])
    scraper = FirecrawlAdapter(
        client=client,
        rate_limiter=SlidingWindowRateLimiter(redis_client=fake_redis),
    )
    tracker = FirecrawlOfertaTrackerAdapter(scraper)
    assert isinstance(tracker, OfertaTrackerPort)

    oferta = _make_oferta(OfertaEstado.PROPUESTA)

    # 1ra comprobación dentro del periodo -> ejecuta y genera evento auditable
    first = await rastrear_oferta_con_lock(
        redis_client=fake_redis,
        tracker=tracker,
        oferta=oferta,
        plataforma="zillow",
        referencia_busqueda="1420 Maple Ave Cleveland OH",
        periodo_segundos=3600,
    )
    assert first is not None
    assert first.senal_detectada == SenalMercadoOferta.UNDER_CONTRACT
    assert first.requiere_revision_humana is True
    assert first.evento.tipo == "tracker_revision_contrato_externo"

    # 2da comprobación dentro del mismo periodo -> omitida (máximo 1 vez por periodo)
    second = await rastrear_oferta_con_lock(
        redis_client=fake_redis,
        tracker=tracker,
        oferta=oferta,
        plataforma="zillow",
        referencia_busqueda="1420 Maple Ave Cleveland OH",
        periodo_segundos=3600,
    )
    assert second is None
    assert len(client.calls) == 1

    # Si otro worker ya mantiene el lock -> no se procesa simultáneamente
    oferta_locked = _make_oferta(OfertaEstado.PROPUESTA)
    await fake_redis.set(f"openclaw:oferta:lock:{oferta_locked.id}", "worker-a", ex=120)
    blocked = await rastrear_oferta_con_lock(
        redis_client=fake_redis,
        tracker=tracker,
        oferta=oferta_locked,
        plataforma="zillow",
        referencia_busqueda="1420 Maple Ave",
    )
    assert blocked is None
    assert len(client.calls) == 1
