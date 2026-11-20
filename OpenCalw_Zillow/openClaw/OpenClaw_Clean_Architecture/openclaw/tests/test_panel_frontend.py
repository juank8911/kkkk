"""Verificación del panel mínimo de jobs, telemetría, mercados y propiedades por barrido."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from uuid import uuid4

import httpx
from fastapi import FastAPI

from app.db.models import Decision, Propiedad
from app.frontend import PANEL_HTML_PATH
from app.frontend import router as panel_router
from app.frontend.markets import listar_mercados_ordenados
from app.frontend.sweep_properties import (
    construir_respuesta_propiedades_barrido,
)


def test_s5_04_panel_html_lists_jobs_details_markets_and_telemetry_safely() -> None:
    """Criterio S5-04: lista jobs, icono de propiedades por barrido, mercados y telemetría."""
    assert PANEL_HTML_PATH.exists()
    html = PANEL_HTML_PATH.read_text(encoding="utf-8")

    # Elementos funcionales requeridos
    assert 'id="jobs-tbody"' in html
    assert 'id="job-detail"' in html
    assert 'id="telemetry-log"' in html
    assert 'id="market-selector"' in html
    assert 'id="markets-tbody"' in html
    assert 'id="job-properties-modal"' in html
    assert 'id="job-properties-tbody"' in html
    assert "btn-icon-props" in html
    assert "/panel/api/barridos" in html
    assert "/propiedades" in html
    assert "/panel/api/mercados" in html
    assert "/v1/openclaw/ws/scraping-feed" in html

    # Higiene de seguridad básica: sin innerHTML con datos no confiables ni secretos fijos
    assert "innerHTML" not in html
    assert "textContent" in html


def test_panel_markets_ranking_has_30_markets_sorted_by_attractiveness() -> None:
    mercados = listar_mercados_ordenados()
    assert len(mercados) == 30

    porcentajes = [m.atractivo_pct for m in mercados]
    assert porcentajes == sorted(porcentajes, reverse=True)
    assert [m.ranking for m in mercados] == list(range(1, 31))
    assert all(0.0 <= m.atractivo_pct <= 100.0 for m in mercados)

    nombres = {m.ciudad for m in mercados}
    for esperada in (
        "Detroit",
        "Cleveland",
        "Flint",
        "Jackson",
        "Baltimore",
        "Indianápolis",
        "Gary y Hammond",
    ):
        assert esperada in nombres


def test_panel_sweep_properties_classification_approval_rejected_and_pending() -> None:
    job_id = uuid4()
    ahora = datetime.now(UTC)

    p_aprob = Propiedad(
        id=uuid4(),
        job_id=job_id,
        plataforma="zillow",
        url_origen="https://www.zillow.com/homedetails/101-main",
        ubicacion="101 Main St, Cleveland, OH 44105",
        ciudad="Cleveland",
        codigo_postal="44105",
        precio_usd=95000.0,
        ingreso_anual_estimado=16800.0,
        descripcion_cruda="Casa duplex",
        created_at=ahora,
    )
    d_aprob = Decision(
        id=uuid4(),
        propiedad_id=p_aprob.id,
        veredicto="comprar",
        cap_rate=0.115,
        capex_estimado=8000.0,
        motivo="ALERTA_ENVIADA",
        detalle={"estado": "ALERTA_ENVIADA", "alerta": {"sent": True, "status": "sent"}},
        created_at=ahora,
    )

    p_rech = Propiedad(
        id=uuid4(),
        job_id=job_id,
        plataforma="redfin",
        url_origen="https://www.redfin.com/OH/Cleveland/202-oak",
        ubicacion="202 Oak Ave, Cleveland, OH 44102",
        ciudad="Cleveland",
        codigo_postal="44102",
        precio_usd=220000.0,
        ingreso_anual_estimado=14000.0,
        descripcion_cruda="Precio alto",
        created_at=ahora,
    )
    d_rech = Decision(
        id=uuid4(),
        propiedad_id=p_rech.id,
        veredicto="descartar",
        cap_rate=0.042,
        capex_estimado=None,
        motivo="DESCARTADO_PRECIO",
        detalle={"estado": "DESCARTADO_PRECIO"},
        created_at=ahora,
    )

    p_pend = Propiedad(
        id=uuid4(),
        job_id=job_id,
        plataforma="realtor",
        url_origen="https://www.realtor.com/realestateandhomes-detail/303-pine",
        ubicacion="303 Pine Rd, Cleveland, OH 44110",
        ciudad="Cleveland",
        codigo_postal="44110",
        precio_usd=88000.0,
        ingreso_anual_estimado=15000.0,
        descripcion_cruda="Recien raspada",
        created_at=ahora,
    )

    resp = construir_respuesta_propiedades_barrido(
        job_id,
        [(p_pend, None), (p_rech, d_rech), (p_aprob, d_aprob)],
    )

    assert resp.resumen.total == 3
    assert resp.resumen.analizadas == 2
    assert resp.resumen.enviadas_aprobacion == 1
    assert resp.resumen.rechazadas_directamente == 1
    assert resp.resumen.pendientes_analisis == 1

    # Orden de prioridad: primero enviadas a aprobación, luego rechazadas, luego pendientes
    assert [p.estado_analisis for p in resp.propiedades] == [
        "enviada_aprobacion",
        "rechazada_directamente",
        "pendiente_analisis",
    ]
    assert resp.propiedades[0].alerta_telegram_enviada is True
    assert resp.propiedades[0].cap_rate_pct == 11.5
    assert resp.propiedades[1].motivo == "DESCARTADO_PRECIO"
    assert resp.propiedades[2].analizada is False


async def test_s5_04_panel_router_serves_html_markets_and_sweep_properties(monkeypatch) -> None:
    app = FastAPI()
    app.include_router(panel_router)

    job_id = uuid4()
    prop = Propiedad(
        id=uuid4(),
        job_id=job_id,
        plataforma="zillow",
        url_origen="https://www.zillow.com/homedetails/101-main",
        ubicacion="101 Main St, Flint, MI 48503",
        ciudad="Flint",
        codigo_postal="48503",
        precio_usd=65000.0,
        ingreso_anual_estimado=13440.0,
        descripcion_cruda="Casa renovada",
        created_at=datetime.now(UTC),
    )
    dec = Decision(
        id=uuid4(),
        propiedad_id=prop.id,
        veredicto="revisar",
        cap_rate=0.128,
        capex_estimado=5000.0,
        motivo="REVISION_LEGAL",
        detalle={"estado": "REVISION_HUMANA", "alerta": {"sent": True}},
        created_at=datetime.now(UTC),
    )

    async def fake_obtener(target_job_id):
        assert target_job_id == job_id
        return construir_respuesta_propiedades_barrido(job_id, [(prop, dec)])

    monkeypatch.setattr(
        sys.modules["app.frontend.router"],
        "obtener_propiedades_por_barrido",
        fake_obtener,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        resp = await client.get("/panel")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "OpenClaw — Panel de Jobs y Telemetría" in resp.text

        resp_mercados = await client.get("/panel/api/mercados")
        assert resp_mercados.status_code == 200
        data = resp_mercados.json()
        assert len(data) == 30
        assert data[0]["ranking"] == 1
        assert data[0]["atractivo_pct"] >= data[-1]["atractivo_pct"]

        resp_props = await client.get(f"/panel/api/barridos/{job_id}/propiedades")
        assert resp_props.status_code == 200
        body = resp_props.json()
        assert body["resumen"]["total"] == 1
        assert body["resumen"]["enviadas_aprobacion"] == 1
        assert body["propiedades"][0]["estado_analisis"] == "enviada_aprobacion"
