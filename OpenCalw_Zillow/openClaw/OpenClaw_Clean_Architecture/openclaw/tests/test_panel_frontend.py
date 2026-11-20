"""Verificación del panel mínimo de jobs, telemetría y ranking de mercados (S5-04)."""

from __future__ import annotations

import httpx
from fastapi import FastAPI

from app.frontend import PANEL_HTML_PATH
from app.frontend import router as panel_router
from app.frontend.markets import listar_mercados_ordenados


def test_s5_04_panel_html_lists_jobs_details_markets_and_telemetry_safely() -> None:
    """Criterio S5-04: lista jobs, selector/tabla de mercados por % atractivo y telemetría."""
    assert PANEL_HTML_PATH.exists()
    html = PANEL_HTML_PATH.read_text(encoding="utf-8")

    # Elementos funcionales requeridos
    assert 'id="jobs-tbody"' in html
    assert 'id="job-detail"' in html
    assert 'id="telemetry-log"' in html
    assert 'id="market-selector"' in html
    assert 'id="markets-tbody"' in html
    assert "/panel/api/barridos" in html
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


async def test_s5_04_panel_router_serves_html_and_markets_endpoint() -> None:
    app = FastAPI()
    app.include_router(panel_router)

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
