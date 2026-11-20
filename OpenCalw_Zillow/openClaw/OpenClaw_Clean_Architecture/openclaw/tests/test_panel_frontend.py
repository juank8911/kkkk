"""Verificación del panel mínimo de jobs y telemetría (S5-04)."""

from __future__ import annotations

import httpx
from fastapi import FastAPI

from app.frontend import PANEL_HTML_PATH
from app.frontend import router as panel_router


def test_s5_04_panel_html_lists_jobs_details_and_telemetry_safely() -> None:
    """Criterio S5-04: lista jobs con estado, permite ver el detalle y consume telemetría."""
    assert PANEL_HTML_PATH.exists()
    html = PANEL_HTML_PATH.read_text(encoding="utf-8")

    # Elementos funcionales requeridos
    assert 'id="jobs-tbody"' in html
    assert 'id="job-detail"' in html
    assert 'id="telemetry-log"' in html
    assert "/v1/openclaw/barridos" in html
    assert "/v1/openclaw/ws/scraping-feed" in html
    assert "X-OpenClaw-Tactical-Key" in html

    # Higiene de seguridad básica: sin innerHTML con datos no confiables ni secretos fijos
    assert "innerHTML" not in html
    assert "textContent" in html


async def test_s5_04_panel_router_serves_html_dashboard() -> None:
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
