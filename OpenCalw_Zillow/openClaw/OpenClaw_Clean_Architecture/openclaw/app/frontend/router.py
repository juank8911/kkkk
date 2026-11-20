"""Router FastAPI ligero para servir el panel mínimo de jobs y telemetría (S5-04)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

PANEL_HTML_PATH = Path(__file__).resolve().parent / "panel.html"

router = APIRouter(tags=["panel"])


@router.get("/panel", response_class=HTMLResponse, summary="Panel mínimo de jobs y telemetría")
async def servir_panel_jobs() -> HTMLResponse:
    """Devuelve el panel HTML autocontenido para listar jobs, ver detalle y leer telemetría WS."""
    html = PANEL_HTML_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html, status_code=200)
