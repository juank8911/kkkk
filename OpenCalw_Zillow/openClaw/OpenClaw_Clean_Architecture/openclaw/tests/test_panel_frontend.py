"""Verificación del panel mínimo de jobs y telemetría (S5-04)."""

from __future__ import annotations

from pathlib import Path

PANEL_PATH = Path(__file__).resolve().parents[1] / "app" / "frontend" / "panel.html"


def test_s5_04_panel_html_lists_jobs_details_and_telemetry_safely() -> None:
    """Criterio S5-04: lista jobs con estado, permite ver el detalle y consume telemetría."""
    assert PANEL_PATH.exists()
    html = PANEL_PATH.read_text(encoding="utf-8")

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
