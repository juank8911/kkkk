"""Pruebas de Fases 105–111 / 117 (Dev 2): ADR-06 y salud de fuentes sin evasión WAF."""

from __future__ import annotations

from pathlib import Path

from app.services.source_health import EstadoFuente, PlatformSourceHealthManager
from tests.conftest import MutableClock

ADR_06_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "adr" / "ADR-06-reconciliacion-fases-105-111.md"
)


def test_fase105_109_adr06_reconciles_missing_phase_numbering() -> None:
    assert ADR_06_PATH.exists()
    content = ADR_06_PATH.read_text(encoding="utf-8")
    for phase_num in ("105", "106", "107", "108", "109", "110–111", "116–117"):
        assert phase_num in content


def test_fase110_111_source_health_stops_insisting_on_blocked_source_and_degrades(
    fake_clock: MutableClock,
) -> None:
    mgr = PlatformSourceHealthManager(
        degraded_threshold=2,
        block_cooldown_seconds=900.0,
        rate_limit_cooldown_seconds=60.0,
        clock=fake_clock,
    )

    assert mgr.puede_consultar("zillow") is True

    # Fallos transitorios -> degradada (aún consultable con cautela)
    mgr.registrar_fallo_transitorio("realtor", motivo="timeout 1")
    snap_deg = mgr.registrar_fallo_transitorio("realtor", motivo="timeout 2")
    assert snap_deg.estado == EstadoFuente.DEGRADADA
    assert mgr.puede_consultar("realtor") is True

    # Rate limit -> limitada durante 60s
    snap_lim = mgr.registrar_rate_limit("redfin", retry_after_seconds=60.0)
    assert snap_lim.estado == EstadoFuente.LIMITADA
    assert mgr.puede_consultar("redfin") is False

    # Bloqueo WAF/403 -> bloqueada sin evasión, worker NO insiste, ofrece alternativas permitidas
    snap_blk = mgr.registrar_bloqueo("zillow", motivo="HTTP 403 WAF")
    assert snap_blk.estado == EstadoFuente.BLOQUEADA
    assert snap_blk.cooldown_restante_seg == 900.0
    assert mgr.puede_consultar("zillow") is False
    assert "zillow" not in snap_blk.alternativas_disponibles
    assert "trulia" in snap_blk.alternativas_disponibles

    # No autorizada -> permanece bloqueada incluso tras avanzar el reloj
    snap_unauth = mgr.marcar_no_autorizada("craigslist", motivo="Restricción ToS regional")
    assert snap_unauth.estado == EstadoFuente.NO_AUTORIZADA
    assert mgr.puede_consultar("craigslist") is False

    # Tras expirar cooldown de 900s -> zillow se recupera controladamente; craigslist sigue no_autorizada
    fake_clock.advance(901.0)
    assert mgr.puede_consultar("zillow") is True
    assert mgr.obtener_estado("zillow").estado == EstadoFuente.DISPONIBLE
    assert mgr.puede_consultar("craigslist") is False
