"""Pruebas de Fase 96 (Dev 2): estratega macroeconómico determinista y aprobación humana."""

from __future__ import annotations

from datetime import date

import pytest

from app.orchestrator.nodes.financial import nodo_analisis_financiero
from app.services.macro_strategist import (
    MacroDataPort,
    MacroIndicador,
    MacroObservacion,
    MacroStrategistService,
    MacroVersionEstado,
    StaticMacroAdapter,
    normalizar_tasa_porcentual,
)


def test_fase96_rejects_unapproved_sources_and_normalizes_rates() -> None:
    with pytest.raises(ValueError):
        StaticMacroAdapter(fuente="random_blog")

    assert normalizar_tasa_porcentual(6.5) == pytest.approx(0.065)
    assert normalizar_tasa_porcentual(0.08) == pytest.approx(0.08)


async def test_fase96_dedupes_observations_requires_approval_and_feeds_graph() -> None:
    svc = MacroStrategistService()
    obs_vac = MacroObservacion(
        fuente="census",
        indicador=MacroIndicador.VACANCIA,
        valor=0.10,
        fecha_observacion=date(2026, 11, 1),
    )
    obs_rate = MacroObservacion(
        fuente="fred",
        indicador=MacroIndicador.TASA_INTERES,
        valor=0.0725,
        fecha_observacion=date(2026, 11, 15),
    )
    obs_inf = MacroObservacion(
        fuente="bls",
        indicador=MacroIndicador.INFLACION,
        valor=0.04,
        fecha_observacion=date(2026, 11, 10),
    )

    adapt_census = StaticMacroAdapter(fuente="census", observaciones=[obs_vac, obs_vac])
    adapt_fred = StaticMacroAdapter(fuente="fred", observaciones=[obs_rate])
    adapt_bls = StaticMacroAdapter(fuente="bls", observaciones=[obs_inf])
    assert isinstance(adapt_fred, MacroDataPort)

    propuesta = await svc.recolectar_y_proponer(
        [adapt_census, adapt_fred, adapt_bls],
        etiqueta="macro-2026-11",
    )

    # Deduplicación: repetir obs_vac no duplica datos
    assert len(svc.observaciones_registradas) == 3
    # No se activa automáticamente sin aprobación humana
    assert propuesta.estado == MacroVersionEstado.PENDIENTE_APROBACION
    assert svc.version_activa.etiqueta == "macro-v1.0"
    assert len(propuesta.cambios) == 4
    assert propuesta.target_cap_rate_pct > 8.0

    # Aprobar con identidad humana activa la versión y alimenta el nodo financiero
    with pytest.raises(ValueError):
        svc.aprobar_y_activar_version(propuesta.version_id, aprobado_por="  ")

    activa = svc.aprobar_y_activar_version(propuesta.version_id, aprobado_por="analista-juank")
    assert activa.estado == MacroVersionEstado.ACTIVA
    assert activa.aprobada_por == "analista-juank"
    assert svc.version_activa.version_id == activa.version_id

    fin_cfg = svc.version_activa.to_financial_config()
    assert fin_cfg.rules_version == "macro-2026-11"
    res = nodo_analisis_financiero(
        {"precio_usd": 100_000.0, "ingreso_anual_estimado": 18_000.0},
        config=fin_cfg,
    )
    assert res["reglas_financieras_version"] == "macro-2026-11"


async def test_fase96_down_source_keeps_last_valid_value_and_emits_alert() -> None:
    svc = MacroStrategistService()
    broken_fred = StaticMacroAdapter(fuente="fred", error=TimeoutError("FRED timeout"))

    propuesta = await svc.recolectar_y_proponer([broken_fred], etiqueta="macro-fallback")
    assert propuesta.mortgage_rate == svc.version_activa.mortgage_rate
    assert any("fuente_caida:fred" in alerta for alerta in propuesta.alertas_fuentes)
    # Sin fuentes válidas verificadas no puede activarse
    with pytest.raises(ValueError):
        svc.aprobar_y_activar_version(propuesta.version_id, aprobado_por="operador")
