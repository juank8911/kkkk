"""Prueba automatizada de la demo de aceptación contra los 8 criterios (S6-06)."""

from __future__ import annotations

from scripts.demo_aceptacion_s6_06 import ejecutar_demo_8_criterios


async def test_s6_06_all_eight_acceptance_criteria_pass() -> None:
    """Criterio S6-06: los ocho criterios de aceptación se demuestran y aprueban en vivo."""
    resultado = await ejecutar_demo_8_criterios()
    assert resultado["total_criterios"] == 8
    assert resultado["criterios_aprobados"] == 8
    assert resultado["aceptacion_global"] is True
    for nombre, detalle in resultado["criterios"].items():
        assert detalle["ok"] is True, f"Falló el criterio {nombre}: {detalle}"
