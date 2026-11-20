"""Nodo financiero parametrizado desde configuración (S4-04).

Cambiar tasas en `FinancialConfig` (o variables `OPENCLAW_FIN_*`) altera el NOI
y el cap rate sin modificar código, dejando trazabilidad completa de entradas y
versión de reglas en el estado resultante.
"""

from __future__ import annotations

from typing import Any

from app.orchestrator.config import FinancialConfig


def nodo_analisis_financiero(
    state: dict[str, Any],
    *,
    config: FinancialConfig | None = None,
) -> dict[str, Any]:
    """Evalúa NOI y cap rate usando tasas parametrizadas desde configuración."""
    cfg = config if config is not None else FinancialConfig()

    precio = float(state.get("precio_usd") or 0.0)
    ingreso = float(state.get("ingreso_anual_estimado") or 0.0)
    hoa_mensual = float(state.get("hoa_mensual_usd") or 0.0)

    if precio <= 0.0 or ingreso <= 0.0:
        return {
            "estado": "RECHAZADO",
            "motivo_financiero": "precio_o_ingreso_invalido",
            "reglas_financieras_version": cfg.rules_version,
        }

    gasto_vacancia = ingreso * cfg.vacancy_rate
    gasto_mantenimiento = ingreso * cfg.maintenance_rate
    gasto_impuestos = precio * cfg.property_tax_rate
    gasto_seguro = cfg.insurance_annual_usd
    gasto_hoa_anual = hoa_mensual * 12.0

    gastos_totales = (
        gasto_vacancia + gasto_mantenimiento + gasto_impuestos + gasto_seguro + gasto_hoa_anual
    )
    noi = round(ingreso - gastos_totales, 2)
    cap_rate = round((noi / precio) * 100.0, 4)

    desglose = {
        "vacancia_usd": round(gasto_vacancia, 2),
        "mantenimiento_usd": round(gasto_mantenimiento, 2),
        "impuestos_usd": round(gasto_impuestos, 2),
        "seguro_usd": round(gasto_seguro, 2),
        "hoa_anual_usd": round(gasto_hoa_anual, 2),
        "gastos_totales_usd": round(gastos_totales, 2),
        "min_cap_rate_pct": cfg.min_cap_rate_pct,
        "rules_version": cfg.rules_version,
    }

    if cap_rate < cfg.min_cap_rate_pct:
        return {
            "estado": "RECHAZADO",
            "cap_rate": cap_rate,
            "noi": noi,
            "motivo_financiero": (
                f"cap_rate {cap_rate:.2f}% por debajo del umbral {cfg.min_cap_rate_pct:.2f}%"
            ),
            "desglose_financiero": desglose,
            "reglas_financieras_version": cfg.rules_version,
        }

    return {
        "estado": "ANALISIS_FINANCIERO_OK",
        "cap_rate": cap_rate,
        "noi": noi,
        "desglose_financiero": desglose,
        "reglas_financieras_version": cfg.rules_version,
    }
