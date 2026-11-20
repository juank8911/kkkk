def nodo_analisis_financiero(state: dict) -> dict:
    precio = state.get("precio_usd", 0)
    ingreso = state.get("ingreso_anual_estimado", 0)
    if precio <= 0 or ingreso <= 0:
        return {"estado": "RECHAZADO"}

    gastos = (ingreso * 0.08) + (ingreso * 0.10) + (precio * 0.02) + 1200
    noi = ingreso - gastos
    cap_rate = (noi / precio) * 100

    if cap_rate < 8.0:
        return {"estado": "RECHAZADO"}
    return {"estado": "ANALISIS_FINANCIERO_OK", "cap_rate": cap_rate, "noi": noi}
