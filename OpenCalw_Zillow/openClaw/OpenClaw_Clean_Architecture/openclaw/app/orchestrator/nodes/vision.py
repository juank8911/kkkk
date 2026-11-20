async def nodo_vision_artificial(state: dict) -> dict:
    capex_estimado = 15000.00
    inversion = state.get("precio_usd", 0) + capex_estimado
    cap_rate_ajustado = (state.get("noi", 0) / inversion) * 100
    if cap_rate_ajustado < 8.0: return {"estado": "RECHAZADO"}
    return {"estado": "INSPECCION_VISUAL_OK", "capex": capex_estimado}