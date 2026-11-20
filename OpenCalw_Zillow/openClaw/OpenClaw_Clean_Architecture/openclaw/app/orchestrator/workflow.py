def enrutador_due_diligence(state: dict):
    precio = state.get("precio_usd", 0)
    if precio == 0 or precio > 150000:
        return "descarte_inmediato"
    return "agente_financiero"


# Compilacion de LangGraph a ser inyectada aqui
