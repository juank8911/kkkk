import requests

from app.core.config import settings


def nodo_alerta_telegram(state: dict):
    if state.get("estado") == "INSPECCION_VISUAL_OK":
        mensaje = (
            "🚨 OPORTUNIDAD HFT DETECTADA 🚨\n"
            f"Precio: ${state.get('precio_usd')}\n"
            "Responde /buy para ejecutar."
        )
        requests.post(
            f"https://api.telegram.org/bot{settings.TG_TOKEN}/sendMessage",
            json={"chat_id": settings.TG_CHAT_ID, "text": mensaje},
        )
    return {"estado": "ALERTA_ENVIADA"}
