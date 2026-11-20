import httpx
from fastapi import APIRouter
from app.core.config import settings

router = APIRouter(prefix="/v1/webhook")

@router.post("/telegram-comando")
async def procesar_comando_telegram(update: dict):
    texto = update.get("message", {}).get("text", "").strip()
    if texto.startswith("/buy"):
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{settings.UAO_SCALPING_API}/request-withdrawal",
                json={"asset": "USDT", "amount": 125000.0, "urgency": "high", "destination": "fiat_bridge_bank"}
            )
        return {"status": "Ejecutando Liquidación UAO_Grid"}
    return {"status": "Comando ignorado"}