"""Webhook de comandos de Telegram.

Salvaguarda del sprint 0: mientras ``BUY_MODE`` no sea ``sandbox``, el comando
``/buy`` se rechaza y no se llama al servicio de liquidez. La implementacion
completa (orden parametrizada, idempotencia y auditoria) es la tarea S2-07.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, status

from app.core.config import settings

logger = logging.getLogger("OpenClaw.Comandos")

router = APIRouter(prefix="/v1/webhook", tags=["comandos"])


@router.post("/telegram-comando", summary="Procesa un comando de Telegram")
async def procesar_comando_telegram(update: dict[str, Any]) -> dict[str, str]:
    texto = str(update.get("message", {}).get("text", "")).strip()

    if not texto.startswith("/buy"):
        return {"status": "Comando ignorado"}

    if settings.BUY_MODE != "sandbox":
        logger.warning("compra_rechazada", extra={"motivo": "buy_mode_disabled"})
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Compra deshabilitada: BUY_MODE no esta en modo sandbox",
        )

    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(
            f"{settings.UAO_SCALPING_API}/request-withdrawal",
            json={
                "asset": "USDT",
                "amount": 125000.0,
                "urgency": "high",
                "destination": "fiat_bridge_bank",
            },
        )

    logger.warning("compra_simulada_enviada", extra={"modo": settings.BUY_MODE})
    return {"status": "Ejecutando Liquidacion UAO_Grid (sandbox)"}
