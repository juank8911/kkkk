"""Webhook de comandos de Telegram (S2-05).

Tres capas antes de ejecutar nada:

1. Autenticacion: ``secret_token`` de Telegram o la API key tactica.
2. Allowlist de chats (``TG_ALLOWED_CHAT_IDS``); vacia significa sin restriccion.
3. Deduplicacion de ``update_id`` en Redis: un update repetido no se procesa dos
   veces. Si Redis no responde se rechaza la peticion (fail-closed), porque el
   coste de un ``/buy`` duplicado es mayor que el de un reintento.

La salvaguarda de ``BUY_MODE`` sigue vigente: mientras no sea ``sandbox``, el
comando ``/buy`` se rechaza y no se llama al servicio de liquidez.
"""

from __future__ import annotations

import hmac
import logging
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.core.config import settings
from app.core.redis_client import redis_client
from app.core.security import CABECERA_API_KEY, hash_api_key
from app.services.telegram_workflow import procesar_interaccion_telegram

logger = logging.getLogger("OpenClaw.Comandos")

router = APIRouter(prefix="/v1/webhook", tags=["comandos"])

CABECERA_SECRETO = "X-Telegram-Bot-Api-Secret-Token"
PREFIJO_DEDUPE = "openclaw:tg:update:"


def _clave_tactica_valida(valor: str | None) -> bool:
    esperada = settings.api_key_hash
    if not esperada or not valor:
        return False
    return hmac.compare_digest(hash_api_key(valor), esperada)


async def autorizar_webhook(
    x_telegram_bot_api_secret_token: Annotated[str | None, Header(alias=CABECERA_SECRETO)] = None,
    x_openclaw_tactical_key: Annotated[str | None, Header(alias=CABECERA_API_KEY)] = None,
) -> str:
    """Acepta el ``secret_token`` de Telegram o, en su defecto, la API key tactica."""
    esperado = settings.TG_WEBHOOK_SECRET.strip()

    if esperado and x_telegram_bot_api_secret_token:
        if hmac.compare_digest(x_telegram_bot_api_secret_token, esperado):
            return "telegram"
        logger.warning("webhook_secreto_invalido")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="El secret_token del webhook no es valido",
        )

    if _clave_tactica_valida(x_openclaw_tactical_key):
        return "api_key"

    if not esperado:
        if settings.APP_ENV == "prod":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="El webhook no tiene secret_token configurado",
            )
        return "sin_secreto_dev"

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Credenciales del webhook invalidas",
    )


async def _update_es_nuevo(update_id: int) -> bool:
    """Marca el update como visto; False si ya se proceso antes."""
    clave = f"{PREFIJO_DEDUPE}{update_id}"
    try:
        creado = await redis_client.set(clave, "1", nx=True, ex=settings.TG_DEDUPE_TTL_SEGUNDOS)
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de Redis se informa igual
        logger.error("dedupe_no_disponible", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se puede verificar la idempotencia del update",
        ) from exc
    return bool(creado)


@router.post(
    "/telegram-comando",
    summary="Procesa un comando de Telegram",
    responses={
        403: {"description": "Chat no autorizado o credenciales invalidas"},
        503: {"description": "Deduplicacion no disponible"},
    },
)
async def procesar_comando_telegram(
    update: dict[str, Any],
    identidad: Annotated[str, Depends(autorizar_webhook)],
) -> dict[str, str]:
    update_id = update.get("update_id")
    if isinstance(update_id, int) and not await _update_es_nuevo(update_id):
        logger.info("update_duplicado", extra={"update_id": update_id, "via": identidad})
        return {"status": "Update duplicado ignorado"}

    callback = update.get("callback_query")
    if isinstance(callback, dict):
        cb_chat_id = ((callback.get("message") or {}).get("chat") or {}).get("id")
        autorizados = settings.chats_autorizados
        if autorizados and cb_chat_id not in autorizados:
            logger.warning("chat_no_autorizado", extra={"chat_id": str(cb_chat_id)})
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Chat no autorizado para comandos",
            )
        resultado = await procesar_interaccion_telegram(update)
        return {"status": str(resultado.get("status") or "ok")}

    mensaje = update.get("message") or {}
    chat_id = (mensaje.get("chat") or {}).get("id")
    autorizados = settings.chats_autorizados
    if autorizados and chat_id not in autorizados:
        logger.warning("chat_no_autorizado", extra={"chat_id": str(chat_id)})
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Chat no autorizado para comandos",
        )

    texto = str(mensaje.get("text", "")).strip()
    if not texto.startswith("/buy"):
        cmd = texto.split(maxsplit=1)[0].lower().split("@", 1)[0] if texto else ""
        if cmd in ("/msg", "/vendedor", "/reporte", "/negociaciones") or (
            texto and not texto.startswith("/")
        ):
            resultado = await procesar_interaccion_telegram(update)
            estado_res = str(resultado.get("status") or "")
            if estado_res in (
                "negociacion_iniciada",
                "comunicacion_vendedor_registrada",
                "reporte_enviado",
            ):
                return {"status": estado_res}
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
