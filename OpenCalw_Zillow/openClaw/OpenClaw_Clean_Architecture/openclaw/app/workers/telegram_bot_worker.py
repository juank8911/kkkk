"""Worker de Telegram: botones interactivos, monitoreo de mensajes y reporte diario 11:00 AM.

Ejecuta dos tareas concurrentes:
1. Long-polling (`getUpdates`) para botones inline (`✅ Aprobar / Negociar`,
   `❌ Rechazar`, `▶️ Iniciar tal cual`), instrucciones de negociación y comandos
   `/msg`, `/vendedor`, `/reporte` y `/negociaciones` (si no hay webhook externo).
2. Programador diario de reporte de estado de negociaciones a las 11:00 AM
   (`NEGOTIATION_REPORT_HOUR=11`, `NEGOTIATION_REPORT_TZ=America/New_York`).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from app.core.config import settings
from app.core.logging import configure_logging
from app.core.redis_client import redis_client
from app.services.negotiation_reporter import enviar_reporte_diario_si_corresponde
from app.services.telegram_workflow import procesar_interaccion_telegram

logger = logging.getLogger("OpenClaw.TelegramBotWorker")

PREFIJO_DEDUPE = "openclaw:tg:update:"
OFFSET_KEY = "openclaw:tg:poll:offset"


def _extraer_chat_id(update: dict[str, Any]) -> int | None:
    cb = update.get("callback_query")
    if isinstance(cb, dict):
        chat_id = ((cb.get("message") or {}).get("chat") or {}).get("id")
        return int(chat_id) if isinstance(chat_id, int) else None
    msg = update.get("message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    return int(chat_id) if isinstance(chat_id, int) else None


async def _es_nuevo(update_id: int) -> bool:
    creado = await redis_client.set(
        f"{PREFIJO_DEDUPE}{update_id}",
        "1",
        nx=True,
        ex=settings.TG_DEDUPE_TTL_SEGUNDOS,
    )
    return bool(creado)


async def _bucle_reporte_diario() -> None:
    """Verifica cada 60 s si ya son las 11:00 AM para emitir el reporte diario."""
    logger.info(
        "reporte_diario_scheduler_inicio",
        extra={
            "hora": f"{settings.NEGOTIATION_REPORT_HOUR:02d}:{settings.NEGOTIATION_REPORT_MINUTE:02d}",
            "tz": settings.NEGOTIATION_REPORT_TZ,
        },
    )
    while True:
        try:
            await enviar_reporte_diario_si_corresponde()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - nunca detener el scheduler
            logger.warning("reporte_diario_scheduler_error", extra={"error": str(exc)})
        await asyncio.sleep(60)


async def _bucle_polling(token: str) -> None:
    offset: int | None = None
    crudo_offset = await redis_client.get(OFFSET_KEY)
    if crudo_offset is not None:
        try:
            offset = int(
                crudo_offset.decode("utf-8") if isinstance(crudo_offset, bytes) else crudo_offset
            )
        except ValueError:
            offset = None

    base_url = f"https://api.telegram.org/bot{token}"
    async with httpx.AsyncClient(timeout=35.0) as client:
        try:
            wh = (await client.get(f"{base_url}/getWebhookInfo")).json()
            if (wh.get("result") or {}).get("url"):
                logger.info("telegram_bot_worker_omitido_por_webhook_activo")
                return
        except Exception as exc:  # noqa: BLE001
            logger.warning("telegram_webhook_check_error", extra={"error": str(exc)})

        while True:
            try:
                params: dict[str, Any] = {
                    "timeout": 25,
                    "allowed_updates": ["message", "callback_query"],
                }
                if offset is not None:
                    params["offset"] = offset
                resp = await client.get(f"{base_url}/getUpdates", params=params)
                resp.raise_for_status()
                body = resp.json()
                updates = body.get("result") or []
                for upd in updates:
                    uid = upd.get("update_id")
                    if isinstance(uid, int):
                        offset = uid + 1
                        await redis_client.set(OFFSET_KEY, str(offset), ex=7 * 86400)
                        if not await _es_nuevo(uid):
                            continue
                    chat_id = _extraer_chat_id(upd)
                    autorizados = settings.chats_autorizados
                    if autorizados and chat_id not in autorizados:
                        logger.warning(
                            "telegram_poll_chat_no_autorizado",
                            extra={"chat_id": str(chat_id)},
                        )
                        continue
                    resultado = await procesar_interaccion_telegram(upd)
                    logger.info("telegram_update_procesado", extra={"resultado": resultado})
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("telegram_poll_error", extra={"error": str(exc)})
                await asyncio.sleep(3)


async def consumir_updates_telegram() -> None:
    configure_logging(settings.LOG_LEVEL, settings.LOG_JSON)
    token = settings.TG_TOKEN.strip()
    if not token:
        logger.warning("telegram_bot_worker_sin_token")
        return

    logger.info("telegram_bot_worker_inicio")
    await asyncio.gather(_bucle_polling(token), _bucle_reporte_diario())


def main() -> None:
    try:
        asyncio.run(consumir_updates_telegram())
    except KeyboardInterrupt:
        logger.info("telegram_bot_worker_detenido")


if __name__ == "__main__":
    main()
