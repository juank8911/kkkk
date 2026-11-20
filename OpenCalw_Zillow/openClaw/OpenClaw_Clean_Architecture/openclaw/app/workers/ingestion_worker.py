"""Worker asincrono de ingestion.

Consume el stream de Redis con un consumer group, procesa cada plataforma en
paralelo y confirma el mensaje con ``XACK``. La recuperacion de mensajes
pendientes y la dead-letter queue llegan en el sprint 3 (tareas S3-01 y S3-02).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from app.core.config import settings
from app.core.logging import configure_logging, job_context
from app.core.redis_client import redis_client
from app.services.scraper import extraer_con_reintentos

logger = logging.getLogger("OpenClaw.Worker")


async def _procesar_mensaje(id_mensaje: str, datos: dict[bytes, bytes]) -> None:
    payload: dict[str, Any] = json.loads(datos[b"payload"].decode("utf-8"))
    plataformas: list[str] = payload.get("plataformas", [])
    query: str = payload.get("query", "")

    with job_context(job_id=str(id_mensaje)):
        logger.info("mensaje_recibido", extra={"plataformas": len(plataformas)})

        tareas = [extraer_con_reintentos(url, query) for url in plataformas]
        resultados = await asyncio.gather(*tareas, return_exceptions=True)

        errores = sum(1 for r in resultados if isinstance(r, BaseException))
        logger.info(
            "mensaje_procesado",
            extra={"plataformas": len(plataformas), "errores": errores},
        )

        await redis_client.xack(
            settings.REDIS_STREAM_SCRAPING,
            settings.REDIS_CONSUMER_GROUP,
            id_mensaje,
        )


async def consumir_cola_scraping() -> None:
    configure_logging(settings.LOG_LEVEL, settings.LOG_JSON)

    logger.info(
        "worker_inicio",
        extra={
            "stream": settings.REDIS_STREAM_SCRAPING,
            "grupo": settings.REDIS_CONSUMER_GROUP,
            "consumidor": settings.REDIS_CONSUMER_NAME,
        },
    )

    try:
        await redis_client.xgroup_create(
            settings.REDIS_STREAM_SCRAPING,
            settings.REDIS_CONSUMER_GROUP,
            mkstream=True,
        )
        logger.info("grupo_creado")
    except Exception:  # noqa: BLE001 - el grupo ya existe en reinicios
        logger.debug("grupo_ya_existe")

    while True:
        try:
            mensajes = await redis_client.xreadgroup(
                groupname=settings.REDIS_CONSUMER_GROUP,
                consumername=settings.REDIS_CONSUMER_NAME,
                streams={settings.REDIS_STREAM_SCRAPING: ">"},
                count=1,
                block=5000,
            )
        except asyncio.CancelledError:
            logger.info("worker_cancelado")
            raise
        except Exception as exc:  # noqa: BLE001 - el bucle debe sobrevivir a Redis
            logger.error("error_lectura_cola", extra={"error": str(exc)})
            await asyncio.sleep(2)
            continue

        if not mensajes:
            continue

        for _stream, alertas in mensajes:
            for id_mensaje, datos in alertas:
                try:
                    await _procesar_mensaje(id_mensaje, datos)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "error_procesando_mensaje",
                        extra={"mensaje": str(id_mensaje), "error": str(exc)},
                    )


def main() -> None:
    try:
        asyncio.run(consumir_cola_scraping())
    except KeyboardInterrupt:
        logger.info("worker_fin")


if __name__ == "__main__":
    main()
