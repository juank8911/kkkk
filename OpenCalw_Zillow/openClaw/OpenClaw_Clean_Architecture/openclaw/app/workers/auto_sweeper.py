"""Programador de barridos automáticos sobre el ranking de mercados.

Este worker solo descubre y encola búsquedas. No crea ofertas, no negocia y no
contacta vendedores; esas acciones permanecen bajo aprobación humana.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from uuid import uuid4

from app.core.config import settings
from app.core.logging import configure_logging
from app.core.redis_client import redis_client
from app.models.contracts import JobResponse, JobStatus
from app.services.job_store import job_store
from app.services.platforms import validate_platforms
from app.frontend.markets import listar_mercados_ordenados

logger = logging.getLogger("OpenClaw.AutoSweeper")
LOCK_KEY = "openclaw:auto-sweeper:cycle-lock"
LAST_DAY_KEY = "openclaw:auto-sweeper:last-day"


def plataformas_configuradas() -> list[str]:
    """Devuelve las plataformas configuradas y valida el catálogo cerrado."""
    candidatas = [item.strip().lower() for item in settings.AUTO_SWEEP_PLATFORMS.split(",") if item.strip()]
    return validate_platforms(candidatas)


async def ejecutar_ciclo() -> int:
    """Encola una sola ciudad del ranking por día y devuelve 0 o 1."""
    if not settings.AUTO_SWEEP_ENABLED:
        logger.info("auto_sweeper_deshabilitado")
        return 0

    adquirido = await redis_client.set(
        LOCK_KEY,
        str(uuid4()),
        nx=True,
        ex=settings.AUTO_SWEEP_LOCK_TTL_SEGUNDOS,
    )
    if not adquirido:
        logger.info("auto_sweeper_ciclo_omitido_lock")
        return 0

    try:
        dia_utc = datetime.now(UTC).date()
        dia_texto = dia_utc.isoformat()
        ultimo_dia = await redis_client.get(LAST_DAY_KEY)
        if isinstance(ultimo_dia, bytes):
            ultimo_dia = ultimo_dia.decode("utf-8")
        if ultimo_dia == dia_texto:
            logger.info("auto_sweeper_dia_ya_procesado", extra={"dia": dia_texto})
            return 0

        plataformas = plataformas_configuradas()
        mercados = listar_mercados_ordenados()
        if not mercados:
            return 0
        mercado = mercados[dia_utc.toordinal() % len(mercados)]
        job = JobResponse(
            job_id=uuid4(),
            status=JobStatus.QUEUED,
            query=mercado.query,
            plataformas=plataformas,
            trace_id=f"auto-market-{mercado.ranking}-{dia_texto}",
        )
        await job_store.guardar(job)
        await redis_client.xadd(
            settings.REDIS_STREAM_SCRAPING,
            {
                "payload": json.dumps(
                    {
                        "job_id": str(job.job_id),
                        "query": mercado.query,
                        "plataformas": plataformas,
                        "max_resultados": 20,
                        "encolado_en": datetime.now(UTC).isoformat(),
                        "origen": "auto_market_sweep",
                        "market_ranking": mercado.ranking,
                        "atractivo_pct": mercado.atractivo_pct,
                        "dia_utc": dia_texto,
                    }
                )
            },
        )
        await redis_client.set(LAST_DAY_KEY, dia_texto, ex=172800)
        logger.info(
            "barrido_automatico_encolado",
            extra={
                "job_id": str(job.job_id),
                "query": mercado.query,
                "market_ranking": mercado.ranking,
                "atractivo_pct": mercado.atractivo_pct,
                "dia_utc": dia_texto,
                "plataformas": len(plataformas),
            },
        )
        return 1
    finally:
        await redis_client.delete(LOCK_KEY)


async def consumir_auto_sweeper() -> None:
    configure_logging(settings.LOG_LEVEL, settings.LOG_JSON)
    logger.info(
        "auto_sweeper_inicio",
        extra={
            "enabled": settings.AUTO_SWEEP_ENABLED,
            "intervalo": settings.AUTO_SWEEP_INTERVAL_SEGUNDOS,
            "mercados": settings.AUTO_SWEEP_MARKET_LIMIT,
        },
    )
    if not settings.AUTO_SWEEP_RUN_ON_START:
        await asyncio.sleep(settings.AUTO_SWEEP_INTERVAL_SEGUNDOS)
    while True:
        try:
            await ejecutar_ciclo()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - el siguiente ciclo debe continuar
            logger.exception("auto_sweeper_ciclo_error", extra={"error": str(exc)})
        await asyncio.sleep(settings.AUTO_SWEEP_INTERVAL_SEGUNDOS)


def main() -> None:
    asyncio.run(consumir_auto_sweeper())


if __name__ == "__main__":
    main()
