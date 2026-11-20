"""Worker periódico del tracker post-oferta (Fase 95)."""

from __future__ import annotations

import asyncio
import logging

from app.core.config import settings
from app.core.redis_client import redis_client
from app.db.base import obtener_sessionmaker
from app.db.uow import UnidadDeTrabajo
from app.services.offer_tracker import (
    FirecrawlOfertaTrackerAdapter,
    OfertaTrackerPort,
    OfertaResponse,
    rastrear_oferta_con_lock,
)
from app.services.scraper import build_default_scraper

logger = logging.getLogger("OpenClaw.OfferTrackerWorker")


async def ejecutar_ciclo(tracker: OfertaTrackerPort, *, limite: int | None = None) -> int:
    """Procesa una tanda de ofertas activas y devuelve el número de consultas."""
    async with obtener_sessionmaker()() as sesion:
        unidad = UnidadDeTrabajo(sesion)
        activas = await unidad.ofertas.listar_activas(
            limite=limite or settings.OFERTA_TRACKER_MAX_OFERTAS_LOTE
        )
        procesadas = 0
        for oferta, propiedad in activas:
            try:
                contrato = OfertaResponse.model_validate(oferta)
                evaluacion = await rastrear_oferta_con_lock(
                    redis_client=redis_client,
                    tracker=tracker,
                    oferta=contrato,
                    plataforma=propiedad.plataforma,
                    referencia_busqueda=propiedad.url_origen,
                    periodo_segundos=settings.OFERTA_TRACKER_INTERVAL_SEGUNDOS,
                    lock_ttl_segundos=settings.OFERTA_TRACKER_LOCK_TTL_SEGUNDOS,
                )
                if evaluacion is not None:
                    await unidad.ofertas.registrar_senal(
                        oferta,
                        senal=evaluacion.senal_detectada.value,
                        notas=evaluacion.motivo,
                    )
                    await unidad.confirmar()
                    procesadas += 1
            except Exception as exc:  # noqa: BLE001 - una oferta no detiene el lote
                await unidad.revertir()
                try:
                    await redis_client.xadd(
                        settings.REDIS_STREAM_DLQ_OFERTAS,
                        {
                            "oferta_id": str(oferta.id),
                            "propiedad_id": str(oferta.propiedad_id),
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                except Exception:  # noqa: BLE001 - la DLQ es best-effort
                    logger.exception("offer_tracker_dlq_error", extra={"oferta_id": str(oferta.id)})
                logger.exception(
                    "oferta_tracker_error",
                    extra={"oferta_id": str(oferta.id), "error": str(exc)},
                )
        await unidad.cerrar()
        return procesadas


async def consumir_tracker() -> None:
    logger.info("offer_tracker_inicio", extra={"intervalo": settings.OFERTA_TRACKER_INTERVAL_SEGUNDOS})
    tracker = FirecrawlOfertaTrackerAdapter(build_default_scraper())
    while True:
        try:
            total = await ejecutar_ciclo(tracker)
            logger.info("offer_tracker_ciclo", extra={"procesadas": total})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - el proceso debe esperar al siguiente ciclo
            logger.exception("offer_tracker_ciclo_error", extra={"error": str(exc)})
        await asyncio.sleep(settings.OFERTA_TRACKER_INTERVAL_SEGUNDOS)


def main() -> None:
    asyncio.run(consumir_tracker())


if __name__ == "__main__":
    main()
