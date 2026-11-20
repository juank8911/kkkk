"""Worker asincrono de ingestion (S3-01, S3-02, S3-03).

Ciclo de vida de un mensaje del stream:

1. Se lee con ``XREADGROUP`` o se recupera con ``XAUTOCLAIM`` si quedo pendiente
   porque el consumidor anterior murio a mitad del trabajo.
2. Se procesa con idempotencia por ``job_id`` + plataforma: un mensaje
   reprocesado no vuelve a raspar ni duplica propiedades.
3. El estado del job solo cambia por transiciones legales de la maquina de
   estados.
4. Si el procesamiento falla, el mensaje NO se confirma: queda pendiente y
   ``XAUTOCLAIM`` lo reentrega tras ``CLAIM_IDLE_MS``. Al agotar
   ``MAX_ENTREGAS`` se copia a la dead-letter queue y se confirma para no
   bloquear la cola.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from uuid import UUID

from redis.exceptions import ResponseError
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.core.logging import configure_logging, job_context
from app.core.redis_client import redis_client
from app.db.models import Job
from app.db.uow import unidad_de_trabajo
from app.models.contracts import JobStatus
from app.orchestrator.decisiones import estado_inicial_desde_orm, evaluar_y_registrar_decision
from app.orchestrator.nodes.telegram_alert import RedisAlertDedupeStore
from app.services import idempotency
from app.services.domain_semaphore import semaforos
from app.services.ingestion_pipeline import procesar_lote_extraccion_con_uow
from app.services.ollama_listing_extractor import (
    enriquecer_ubicacion_extraccion,
    extraction_to_markdown,
    extraer_viviendas_ollama,
)
from app.services.platforms import InvalidPlatformError, get_platform_spec
from app.services.scraper import extraer_con_reintentos
from app.workers.state_machine import TransicionInvalida, aplicar_transicion

logger = logging.getLogger("OpenClaw.Worker")

PREFIJO_ENTREGAS = "openclaw:entregas:"
TTL_ENTREGAS = 7 * 24 * 3600
MAX_LOTES_RECUPERACION = 10


def _estado_final(resultados: list[dict[str, Any]]) -> JobStatus:
    exitos = sum(1 for r in resultados if r.get("status") == "exito")
    if exitos == len(resultados):
        return JobStatus.COMPLETED
    if exitos == 0:
        return JobStatus.FAILED
    return JobStatus.PARTIAL


def debe_enviar_a_dlq(entregas: int, maximo: int | None = None) -> bool:
    """True cuando el mensaje ya consumio todas sus entregas."""
    limite = settings.MAX_ENTREGAS if maximo is None else maximo
    return entregas >= limite


def _texto(valor: Any) -> str:
    if isinstance(valor, bytes):
        return valor.decode("utf-8", errors="replace")
    return str(valor)


async def _entregas(id_mensaje: str) -> int:
    crudo = await redis_client.get(f"{PREFIJO_ENTREGAS}{id_mensaje}")
    try:
        return int(_texto(crudo)) if crudo is not None else 0
    except ValueError:
        return 0


async def _registrar_entrega(id_mensaje: str) -> int:
    clave = f"{PREFIJO_ENTREGAS}{id_mensaje}"
    total = int(await redis_client.incr(clave))
    await redis_client.expire(clave, TTL_ENTREGAS)
    return total


async def _olvidar_entregas(id_mensaje: str) -> None:
    await redis_client.delete(f"{PREFIJO_ENTREGAS}{id_mensaje}")


async def _extraer_plataforma(clave: str, query: str, job_id: UUID | None) -> dict[str, Any]:
    try:
        spec = get_platform_spec(clave)
    except InvalidPlatformError:
        logger.warning("plataforma_invalida", extra={"plataforma": clave})
        return {"plataforma": clave, "status": "plataforma_invalida"}

    if job_id is not None and not await idempotency.reclamar(job_id, clave):
        return {"plataforma": clave, "status": "omitido_idempotente"}

    async with semaforos.turno(spec.domain):
        resultado = await extraer_con_reintentos(f"https://{spec.domain}", query)
    return {"plataforma": clave, **resultado}


async def _persistir_resultado(
    job_id: UUID,
    resultado: dict[str, Any],
    *,
    query: str,
    plataformas: list[str],
) -> dict[str, Any]:
    """Persiste los anuncios extraídos; un job exitoso no queda solo en Redis."""
    if resultado.get("status") != "exito":
        return resultado
    data = resultado.get("data") or {}
    markdown = str(data.get("markdown") or "").strip()
    if not markdown:
        return {**resultado, "persistencia": {"persistidas": 0, "motivo": "markdown_vacio"}}

    markdown_para_pipeline = markdown
    extraction, extraction_error = await extraer_viviendas_ollama(
        markdown,
        query=query,
        platform=str(resultado.get("plataforma") or "desconocida"),
        url_origen=resultado.get("dominio"),
    )
    if extraction is not None:
        extraction = await enriquecer_ubicacion_extraccion(extraction, query=query)
        markdown_para_pipeline = extraction_to_markdown(
            extraction,
            platform=str(resultado.get("plataforma") or "desconocida"),
            url_origen=resultado.get("dominio"),
            query=query,
        )
        if not markdown_para_pipeline:
            return {
                **resultado,
                "status": "sin_propiedades",
                "extraccion": {"modelo": settings.OLLAMA_MODEL, "viviendas": 0},
                "persistencia": {"persistidas": 0, "motivo": "ollama_sin_viviendas"},
            }

    try:
        async with unidad_de_trabajo() as unidad:
            if await unidad.jobs.obtener(job_id) is None:
                await unidad.jobs.crear(
                    Job(
                        job_id=job_id,
                        status=JobStatus.RUNNING.value,
                        query=query[:200],
                        plataformas=plataformas,
                    )
                )
            summary = await procesar_lote_extraccion_con_uow(
                unidad,
                job_id=job_id,
                plataforma=str(resultado.get("plataforma") or "desconocida"),
                markdown=markdown_para_pipeline,
                url_origen=resultado.get("dominio"),
                require_county=False,
            )

        evaluadas = 0
        if summary.propiedad_ids:
            dedupe_store = RedisAlertDedupeStore(redis_client)
            async with unidad_de_trabajo() as unidad_eval:
                propiedades_job = await unidad_eval.propiedades.listar_por_job(job_id)
                ids_nuevos = set(summary.propiedad_ids)
                for prop in propiedades_job:
                    if prop.id not in ids_nuevos:
                        continue
                    try:
                        await evaluar_y_registrar_decision(
                            unidad_eval,
                            propiedad_id=prop.id,
                            estado_inicial=estado_inicial_desde_orm(prop),
                            job_id=job_id,
                            dedupe_store=dedupe_store,
                        )
                        evaluadas += 1
                    except Exception as eval_exc:  # noqa: BLE001 - una propiedad no frena el lote
                        logger.warning(
                            "evaluacion_propiedad_error",
                            extra={"propiedad_id": str(prop.id), "error": str(eval_exc)},
                        )

        persistencia = {**summary.model_dump(mode="json"), "evaluadas": evaluadas}
        logger.info(
            "resultado_scraper_persistido",
            extra={
                "job_id": str(job_id),
                "plataforma": resultado.get("plataforma"),
                "persistidas": summary.persistidas,
            },
        )
        extra = (
            {"modelo": settings.OLLAMA_MODEL, "viviendas": len(extraction.viviendas_en_venta)}
            if extraction
            else {"fallback": extraction_error or "ollama_no_configurado"}
        )
        if summary.normalizadas == 0:
            return {
                **resultado,
                "status": "sin_propiedades",
                "extraccion": extra,
                "persistencia": persistencia,
            }
        return {**resultado, "extraccion": extra, "persistencia": persistencia}
    except (SQLAlchemyError, ValueError) as exc:
        logger.exception("persistencia_ingesta_error", extra={"job_id": str(job_id)})
        return {**resultado, "status": "persistencia_error", "error_persistencia": str(exc)}


async def _procesar_mensaje(id_mensaje: str, datos: dict[bytes, bytes]) -> None:
    """Ejecuta el trabajo del mensaje. No confirma: eso lo hace el bucle."""
    crudo = datos.get(b"payload", b"{}")
    payload: dict[str, Any] = json.loads(_texto(crudo))
    job_id_crudo = payload.get("job_id")
    job_id = UUID(str(job_id_crudo)) if job_id_crudo else None
    query: str = payload.get("query", "")
    plataformas: list[str] = payload.get("plataformas", [])

    with job_context(job_id=str(job_id or id_mensaje)):
        if job_id is not None:
            try:
                await aplicar_transicion(job_id, JobStatus.RUNNING)
            except TransicionInvalida:
                logger.info("job_ya_cerrado", extra={"job_id": str(job_id)})
                return

        tareas = [_extraer_plataforma(clave, query, job_id) for clave in plataformas]
        resultados = await asyncio.gather(*tareas, return_exceptions=True)

        normalizados: list[dict[str, Any]] = []
        for clave, resultado in zip(plataformas, resultados, strict=False):
            if isinstance(resultado, BaseException):
                normalizados.append({"plataforma": clave, "status": "fallo_critico"})
            else:
                normalizados.append(resultado)

        if job_id is not None:
            normalizados = [
                await _persistir_resultado(
                    job_id,
                    resultado,
                    query=query,
                    plataformas=plataformas,
                )
                for resultado in normalizados
            ]

        progress = {r["plataforma"]: str(r.get("status", "desconocido")) for r in normalizados}
        estado = _estado_final(normalizados)

        if job_id is not None:
            try:
                await aplicar_transicion(job_id, estado, progress=progress)
            except TransicionInvalida:
                logger.warning("estado_final_rechazado", extra={"job_id": str(job_id)})
            try:
                async with unidad_de_trabajo() as unidad:
                    await unidad.jobs.actualizar_estado(job_id, estado.value)
            except SQLAlchemyError:
                logger.exception("estado_job_sql_no_actualizado", extra={"job_id": str(job_id)})

        logger.info(
            "mensaje_procesado",
            extra={"plataformas": len(plataformas), "estado": estado.value},
        )


async def _enviar_a_dlq(
    id_mensaje: str, datos: dict[bytes, bytes], motivo: str, entregas: int
) -> None:
    await redis_client.xadd(
        settings.REDIS_STREAM_DLQ,
        {
            "mensaje_original": id_mensaje,
            "payload": _texto(datos.get(b"payload", b"")),
            "motivo": motivo[:500],
            "entregas": str(entregas),
        },
        maxlen=10000,
        approximate=True,
    )
    logger.error(
        "mensaje_a_dlq",
        extra={"mensaje": id_mensaje, "entregas": entregas, "motivo": motivo[:200]},
    )


async def _confirmar(id_mensaje: str) -> None:
    await redis_client.xack(
        settings.REDIS_STREAM_SCRAPING, settings.REDIS_CONSUMER_GROUP, id_mensaje
    )


async def _manejar_mensaje(id_mensaje: str, datos: dict[bytes, bytes]) -> None:
    """Procesa, reintenta o manda a DLQ; confirma solo cuando corresponde."""
    try:
        await _procesar_mensaje(id_mensaje, datos)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - cualquier fallo debe poder reintentarse
        entregas = await _registrar_entrega(id_mensaje)
        if debe_enviar_a_dlq(entregas):
            await _enviar_a_dlq(id_mensaje, datos, f"{type(exc).__name__}: {exc}", entregas)
            await _confirmar(id_mensaje)
            await _olvidar_entregas(id_mensaje)
        else:
            logger.warning(
                "mensaje_reintentable",
                extra={
                    "mensaje": id_mensaje,
                    "entregas": entregas,
                    "maximo": settings.MAX_ENTREGAS,
                    "error": str(exc),
                },
            )
        return

    await _confirmar(id_mensaje)
    await _olvidar_entregas(id_mensaje)


async def _recuperar_pendientes() -> int:
    """Reclama mensajes pendientes de consumidores caidos. Devuelve cuantos."""
    recuperados = 0
    cursor = "0-0"
    for _ in range(MAX_LOTES_RECUPERACION):
        try:
            siguiente, mensajes, _eliminados = await redis_client.xautoclaim(
                name=settings.REDIS_STREAM_SCRAPING,
                groupname=settings.REDIS_CONSUMER_GROUP,
                consumername=settings.REDIS_CONSUMER_NAME,
                min_idle_time=settings.CLAIM_IDLE_MS,
                start_id=cursor,
                count=10,
            )
        except ResponseError as exc:
            logger.error("error_xautoclaim", extra={"error": str(exc)})
            return recuperados

        for id_mensaje, datos in mensajes:
            await _manejar_mensaje(id_mensaje, datos)
            recuperados += 1

        nuevo_cursor = _texto(siguiente)
        if not mensajes or nuevo_cursor in {"0-0", "0"}:
            break
        cursor = nuevo_cursor

    if recuperados:
        logger.info("pendientes_recuperados", extra={"total": recuperados})
    return recuperados


async def _asegurar_grupo() -> None:
    try:
        await redis_client.xgroup_create(
            settings.REDIS_STREAM_SCRAPING,
            settings.REDIS_CONSUMER_GROUP,
            mkstream=True,
        )
        logger.info("grupo_creado")
    except ResponseError as exc:
        if "BUSYGROUP" in str(exc):
            logger.debug("grupo_ya_existe")
        else:
            raise


async def consumir_cola_scraping() -> None:
    configure_logging(settings.LOG_LEVEL, settings.LOG_JSON)

    logger.info(
        "worker_inicio",
        extra={
            "stream": settings.REDIS_STREAM_SCRAPING,
            "grupo": settings.REDIS_CONSUMER_GROUP,
            "consumidor": settings.REDIS_CONSUMER_NAME,
            "max_entregas": settings.MAX_ENTREGAS,
            "claim_idle_ms": settings.CLAIM_IDLE_MS,
        },
    )

    await _asegurar_grupo()

    while True:
        try:
            await _recuperar_pendientes()
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
                await _manejar_mensaje(id_mensaje, datos)


def main() -> None:
    try:
        asyncio.run(consumir_cola_scraping())
    except KeyboardInterrupt:
        logger.info("worker_fin")


if __name__ == "__main__":
    main()
