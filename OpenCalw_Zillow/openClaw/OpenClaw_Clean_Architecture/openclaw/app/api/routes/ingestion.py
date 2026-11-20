"""Rutas de ingesta: crear un barrido y consultar su estado (S1-03 y S1-04)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.core.redis_client import redis_client
from app.models.contracts import BarridoRequest, ErrorResponse, JobResponse, JobStatus
from app.services.job_store import job_store

logger = logging.getLogger("OpenClaw.Ingesta")

router = APIRouter(prefix="/v1/openclaw", tags=["ingesta"])


@router.post(
    "/barridos",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResponse,
    summary="Encola un barrido inmobiliario",
    responses={
        422: {"model": ErrorResponse, "description": "Payload invalido"},
        503: {"model": ErrorResponse, "description": "Broker no disponible"},
    },
)
async def crear_barrido(payload: BarridoRequest) -> JobResponse:
    trace_id = (payload.metadata or {}).get("trace_id")

    job = JobResponse(
        job_id=uuid4(),
        status=JobStatus.QUEUED,
        query=payload.query,
        plataformas=payload.plataformas,
        trace_id=trace_id,
    )

    await job_store.guardar(job)

    mensaje = {
        "job_id": str(job.job_id),
        "query": payload.query,
        "plataformas": payload.plataformas,
        "max_resultados": payload.max_resultados_por_plataforma,
        "encolado_en": datetime.now(timezone.utc).isoformat(),
    }

    try:
        await redis_client.xadd(settings.REDIS_STREAM_SCRAPING, {"payload": json.dumps(mensaje)})
    except Exception as exc:  # noqa: BLE001 - cualquier fallo del broker se informa igual
        await job_store.marcar(job.job_id, JobStatus.FAILED, error="broker_no_disponible")
        logger.error(
            "fallo_encolando_barrido",
            extra={"job_id": str(job.job_id), "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El broker de mensajes no esta disponible",
        ) from exc

    logger.info(
        "barrido_encolado",
        extra={"job_id": str(job.job_id), "plataformas": len(payload.plataformas)},
    )
    return job


@router.get(
    "/barridos/{job_id}",
    response_model=JobResponse,
    summary="Consulta el estado de un barrido",
    responses={404: {"model": ErrorResponse, "description": "Job no encontrado"}},
)
async def consultar_barrido(job_id: UUID) -> JobResponse:
    job = await job_store.obtener(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job no encontrado",
        )
    return job
