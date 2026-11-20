"""Router FastAPI ligero para servir el panel mínimo de jobs y telemetría (S5-04)."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import HTMLResponse

from app.api.routes import ingestion
from app.core.redis_client import redis_client
from app.frontend.markets import MercadoAtractivo, listar_mercados_ordenados
from app.frontend.sweep_properties import (
    PropiedadesBarridoResponse,
    obtener_propiedades_por_barrido,
)
from app.models.contracts import BarridoRequest, JobResponse
from app.services.market_data import fuentes_directas_mercado

logger = logging.getLogger("OpenClaw.Panel")

PANEL_HTML_PATH = Path(__file__).resolve().parent / "panel.html"

router = APIRouter(tags=["panel"])


@router.get("/panel", response_class=HTMLResponse, summary="Panel mínimo de jobs y telemetría")
async def servir_panel_jobs() -> HTMLResponse:
    """Devuelve el panel HTML autocontenido para listar jobs, ver detalle y leer telemetría WS."""
    html = PANEL_HTML_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html, status_code=200)


@router.get(
    "/panel/api/mercados",
    response_model=list[MercadoAtractivo],
    summary="Lista los 30 mercados objetivo ordenados por porcentaje de atractivo",
)
async def listar_mercados_panel() -> list[MercadoAtractivo]:
    """Devuelve el ranking de mercados calculado por rendimiento, precio, renta y factores."""
    return listar_mercados_ordenados()


@router.get(
    "/panel/api/datos-mercado",
    summary="Fuentes directas y auditables de datos de mercado",
)
async def telemetria_datos_mercado_panel() -> dict[str, object]:
    """Expone solo catálogo/metadatos de fuentes, nunca credenciales ni secretos."""
    return {"fuentes": fuentes_directas_mercado(), "decisiones_automaticas": False}


@router.get(
    "/panel/api/barridos",
    response_model=list[JobResponse],
    summary="Lista los jobs recientes para el panel",
)
async def listar_barridos_panel(
    limit: int = Query(default=50, ge=1, le=100),
) -> list[JobResponse]:
    """Carga jobs automáticos y manuales desde Redis sin exponer secretos."""
    jobs: list[JobResponse] = []
    async for clave in redis_client.scan_iter(match="openclaw:job:*", count=100):
        crudo = await redis_client.get(clave)
        if crudo is None:
            continue
        if isinstance(crudo, bytes):
            crudo = crudo.decode("utf-8")
        try:
            jobs.append(JobResponse.model_validate_json(crudo))
        except ValueError:
            continue
    jobs.sort(key=lambda job: job.created_at, reverse=True)
    return jobs[:limit]


@router.post(
    "/panel/api/barridos",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Crea un barrido desde el panel sin exponer la API key",
)
async def crear_barrido_desde_panel(payload: BarridoRequest) -> JobResponse:
    """Encola un barrido usando internamente la misma lógica protegida de la API."""
    return await ingestion.crear_barrido(payload)


@router.get(
    "/panel/api/barridos/{job_id}",
    response_model=JobResponse,
    summary="Consulta un job desde el panel sin exponer la API key",
)
async def consultar_barrido_desde_panel(job_id: UUID) -> JobResponse:
    """Consulta el estado del job sin enviar la clave táctica al navegador."""
    return await ingestion.consultar_barrido(job_id)


@router.get(
    "/panel/api/barridos/{job_id}/propiedades",
    response_model=PropiedadesBarridoResponse,
    summary="Lista las propiedades de un barrido clasificadas por estado de análisis",
)
async def listar_propiedades_barrido_panel(job_id: UUID) -> PropiedadesBarridoResponse:
    """Devuelve las propiedades persistidas del barrido: enviadas a aprobación, rechazadas y pendientes."""
    try:
        return await obtener_propiedades_por_barrido(job_id)
    except Exception as exc:  # noqa: BLE001 - fallo de BD reportado como 503 controlado
        logger.warning(
            "panel_propiedades_barrido_error",
            extra={"job_id": str(job_id), "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudieron consultar las propiedades del barrido en PostgreSQL",
        ) from exc
