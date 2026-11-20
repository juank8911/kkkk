"""Tracker post-oferta de la Fase 95.

El tracker solo propone transiciones respaldadas por señales explícitas. Una
señal ambigua se audita y no muta el estado de la oferta.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from app.core.config import settings
from app.core.redis_client import redis_client
from app.db.models import Oferta, Propiedad
from app.db.uow import UnidadDeTrabajo
from app.models.offers import OfertaEstado

logger = logging.getLogger("OpenClaw.OfferTracker")

PREFIJO_LOCK = "openclaw:oferta:tracker:lock:"


class SenalOferta(StrEnum):
    SIN_CAMBIO = "sin_cambio"
    PENDIENTE = "pendiente"
    BAJO_CONTRATO = "bajo_contrato"
    VENDIDA = "vendida"
    RETIRADA = "retirada"
    AMBIGUA = "ambigua"
    NO_DISPONIBLE = "no_disponible"


@dataclass(frozen=True, slots=True)
class ResultadoSeguimiento:
    senal: SenalOferta
    nuevo_estado: OfertaEstado | None
    motivo: str


class OfertaTrackerPort(Protocol):
    async def consultar(self, url: str, plataforma: str) -> str:
        """Devuelve el texto visible de la ficha, sin mutar el dominio."""
        ...


class ScraperOfertaTracker:
    """Adaptador que consulta una ficha usando el puerto de scraping existente."""

    def __init__(self, scraper: object) -> None:
        self.scraper = scraper

    async def consultar(self, url: str, plataforma: str) -> str:
        resultado = await self.scraper.scrape_url(url, plataforma)  # type: ignore[attr-defined]
        return resultado.markdown if resultado.ok else ""


def interpretar_senal(texto: str, estado_actual: str) -> ResultadoSeguimiento:
    """Interpreta señales explícitas; nunca convierte una señal incierta en cierre."""
    normalizado = " ".join(texto.lower().split())
    estado = OfertaEstado(estado_actual)
    if not normalizado:
        return ResultadoSeguimiento(SenalOferta.NO_DISPONIBLE, None, "ficha_sin_contenido")
    if any(valor in normalizado for valor in ("withdrawn", "removed", "listing removed", "retirada")):
        if estado in {OfertaEstado.PROPUESTA, OfertaEstado.CONTRAOFERTA, OfertaEstado.ACEPTADA}:
            return ResultadoSeguimiento(SenalOferta.RETIRADA, OfertaEstado.CANCELADA, "ficha_retirada")
        return ResultadoSeguimiento(SenalOferta.RETIRADA, None, "oferta_terminal")
    if any(valor in normalizado for valor in ("sold", "closed", "vendida")):
        if estado is OfertaEstado.ACEPTADA:
            return ResultadoSeguimiento(SenalOferta.VENDIDA, OfertaEstado.CERRADA, "venta_confirmada")
        return ResultadoSeguimiento(SenalOferta.VENDIDA, None, "venta_sin_oferta_aceptada")
    if "under contract" in normalizado or "bajo contrato" in normalizado:
        return ResultadoSeguimiento(SenalOferta.BAJO_CONTRATO, None, "senal_informativa")
    if "pending" in normalizado or "pendiente" in normalizado:
        return ResultadoSeguimiento(SenalOferta.PENDIENTE, None, "senal_informativa")
    return ResultadoSeguimiento(SenalOferta.AMBIGUA, None, "sin_senal_reconocida")


async def adquirir_lock(oferta_id: UUID) -> str | None:
    token = secrets.token_urlsafe(16)
    creado = await redis_client.set(
        f"{PREFIJO_LOCK}{oferta_id}",
        token,
        nx=True,
        ex=settings.OFERTA_TRACKER_LOCK_TTL_SEGUNDOS,
    )
    return token if creado else None


async def liberar_lock(oferta_id: UUID, token: str) -> None:
    clave = f"{PREFIJO_LOCK}{oferta_id}"
    actual = await redis_client.get(clave)
    valor_actual = actual.decode("utf-8") if isinstance(actual, bytes) else str(actual)
    if actual is not None and valor_actual == token:
        await redis_client.delete(clave)


async def rastrear_oferta(
    unidad: UnidadDeTrabajo,
    oferta: Oferta,
    propiedad: Propiedad,
    tracker: OfertaTrackerPort,
) -> ResultadoSeguimiento:
    """Consulta una oferta y persiste una transición solo si es válida y explícita."""
    token = await adquirir_lock(oferta.id)
    if token is None:
        return ResultadoSeguimiento(SenalOferta.SIN_CAMBIO, None, "lock_no_disponible")
    try:
        try:
            texto = await tracker.consultar(propiedad.url_origen, propiedad.plataforma)
            resultado = interpretar_senal(texto, oferta.estado)
        except Exception as exc:  # noqa: BLE001 - el tracker no debe detener el lote
            resultado = ResultadoSeguimiento(SenalOferta.NO_DISPONIBLE, None, f"consulta_fallida:{type(exc).__name__}")
        if resultado.nuevo_estado is not None:
            await unidad.ofertas.cambiar_estado(
                oferta,
                resultado.nuevo_estado,
                actor="offer-tracker",
                notas=f"Señal automática: {resultado.senal.value}; {resultado.motivo}",
            )
        else:
            await unidad.ofertas.registrar_senal(
                oferta,
                senal=resultado.senal.value,
                notas=resultado.motivo,
            )
        await unidad.confirmar()
        return resultado
    finally:
        await liberar_lock(oferta.id, token)
