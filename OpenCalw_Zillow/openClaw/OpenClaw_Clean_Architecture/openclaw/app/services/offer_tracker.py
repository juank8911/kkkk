"""Tracker post-oferta unificado de la Fase 95 (Dev 1 + Dev 2).

Combina:
- Contrato del worker periódico (`OfertaTrackerPort`, `ScraperOfertaTracker`,
  `SenalOferta`, `ResultadoSeguimiento`, `interpretar_senal`, `adquirir_lock`,
  `liberar_lock`, `rastrear_oferta`).
- Contrato enriquecido de dominio Dev 2 (`SenalMercadoOferta`,
  `ObservacionTrackerOferta`, `EvaluacionSeguimientoOferta`,
  `FirecrawlOfertaTrackerAdapter`, `detectar_senales_mercado`,
  `formatear_aviso_tracker_oferta`, `evaluar_observacion_oferta`,
  `rastrear_oferta_con_lock`).
"""

from __future__ import annotations

import logging
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.core.redis_client import redis_client
from app.db.models import Oferta, Propiedad
from app.db.uow import UnidadDeTrabajo
from app.integrations.scraper_port import ScraperPort
from app.models.offers import OfertaEstado, OfertaEventoResponse, OfertaResponse
from app.services.offers import TRANSICIONES_OFERTA

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


@runtime_checkable
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
    if any(
        valor in normalizado for valor in ("withdrawn", "removed", "listing removed", "retirada")
    ):
        if estado in {OfertaEstado.PROPUESTA, OfertaEstado.CONTRAOFERTA, OfertaEstado.ACEPTADA}:
            return ResultadoSeguimiento(
                SenalOferta.RETIRADA, OfertaEstado.CANCELADA, "ficha_retirada"
            )
        return ResultadoSeguimiento(SenalOferta.RETIRADA, None, "oferta_terminal")
    if any(valor in normalizado for valor in ("sold", "closed", "vendida")):
        if estado is OfertaEstado.ACEPTADA:
            return ResultadoSeguimiento(
                SenalOferta.VENDIDA, OfertaEstado.CERRADA, "venta_confirmada"
            )
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
            resultado = ResultadoSeguimiento(
                SenalOferta.NO_DISPONIBLE, None, f"consulta_fallida:{type(exc).__name__}"
            )
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


# ---------------------------------------------------------------------------
# Capa de dominio enriquecida Dev 2 (observaciones, detección de conflictos,
# plantilla Telegram informativa y control de periodo mínimo)
# ---------------------------------------------------------------------------


class SenalMercadoOferta(StrEnum):
    """Señales normalizadas detectadas en portales inmobiliarios para una oferta activa."""

    ACTIVE = "active"
    PENDING = "pending"
    UNDER_CONTRACT = "under_contract"
    CONTINGENT = "contingent"
    SOLD = "sold"
    WITHDRAWN = "withdrawn"
    AMBIGUA = "ambigua"
    FUENTE_NO_DISPONIBLE = "fuente_no_disponible"


_SIGNAL_PATTERNS: Final[tuple[tuple[SenalMercadoOferta, re.Pattern[str]], ...]] = (
    (
        SenalMercadoOferta.SOLD,
        re.compile(r"(?i)\b(?:sold|closed\s+sale|recently\s+sold|vendid[oa])\b"),
    ),
    (
        SenalMercadoOferta.UNDER_CONTRACT,
        re.compile(r"(?i)\b(?:under\s+contract|in\s+contract|bajo\s+contrato)\b"),
    ),
    (
        SenalMercadoOferta.PENDING,
        re.compile(r"(?i)\b(?:pending(?:\s+sale)?|sale\s+pending|pendiente\s+de\s+cierre)\b"),
    ),
    (
        SenalMercadoOferta.CONTINGENT,
        re.compile(r"(?i)\b(?:contingent|accepting\s+backup\s+offers?|contingente)\b"),
    ),
    (
        SenalMercadoOferta.WITHDRAWN,
        re.compile(
            r"(?i)\b(?:withdrawn|off\s+market|listing\s+removed|expired|retirad[oa]\s+del\s+mercado)\b"
        ),
    ),
    (
        SenalMercadoOferta.ACTIVE,
        re.compile(r"(?i)\b(?:for\s+sale|active\s+listing|en\s+venta|back\s+on\s+market)\b"),
    ),
)

_ESTADOS_ACTIVOS: Final[frozenset[OfertaEstado]] = frozenset(
    {OfertaEstado.PROPUESTA, OfertaEstado.CONTRAOFERTA, OfertaEstado.ACEPTADA}
)


class ObservacionTrackerOferta(BaseModel):
    """Observación cruda retornada por un adaptador de seguimiento de oferta."""

    model_config = ConfigDict(extra="forbid")

    oferta_id: UUID
    plataforma: str
    url_fuente: str | None = None
    markdown: str = ""
    disponible: bool = True
    error: str | None = None
    observado_en: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvaluacionSeguimientoOferta(BaseModel):
    """Resultado auditable de evaluar una señal de mercado sobre una oferta."""

    model_config = ConfigDict(extra="forbid")

    oferta_id: UUID
    estado_actual: OfertaEstado
    senal_detectada: SenalMercadoOferta
    senales_coincidentes: list[SenalMercadoOferta] = Field(default_factory=list)
    estado_sugerido: OfertaEstado | None = None
    transicion_aplicable: bool = False
    requiere_revision_humana: bool = False
    motivo: str
    evento: OfertaEventoResponse
    aviso_telegram: str | None = None
    accion_comercial_automatica: bool = False


class FirecrawlOfertaTrackerAdapter:
    """Adaptador unificado respaldado por `ScraperPort` (compatible con worker y dominio)."""

    def __init__(self, scraper: ScraperPort) -> None:
        self._scraper = scraper

    async def consultar(self, url: str, plataforma: str) -> str:
        result = await self._scraper.scrape(plataforma, url)
        return result.markdown if result.ok else ""

    async def consultar_estado_anuncio(
        self,
        *,
        oferta_id: UUID,
        plataforma: str,
        referencia_busqueda: str,
    ) -> ObservacionTrackerOferta:
        result = await self._scraper.scrape(plataforma, referencia_busqueda)
        if not result.ok:
            return ObservacionTrackerOferta(
                oferta_id=oferta_id,
                plataforma=plataforma,
                url_fuente=result.url,
                markdown="",
                disponible=False,
                error=result.error.message if result.error else result.status.value,
            )
        return ObservacionTrackerOferta(
            oferta_id=oferta_id,
            plataforma=plataforma,
            url_fuente=result.url,
            markdown=result.markdown,
            disponible=True,
            error=None,
        )


def detectar_senales_mercado(markdown: str) -> tuple[SenalMercadoOferta, list[SenalMercadoOferta]]:
    """Extrae señales de mercado desde markdown; si hay conflicto o vacío devuelve `ambigua`."""
    texto = (markdown or "").strip()
    if not texto:
        return SenalMercadoOferta.AMBIGUA, []

    encontradas: list[SenalMercadoOferta] = []
    for senal, pattern in _SIGNAL_PATTERNS:
        if pattern.search(texto):
            encontradas.append(senal)

    if not encontradas:
        return SenalMercadoOferta.AMBIGUA, []

    grupo_cierre = {SenalMercadoOferta.SOLD, SenalMercadoOferta.WITHDRAWN}
    grupo_activo = {
        SenalMercadoOferta.ACTIVE,
        SenalMercadoOferta.PENDING,
        SenalMercadoOferta.UNDER_CONTRACT,
        SenalMercadoOferta.CONTINGENT,
    }
    if (set(encontradas) & grupo_cierre) and (set(encontradas) & grupo_activo):
        return SenalMercadoOferta.AMBIGUA, encontradas

    if len(set(encontradas)) == 1:
        return encontradas[0], encontradas

    if set(encontradas) <= {
        SenalMercadoOferta.PENDING,
        SenalMercadoOferta.UNDER_CONTRACT,
        SenalMercadoOferta.CONTINGENT,
    }:
        return SenalMercadoOferta.UNDER_CONTRACT, encontradas

    return SenalMercadoOferta.AMBIGUA, encontradas


def formatear_aviso_tracker_oferta(
    *,
    oferta: OfertaResponse,
    senal: SenalMercadoOferta,
    estado_sugerido: OfertaEstado | None,
    requiere_revision_humana: bool,
    motivo: str,
    url_fuente: str | None = None,
) -> str:
    """Plantilla de aviso Telegram puramente informativa (sin acciones comerciales automáticas)."""
    revision_txt = (
        "SÍ — Requiere confirmación humana antes de cambiar estado."
        if requiere_revision_humana
        else "No (seguimiento informativo)."
    )
    sugerido_txt = estado_sugerido.value if estado_sugerido is not None else "sin cambio automático"
    fuente_txt = url_fuente or "fuente interna"
    return (
        "SEGUIMIENTO POST-OFERTA OPENCLAW (INFORMATIVO)\n"
        f"Oferta: {oferta.id} | Propiedad: {oferta.propiedad_id}\n"
        f"Estado actual: {oferta.estado.value} | Importe: ${oferta.importe_usd:,.0f}\n"
        f"Señal detectada: {senal.value}\n"
        f"Transición sugerida: {sugerido_txt}\n"
        f"Revisión humana: {revision_txt}\n"
        f"Motivo: {motivo}\n"
        f"Fuente: {fuente_txt}\n"
        "Nota: No se ha enviado ninguna comunicación ni acción comercial vinculante."
    )


def evaluar_observacion_oferta(
    oferta: OfertaResponse,
    observacion: ObservacionTrackerOferta,
    *,
    actor: str = "tracker_fase95",
) -> EvaluacionSeguimientoOferta:
    """Mapea una observación externa a una transición válida o revisión humana obligatoria."""
    now = observacion.observado_en
    actual = oferta.estado

    if not observacion.disponible:
        motivo = f"Fuente no disponible ({observacion.error or 'sin detalle'}); estado conservado."
        evento = OfertaEventoResponse(
            tipo="tracker_fuente_no_disponible",
            estado_anterior=actual,
            estado_nuevo=actual,
            actor=actor,
            ocurrido_en=now,
        )
        return EvaluacionSeguimientoOferta(
            oferta_id=oferta.id,
            estado_actual=actual,
            senal_detectada=SenalMercadoOferta.FUENTE_NO_DISPONIBLE,
            estado_sugerido=None,
            transicion_aplicable=False,
            requiere_revision_humana=True,
            motivo=motivo,
            evento=evento,
            aviso_telegram=formatear_aviso_tracker_oferta(
                oferta=oferta,
                senal=SenalMercadoOferta.FUENTE_NO_DISPONIBLE,
                estado_sugerido=None,
                requiere_revision_humana=True,
                motivo=motivo,
                url_fuente=observacion.url_fuente,
            ),
            accion_comercial_automatica=False,
        )

    senal, coincidentes = detectar_senales_mercado(observacion.markdown)

    if actual not in _ESTADOS_ACTIVOS:
        motivo = f"Oferta en estado terminal '{actual.value}'; no se aplican transiciones."
        evento = OfertaEventoResponse(
            tipo="tracker_oferta_terminal_omitida",
            estado_anterior=actual,
            estado_nuevo=actual,
            actor=actor,
            ocurrido_en=now,
        )
        return EvaluacionSeguimientoOferta(
            oferta_id=oferta.id,
            estado_actual=actual,
            senal_detectada=senal,
            senales_coincidentes=coincidentes,
            estado_sugerido=None,
            transicion_aplicable=False,
            requiere_revision_humana=False,
            motivo=motivo,
            evento=evento,
            aviso_telegram=None,
            accion_comercial_automatica=False,
        )

    estado_sugerido: OfertaEstado | None = None
    transicion_aplicable = False
    requiere_revision = False
    motivo = ""
    tipo_evento = "tracker_comprobacion"

    if senal == SenalMercadoOferta.AMBIGUA:
        requiere_revision = True
        tipo_evento = "tracker_senal_ambigua"
        motivo = (
            "Señal ambigua o contradictoria en el portal; no se cierra ni cancela "
            "la oferta sin validación humana."
        )
    elif senal == SenalMercadoOferta.ACTIVE:
        tipo_evento = "tracker_sin_cambios"
        motivo = "Anuncio sigue activo en el portal; sin cambio de estado."
    elif senal in (
        SenalMercadoOferta.PENDING,
        SenalMercadoOferta.UNDER_CONTRACT,
        SenalMercadoOferta.CONTINGENT,
    ):
        if actual == OfertaEstado.ACEPTADA:
            tipo_evento = "tracker_en_curso_confirmado"
            motivo = (
                f"Portal muestra '{senal.value}', consistente con oferta aceptada; "
                "cierre sujeto a confirmación humana."
            )
        else:
            requiere_revision = True
            tipo_evento = "tracker_revision_contrato_externo"
            motivo = (
                f"Portal cambió a '{senal.value}' mientras nuestra oferta está en "
                f"'{actual.value}'; verificar si corresponde a nuestra oferta o a un tercero."
            )
    elif senal == SenalMercadoOferta.SOLD:
        requiere_revision = True
        if actual == OfertaEstado.ACEPTADA:
            estado_sugerido = OfertaEstado.CERRADA
            transicion_aplicable = estado_sugerido in TRANSICIONES_OFERTA[actual]
            tipo_evento = "tracker_cierre_detectado"
            motivo = (
                "Portal indica propiedad vendida ('sold') sobre oferta aceptada; "
                "se sugiere transición a 'cerrada' con confirmación humana."
            )
        else:
            estado_sugerido = OfertaEstado.RECHAZADA
            transicion_aplicable = estado_sugerido in TRANSICIONES_OFERTA[actual]
            tipo_evento = "tracker_vendida_tercero_sospecha"
            motivo = (
                f"Portal indica 'sold' con oferta en '{actual.value}'; "
                "revisión humana requerida antes de marcar rechazada."
            )
    elif senal == SenalMercadoOferta.WITHDRAWN:
        requiere_revision = True
        estado_sugerido = OfertaEstado.CANCELADA
        transicion_aplicable = estado_sugerido in TRANSICIONES_OFERTA[actual]
        tipo_evento = "tracker_retirada_detectada"
        motivo = (
            "Anuncio retirado o fuera de mercado ('withdrawn'); requiere revisión humana "
            "antes de cancelar la oferta."
        )

    evento = OfertaEventoResponse(
        tipo=tipo_evento,
        estado_anterior=actual,
        estado_nuevo=actual,
        actor=actor,
        ocurrido_en=now,
    )
    aviso = (
        formatear_aviso_tracker_oferta(
            oferta=oferta,
            senal=senal,
            estado_sugerido=estado_sugerido,
            requiere_revision_humana=requiere_revision,
            motivo=motivo,
            url_fuente=observacion.url_fuente,
        )
        if requiere_revision or estado_sugerido is not None
        else None
    )

    return EvaluacionSeguimientoOferta(
        oferta_id=oferta.id,
        estado_actual=actual,
        senal_detectada=senal,
        senales_coincidentes=coincidentes,
        estado_sugerido=estado_sugerido,
        transicion_aplicable=transicion_aplicable,
        requiere_revision_humana=requiere_revision,
        motivo=motivo,
        evento=evento,
        aviso_telegram=aviso,
        accion_comercial_automatica=False,
    )


async def rastrear_oferta_con_lock(
    *,
    redis_client: Any,
    tracker: FirecrawlOfertaTrackerAdapter,
    oferta: OfertaResponse,
    plataforma: str,
    referencia_busqueda: str,
    periodo_segundos: int = 3600,
    lock_ttl_segundos: int = 120,
) -> EvaluacionSeguimientoOferta | None:
    """Coordina lock exclusivo y ventana mínima entre comprobaciones por oferta en Redis."""
    lock_key = f"openclaw:oferta:lock:{oferta.id}"
    cooldown_key = f"openclaw:oferta:checked:{oferta.id}"

    acquired = await redis_client.set(lock_key, "1", nx=True, ex=lock_ttl_segundos)
    if not acquired:
        return None

    try:
        already_checked = await redis_client.exists(cooldown_key)
        if already_checked:
            return None

        obs = await tracker.consultar_estado_anuncio(
            oferta_id=oferta.id,
            plataforma=plataforma,
            referencia_busqueda=referencia_busqueda,
        )
        evaluacion = evaluar_observacion_oferta(oferta, obs)
        await redis_client.set(cooldown_key, "1", ex=periodo_segundos)
        return evaluacion
    finally:
        await redis_client.delete(lock_key)
