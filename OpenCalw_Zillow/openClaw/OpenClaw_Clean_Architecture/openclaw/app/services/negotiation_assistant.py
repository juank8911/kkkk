"""Negociador asistido con NLP supervisado — Fase 101/104 (Dev 2).

Reglas obligatorias (PLAN_IMPLEMENTACION_FASES_FALTANTES_2DEVS.md):
- La IA únicamente propone borradores; una persona siempre decide, aprueba o rechaza.
- No infiere ni utiliza atributos protegidos (Fair Housing Act) ni proxies discriminatorios.
- Cada sugerencia incluye explicación explícita y las variables económicas usadas.
- Umbral de confianza con fallback manual cuando el mensaje es ambiguo o incompleto.
- Integración con LOI exclusivamente como borrador versionado (`LoiDraftService`);
  no existe ninguna función de envío automático por correo o SMS.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Final
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.services.loi_generator import LoiDocumentVersion, LoiDraftInput, LoiDraftService

_MONEY_RE: Final[re.Pattern[str]] = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)\s*([Kk])?")
_DAYS_RE: Final[re.Pattern[str]] = re.compile(r"(?i)\b(\d{1,3})\s*(?:days?|d[ií]as)\b")
_CREDIT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:credit|cr[eé]dito|concession|closing\s+cost)[^\n$]{0,30}\$\s*([\d,]+(?:\.\d+)?)"
)

# Patrones de atributos protegidos (Fair Housing Act / ECOA) que JAMÁS se usan para decidir
_PROTECTED_ATTR_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "race_ethnicity",
        re.compile(r"(?i)\b(?:race|ethnic|hispanic|latino|caucasian|african[- ]american|asian)\b"),
    ),
    (
        "religion",
        re.compile(r"(?i)\b(?:religion|church|mosque|synagogue|christian|muslim|jewish)\b"),
    ),
    (
        "familial_status",
        re.compile(r"(?i)\b(?:pregnant|children|kids|family\s+status|married|single\s+mother)\b"),
    ),
    ("disability", re.compile(r"(?i)\b(?:disability|disabled|wheelchair|handicap)\b")),
    (
        "national_origin",
        re.compile(r"(?i)\b(?:immigrant|citizenship|national\s+origin|foreigner)\b"),
    ),
    (
        "sex_gender",
        re.compile(r"(?i)\b(?:gender|female\s+seller|male\s+seller|sexual\s+orientation)\b"),
    ),
)


class PropuestaNegociacionEstado(StrEnum):
    """Estados auditables de revisión humana para una contraoferta asistida."""

    BORRADOR = "borrador"
    REVISAR = "revisar"
    APROBADO = "aprobado"
    RECHAZADO = "rechazado"


class ExtraccionMensajeVendedor(BaseModel):
    """Datos puramente comerciales extraídos de un mensaje del vendedor/agente."""

    model_config = ConfigDict(extra="forbid")

    precio_solicitado_usd: float | None = Field(default=None, gt=0)
    credito_cierre_usd: float = Field(default=0.0, ge=0)
    dias_cierre_solicitados: int | None = Field(default=None, ge=1, le=365)
    menciona_as_is: bool = False
    menciona_inspeccion: bool = False
    confianza_extraccion: float = Field(..., ge=0.0, le=1.0)
    atributos_protegidos_ignorados: list[str] = Field(default_factory=list)


class AlternativaContraoferta(BaseModel):
    """Opción de contraoferta explicable generada por reglas económicas transparentes."""

    model_config = ConfigDict(extra="forbid")

    perfil: str
    importe_sugerido_usd: float = Field(..., gt=0)
    cap_rate_proyectado_pct: float
    dias_cierre_sugeridos: int = Field(..., ge=7, le=120)
    variables_usadas: dict[str, float]
    explicacion: str


class PropuestaNegociacion(BaseModel):
    """Borrador supervisado de contraoferta listo para revisión y decisión humana."""

    model_config = ConfigDict(extra="forbid")

    propuesta_id: UUID = Field(default_factory=uuid4)
    propiedad_id: UUID
    estado: PropuestaNegociacionEstado = PropuestaNegociacionEstado.REVISAR
    fallback_manual: bool = False
    motivo_fallback: str | None = None
    extraccion: ExtraccionMensajeVendedor
    alternativas: list[AlternativaContraoferta] = Field(default_factory=list)
    requiere_aprobacion_humana: bool = True
    envio_automatico_habilitado: bool = False
    revisada_por: str | None = None
    importe_corregido_humano_usd: float | None = None
    notas_operador: str | None = None
    loi_borrador: LoiDocumentVersion | None = None
    creada_en: datetime = Field(default_factory=lambda: datetime.now(UTC))


def extraer_datos_comerciales_mensaje(mensaje: str) -> ExtraccionMensajeVendedor:
    """Extrae solo señales económicas/contractuales e ignora cualquier atributo protegido."""
    texto = (mensaje or "").strip()
    protegidos: list[str] = []
    texto_sanitizado = texto
    for categoria, pattern in _PROTECTED_ATTR_PATTERNS:
        if pattern.search(texto_sanitizado):
            protegidos.append(categoria)
            texto_sanitizado = pattern.sub("[IGNORADO_FAIR_HOUSING]", texto_sanitizado)

    credito = 0.0
    credit_match = _CREDIT_RE.search(texto_sanitizado)
    if credit_match:
        credito = float(credit_match.group(1).replace(",", ""))

    precios_candidatos: list[float] = []
    for match in _MONEY_RE.finditer(texto_sanitizado):
        raw = float(match.group(1).replace(",", ""))
        if (match.group(2) or "").lower() == "k":
            raw *= 1_000.0
        if raw >= 10_000.0 and raw != credito:
            precios_candidatos.append(round(raw, 2))

    precio_solicitado = precios_candidatos[0] if precios_candidatos else None

    days_match = _DAYS_RE.search(texto_sanitizado)
    dias_cierre = int(days_match.group(1)) if days_match else None

    lower = texto_sanitizado.lower()
    menciona_as_is = "as-is" in lower or "as is" in lower or "como esta" in lower
    menciona_inspeccion = "inspection" in lower or "inspeccion" in lower or "inspección" in lower

    if precio_solicitado is not None and dias_cierre is not None:
        confianza = 0.9
    elif precio_solicitado is not None:
        confianza = 0.78
    elif dias_cierre is not None or menciona_as_is or menciona_inspeccion:
        confianza = 0.45
    else:
        confianza = 0.15

    return ExtraccionMensajeVendedor(
        precio_solicitado_usd=precio_solicitado,
        credito_cierre_usd=credito,
        dias_cierre_solicitados=dias_cierre,
        menciona_as_is=menciona_as_is,
        menciona_inspeccion=menciona_inspeccion,
        confianza_extraccion=confianza,
        atributos_protegidos_ignorados=protegidos,
    )


class SupervisedNegotiationService:
    """Genera alternativas explicables de contraoferta y borradores LOI bajo control humano."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.65,
        loi_service: LoiDraftService | None = None,
    ) -> None:
        self._min_confidence = min_confidence
        self._loi = loi_service if loi_service is not None else LoiDraftService()
        self._propuestas: dict[UUID, PropuestaNegociacion] = {}

    def analizar_y_proponer(
        self,
        *,
        propiedad_id: UUID,
        mensaje_vendedor: str,
        oferta_actual_usd: float,
        noi_anual_usd: float,
        capex_estimado_usd: float,
        min_cap_rate_pct: float = 8.0,
    ) -> PropuestaNegociacion:
        """Analiza el mensaje, genera alternativas explicables o degrada a fallback manual."""
        extraccion = extraer_datos_comerciales_mensaje(mensaje_vendedor)

        if (
            extraccion.confianza_extraccion < self._min_confidence
            or extraccion.precio_solicitado_usd is None
        ):
            propuesta = PropuestaNegociacion(
                propiedad_id=propiedad_id,
                estado=PropuestaNegociacionEstado.REVISAR,
                fallback_manual=True,
                motivo_fallback=(
                    f"Confianza insuficiente ({extraccion.confianza_extraccion:.2f} < "
                    f"{self._min_confidence:.2f}) o sin cifra de precio explícita; "
                    "se requiere revisión manual del operador."
                ),
                extraccion=extraccion,
                alternativas=[],
            )
            self._propuestas[propuesta.propuesta_id] = propuesta
            return propuesta

        # Precio techo económico para respetar min_cap_rate_pct considerando CAPEX
        precio_max_rentable = max(
            1.0, round((noi_anual_usd / (min_cap_rate_pct / 100.0)) - capex_estimado_usd, 2)
        )
        pedido = extraccion.precio_solicitado_usd
        dias_base = extraccion.dias_cierre_solicitados or 30

        candidatos = [
            ("conservadora", round(min(oferta_actual_usd * 1.02, precio_max_rentable), 2)),
            (
                "equilibrada",
                round(min((oferta_actual_usd + pedido) / 2.0, precio_max_rentable), 2),
            ),
            ("limite_cap_rate", round(min(pedido, precio_max_rentable), 2)),
        ]

        alternativas: list[AlternativaContraoferta] = []
        vistos: set[float] = set()
        for perfil, importe in candidatos:
            if importe <= 0 or importe in vistos:
                continue
            vistos.add(importe)
            cap_proyectado = round((noi_anual_usd / (importe + capex_estimado_usd)) * 100.0, 2)
            alternativas.append(
                AlternativaContraoferta(
                    perfil=perfil,
                    importe_sugerido_usd=importe,
                    cap_rate_proyectado_pct=cap_proyectado,
                    dias_cierre_sugeridos=max(14, min(60, dias_base)),
                    variables_usadas={
                        "oferta_actual_usd": oferta_actual_usd,
                        "precio_solicitado_usd": pedido,
                        "noi_anual_usd": noi_anual_usd,
                        "capex_estimado_usd": capex_estimado_usd,
                        "precio_max_rentable_usd": precio_max_rentable,
                        "min_cap_rate_pct": min_cap_rate_pct,
                    },
                    explicacion=(
                        f"Opción {perfil}: ${importe:,.2f} mantiene un cap rate proyectado "
                        f"de {cap_proyectado:.2f}% (umbral mínimo {min_cap_rate_pct:.2f}%) "
                        f"considerando CAPEX estimado de ${capex_estimado_usd:,.0f}."
                    ),
                )
            )

        propuesta = PropuestaNegociacion(
            propiedad_id=propiedad_id,
            estado=PropuestaNegociacionEstado.REVISAR,
            fallback_manual=False,
            extraccion=extraccion,
            alternativas=alternativas,
        )
        self._propuestas[propuesta.propuesta_id] = propuesta
        return propuesta

    def resolver_revision_humana(
        self,
        propuesta_id: UUID,
        *,
        decision: PropuestaNegociacionEstado,
        operador: str,
        importe_elegido_usd: float | None = None,
        notas: str | None = None,
        generar_borrador_loi: dict[str, str] | None = None,
    ) -> PropuestaNegociacion:
        """Registra la decisión del operador humano (aprobar/corregir/rechazar) sin envío automático."""
        if decision not in (
            PropuestaNegociacionEstado.APROBADO,
            PropuestaNegociacionEstado.RECHAZADO,
        ):
            raise ValueError("La decisión humana debe ser 'aprobado' o 'rechazado'.")
        if not operador.strip():
            raise ValueError("Se requiere identificar al operador humano.")

        actual = self._propuestas[propuesta_id]
        loi_doc: LoiDocumentVersion | None = None
        if decision == PropuestaNegociacionEstado.APROBADO and generar_borrador_loi is not None:
            if importe_elegido_usd is None or importe_elegido_usd <= 0:
                raise ValueError("Se requiere importe_elegido_usd > 0 para emitir borrador LOI.")
            loi_input = LoiDraftInput(
                propiedad_id=actual.propiedad_id,
                estado_us=generar_borrador_loi["estado_us"],
                ubicacion=generar_borrador_loi["ubicacion"],
                comprador=generar_borrador_loi["comprador"],
                vendedor=generar_borrador_loi["vendedor"],
                importe_usd=importe_elegido_usd,
                fecha_vencimiento=date.today() + timedelta(days=7),
                creado_por=operador.strip(),
            )
            loi_doc, _ = self._loi.generar_borrador(loi_input)

        actualizada = actual.model_copy(
            update={
                "estado": decision,
                "revisada_por": operador.strip(),
                "importe_corregido_humano_usd": importe_elegido_usd,
                "notas_operador": notas,
                "loi_borrador": loi_doc,
            }
        )
        self._propuestas[propuesta_id] = actualizada
        return actualizada
