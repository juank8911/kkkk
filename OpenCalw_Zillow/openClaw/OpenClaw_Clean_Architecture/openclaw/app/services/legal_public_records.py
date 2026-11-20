"""Auditoría legal ampliada sobre fuentes públicas permitidas — Fase 97 (Dev 2).

Extiende `app.orchestrator.nodes.legal_audit` integrando `PublicRecordPort` con:
- Allowlist estricta de dominios públicos oficiales de condados/tax assessors.
- Hash SHA-256, URL y timestamp de cada documento consultado.
- Separación explícita de estados: `sin_evidencia`, `riesgo_detectado`,
  `fuente_no_disponible` y `requiere_abogado`.
- Regla de oro: la ausencia de datos nunca se interpreta como ausencia de riesgo.
- Disclaimer obligatorio en todo resultado.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from app.orchestrator.nodes.legal_audit import LegalSignal, auditar_riesgo_legal

LEGAL_DISCLAIMER: Final[str] = (
    "Análisis legal asistido de carácter informativo; no constituye opinión jurídica "
    "ni sustituye un estudio ni seguro de título (title insurance)."
)

DEFAULT_ALLOWED_PUBLIC_RECORD_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        "myplace.cuyahogacounty.gov",
        "treasurer.cuyahogacounty.gov",
        "propertyappraiser.miamidade.gov",
        "miamidade.county-taxes.com",
        "treasurer.waynecounty.com",
        "ocpafl.org",
        "hcpafl.org",
    }
)


class EstadoAuditoriaPublica(StrEnum):
    """Estados exclusivos exigidos por la Fase 97."""

    SIN_EVIDENCIA = "sin_evidencia"
    RIESGO_DETECTADO = "riesgo_detectado"
    FUENTE_NO_DISPONIBLE = "fuente_no_disponible"
    REQUIERE_ABOGADO = "requiere_abogado"


class PublicRecordSourceNotAllowedError(ValueError):
    """Se lanza cuando se intenta consultar un dominio fuera de la allowlist permitida."""


class PublicRecordDocument(BaseModel):
    """Evidencia cruda devuelta por un adaptador de registros públicos permitidos."""

    model_config = ConfigDict(extra="forbid")

    condado: str
    url_fuente: str
    contenido_texto: str = ""
    disponible: bool = True
    error: str | None = None
    consultado_en: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def content_sha256(self) -> str:
        return hashlib.sha256(self.contenido_texto.encode("utf-8")).hexdigest()


class EvidenciaLegalPublica(BaseModel):
    """Señal legal trazable a documento público, fecha y hash de contenido."""

    model_config = ConfigDict(extra="forbid")

    senal: LegalSignal
    url_fuente: str
    fecha_consulta: datetime
    documento_sha256: str


class AuditoriaLegalAmpliadaResult(BaseModel):
    """Resultado consolidado de auditoría legal ampliada (Fase 97)."""

    model_config = ConfigDict(extra="forbid")

    estado: EstadoAuditoriaPublica
    condado: str
    requiere_revision_humana: bool
    evidencias: list[EvidenciaLegalPublica] = Field(default_factory=list)
    fuentes_consultadas: list[str] = Field(default_factory=list)
    hashes_documentos: dict[str, str] = Field(default_factory=dict)
    consultado_en: datetime = Field(default_factory=lambda: datetime.now(UTC))
    resumen: str
    disclaimer: str = LEGAL_DISCLAIMER


@runtime_checkable
class PublicRecordPort(Protocol):
    """Puerto de consulta a registros públicos oficiales con allowlist."""

    async def consultar_registros(
        self,
        *,
        condado: str,
        ubicacion: str,
        parcel_id: str | None = None,
    ) -> list[PublicRecordDocument]: ...


def validar_dominio_registro_publico(
    url: str,
    *,
    allowed_domains: frozenset[str] = DEFAULT_ALLOWED_PUBLIC_RECORD_DOMAINS,
) -> str:
    """Verifica que la URL pertenezca a un dominio oficial de la allowlist (HTTPS)."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host:
        raise PublicRecordSourceNotAllowedError(
            f"URL de registro público inválida o no HTTPS: {url!r}"
        )
    if not any(host == d or host.endswith(f".{d}") for d in allowed_domains):
        raise PublicRecordSourceNotAllowedError(
            f"Dominio '{host}' fuera de la allowlist de registros públicos permitidos."
        )
    return host


class CountyTaxAssessorAdapter:
    """Adaptador offline/controlado por condado respetando estrictamente la allowlist."""

    def __init__(
        self,
        records_by_county: dict[str, list[PublicRecordDocument]],
        *,
        allowed_domains: frozenset[str] = DEFAULT_ALLOWED_PUBLIC_RECORD_DOMAINS,
    ) -> None:
        self._records = {k.strip().lower(): list(v) for k, v in records_by_county.items()}
        self._allowed = allowed_domains

    async def consultar_registros(
        self,
        *,
        condado: str,
        ubicacion: str,
        parcel_id: str | None = None,
    ) -> list[PublicRecordDocument]:
        docs = self._records.get(condado.strip().lower(), [])
        validated: list[PublicRecordDocument] = []
        for doc in docs:
            validar_dominio_registro_publico(doc.url_fuente, allowed_domains=self._allowed)
            validated.append(doc)
        return validated


class ExtendedLegalAuditService:
    """Combina texto del anuncio y registros públicos permitidos en 4 estados claros."""

    def __init__(
        self,
        *,
        public_record_port: PublicRecordPort,
        allowed_domains: frozenset[str] = DEFAULT_ALLOWED_PUBLIC_RECORD_DOMAINS,
    ) -> None:
        self._port = public_record_port
        self._allowed = allowed_domains

    async def auditar_propiedad(
        self,
        *,
        condado: str,
        ubicacion: str,
        descripcion_anuncio: str = "",
        parcel_id: str | None = None,
    ) -> AuditoriaLegalAmpliadaResult:
        docs = await self._port.consultar_registros(
            condado=condado,
            ubicacion=ubicacion,
            parcel_id=parcel_id,
        )

        fuentes: list[str] = []
        hashes: dict[str, str] = {}
        evidencias: list[EvidenciaLegalPublica] = []
        alguna_fuente_caida = False

        for doc in docs:
            validar_dominio_registro_publico(doc.url_fuente, allowed_domains=self._allowed)
            fuentes.append(doc.url_fuente)
            if not doc.disponible:
                alguna_fuente_caida = True
                continue
            doc_hash = doc.content_sha256()
            hashes[doc.url_fuente] = doc_hash
            audit_doc = auditar_riesgo_legal(doc.contenido_texto)
            for senal in audit_doc.senales_detectadas:
                evidencias.append(
                    EvidenciaLegalPublica(
                        senal=senal,
                        url_fuente=doc.url_fuente,
                        fecha_consulta=doc.consultado_en,
                        documento_sha256=doc_hash,
                    )
                )

        if descripcion_anuncio.strip():
            anuncio_hash = hashlib.sha256(descripcion_anuncio.encode("utf-8")).hexdigest()
            audit_anuncio = auditar_riesgo_legal(descripcion_anuncio)
            for senal in audit_anuncio.senales_detectadas:
                evidencias.append(
                    EvidenciaLegalPublica(
                        senal=senal,
                        url_fuente="anuncio://descripcion_cruda",
                        fecha_consulta=datetime.now(UTC),
                        documento_sha256=anuncio_hash,
                    )
                )

        if evidencias:
            categorias_altas = {
                "tax_lien",
                "hoa_lien",
                "mechanic_lien",
                "general_lien",
                "back_taxes",
                "foreclosure_lis_pendens",
            }
            tiene_riesgo_abogado = (
                any(ev.senal.categoria in categorias_altas for ev in evidencias)
                or len(evidencias) >= 2
            )
            estado = (
                EstadoAuditoriaPublica.REQUIERE_ABOGADO
                if tiene_riesgo_abogado
                else EstadoAuditoriaPublica.RIESGO_DETECTADO
            )
            cats = sorted({ev.senal.categoria for ev in evidencias})
            return AuditoriaLegalAmpliadaResult(
                estado=estado,
                condado=condado,
                requiere_revision_humana=True,
                evidencias=evidencias,
                fuentes_consultadas=fuentes,
                hashes_documentos=hashes,
                resumen=(
                    f"Señales legales detectadas ({', '.join(cats)}); "
                    "revisión humana/legal obligatoria antes de ofertar."
                ),
            )

        if not docs or alguna_fuente_caida:
            return AuditoriaLegalAmpliadaResult(
                estado=EstadoAuditoriaPublica.FUENTE_NO_DISPONIBLE,
                condado=condado,
                requiere_revision_humana=True,
                evidencias=[],
                fuentes_consultadas=fuentes,
                hashes_documentos=hashes,
                resumen=(
                    "Fuente pública no disponible o sin cobertura para el condado; "
                    "la ausencia de datos NO implica ausencia de riesgo."
                ),
            )

        return AuditoriaLegalAmpliadaResult(
            estado=EstadoAuditoriaPublica.SIN_EVIDENCIA,
            condado=condado,
            requiere_revision_humana=False,
            evidencias=[],
            fuentes_consultadas=fuentes,
            hashes_documentos=hashes,
            resumen=(
                "Sin evidencia de gravámenes o morosidad en los documentos consultados; "
                "la ausencia de evidencia no garantiza título limpio sin title search formal."
            ),
        )
