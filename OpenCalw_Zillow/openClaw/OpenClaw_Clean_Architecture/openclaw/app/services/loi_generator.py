"""Generador determinista de borradores LOI no vinculantes en PDF — Fase 98/102 (Dev 2).

Reglas obligatorias (PLAN_IMPLEMENTACION_FASES_FALTANTES_2DEVS.md):
- Plantillas versionadas por estado de EE. UU. sujetas a marca de revisión legal previa.
- Todo documento contiene cláusula `NON-BINDING LETTER OF INTENT` y disclaimer explícito.
- Generación PDF 1.4 determinista (idéntica entrada => idénticos bytes y SHA-256).
- Toda modificación crea una nueva versión inmutable (`version + 1`).
- No puede pasar a `aprobado_para_envio` sin confirmación/firma humana explícita.
- No existe ninguna función ni endpoint de envío automático.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Final
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

NON_BINDING_DISCLAIMER: Final[str] = (
    "NON-BINDING LETTER OF INTENT: This draft is for discussion purposes only, "
    "creates no legal or binding obligation on any party, and requires review and "
    "execution by licensed counsel and an authorized human signatory before delivery."
)


class LoiEstado(StrEnum):
    """Estados del ciclo de vida de un borrador LOI."""

    BORRADOR = "borrador"
    EN_REVISION_LEGAL = "en_revision_legal"
    APROBADO_PARA_ENVIO = "aprobado_para_envio"
    RECHAZADO = "rechazado"


@dataclass(frozen=True, slots=True)
class StateLoiTemplate:
    """Plantilla versionada de LOI específica por estado de EE. UU."""

    state_code: str
    template_version: str
    legal_reviewed: bool
    governing_law_clause: str
    required_contingencies: tuple[str, ...]


STATE_LOI_TEMPLATES: Final[dict[str, StateLoiTemplate]] = {
    "OH": StateLoiTemplate(
        state_code="OH",
        template_version="OH-LOI-v1.0",
        legal_reviewed=True,
        governing_law_clause="Subject to Ohio Revised Code real property formal purchase contract.",
        required_contingencies=("Inspection (10 business days)", "Clear & Marketable Title"),
    ),
    "FL": StateLoiTemplate(
        state_code="FL",
        template_version="FL-LOI-v1.0",
        legal_reviewed=True,
        governing_law_clause="Subject to execution of FAR/BAR As-Is or Standard Florida contract.",
        required_contingencies=(
            "Inspection (15 days)",
            "Title & Municipal Lien Search",
            "HOA/Condo Disclosure Review",
        ),
    ),
    "MI": StateLoiTemplate(
        state_code="MI",
        template_version="MI-LOI-v1.0",
        legal_reviewed=True,
        governing_law_clause="Subject to formal Michigan Purchase Agreement execution.",
        required_contingencies=("Inspection (10 business days)", "Title Insurance Commitment"),
    ),
    "TX": StateLoiTemplate(
        state_code="TX",
        template_version="TX-LOI-v1.0",
        legal_reviewed=True,
        governing_law_clause="Subject to TREC promulgated formal contract execution.",
        required_contingencies=("Option Period Inspection", "Title Commitment & Survey Review"),
    ),
}


class LoiDraftInput(BaseModel):
    """Datos estructurados requeridos para generar un borrador LOI determinista."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    propiedad_id: UUID
    estado_us: str = Field(..., min_length=2, max_length=2)
    ubicacion: str = Field(..., min_length=5, max_length=300)
    comprador: str = Field(..., min_length=2, max_length=160)
    vendedor: str = Field(..., min_length=2, max_length=160)
    importe_usd: float = Field(..., gt=0, le=100_000_000)
    deposito_seriedad_usd: float = Field(default=1_000.0, ge=0)
    fecha_vencimiento: date
    contingencias_adicionales: list[str] = Field(default_factory=list)
    creado_por: str = Field(..., min_length=1, max_length=120)


class LoiDocumentVersion(BaseModel):
    """Versión inmutable y verificable de un borrador LOI con su PDF determinista."""

    model_config = ConfigDict(extra="forbid")

    document_id: UUID = Field(default_factory=uuid4)
    propiedad_id: UUID
    version: int = Field(..., ge=1)
    template_version: str
    estado_us: str
    estado: LoiEstado = LoiEstado.BORRADOR
    datos: LoiDraftInput
    texto_renderizado: str
    pdf_sha256: str
    pdf_size_bytes: int = Field(..., gt=0)
    non_binding: bool = True
    creado_por: str
    creado_en: datetime = Field(default_factory=lambda: datetime.now(UTC))
    revisado_por: str | None = None
    aprobado_por: str | None = None
    aprobado_en: datetime | None = None


def _escape_pdf_text(text: str) -> str:
    ascii_safe = text.encode("latin-1", "replace").decode("latin-1")
    return ascii_safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def renderizar_pdf_determinista(lines: list[str]) -> bytes:
    """Genera un documento PDF 1.4 válido y 100% determinista a partir de líneas de texto."""
    text_ops = ["BT", "/F1 10 Tf", "50 760 Td", "14 TL"]
    for idx, line in enumerate(lines):
        escaped = _escape_pdf_text(line)
        if idx == 0:
            text_ops.append(f"({escaped}) Tj")
        else:
            text_ops.append(f"T* ({escaped}) Tj")
    text_ops.append("ET")
    stream_bytes = ("\n".join(text_ops) + "\n").encode("latin-1")

    objects: list[bytes] = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
        ),
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        (
            f"5 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"endstream\nendobj\n"
        ),
    ]

    buf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for obj in objects:
        offsets.append(len(buf))
        buf.extend(obj)

    xref_pos = len(buf)
    buf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    buf.extend(b"0000000000 65535 f \n")
    for off in offsets:
        buf.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    buf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(buf)


class LoiDraftService:
    """Servicio de borradores LOI versionados con PDF determinista y aprobación humana."""

    def __init__(
        self,
        *,
        templates: dict[str, StateLoiTemplate] | None = None,
    ) -> None:
        self._templates = dict(templates) if templates is not None else dict(STATE_LOI_TEMPLATES)
        self._versions_by_property: dict[UUID, list[LoiDocumentVersion]] = {}
        self._pdf_store: dict[UUID, bytes] = {}

    def generar_borrador(self, datos: LoiDraftInput) -> tuple[LoiDocumentVersion, bytes]:
        """Crea una nueva versión inmutable de borrador LOI y su PDF reproducible."""
        state_key = datos.estado_us.upper()
        template = self._templates.get(state_key)
        if template is None:
            valid = ", ".join(sorted(self._templates))
            raise ValueError(
                f"No existe plantilla LOI aprobada para el estado '{state_key}'. "
                f"Estados disponibles: [{valid}]."
            )
        if not template.legal_reviewed:
            raise ValueError(
                f"La plantilla '{template.template_version}' no ha superado revisión legal."
            )

        existing = self._versions_by_property.get(datos.propiedad_id, [])
        next_version = len(existing) + 1
        contingencias = [*template.required_contingencies, *datos.contingencias_adicionales]

        lines = [
            f"NON-BINDING LETTER OF INTENT (Template {template.template_version} / Draft v{next_version})",
            f"Property: {datos.ubicacion} ({state_key})",
            f"Buyer: {datos.comprador} | Seller: {datos.vendedor}",
            f"Proposed Purchase Price: ${datos.importe_usd:,.2f} USD",
            f"Proposed Earnest Money Deposit: ${datos.deposito_seriedad_usd:,.2f} USD",
            f"Expiration Date: {datos.fecha_vencimiento.isoformat()}",
            f"Governing Clause: {template.governing_law_clause}",
            "Contingencies: " + "; ".join(contingencias),
            "Human / Counsel Review Status: PENDING SIGNATURE & APPROVAL",
            NON_BINDING_DISCLAIMER,
        ]
        texto = "\n".join(lines)
        pdf_bytes = renderizar_pdf_determinista(lines)
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

        doc = LoiDocumentVersion(
            propiedad_id=datos.propiedad_id,
            version=next_version,
            template_version=template.template_version,
            estado_us=state_key,
            estado=LoiEstado.BORRADOR,
            datos=datos,
            texto_renderizado=texto,
            pdf_sha256=pdf_hash,
            pdf_size_bytes=len(pdf_bytes),
            non_binding=True,
            creado_por=datos.creado_por,
        )
        self._versions_by_property.setdefault(datos.propiedad_id, []).append(doc)
        self._pdf_store[doc.document_id] = pdf_bytes
        return doc, pdf_bytes

    def aprobar_para_envio(
        self,
        *,
        propiedad_id: UUID,
        version: int,
        aprobado_por: str,
        confirmacion_humana: bool,
        revisado_por: str | None = None,
    ) -> LoiDocumentVersion:
        """Marca un borrador como aprobado solo con confirmación/firma humana explícita."""
        if not confirmacion_humana or not aprobado_por.strip():
            raise ValueError(
                "No se puede marcar 'aprobado_para_envio' sin confirmación humana explícita."
            )
        versions = self._versions_by_property.get(propiedad_id, [])
        for idx, doc in enumerate(versions):
            if doc.version == version:
                updated = doc.model_copy(
                    update={
                        "estado": LoiEstado.APROBADO_PARA_ENVIO,
                        "revisado_por": revisado_por or aprobado_por.strip(),
                        "aprobado_por": aprobado_por.strip(),
                        "aprobado_en": datetime.now(UTC),
                    }
                )
                versions[idx] = updated
                return updated
        raise KeyError(f"Versión LOI v{version} no encontrada para propiedad {propiedad_id}.")

    def listar_versiones(self, propiedad_id: UUID) -> list[LoiDocumentVersion]:
        return list(self._versions_by_property.get(propiedad_id, []))

    def obtener_pdf(self, document_id: UUID) -> bytes:
        return self._pdf_store[document_id]
