"""Contratos Pydantic de API compartidos entre Dev A y Dev B (S1-02 / ADR-01).

Define BarridoRequest, JobResponse, JobStatus y ErrorResponse para su publicación
en el esquema OpenAPI y validación uniforme de payloads.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.platforms import VALID_PLATFORM_KEYS, validate_platforms


class JobStatus(str, Enum):
    """Estados válidos del ciclo de vida de un job de barrido."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class BarridoRequest(BaseModel):
    """Payload de entrada para iniciar un barrido inmobiliario (POST /v1/openclaw/barridos)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Consulta geográfica o criterio de búsqueda (ej. 'Miami, FL' o '33101').",
        examples=["Miami, FL"],
    )
    plataformas: list[str] = Field(
        ...,
        min_length=1,
        max_length=len(VALID_PLATFORM_KEYS),
        description=(
            "Lista de claves del catálogo cerrado de plataformas: "
            + ", ".join(VALID_PLATFORM_KEYS)
        ),
        examples=[["zillow", "realtor", "redfin"]],
    )
    max_resultados_por_plataforma: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Límite máximo de propiedades a extraer por plataforma.",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Metadatos opcionales de trazabilidad enviados por el cliente.",
    )

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("El campo 'query' debe tener al menos 2 caracteres útiles.")
        if any(ord(ch) < 32 for ch in cleaned):
            raise ValueError("El campo 'query' contiene caracteres de control no permitidos.")
        return cleaned

    @field_validator("plataformas")
    @classmethod
    def _validate_plataformas(cls, value: list[str]) -> list[str]:
        return validate_platforms(value)


class JobResponse(BaseModel):
    """Respuesta estándar al encolar o consultar el estado de un job de barrido."""

    model_config = ConfigDict(extra="forbid")

    job_id: UUID = Field(..., description="Identificador único del job de barrido.")
    status: JobStatus = Field(..., description="Estado actual del job.")
    query: str = Field(..., description="Consulta solicitada para el barrido.")
    plataformas: list[str] = Field(..., description="Claves de plataformas validadas.")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Marca de tiempo UTC de creación del job.",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Marca de tiempo UTC de la última transición de estado.",
    )
    trace_id: str | None = Field(
        default=None,
        description="Identificador de trazabilidad distribuida asociado a la solicitud.",
    )
    progress: dict[str, str] = Field(
        default_factory=dict,
        description="Estado por plataforma dentro del job.",
    )
    error: str | None = Field(
        default=None,
        description="Resumen de error cuando el job finaliza en estado failed o partial.",
    )


class ErrorDetail(BaseModel):
    """Detalle estructurado de un error individual (campo o causa específica)."""

    model_config = ConfigDict(extra="forbid")

    field: str | None = Field(default=None, description="Campo afectado, si aplica.")
    reason: str = Field(..., description="Descripción del motivo del error.")
    valid_values: list[str] | None = Field(
        default=None,
        description="Lista de valores válidos cuando el error proviene de un catálogo cerrado.",
    )


class ErrorResponse(BaseModel):
    """Envelope uniforme de errores HTTP para toda la API de OpenClaw."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        ...,
        description="Código máquina del error (ej. validation_error, invalid_platform, not_found).",
        examples=["invalid_platform"],
    )
    message: str = Field(..., description="Mensaje legible explicando el error.")
    details: list[ErrorDetail] | dict[str, Any] | None = Field(
        default=None,
        description="Detalles adicionales estructurados del error.",
    )
    trace_id: str | None = Field(
        default=None,
        description="Identificador de trazabilidad para correlación en logs.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Marca de tiempo UTC en la que ocurrió el error.",
    )
