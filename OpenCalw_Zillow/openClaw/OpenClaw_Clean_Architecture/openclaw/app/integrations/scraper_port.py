"""Puerto explícito de extracción (ScraperPort) y errores tipados (S1-08).

Desacopla el dominio y el worker del proveedor concreto (Firecrawl), permitiendo
pruebas de contrato deterministas sin red mediante dobles y cassettes.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ScrapeStatus(StrEnum):
    """Estado estructurado del resultado de una operación de extracción."""

    OK = "ok"
    RATE_LIMITED = "rate_limited"
    BLOCKED = "blocked"
    TIMEOUT = "timeout"
    UPSTREAM_ERROR = "upstream_error"
    RETRIES_EXHAUSTED = "retries_exhausted"
    INVALID_PLATFORM = "invalid_platform"


class ScrapeErrorCode(StrEnum):
    """Códigos máquina de error tipados para fallos de extracción."""

    RATE_LIMITED = "rate_limited"
    REDIS_UNAVAILABLE = "redis_unavailable"
    BLOCKED = "blocked"
    TIMEOUT = "timeout"
    UPSTREAM_ERROR = "upstream_error"
    RETRIES_EXHAUSTED = "retries_exhausted"
    INVALID_PLATFORM = "invalid_platform"
    EMPTY_CONTENT = "empty_content"


class TypedScrapeError(BaseModel):
    """Error tipado serializable devuelto cuando falla una extracción."""

    model_config = ConfigDict(extra="forbid")

    code: ScrapeErrorCode
    message: str
    platform: str
    url: str | None = None
    attempts: int = Field(default=0, ge=0)
    retryable: bool = False
    http_status: int | None = None
    cause_type: str | None = None


class ScrapeResult(BaseModel):
    """Resultado estructurado de una extracción a través de ScraperPort."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    status: ScrapeStatus
    platform: str
    url: str | None = None
    markdown: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    attempts: int = Field(default=0, ge=0)
    elapsed_ms: float = Field(default=0.0, ge=0.0)
    error: TypedScrapeError | None = None


class ScraperError(Exception):
    """Excepción base tipada del puerto de extracción."""

    def __init__(self, error: TypedScrapeError) -> None:
        self.error = error
        super().__init__(error.message)


class ScraperRateLimitedError(ScraperError):
    """Error tipado cuando el rate limiter (o Redis fail-closed) bloquea la solicitud."""


class ScraperTimeoutError(ScraperError):
    """Error tipado cuando el proveedor excede el tiempo máximo permitido."""


class ScraperBlockedError(ScraperError):
    """Error tipado cuando el destino o WAF devuelve bloqueo (403/429/acceso denegado)."""


class ScraperUpstreamError(ScraperError):
    """Error tipado ante fallos 5xx o de red del proveedor."""


class ScraperRetriesExhaustedError(ScraperError):
    """Error tipado cuando se agotan todos los reintentos configurados."""


@runtime_checkable
class ScraperPort(Protocol):
    """Puerto de extracción que deben implementar todos los adaptadores de scraping."""

    async def scrape(self, platform: str, query: str) -> ScrapeResult:
        """Extrae contenido markdown para una clave de plataforma del catálogo y un query."""
        ...
