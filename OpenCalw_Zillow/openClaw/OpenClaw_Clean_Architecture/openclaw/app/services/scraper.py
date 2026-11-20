"""Fachada de extracción basada en ScraperPort y FirecrawlAdapter (S1-08)."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from app.core.config import settings
from app.integrations.firecrawl_adapter import FirecrawlAdapter
from app.integrations.scraper_port import ScrapeResult, ScraperPort, ScrapeStatus
from app.services.platforms import PLATFORM_CATALOG

logger = logging.getLogger("OpenClaw.Scraper")


def _resolve_platform_key(platform_or_url: str) -> str:
    """Resuelve una clave del catálogo o una URL canónica del mismo catálogo (ADR-01).

    Permite compatibilidad con el worker legado que pasa `https://<spec.domain>`
    sin abrir el catálogo a dominios externos (evita SSRF).
    """
    cleaned = platform_or_url.strip().lower()
    if cleaned in PLATFORM_CATALOG:
        return cleaned

    parsed = urlparse(cleaned)
    host = (parsed.hostname or "").lower()
    if host:
        for key, spec in PLATFORM_CATALOG.items():
            if host == spec.domain or host.endswith(f".{spec.domain}"):
                return key
    return platform_or_url


def build_default_scraper(
    *,
    client: Any | None = None,
    timeout_seconds: float = 15.0,
    max_attempts: int = 3,
) -> FirecrawlAdapter:
    """Construye una instancia de FirecrawlAdapter con la configuración activa."""
    return FirecrawlAdapter(
        client=client,
        api_key=getattr(settings, "FIRECRAWL_API_KEY", None),
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
    )


async def extraer_plataforma(
    platform: str,
    query: str,
    *,
    scraper: ScraperPort | None = None,
) -> ScrapeResult:
    """Extrae el markdown de una plataforma del catálogo devolviendo ScrapeResult tipado."""
    active_scraper = scraper if scraper is not None else build_default_scraper()
    return await active_scraper.scrape(platform=platform, query=query)


async def extraer_con_reintentos(
    platform: str,
    query: str,
    max_intentos: int = 3,
    *,
    scraper: ScraperPort | None = None,
) -> dict[str, Any]:
    """Wrapper compatible con el worker delegando en FirecrawlAdapter."""
    resolved_platform = _resolve_platform_key(platform)
    active_scraper = (
        scraper if scraper is not None else build_default_scraper(max_attempts=max_intentos)
    )
    result = await active_scraper.scrape(platform=resolved_platform, query=query)
    if result.ok:
        return {
            "dominio": result.url or platform,
            "plataforma": result.platform,
            "status": "exito",
            "data": {"markdown": result.markdown, "metadata": result.metadata},
            "attempts": result.attempts,
        }
    if result.status == ScrapeStatus.RATE_LIMITED:
        return {
            "dominio": result.url or platform,
            "plataforma": result.platform,
            "status": "rate_limited",
            "error": result.error.model_dump() if result.error else None,
        }
    return {
        "dominio": result.url or platform,
        "plataforma": result.platform,
        "status": result.status.value,
        "error": result.error.model_dump() if result.error else None,
    }
