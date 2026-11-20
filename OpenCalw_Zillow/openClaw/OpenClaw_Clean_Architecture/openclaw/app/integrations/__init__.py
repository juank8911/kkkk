"""Integraciones externas de OpenClaw (puertos y adaptadores)."""

from app.integrations.firecrawl_adapter import FirecrawlAdapter
from app.integrations.scraper_port import (
    ScrapeErrorCode,
    ScraperBlockedError,
    ScraperError,
    ScrapeResult,
    ScraperPort,
    ScraperRateLimitedError,
    ScraperRetriesExhaustedError,
    ScraperTimeoutError,
    ScraperUpstreamError,
    ScrapeStatus,
    TypedScrapeError,
)

__all__ = [
    "FirecrawlAdapter",
    "ScrapeErrorCode",
    "ScrapeResult",
    "ScrapeStatus",
    "ScraperBlockedError",
    "ScraperError",
    "ScraperPort",
    "ScraperRateLimitedError",
    "ScraperRetriesExhaustedError",
    "ScraperTimeoutError",
    "ScraperUpstreamError",
    "TypedScrapeError",
]
