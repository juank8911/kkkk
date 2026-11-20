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
from app.integrations.uao_client import (
    CircuitBreakerState,
    InMemoryRetryBuffer,
    OrderRetryBufferPort,
    PendingRetryEntry,
    RedisRetryBuffer,
    UaoCircuitBreaker,
    UaoClient,
    UaoDispatchResult,
    UaoLiquidityPort,
    UaoWithdrawalRequest,
)

__all__ = [
    "CircuitBreakerState",
    "FirecrawlAdapter",
    "InMemoryRetryBuffer",
    "OrderRetryBufferPort",
    "PendingRetryEntry",
    "RedisRetryBuffer",
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
    "UaoCircuitBreaker",
    "UaoClient",
    "UaoDispatchResult",
    "UaoLiquidityPort",
    "UaoWithdrawalRequest",
]
