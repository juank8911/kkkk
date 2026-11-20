"""Servicios de dominio de OpenClaw (catálogo, rate limiter, scraper y normalizador)."""

from app.services.normalizer import (
    MarkdownPropertyNormalizer,
    NormalizationBatch,
    NormalizationError,
    RejectedListing,
    normalizar_markdown_propiedades,
    normalizar_propiedad_unica,
)
from app.services.platforms import (
    PLATFORM_CATALOG,
    VALID_PLATFORM_KEYS,
    InvalidPlatformError,
    InvalidQueryError,
    PlatformSpec,
    build_platform_url,
    get_default_domain_limits,
    get_platform_spec,
    sanitize_query_slug,
    validate_platforms,
)
from app.services.rate_limiter import (
    LIMITES_WAF,
    RateLimitDecision,
    SlidingWindowRateLimiter,
    adquirir_permiso_extraccion,
)

__all__ = [
    "InvalidPlatformError",
    "InvalidQueryError",
    "LIMITES_WAF",
    "MarkdownPropertyNormalizer",
    "NormalizationBatch",
    "NormalizationError",
    "PLATFORM_CATALOG",
    "PlatformSpec",
    "RateLimitDecision",
    "RejectedListing",
    "SlidingWindowRateLimiter",
    "VALID_PLATFORM_KEYS",
    "adquirir_permiso_extraccion",
    "build_platform_url",
    "get_default_domain_limits",
    "get_platform_spec",
    "normalizar_markdown_propiedades",
    "normalizar_propiedad_unica",
    "sanitize_query_slug",
    "validate_platforms",
]
