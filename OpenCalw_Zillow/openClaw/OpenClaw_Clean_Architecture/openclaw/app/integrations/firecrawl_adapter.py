"""Adaptador Firecrawl sobre ScraperPort con timeout, backoff y errores tipados (S1-08).

Respeta estrictamente:
- Catálogo cerrado de 9 plataformas (ADR-01 / S1-07).
- Rate limiting de ventana deslizante con política fail-closed (S1-09).
- Timeout acotado por intento y backoff exponencial con reintentos limitados.
- No intenta evadir controles de acceso ni CAPTCHAs: clasifica bloqueos como
  `ScrapeStatus.BLOCKED` con error tipado.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

import httpx

from app.integrations.scraper_port import (
    ScrapeErrorCode,
    ScraperBlockedError,
    ScraperError,
    ScrapeResult,
    ScraperRateLimitedError,
    ScraperRetriesExhaustedError,
    ScraperTimeoutError,
    ScraperUpstreamError,
    ScrapeStatus,
    TypedScrapeError,
)
from app.services.platforms import (
    InvalidPlatformError,
    InvalidQueryError,
    build_platform_url,
    get_platform_spec,
)
from app.services.rate_limiter import SlidingWindowRateLimiter

logger = logging.getLogger("OpenClaw.FirecrawlAdapter")

SleepFn = Callable[[float], Awaitable[None]]

_BLOCKED_HTTP_STATUSES = frozenset({401, 403, 407, 429, 451})
_BLOCKED_KEYWORDS = (
    "access denied",
    "blocked",
    "captcha",
    "cloudflare",
    "forbidden",
    "perimeterx",
    "too many requests",
    "waf",
)


class FirecrawlAdapter:
    """Implementación de ScraperPort usando un cliente Firecrawl inyectable."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        api_key: str | None = None,
        rate_limiter: SlidingWindowRateLimiter | None = None,
        timeout_seconds: float = 15.0,
        max_attempts: int = 3,
        base_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 16.0,
        sleep_fn: SleepFn | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds debe ser mayor que 0.")
        if max_attempts < 1:
            raise ValueError("max_attempts debe ser al menos 1.")
        if base_backoff_seconds < 0:
            raise ValueError("base_backoff_seconds no puede ser negativo.")

        self._client = client
        self._api_key = api_key
        self._rate_limiter = (
            rate_limiter if rate_limiter is not None else SlidingWindowRateLimiter()
        )
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._base_backoff_seconds = base_backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._sleep: SleepFn = sleep_fn if sleep_fn is not None else asyncio.sleep

    def _get_client(self) -> Any:
        if self._client is None:
            from firecrawl import FirecrawlApp  # Lazy import para entornos sin firecrawl-py

            if not self._api_key:
                raise RuntimeError(
                    "Se requiere FIRECRAWL_API_KEY o un cliente Firecrawl inyectado."
                )
            self._client = FirecrawlApp(api_key=self._api_key)
        return self._client

    async def scrape_url(self, url: str, platform: str) -> ScrapeResult:
        """Extrae una ficha ya registrada, validando HTTPS y el dominio del catálogo."""
        started = time.perf_counter()
        try:
            spec = get_platform_spec(platform)
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
            if parsed.scheme != "https" or not (host == spec.domain or host.endswith(f".{spec.domain}")):
                raise InvalidQueryError(f"La URL no corresponde al dominio autorizado '{spec.domain}'.")
            decision = await self._rate_limiter.check_and_acquire(spec.domain)
            if not decision.allowed:
                return ScrapeResult(
                    ok=False,
                    status=ScrapeStatus.RATE_LIMITED,
                    platform=spec.key,
                    url=url,
                    error=TypedScrapeError(
                        code=ScrapeErrorCode.RATE_LIMITED,
                        message="Límite de tasa o Redis no disponible.",
                        platform=spec.key,
                        url=url,
                        retryable=True,
                    ),
                )
            raw_response = await asyncio.wait_for(self._invoke_firecrawl(url), timeout=self._timeout_seconds)
            markdown, metadata, _ = _extract_response_payload(raw_response)
            return ScrapeResult(
                ok=True,
                status=ScrapeStatus.OK,
                platform=spec.key,
                url=url,
                markdown=markdown,
                metadata=metadata,
                attempts=1,
                elapsed_ms=_elapsed_ms(started),
            )
        except Exception as exc:  # noqa: BLE001 - se devuelve error tipado al tracker
            return ScrapeResult(
                ok=False,
                status=ScrapeStatus.UPSTREAM_ERROR,
                platform=platform,
                url=url,
                attempts=1,
                elapsed_ms=_elapsed_ms(started),
                error=TypedScrapeError(
                    code=ScrapeErrorCode.UPSTREAM_ERROR,
                    message=str(exc),
                    platform=platform,
                    url=url,
                    retryable=True,
                    cause_type=type(exc).__name__,
                ),
            )

    async def scrape(self, platform: str, query: str) -> ScrapeResult:
        """Ejecuta la extracción con validación de catálogo, rate limit, timeout y backoff."""
        started = time.perf_counter()

        try:
            spec = get_platform_spec(platform)
            target_url = build_platform_url(spec.key, query)
        except (InvalidPlatformError, InvalidQueryError) as exc:
            elapsed_ms = _elapsed_ms(started)
            return ScrapeResult(
                ok=False,
                status=ScrapeStatus.INVALID_PLATFORM,
                platform=platform,
                url=None,
                attempts=0,
                elapsed_ms=elapsed_ms,
                error=TypedScrapeError(
                    code=ScrapeErrorCode.INVALID_PLATFORM,
                    message=str(exc),
                    platform=platform,
                    url=None,
                    attempts=0,
                    retryable=False,
                    cause_type=type(exc).__name__,
                ),
            )

        decision = await self._rate_limiter.check_and_acquire(spec.domain)
        if not decision.allowed:
            elapsed_ms = _elapsed_ms(started)
            code = (
                ScrapeErrorCode.REDIS_UNAVAILABLE
                if decision.reason == "redis_unavailable"
                else ScrapeErrorCode.RATE_LIMITED
            )
            msg = (
                f"Rate limiter fail-closed: Redis no disponible para '{spec.domain}'."
                if decision.reason == "redis_unavailable"
                else (
                    f"Límite de tasa excedido para '{spec.domain}' "
                    f"(reintentar en {decision.retry_after_seconds}s)."
                )
            )
            return ScrapeResult(
                ok=False,
                status=ScrapeStatus.RATE_LIMITED,
                platform=spec.key,
                url=target_url,
                attempts=0,
                elapsed_ms=elapsed_ms,
                error=TypedScrapeError(
                    code=code,
                    message=msg,
                    platform=spec.key,
                    url=target_url,
                    attempts=0,
                    retryable=True,
                ),
            )

        last_error: TypedScrapeError | None = None
        last_status: ScrapeStatus = ScrapeStatus.UPSTREAM_ERROR

        for attempt in range(1, self._max_attempts + 1):
            try:
                raw_response = await asyncio.wait_for(
                    self._invoke_firecrawl(target_url),
                    timeout=self._timeout_seconds,
                )
                markdown, metadata, http_status = _extract_response_payload(raw_response)

                if http_status in _BLOCKED_HTTP_STATUSES or _looks_like_block(markdown, metadata):
                    elapsed_ms = _elapsed_ms(started)
                    err = TypedScrapeError(
                        code=ScrapeErrorCode.BLOCKED,
                        message=(
                            f"Acceso bloqueado por la plataforma '{spec.key}' "
                            f"(HTTP {http_status or 'N/A'})."
                        ),
                        platform=spec.key,
                        url=target_url,
                        attempts=attempt,
                        retryable=False,
                        http_status=http_status,
                    )
                    return ScrapeResult(
                        ok=False,
                        status=ScrapeStatus.BLOCKED,
                        platform=spec.key,
                        url=target_url,
                        metadata=metadata,
                        attempts=attempt,
                        elapsed_ms=elapsed_ms,
                        error=err,
                    )

                if not markdown.strip():
                    last_status = ScrapeStatus.UPSTREAM_ERROR
                    last_error = TypedScrapeError(
                        code=ScrapeErrorCode.EMPTY_CONTENT,
                        message=f"Firecrawl devolvió markdown vacío para '{spec.key}'.",
                        platform=spec.key,
                        url=target_url,
                        attempts=attempt,
                        retryable=True,
                        http_status=http_status,
                    )
                else:
                    return ScrapeResult(
                        ok=True,
                        status=ScrapeStatus.OK,
                        platform=spec.key,
                        url=target_url,
                        markdown=markdown,
                        metadata=metadata,
                        attempts=attempt,
                        elapsed_ms=_elapsed_ms(started),
                        error=None,
                    )

            except (TimeoutError, httpx.TimeoutException) as exc:
                last_status = ScrapeStatus.TIMEOUT
                last_error = TypedScrapeError(
                    code=ScrapeErrorCode.TIMEOUT,
                    message=(
                        f"Timeout tras {self._timeout_seconds}s consultando '{spec.key}' "
                        f"(intento {attempt}/{self._max_attempts})."
                    ),
                    platform=spec.key,
                    url=target_url,
                    attempts=attempt,
                    retryable=True,
                    cause_type=type(exc).__name__,
                )
            except Exception as exc:
                http_status = _extract_http_status(exc)
                if http_status in _BLOCKED_HTTP_STATUSES or _exception_is_blocked(exc):
                    elapsed_ms = _elapsed_ms(started)
                    err = TypedScrapeError(
                        code=ScrapeErrorCode.BLOCKED,
                        message=f"Solicitud bloqueada en '{spec.key}': {exc}",
                        platform=spec.key,
                        url=target_url,
                        attempts=attempt,
                        retryable=False,
                        http_status=http_status,
                        cause_type=type(exc).__name__,
                    )
                    return ScrapeResult(
                        ok=False,
                        status=ScrapeStatus.BLOCKED,
                        platform=spec.key,
                        url=target_url,
                        attempts=attempt,
                        elapsed_ms=elapsed_ms,
                        error=err,
                    )

                last_status = ScrapeStatus.UPSTREAM_ERROR
                last_error = TypedScrapeError(
                    code=ScrapeErrorCode.UPSTREAM_ERROR,
                    message=f"Error del proveedor Firecrawl en '{spec.key}': {exc}",
                    platform=spec.key,
                    url=target_url,
                    attempts=attempt,
                    retryable=True,
                    http_status=http_status,
                    cause_type=type(exc).__name__,
                )

            if attempt < self._max_attempts:
                delay = min(
                    self._base_backoff_seconds * (2 ** (attempt - 1)),
                    self._max_backoff_seconds,
                )
                logger.warning(
                    "Reintento %d/%d para '%s' tras fallo '%s' (backoff=%.2fs).",
                    attempt,
                    self._max_attempts,
                    spec.key,
                    last_error.code.value if last_error else "unknown",
                    delay,
                )
                await self._sleep(delay)

        assert last_error is not None
        final_code = (
            ScrapeErrorCode.TIMEOUT
            if last_status == ScrapeStatus.TIMEOUT
            else ScrapeErrorCode.RETRIES_EXHAUSTED
        )
        final_status = (
            ScrapeStatus.TIMEOUT
            if last_status == ScrapeStatus.TIMEOUT
            else ScrapeStatus.RETRIES_EXHAUSTED
        )
        exhausted_error = TypedScrapeError(
            code=final_code,
            message=(
                f"Se agotaron {self._max_attempts} intentos para '{spec.key}'. "
                f"Último error: {last_error.message}"
            ),
            platform=spec.key,
            url=target_url,
            attempts=self._max_attempts,
            retryable=False,
            http_status=last_error.http_status,
            cause_type=last_error.cause_type,
        )
        return ScrapeResult(
            ok=False,
            status=final_status,
            platform=spec.key,
            url=target_url,
            attempts=self._max_attempts,
            elapsed_ms=_elapsed_ms(started),
            error=exhausted_error,
        )

    async def scrape_or_raise(self, platform: str, query: str) -> ScrapeResult:
        """Variante que lanza la subclase tipada de ScraperError cuando ok=False."""
        result = await self.scrape(platform, query)
        if result.ok:
            return result
        assert result.error is not None
        raise _to_exception(result.error)

    async def _invoke_firecrawl(self, target_url: str) -> Any:
        client = self._get_client()
        params = {"formats": ["markdown"], "onlyMainContent": True}

        if hasattr(client, "scrape_url"):
            fn = client.scrape_url
            if inspect.iscoroutinefunction(fn):
                return await fn(target_url, params=params)
            return await asyncio.to_thread(fn, target_url, params=params)

        if hasattr(client, "scrape"):
            fn = client.scrape
            if inspect.iscoroutinefunction(fn):
                return await fn(target_url, formats=["markdown"])
            return await asyncio.to_thread(fn, target_url, formats=["markdown"])

        raise RuntimeError(
            "El cliente Firecrawl no expone un método compatible ('scrape_url' o 'scrape')."
        )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)


def _extract_response_payload(raw: Any) -> tuple[str, dict[str, Any], int | None]:
    if isinstance(raw, dict):
        nested = raw.get("data")
        data: dict[str, Any] = nested if isinstance(nested, dict) else raw
        markdown = str(data.get("markdown") or raw.get("markdown") or "")
        metadata = dict(data.get("metadata") or raw.get("metadata") or {})
        http_status = _coerce_int(
            metadata.get("statusCode")
            or metadata.get("status_code")
            or raw.get("statusCode")
            or raw.get("status_code")
        )
        return markdown, metadata, http_status

    markdown = str(getattr(raw, "markdown", "") or "")
    raw_meta = getattr(raw, "metadata", None)
    metadata = dict(raw_meta) if isinstance(raw_meta, dict) else {}
    http_status = _coerce_int(
        metadata.get("statusCode")
        or metadata.get("status_code")
        or getattr(raw, "status_code", None)
    )
    return markdown, metadata, http_status


def _extract_http_status(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    if response is not None:
        coerced_response = _coerce_int(getattr(response, "status_code", None))
        if coerced_response is not None:
            return coerced_response
    for attr in ("status_code", "statusCode", "http_status"):
        val = getattr(exc, attr, None)
        coerced = _coerce_int(val)
        if coerced is not None:
            return coerced
    return None


def _looks_like_block(markdown: str, metadata: dict[str, Any]) -> bool:
    if metadata.get("blocked") is True:
        return True
    sample = f"{metadata.get('title', '')} {markdown[:500]}".lower()
    if len(markdown.strip()) < 400:
        return any(keyword in sample for keyword in _BLOCKED_KEYWORDS)
    return False


def _exception_is_blocked(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(keyword in text for keyword in _BLOCKED_KEYWORDS)


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_exception(err: TypedScrapeError) -> ScraperError:
    mapping: dict[ScrapeErrorCode, type[ScraperError]] = {
        ScrapeErrorCode.RATE_LIMITED: ScraperRateLimitedError,
        ScrapeErrorCode.REDIS_UNAVAILABLE: ScraperRateLimitedError,
        ScrapeErrorCode.BLOCKED: ScraperBlockedError,
        ScrapeErrorCode.TIMEOUT: ScraperTimeoutError,
        ScrapeErrorCode.RETRIES_EXHAUSTED: ScraperRetriesExhaustedError,
        ScrapeErrorCode.UPSTREAM_ERROR: ScraperUpstreamError,
    }
    exc_cls = mapping.get(err.code, ScraperError)
    return exc_cls(err)
