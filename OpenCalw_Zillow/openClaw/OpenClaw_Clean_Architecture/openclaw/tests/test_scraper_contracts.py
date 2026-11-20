"""Pruebas de contrato del scraper con cassettes sin red (S1-08 y S1-10).

Cubre:
- Éxito en múltiples plataformas desde cassettes locales (sin tráfico de red).
- Bloqueo (403 / WAF y 429) con clasificación tipada sin evasión.
- Timeout con backoff exponencial agotando reintentos y devolviendo error tipado.
- Agotamiento de reintentos por error 503 y recuperación tras fallo transitorio.
- Integración con rate limiter fail-closed (límite excedido y Redis caído).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import fakeredis.aioredis
import httpx
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.integrations import (
    FirecrawlAdapter,
    ScrapeErrorCode,
    ScraperBlockedError,
    ScraperPort,
    ScraperRetriesExhaustedError,
    ScraperTimeoutError,
    ScrapeStatus,
    TypedScrapeError,
)
from app.services.rate_limiter import SlidingWindowRateLimiter
from app.services.scraper import extraer_con_reintentos
from tests.conftest import MutableClock

CASSETTES_DIR = Path(__file__).parent / "cassettes" / "scraper"


def load_cassette(name: str) -> dict[str, Any]:
    path = CASSETTES_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


class CassetteFirecrawlClient:
    """Cliente Firecrawl determinista respaldado por cassettes locales (cero red)."""

    def __init__(self, cassettes: list[dict[str, Any]]) -> None:
        self._cassettes = list(cassettes)
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def scrape_url(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((url, params))
        idx = min(len(self.calls) - 1, len(self._cassettes) - 1)
        cassette = self._cassettes[idx]
        behavior = cassette["behavior"]

        if behavior == "response":
            return cassette["response"]

        if behavior == "timeout":
            await asyncio.sleep(float(cassette.get("sleep_seconds", 0.2)))
            return {"data": {"markdown": "too late"}}

        if behavior == "http_error":
            status_code = int(cassette["status_code"])
            req = httpx.Request("POST", url)
            resp = httpx.Response(status_code=status_code, request=req, text=cassette["message"])
            raise httpx.HTTPStatusError(cassette["message"], request=req, response=resp)

        raise RuntimeError(f"Comportamiento de cassette desconocido: {behavior}")


class _DownRedis:
    def pipeline(self, transaction: bool = True) -> _DownRedis:
        return self

    async def __aenter__(self) -> _DownRedis:
        raise RedisConnectionError("Redis down")

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def _make_sleep_recorder() -> tuple[list[float], Any]:
    recorded: list[float] = []

    async def _sleep(seconds: float) -> None:
        recorded.append(seconds)

    return recorded, _sleep


def test_firecrawl_adapter_implements_scraper_port(
    fake_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    limiter = SlidingWindowRateLimiter(redis_client=fake_redis)
    adapter = FirecrawlAdapter(client=CassetteFirecrawlClient([]), rate_limiter=limiter)
    assert isinstance(adapter, ScraperPort)


@pytest.mark.parametrize(
    ("cassette_name", "platform", "query", "expected_snippet"),
    [
        ("zillow_success", "zillow", "Cleveland OH", "1420 Maple Ave"),
        ("realtor_success", "realtor", "Detroit MI", "8842 Oakwood Dr"),
    ],
)
async def test_cassette_scrape_success_offline(
    fake_redis: fakeredis.aioredis.FakeRedis,
    fake_clock: MutableClock,
    cassette_name: str,
    platform: str,
    query: str,
    expected_snippet: str,
) -> None:
    cassette = load_cassette(cassette_name)
    client = CassetteFirecrawlClient([cassette])
    limiter = SlidingWindowRateLimiter(redis_client=fake_redis, clock=fake_clock)
    delays, sleep_fn = _make_sleep_recorder()

    adapter = FirecrawlAdapter(
        client=client,
        rate_limiter=limiter,
        timeout_seconds=1.0,
        max_attempts=3,
        sleep_fn=sleep_fn,
    )

    result = await adapter.scrape(platform, query)
    assert result.ok is True
    assert result.status == ScrapeStatus.OK
    assert result.attempts == 1
    assert result.error is None
    assert expected_snippet in result.markdown
    assert delays == []
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("cassette_name", "platform", "query", "expected_http"),
    [
        ("redfin_blocked_403", "redfin", "Miami FL", 403),
        ("trulia_rate_limited_429", "trulia", "Tampa FL", 429),
    ],
)
async def test_cassette_scrape_blocked_returns_typed_error_without_evasion(
    fake_redis: fakeredis.aioredis.FakeRedis,
    fake_clock: MutableClock,
    cassette_name: str,
    platform: str,
    query: str,
    expected_http: int,
) -> None:
    cassette = load_cassette(cassette_name)
    client = CassetteFirecrawlClient([cassette])
    limiter = SlidingWindowRateLimiter(redis_client=fake_redis, clock=fake_clock)
    delays, sleep_fn = _make_sleep_recorder()

    adapter = FirecrawlAdapter(
        client=client,
        rate_limiter=limiter,
        timeout_seconds=1.0,
        max_attempts=3,
        sleep_fn=sleep_fn,
    )

    result = await adapter.scrape(platform, query)
    assert result.ok is False
    assert result.status == ScrapeStatus.BLOCKED
    assert result.attempts == 1
    assert isinstance(result.error, TypedScrapeError)
    assert result.error.code == ScrapeErrorCode.BLOCKED
    assert result.error.http_status == expected_http
    assert result.error.retryable is False
    assert delays == []  # No reintenta ni evade bloqueos WAF/403/429

    with pytest.raises(ScraperBlockedError) as exc_info:
        await adapter.scrape_or_raise(platform, query)
    assert exc_info.value.error.code == ScrapeErrorCode.BLOCKED


async def test_cassette_timeout_exhausts_retries_with_exponential_backoff(
    fake_redis: fakeredis.aioredis.FakeRedis,
    fake_clock: MutableClock,
) -> None:
    """Criterio S1-08 & S1-10: timeout agota reintentos con backoff y devuelve error tipado."""
    cassette = load_cassette("loopnet_timeout")
    client = CassetteFirecrawlClient([cassette])
    limiter = SlidingWindowRateLimiter(redis_client=fake_redis, clock=fake_clock)
    delays, sleep_fn = _make_sleep_recorder()

    adapter = FirecrawlAdapter(
        client=client,
        rate_limiter=limiter,
        timeout_seconds=0.02,
        max_attempts=3,
        base_backoff_seconds=1.5,
        sleep_fn=sleep_fn,
    )

    result = await adapter.scrape("loopnet", "Dallas TX")
    assert result.ok is False
    assert result.status == ScrapeStatus.TIMEOUT
    assert result.attempts == 3
    assert isinstance(result.error, TypedScrapeError)
    assert result.error.code == ScrapeErrorCode.TIMEOUT
    assert result.error.attempts == 3
    assert result.error.retryable is False
    assert delays == [1.5, 3.0]
    assert len(client.calls) == 3

    with pytest.raises(ScraperTimeoutError):
        await adapter.scrape_or_raise("loopnet", "Dallas TX")


async def test_cassette_upstream_503_exhausts_retries_and_returns_typed_error(
    fake_redis: fakeredis.aioredis.FakeRedis,
    fake_clock: MutableClock,
) -> None:
    cassette = load_cassette("zillow_upstream_503")
    client = CassetteFirecrawlClient([cassette])
    limiter = SlidingWindowRateLimiter(redis_client=fake_redis, clock=fake_clock)
    delays, sleep_fn = _make_sleep_recorder()

    adapter = FirecrawlAdapter(
        client=client,
        rate_limiter=limiter,
        timeout_seconds=1.0,
        max_attempts=3,
        base_backoff_seconds=1.0,
        sleep_fn=sleep_fn,
    )

    result = await adapter.scrape("zillow", "Orlando FL")
    assert result.ok is False
    assert result.status == ScrapeStatus.RETRIES_EXHAUSTED
    assert result.attempts == 3
    assert isinstance(result.error, TypedScrapeError)
    assert result.error.code == ScrapeErrorCode.RETRIES_EXHAUSTED
    assert result.error.http_status == 503
    assert delays == [1.0, 2.0]

    with pytest.raises(ScraperRetriesExhaustedError):
        await adapter.scrape_or_raise("zillow", "Orlando FL")


async def test_transient_503_then_success_recovers_on_retry(
    fake_redis: fakeredis.aioredis.FakeRedis,
    fake_clock: MutableClock,
) -> None:
    client = CassetteFirecrawlClient(
        [
            load_cassette("zillow_upstream_503"),
            load_cassette("zillow_success"),
        ]
    )
    limiter = SlidingWindowRateLimiter(redis_client=fake_redis, clock=fake_clock)
    delays, sleep_fn = _make_sleep_recorder()

    adapter = FirecrawlAdapter(
        client=client,
        rate_limiter=limiter,
        timeout_seconds=1.0,
        max_attempts=3,
        base_backoff_seconds=0.5,
        sleep_fn=sleep_fn,
    )

    result = await adapter.scrape("zillow", "Cleveland OH")
    assert result.ok is True
    assert result.status == ScrapeStatus.OK
    assert result.attempts == 2
    assert delays == [0.5]
    assert "1420 Maple Ave" in result.markdown


async def test_rate_limited_and_redis_down_never_call_firecrawl(
    fake_redis: fakeredis.aioredis.FakeRedis,
    fake_clock: MutableClock,
) -> None:
    client = CassetteFirecrawlClient([load_cassette("zillow_success")])
    limiter = SlidingWindowRateLimiter(
        redis_client=fake_redis,
        limits={"zillow.com": 1, "default": 1},
        clock=fake_clock,
    )
    adapter = FirecrawlAdapter(client=client, rate_limiter=limiter)

    first = await adapter.scrape("zillow", "Cleveland OH")
    second = await adapter.scrape("zillow", "Cleveland OH")
    assert first.ok is True
    assert second.ok is False
    assert second.status == ScrapeStatus.RATE_LIMITED
    assert second.error is not None
    assert second.error.code == ScrapeErrorCode.RATE_LIMITED
    assert len(client.calls) == 1

    # Redis caído -> fail-closed
    down_adapter = FirecrawlAdapter(
        client=client,
        rate_limiter=SlidingWindowRateLimiter(redis_client=_DownRedis(), clock=fake_clock),
    )
    blocked_by_redis = await down_adapter.scrape("zillow", "Cleveland OH")
    assert blocked_by_redis.ok is False
    assert blocked_by_redis.status == ScrapeStatus.RATE_LIMITED
    assert blocked_by_redis.error is not None
    assert blocked_by_redis.error.code == ScrapeErrorCode.REDIS_UNAVAILABLE
    assert len(client.calls) == 1


async def test_legacy_scraper_facade_delegates_to_adapter(
    fake_redis: fakeredis.aioredis.FakeRedis,
    fake_clock: MutableClock,
) -> None:
    client = CassetteFirecrawlClient([load_cassette("zillow_success")])
    limiter = SlidingWindowRateLimiter(redis_client=fake_redis, clock=fake_clock)
    adapter = FirecrawlAdapter(client=client, rate_limiter=limiter)

    legacy = await extraer_con_reintentos("zillow", "Cleveland OH", scraper=adapter)
    assert legacy["status"] == "exito"
    assert legacy["plataforma"] == "zillow"
    assert "1420 Maple Ave" in legacy["data"]["markdown"]
