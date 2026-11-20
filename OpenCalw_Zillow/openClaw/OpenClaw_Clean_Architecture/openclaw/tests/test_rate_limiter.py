"""Pruebas del rate limiter de ventana deslizante con política fail-closed (S1-09)."""

from __future__ import annotations

import fakeredis.aioredis
from redis.exceptions import ConnectionError as RedisConnectionError

from app.services.rate_limiter import SlidingWindowRateLimiter, adquirir_permiso_extraccion
from tests.conftest import MutableClock


class BrokenRedisClient:
    """Dobles de Redis caído para verificar política fail-closed."""

    def pipeline(self, transaction: bool = True) -> BrokenRedisClient:
        return self

    async def __aenter__(self) -> BrokenRedisClient:
        raise RedisConnectionError("Redis connection refused")

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


async def test_sliding_window_enforces_limit_and_recovers_after_window(
    fake_redis: fakeredis.aioredis.FakeRedis,
    fake_clock: MutableClock,
) -> None:
    """Superar el límite devuelve rate_limited y al deslizar la ventana vuelve a permitir."""
    limiter = SlidingWindowRateLimiter(
        redis_client=fake_redis,
        limits={"zillow.com": 3, "default": 5},
        window_seconds=60,
        clock=fake_clock,
    )

    d1 = await limiter.check_and_acquire("zillow.com")
    fake_clock.advance(10)
    d2 = await limiter.check_and_acquire("https://www.zillow.com/homes/Miami_rb/")
    fake_clock.advance(10)
    d3 = await limiter.check_and_acquire("www.zillow.com")

    assert [d1.allowed, d2.allowed, d3.allowed] == [True, True, True]
    assert d1.remaining == 2
    assert d2.remaining == 1
    assert d3.remaining == 0

    # 4ta solicitud dentro de los 60s -> rate_limited
    d4 = await limiter.check_and_acquire("zillow.com")
    assert d4.allowed is False
    assert d4.status == "rate_limited"
    assert d4.reason == "limit_exceeded"
    assert d4.remaining == 0
    assert d4.retry_after_seconds > 0

    # Avanzar 41s (total 61s desde d1, pero d2 y d3 siguen dentro de la ventana) -> 1 hueco libre
    fake_clock.advance(41)
    d5 = await limiter.check_and_acquire("zillow.com")
    assert d5.allowed is True
    assert d5.status == "allowed"
    assert d5.remaining == 0

    # Siguiente inmediata vuelve a superar el límite de 3
    d6 = await limiter.check_and_acquire("zillow.com")
    assert d6.allowed is False
    assert d6.status == "rate_limited"


async def test_domains_are_isolated_independently(
    fake_redis: fakeredis.aioredis.FakeRedis,
    fake_clock: MutableClock,
) -> None:
    limiter = SlidingWindowRateLimiter(
        redis_client=fake_redis,
        limits={"zillow.com": 1, "realtor.com": 2, "default": 2},
        window_seconds=60,
        clock=fake_clock,
    )

    assert (await limiter.check_and_acquire("zillow.com")).allowed is True
    assert (await limiter.check_and_acquire("zillow.com")).status == "rate_limited"
    # realtor.com mantiene su propia cuota
    assert (await limiter.check_and_acquire("realtor.com")).allowed is True
    assert (await limiter.check_and_acquire("realtor.com")).allowed is True
    assert (await limiter.check_and_acquire("realtor.com")).status == "rate_limited"


async def test_redis_down_blocks_fail_closed(fake_clock: MutableClock) -> None:
    """Criterio S1-09: Redis caído bloquea (política fail-closed)."""
    limiter = SlidingWindowRateLimiter(
        redis_client=BrokenRedisClient(),
        limits={"zillow.com": 15, "default": 20},
        window_seconds=60,
        clock=fake_clock,
    )

    decision = await limiter.check_and_acquire("zillow.com")
    assert decision.allowed is False
    assert decision.status == "rate_limited"
    assert decision.reason == "redis_unavailable"
    assert decision.remaining == 0

    # Wrapper legado también debe respetar fail-closed
    permitted = await adquirir_permiso_extraccion("zillow.com", limiter=limiter)
    assert permitted is False
