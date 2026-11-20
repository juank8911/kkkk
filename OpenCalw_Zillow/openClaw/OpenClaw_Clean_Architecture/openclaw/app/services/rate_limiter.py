"""Rate limiter de ventana deslizante en Redis con política fail-closed (S1-09).

Garantiza que:
- Superar el límite configurado por dominio en la ventana deslizante devuelve
  `status="rate_limited"`.
- Si Redis no está disponible o falla cualquier operación, la política
  **fail-closed** bloquea la solicitud (`allowed=False`, `status="rate_limited"`,
  `reason="redis_unavailable"`).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from app.core.redis_client import redis_client as default_redis_client
from app.services.platforms import get_default_domain_limits

logger = logging.getLogger("OpenClaw.RateLimiter")

LIMITES_WAF: dict[str, int] = get_default_domain_limits()


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Resultado tipado de una evaluación del limitador de tasa."""

    allowed: bool
    status: Literal["allowed", "rate_limited"]
    domain: str
    limit: int
    remaining: int
    retry_after_seconds: float
    reason: Literal["ok", "limit_exceeded", "redis_unavailable"]


class SlidingWindowRateLimiter:
    """Limitador de tasa de ventana deslizante respaldado por Redis ZSET (fail-closed)."""

    def __init__(
        self,
        redis_client: Any | None = None,
        limits: Mapping[str, int] | None = None,
        window_seconds: int = 60,
        key_prefix: str = "ratelimit:openclaw:sw",
        clock: Callable[[], float] | None = None,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds debe ser mayor que 0.")
        self._redis = redis_client if redis_client is not None else default_redis_client
        self._limits: dict[str, int] = dict(limits) if limits is not None else dict(LIMITES_WAF)
        self._window_seconds = window_seconds
        self._key_prefix = key_prefix
        self._clock = clock if clock is not None else time.time

    @property
    def window_seconds(self) -> int:
        return self._window_seconds

    def get_limit_for_domain(self, domain: str) -> int:
        normalized = _normalize_domain(domain)
        return int(self._limits.get(normalized, self._limits.get("default", 20)))

    async def check_and_acquire(self, domain: str) -> RateLimitDecision:
        """Evalúa la ventana deslizante y registra la petición si está permitida.

        Política fail-closed: cualquier error de Redis rechaza la solicitud.
        """
        normalized_domain = _normalize_domain(domain)
        limit = self.get_limit_for_domain(normalized_domain)
        if limit <= 0:
            return RateLimitDecision(
                allowed=False,
                status="rate_limited",
                domain=normalized_domain,
                limit=limit,
                remaining=0,
                retry_after_seconds=float(self._window_seconds),
                reason="limit_exceeded",
            )

        now_ms = int(self._clock() * 1000)
        window_ms = self._window_seconds * 1000
        window_start_ms = now_ms - window_ms
        key = f"{self._key_prefix}:{normalized_domain}"
        member = f"{now_ms}:{uuid.uuid4().hex}"

        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.zremrangebyscore(key, "-inf", window_start_ms)
                pipe.zadd(key, {member: now_ms})
                pipe.zcard(key)
                pipe.pexpire(key, window_ms + 1000)
                results = await pipe.execute()

            current_count = int(results[2])

            if current_count > limit:
                # Retira el miembro recién agregado para no inflar la ventana al rechazar.
                await self._redis.zrem(key, member)
                oldest = await self._redis.zrange(key, 0, 0, withscores=True)
                retry_after = float(self._window_seconds)
                if oldest:
                    oldest_ts_ms = float(oldest[0][1])
                    retry_after = max(0.05, (oldest_ts_ms + window_ms - now_ms) / 1000.0)

                logger.warning(
                    "Rate limit excedido para dominio '%s' (limite=%d/%ds, retry_after=%.2fs).",
                    normalized_domain,
                    limit,
                    self._window_seconds,
                    retry_after,
                )
                return RateLimitDecision(
                    allowed=False,
                    status="rate_limited",
                    domain=normalized_domain,
                    limit=limit,
                    remaining=0,
                    retry_after_seconds=round(retry_after, 3),
                    reason="limit_exceeded",
                )

            remaining = max(0, limit - current_count)
            return RateLimitDecision(
                allowed=True,
                status="allowed",
                domain=normalized_domain,
                limit=limit,
                remaining=remaining,
                retry_after_seconds=0.0,
                reason="ok",
            )
        except Exception as exc:
            logger.error(
                "Redis no disponible en rate limiter para '%s'; aplicando política fail-closed: %s",
                normalized_domain,
                exc,
            )
            return RateLimitDecision(
                allowed=False,
                status="rate_limited",
                domain=normalized_domain,
                limit=limit,
                remaining=0,
                retry_after_seconds=float(self._window_seconds),
                reason="redis_unavailable",
            )


def _normalize_domain(domain: str) -> str:
    cleaned = domain.strip().lower()
    if "://" in cleaned:
        cleaned = cleaned.split("://", 1)[1]
    cleaned = cleaned.split("/", 1)[0]
    if cleaned.startswith("www."):
        cleaned = cleaned[4:]
    return cleaned


_default_limiter = SlidingWindowRateLimiter()


async def adquirir_permiso_extraccion(
    dominio: str,
    *,
    limiter: SlidingWindowRateLimiter | None = None,
) -> bool:
    """Wrapper booleano compatible con consumidores existentes (política fail-closed)."""
    active_limiter = limiter if limiter is not None else _default_limiter
    decision = await active_limiter.check_and_acquire(dominio)
    return decision.allowed
