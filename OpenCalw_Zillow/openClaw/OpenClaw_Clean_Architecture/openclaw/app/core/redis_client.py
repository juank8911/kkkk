"""Cliente Redis compartido por API y worker.

Se mantiene ``decode_responses=False`` porque el worker lee el campo ``payload``
como bytes, tal como lo entrega ``XREADGROUP``.
El ``socket_timeout`` se deja en ``None`` a proposito: ``XREADGROUP`` bloquea
hasta 5 segundos y un timeout de socket mas corto lo interrumpiria.
"""

from __future__ import annotations

import redis.asyncio as redis

from app.core.config import settings

redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    decode_responses=False,
    socket_connect_timeout=2,
    socket_timeout=None,
    health_check_interval=30,
    retry_on_timeout=False,
)
