"""Pruebas de idempotencia por ``job_id`` y plataforma (S3-02)."""

from __future__ import annotations

from uuid import uuid4

import fakeredis.aioredis
import pytest

from app.services import idempotency

JOB_ID = uuid4()


@pytest.fixture(autouse=True)
def redis_falso(
    monkeypatch: pytest.MonkeyPatch, fake_redis: fakeredis.aioredis.FakeRedis
) -> fakeredis.aioredis.FakeRedis:
    monkeypatch.setattr(idempotency, "redis_client", fake_redis)
    return fake_redis


def test_clave_estable_por_job_y_plataforma() -> None:
    assert idempotency.clave(JOB_ID, "zillow") == f"openclaw:idem:{JOB_ID}:zillow"
    assert idempotency.clave(JOB_ID, "zillow") != idempotency.clave(JOB_ID, "redfin")


async def test_primera_reclamacion_gana() -> None:
    assert await idempotency.reclamar(JOB_ID, "zillow") is True


async def test_segunda_reclamacion_se_omite() -> None:
    assert await idempotency.reclamar(JOB_ID, "zillow") is True
    assert await idempotency.reclamar(JOB_ID, "zillow") is False


async def test_plataformas_distintas_no_se_bloquean() -> None:
    assert await idempotency.reclamar(JOB_ID, "zillow") is True
    assert await idempotency.reclamar(JOB_ID, "redfin") is True


async def test_liberar_permite_reintento() -> None:
    await idempotency.reclamar(JOB_ID, "trulia")
    assert await idempotency.ya_realizado(JOB_ID, "trulia") is True
    await idempotency.liberar(JOB_ID, "trulia")
    assert await idempotency.ya_realizado(JOB_ID, "trulia") is False
    assert await idempotency.reclamar(JOB_ID, "trulia") is True


async def test_clave_con_ttl() -> None:
    await idempotency.reclamar(JOB_ID, "loopnet")
    ttl = await idempotency.redis_client.ttl(idempotency.clave(JOB_ID, "loopnet"))
    assert ttl > 0
