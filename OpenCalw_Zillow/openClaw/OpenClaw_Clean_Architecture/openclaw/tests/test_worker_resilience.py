"""Pruebas de resiliencia del worker sobre Redis Streams (S3-08).

Cubre los cuatro escenarios exigidos por el criterio de aceptación de S3-08:
1. Pérdida / caída de un worker a mitad de proceso: recuperación de pendientes sin pérdida.
2. Reintentos controlados ante fallos transitorios.
3. Envío a Dead-Letter Queue (DLQ) tras agotar reintentos + XACK del stream principal.
4. Idempotencia frente a entregas duplicadas por `(job_id, plataforma)`.
"""

from __future__ import annotations

import json
from contextlib import suppress
from typing import Any

import fakeredis.aioredis


class ResilientStreamWorkerHarness:
    """Modelo de consumer group + recuperación de pendientes + DLQ + idempotencia (ADR-02 / S3-08)."""

    def __init__(
        self,
        redis_client: fakeredis.aioredis.FakeRedis,
        *,
        stream: str = "openclaw:streams:barridos",
        dlq_stream: str = "openclaw:streams:barridos:dlq",
        group: str = "openclaw_scrapers",
        max_retries: int = 3,
    ) -> None:
        self.redis = redis_client
        self.stream = stream
        self.dlq_stream = dlq_stream
        self.group = group
        self.max_retries = max_retries
        self.persisted_properties: list[tuple[str, str]] = []

    async def ensure_group(self) -> None:
        with suppress(Exception):
            await self.redis.xgroup_create(self.stream, self.group, id="0", mkstream=True)

    async def publish(self, job_id: str, plataforma: str, query: str) -> Any:
        payload = json.dumps({"job_id": job_id, "plataforma": plataforma, "query": query})
        return await self.redis.xadd(self.stream, {"payload": payload})

    async def read_without_ack_simulating_crash(self, consumer: str) -> list[tuple[bytes, dict]]:
        """Simula un worker que lee un mensaje y muere antes de XACK."""
        msgs = await self.redis.xreadgroup(
            groupname=self.group,
            consumername=consumer,
            streams={self.stream: ">"},
            count=1,
        )
        if not msgs:
            return []
        return list(msgs[0][1])

    async def recover_pending_for_consumer(
        self,
        new_consumer: str,
        *,
        min_idle_ms: int = 0,
    ) -> list[tuple[bytes, dict]]:
        """Reclama mensajes pendientes de workers caídos mediante XCLAIM / XAUTOCLAIM."""
        pending = await self.redis.xpending_range(self.stream, self.group, "-", "+", count=50)
        if not pending:
            return []
        ids = [entry["message_id"] for entry in pending]
        claimed = await self.redis.xclaim(
            self.stream,
            self.group,
            new_consumer,
            min_idle_time=min_idle_ms,
            message_ids=ids,
        )
        return list(claimed)

    async def process_message(
        self,
        msg_id: bytes | str,
        fields: dict[bytes, bytes],
        *,
        processor,
    ) -> str:
        payload = json.loads(fields[b"payload"].decode("utf-8"))
        job_id = payload["job_id"]
        plataforma = payload["plataforma"]
        idem_key = f"openclaw:processed:{job_id}:{plataforma}"
        retry_key = f"openclaw:retries:{job_id}:{plataforma}"

        # Idempotencia por (job_id, plataforma)
        already_done = await self.redis.get(idem_key)
        if already_done:
            await self.redis.xack(self.stream, self.group, msg_id)
            return "duplicate_skipped"

        try:
            await processor(payload)
            await self.redis.set(idem_key, "1", ex=3600)
            self.persisted_properties.append((job_id, plataforma))
            await self.redis.xack(self.stream, self.group, msg_id)
            return "processed"
        except Exception as exc:
            attempts = int(await self.redis.incr(retry_key))
            if attempts >= self.max_retries:
                await self.redis.xadd(
                    self.dlq_stream,
                    {
                        "job_id": job_id,
                        "plataforma": plataforma,
                        "attempts": str(attempts),
                        "error": str(exc),
                        "payload": fields[b"payload"].decode("utf-8"),
                    },
                )
                await self.redis.xack(self.stream, self.group, msg_id)
                return "sent_to_dlq"
            return "retry_pending"


async def test_s3_08_worker_crash_recovers_pending_without_message_loss(
    fake_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    harness = ResilientStreamWorkerHarness(fake_redis)
    await harness.ensure_group()
    await harness.publish("job-crash-1", "zillow", "Cleveland OH")

    # Worker 1 lee el mensaje y muere antes de XACK
    crashed_msgs = await harness.read_without_ack_simulating_crash("worker-dead")
    assert len(crashed_msgs) == 1

    # Worker 2 reclama el mensaje pendiente y lo procesa sin pérdida
    recovered = await harness.recover_pending_for_consumer("worker-alive")
    assert len(recovered) == 1

    async def ok_processor(payload: dict[str, Any]) -> None:
        assert payload["job_id"] == "job-crash-1"

    msg_id, fields = recovered[0]
    outcome = await harness.process_message(msg_id, fields, processor=ok_processor)
    assert outcome == "processed"
    assert harness.persisted_properties == [("job-crash-1", "zillow")]

    # No quedan mensajes pendientes tras XACK
    remaining = await fake_redis.xpending_range(harness.stream, harness.group, "-", "+", count=10)
    assert remaining == []


async def test_s3_08_retries_then_routes_poison_message_to_dlq(
    fake_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    harness = ResilientStreamWorkerHarness(fake_redis, max_retries=3)
    await harness.ensure_group()
    await harness.publish("job-poison-1", "redfin", "Miami FL")

    msgs = await harness.read_without_ack_simulating_crash("worker-1")
    msg_id, fields = msgs[0]

    async def failing_processor(payload: dict[str, Any]) -> None:
        raise RuntimeError("Upstream parser crash")

    assert (
        await harness.process_message(msg_id, fields, processor=failing_processor)
        == "retry_pending"
    )
    assert (
        await harness.process_message(msg_id, fields, processor=failing_processor)
        == "retry_pending"
    )
    assert (
        await harness.process_message(msg_id, fields, processor=failing_processor) == "sent_to_dlq"
    )

    dlq_entries = await fake_redis.xrange(harness.dlq_stream, "-", "+")
    assert len(dlq_entries) == 1
    assert dlq_entries[0][1][b"job_id"] == b"job-poison-1"
    assert dlq_entries[0][1][b"attempts"] == b"3"
    assert b"Upstream parser crash" in dlq_entries[0][1][b"error"]

    # Confirmado en el stream principal para no atascar el grupo
    remaining = await fake_redis.xpending_range(harness.stream, harness.group, "-", "+", count=10)
    assert remaining == []


async def test_s3_08_duplicate_delivery_is_idempotent(
    fake_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    harness = ResilientStreamWorkerHarness(fake_redis)
    await harness.ensure_group()

    # Dos mensajes con el mismo (job_id, plataforma)
    await harness.publish("job-dup-1", "realtor", "Tampa FL")
    await harness.publish("job-dup-1", "realtor", "Tampa FL")

    executions = 0

    async def counting_processor(payload: dict[str, Any]) -> None:
        nonlocal executions
        executions += 1

    first = await harness.read_without_ack_simulating_crash("worker-1")
    second = await harness.read_without_ack_simulating_crash("worker-1")

    assert await harness.process_message(*first[0], processor=counting_processor) == "processed"
    assert (
        await harness.process_message(*second[0], processor=counting_processor)
        == "duplicate_skipped"
    )
    assert executions == 1
    assert harness.persisted_properties == [("job-dup-1", "realtor")]
