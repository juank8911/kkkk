"""Pruebas del ciclo de reintentos, DLQ y recuperacion de pendientes (S3-01)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import fakeredis.aioredis
import pytest

from app.core.config import settings
from app.models.contracts import JobResponse, JobStatus
from app.workers import ingestion_worker, state_machine

JOB_ID = uuid4()


def _payload(job_id: UUID = JOB_ID) -> bytes:
    return json.dumps(
        {"job_id": str(job_id), "query": "Cleveland OH", "plataformas": ["zillow"]}
    ).encode("utf-8")


class FakeJobStore:
    def __init__(self, estado: JobStatus = JobStatus.QUEUED) -> None:
        self.job: JobResponse = JobResponse(
            job_id=JOB_ID,
            status=estado,
            query="Cleveland OH",
            plataformas=["zillow"],
            created_at=datetime.now(timezone.utc),
        )
        self.llamadas: list[JobStatus] = []

    async def obtener(self, job_id: UUID) -> JobResponse | None:
        return self.job if job_id == JOB_ID else None

    async def marcar(
        self,
        job_id: UUID,
        estado: JobStatus,
        *,
        progress: dict[str, str] | None = None,
        error: str | None = None,
    ) -> None:
        self.llamadas.append(estado)
        self.job = self.job.model_copy(update={"status": estado})


@pytest.fixture
def redis_falso(
    monkeypatch: pytest.MonkeyPatch, fake_redis: fakeredis.aioredis.FakeRedis
) -> fakeredis.aioredis.FakeRedis:
    monkeypatch.setattr(ingestion_worker, "redis_client", fake_redis)
    monkeypatch.setattr(settings, "MAX_ENTREGAS", 3)
    return fake_redis


@pytest.fixture
def job_store_falso(monkeypatch: pytest.MonkeyPatch) -> FakeJobStore:
    falso = FakeJobStore()
    monkeypatch.setattr(state_machine, "job_store", falso)
    return falso


async def _mensaje_pendiente(
    redis: fakeredis.aioredis.FakeRedis,
) -> str:
    await redis.xgroup_create(
        settings.REDIS_STREAM_SCRAPING, settings.REDIS_CONSUMER_GROUP, id="0", mkstream=True
    )
    id_mensaje = await redis.xadd(settings.REDIS_STREAM_SCRAPING, {"payload": _payload()})
    await redis.xreadgroup(
        groupname=settings.REDIS_CONSUMER_GROUP,
        consumername=settings.REDIS_CONSUMER_NAME,
        streams={settings.REDIS_STREAM_SCRAPING: ">"},
        count=1,
    )
    return id_mensaje


def test_debe_enviar_a_dlq_al_agotar_entregas() -> None:
    assert ingestion_worker.debe_enviar_a_dlq(1, maximo=3) is False
    assert ingestion_worker.debe_enviar_a_dlq(2, maximo=3) is False
    assert ingestion_worker.debe_enviar_a_dlq(3, maximo=3) is True
    assert ingestion_worker.debe_enviar_a_dlq(4, maximo=3) is True


def test_estado_final_segun_resultados() -> None:
    assert ingestion_worker._estado_final([{"status": "exito"}]) is JobStatus.COMPLETED
    assert (
        ingestion_worker._estado_final([{"status": "exito"}, {"status": "fallo"}])
        is JobStatus.PARTIAL
    )
    assert ingestion_worker._estado_final([{"status": "fallo"}]) is JobStatus.FAILED


async def test_mensaje_exitoso_se_confirma(
    redis_falso: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    id_mensaje = await _mensaje_pendiente(redis_falso)

    async def ok(_id: str, _datos: dict[bytes, bytes]) -> None:
        return None

    monkeypatch.setattr(ingestion_worker, "_procesar_mensaje", ok)
    await ingestion_worker._manejar_mensaje(id_mensaje, {b"payload": _payload()})

    pendientes = await redis_falso.xpending_range(
        settings.REDIS_STREAM_SCRAPING, settings.REDIS_CONSUMER_GROUP, "-", "+", count=10
    )
    assert pendientes == []
    assert await redis_falso.get(f"{ingestion_worker.PREFIJO_ENTREGAS}{id_mensaje}") is None


async def test_fallo_no_confirma_hasta_agotar_entregas(
    redis_falso: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    id_mensaje = await _mensaje_pendiente(redis_falso)

    async def falla(_id: str, _datos: dict[bytes, bytes]) -> None:
        raise RuntimeError("Firecrawl no responde")

    monkeypatch.setattr(ingestion_worker, "_procesar_mensaje", falla)

    await ingestion_worker._manejar_mensaje(id_mensaje, {b"payload": _payload()})
    pendientes = await redis_falso.xpending_range(
        settings.REDIS_STREAM_SCRAPING, settings.REDIS_CONSUMER_GROUP, "-", "+", count=10
    )
    assert len(pendientes) == 1, "el mensaje debe seguir pendiente para reentrega"

    await ingestion_worker._manejar_mensaje(id_mensaje, {b"payload": _payload()})
    await ingestion_worker._manejar_mensaje(id_mensaje, {b"payload": _payload()})

    dlq = await redis_falso.xrange(settings.REDIS_STREAM_DLQ, "-", "+")
    assert len(dlq) == 1
    campos = dlq[0][1]
    assert campos[b"entregas"] == b"3"
    assert b"Firecrawl no responde" in campos[b"motivo"]
    assert json.loads(campos[b"payload"].decode("utf-8"))["query"] == "Cleveland OH"

    pendientes = await redis_falso.xpending_range(
        settings.REDIS_STREAM_SCRAPING, settings.REDIS_CONSUMER_GROUP, "-", "+", count=10
    )
    assert pendientes == [], "tras la DLQ se confirma para no atascar el grupo"


async def test_job_cerrado_no_se_reprocesa(
    redis_falso: fakeredis.aioredis.FakeRedis,
    job_store_falso: FakeJobStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_store_falso.job = job_store_falso.job.model_copy(
        update={"status": JobStatus.COMPLETED}
    )

    ejecuciones: list[str] = []

    async def extraer(clave: str, query: str, job_id: UUID | None) -> dict[str, Any]:
        ejecuciones.append(clave)
        return {"plataforma": clave, "status": "exito"}

    monkeypatch.setattr(ingestion_worker, "_extraer_plataforma", extraer)
    id_mensaje = await _mensaje_pendiente(redis_falso)
    await ingestion_worker._procesar_mensaje(id_mensaje, {b"payload": _payload()})

    assert ejecuciones == []
    assert job_store_falso.llamadas == []


async def test_plataforma_omitida_por_idempotencia(
    redis_falso: fakeredis.aioredis.FakeRedis,
    job_store_falso: FakeJobStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import idempotency

    monkeypatch.setattr(idempotency, "redis_client", redis_falso)

    llamadas: list[str] = []

    async def extraer_con_reintentos(url: str, query: str) -> dict[str, Any]:
        llamadas.append(url)
        return {"status": "exito"}

    monkeypatch.setattr(ingestion_worker, "extraer_con_reintentos", extraer_con_reintentos)

    primero = await ingestion_worker._extraer_plataforma("zillow", "Cleveland OH", JOB_ID)
    segundo = await ingestion_worker._extraer_plataforma("zillow", "Cleveland OH", JOB_ID)

    assert primero["status"] == "exito"
    assert segundo["status"] == "omitido_idempotente"
    assert len(llamadas) == 1


async def test_recuperar_pendientes_reclama_y_procesa(
    redis_falso: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not hasattr(redis_falso, "xautoclaim"):
        pytest.skip("fakeredis sin soporte de XAUTOCLAIM")

    await _mensaje_pendiente(redis_falso)
    procesados: list[str] = []

    async def ok(id_mensaje: str, _datos: dict[bytes, bytes]) -> None:
        procesados.append(id_mensaje)

    monkeypatch.setattr(ingestion_worker, "_procesar_mensaje", ok)
    monkeypatch.setattr(settings, "CLAIM_IDLE_MS", 0)

    recuperados = await ingestion_worker._recuperar_pendientes()

    assert recuperados == 1
    assert len(procesados) == 1
