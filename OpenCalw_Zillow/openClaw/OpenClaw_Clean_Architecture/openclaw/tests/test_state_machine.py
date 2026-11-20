"""Pruebas de la maquina de estados del job (S3-03)."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from app.models.contracts import JobResponse, JobStatus
from app.workers import state_machine
from app.workers.state_machine import (
    TransicionInvalida,
    aplicar_transicion,
    es_terminal,
    transicion_valida,
)

JOB_ID = uuid4()


class FakeJobStore:
    """Almacen en memoria con la misma interfaz que ``JobStore``."""

    def __init__(self, estado: JobStatus = JobStatus.QUEUED) -> None:
        self.job: JobResponse | None = JobResponse(
            job_id=JOB_ID,
            status=estado,
            query="Cleveland OH",
            plataformas=["zillow"],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.llamadas: list[JobStatus] = []

    async def obtener(self, job_id: UUID) -> JobResponse | None:
        if self.job is None or job_id != JOB_ID:
            return None
        return self.job

    async def marcar(
        self,
        job_id: UUID,
        estado: JobStatus,
        *,
        progress: dict[str, str] | None = None,
        error: str | None = None,
    ) -> None:
        self.llamadas.append(estado)
        if self.job is not None:
            cambios: dict[str, object] = {"status": estado, "updated_at": datetime.now(timezone.utc)}
            if progress is not None:
                cambios["progress"] = progress
            self.job = self.job.model_copy(update=cambios)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeJobStore:
    falso = FakeJobStore()
    monkeypatch.setattr(state_machine, "job_store", falso)
    return falso


def test_transiciones_declaradas() -> None:
    assert transicion_valida(JobStatus.QUEUED, JobStatus.RUNNING)
    assert transicion_valida(JobStatus.RUNNING, JobStatus.PARTIAL)
    assert transicion_valida(JobStatus.FAILED, JobStatus.QUEUED)
    assert not transicion_valida(JobStatus.QUEUED, JobStatus.COMPLETED)
    assert not transicion_valida(JobStatus.COMPLETED, JobStatus.RUNNING)


def test_estados_terminales() -> None:
    assert es_terminal(JobStatus.COMPLETED)
    assert es_terminal(JobStatus.PARTIAL)
    assert es_terminal(JobStatus.FAILED)
    assert not es_terminal(JobStatus.RUNNING)


async def test_aplicar_transicion_valida(store: FakeJobStore) -> None:
    actualizado = await aplicar_transicion(JOB_ID, JobStatus.RUNNING)
    assert actualizado is not None
    assert actualizado.status is JobStatus.RUNNING
    assert store.llamadas == [JobStatus.RUNNING]


async def test_transicion_ilegal_no_toca_el_estado(store: FakeJobStore) -> None:
    with pytest.raises(TransicionInvalida):
        await aplicar_transicion(JOB_ID, JobStatus.COMPLETED)
    assert store.llamadas == []
    assert store.job is not None
    assert store.job.status is JobStatus.QUEUED


async def test_job_inexistente_devuelve_none(store: FakeJobStore) -> None:
    assert await aplicar_transicion(uuid4(), JobStatus.RUNNING) is None


async def test_reintento_tras_fallo_es_legal(store: FakeJobStore) -> None:
    await aplicar_transicion(JOB_ID, JobStatus.RUNNING)
    await aplicar_transicion(JOB_ID, JobStatus.FAILED)
    await aplicar_transicion(JOB_ID, JobStatus.QUEUED)
    assert store.job is not None
    assert store.job.status is JobStatus.QUEUED
