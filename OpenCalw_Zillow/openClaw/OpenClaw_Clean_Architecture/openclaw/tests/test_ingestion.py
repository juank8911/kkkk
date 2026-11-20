"""Pruebas de los endpoints de ingesta (S1-03 y S1-04)."""

import uuid
from typing import Any

import pytest

from app.api.routes import ingestion


class FakeJobStore:
    """Dobles en memoria para no depender de Redis en las pruebas."""

    def __init__(self) -> None:
        self.jobs: dict[uuid.UUID, Any] = {}

    async def guardar(self, job) -> None:
        self.jobs[job.job_id] = job

    async def obtener(self, job_id):
        return self.jobs.get(job_id)

    async def marcar(self, job_id, estado, *, progress=None, error=None) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return
        self.jobs[job_id] = job.model_copy(update={"status": estado, "error": error})


@pytest.fixture()
def almacen(monkeypatch):
    falso = FakeJobStore()
    monkeypatch.setattr(ingestion, "job_store", falso)
    return falso


@pytest.fixture()
def broker(monkeypatch):
    llamadas: list[tuple[str, dict]] = []

    async def xadd(nombre, campos):
        llamadas.append((nombre, campos))
        return b"1-0"

    monkeypatch.setattr(ingestion.redis_client, "xadd", xadd)
    return llamadas


def test_crear_barrido_devuelve_202_y_encola(client, almacen, broker):
    respuesta = client.post(
        "/v1/openclaw/barridos",
        json={"query": "Miami, FL", "plataformas": ["zillow", "redfin"]},
    )

    assert respuesta.status_code == 202
    cuerpo = respuesta.json()
    assert cuerpo["status"] == "queued"
    assert cuerpo["plataformas"] == ["zillow", "redfin"]
    uuid.UUID(cuerpo["job_id"])

    assert len(broker) == 1
    assert len(almacen.jobs) == 1


def test_el_mensaje_encolado_lleva_el_job_id(client, almacen, broker):
    respuesta = client.post(
        "/v1/openclaw/barridos",
        json={"query": "Austin", "plataformas": ["zillow"]},
    )

    _, campos = broker[0]
    assert respuesta.json()["job_id"] in campos["payload"]


def test_payload_sin_plataformas_devuelve_422(client, almacen, broker):
    respuesta = client.post("/v1/openclaw/barridos", json={"query": "Miami"})

    assert respuesta.status_code == 422
    assert respuesta.json()["code"] == "validation_error"


def test_plataforma_fuera_del_catalogo_devuelve_422(client, almacen, broker):
    respuesta = client.post(
        "/v1/openclaw/barridos",
        json={"query": "Miami", "plataformas": ["idealista"]},
    )

    assert respuesta.status_code == 422
    assert respuesta.json()["code"] == "validation_error"


def test_consulta_de_job_existente(client, almacen, broker):
    creado = client.post(
        "/v1/openclaw/barridos",
        json={"query": "Tampa", "plataformas": ["zillow"]},
    ).json()

    respuesta = client.get(f"/v1/openclaw/barridos/{creado['job_id']}")

    assert respuesta.status_code == 200
    assert respuesta.json()["query"] == "Tampa"


def test_consulta_de_job_inexistente_devuelve_404(client, almacen, broker):
    respuesta = client.get(f"/v1/openclaw/barridos/{uuid.uuid4()}")

    assert respuesta.status_code == 404
    assert respuesta.json()["code"] == "not_found"


def test_broker_caido_devuelve_503_y_marca_fallido(client, almacen, monkeypatch):
    async def xadd_falla(nombre, campos):
        raise ConnectionError("broker caido")

    monkeypatch.setattr(ingestion.redis_client, "xadd", xadd_falla)

    respuesta = client.post(
        "/v1/openclaw/barridos",
        json={"query": "Orlando", "plataformas": ["zillow"]},
    )

    assert respuesta.status_code == 503
    assert respuesta.json()["code"] == "service_unavailable"
    assert next(iter(almacen.jobs.values())).status.value == "failed"
