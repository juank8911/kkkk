"""Pruebas de la autenticacion por API key tactica (S2-01 / ADR-04)."""

from __future__ import annotations

import uuid

import pytest

from app.api.routes import ingestion
from app.core.config import settings
from app.core.security import CABECERA_API_KEY, hash_api_key

from tests.conftest import CLAVE_PRUEBA


class FakeJobStore:
    """Almacen en memoria para no depender de Redis."""

    def __init__(self) -> None:
        self.jobs: dict[uuid.UUID, object] = {}

    async def guardar(self, job) -> None:
        self.jobs[job.job_id] = job

    async def obtener(self, job_id):
        return self.jobs.get(job_id)

    async def marcar(self, job_id, estado, *, progress=None, error=None) -> None:
        return None


@pytest.fixture()
def almacen(monkeypatch: pytest.MonkeyPatch) -> FakeJobStore:
    falso = FakeJobStore()
    monkeypatch.setattr(ingestion, "job_store", falso)
    monkeypatch.setattr(ingestion.redis_client, "xadd", _xadd_ok)
    return falso


async def _xadd_ok(nombre: str, campos: dict) -> bytes:
    return b"1-0"


def test_hash_es_estable_y_distingue_claves() -> None:
    assert hash_api_key("abc") == hash_api_key(" abc ")
    assert hash_api_key("abc") != hash_api_key("abd")
    assert len(hash_api_key("abc")) == 64


def test_huella_precomputada_tiene_prioridad(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "OPENCLAW_API_KEY", "otra-clave")
    monkeypatch.setattr(settings, "OPENCLAW_API_KEY_SHA256", "F" * 64)
    assert settings.api_key_hash == "f" * 64


def test_sin_cabecera_devuelve_401(anon_client) -> None:
    respuesta = anon_client.post(
        "/v1/openclaw/barridos",
        json={"query": "Miami", "plataformas": ["zillow"]},
    )

    assert respuesta.status_code == 401
    assert respuesta.json()["code"] == "unauthorized"
    assert respuesta.headers["www-authenticate"] == CABECERA_API_KEY


def test_cabecera_vacia_devuelve_401(anon_client) -> None:
    respuesta = anon_client.get(
        f"/v1/openclaw/barridos/{uuid.uuid4()}",
        headers={CABECERA_API_KEY: "   "},
    )

    assert respuesta.status_code == 401


def test_clave_invalida_devuelve_403(anon_client) -> None:
    respuesta = anon_client.post(
        "/v1/openclaw/barridos",
        json={"query": "Miami", "plataformas": ["zillow"]},
        headers={CABECERA_API_KEY: "clave-equivocada"},
    )

    assert respuesta.status_code == 403
    assert respuesta.json()["code"] == "forbidden"


def test_clave_valida_permite_el_barrido(client, almacen) -> None:
    respuesta = client.post(
        "/v1/openclaw/barridos",
        json={"query": "Miami", "plataformas": ["zillow"]},
    )

    assert respuesta.status_code == 202
    assert len(almacen.jobs) == 1


def test_sin_clave_configurada_devuelve_503(anon_client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "OPENCLAW_API_KEY", "")
    monkeypatch.setattr(settings, "OPENCLAW_API_KEY_SHA256", "")

    respuesta = anon_client.post(
        "/v1/openclaw/barridos",
        json={"query": "Miami", "plataformas": ["zillow"]},
    )

    assert respuesta.status_code == 503
    assert respuesta.json()["code"] == "service_unavailable"


def test_health_sigue_siendo_publico(anon_client) -> None:
    respuesta = anon_client.get("/health")

    assert respuesta.status_code in {200, 503}


def test_clave_configurada_en_claro_coincide() -> None:
    assert settings.api_key_hash == hash_api_key(CLAVE_PRUEBA)
