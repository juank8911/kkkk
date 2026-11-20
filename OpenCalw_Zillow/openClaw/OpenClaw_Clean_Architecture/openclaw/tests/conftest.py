"""Configuracion comun de las pruebas.

Las variables se definen antes de importar la aplicacion para que ``Settings``
se construya con un entorno de pruebas y nunca lea secretos reales.
"""

import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("FIRECRAWL_API_KEY", "test-key")
os.environ.setdefault("OPENCLAW_API_KEY", "test-key")
os.environ.setdefault("REDIS_HOST", "127.0.0.1")
os.environ.setdefault("LOG_JSON", "false")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.api.main import app  # noqa: E402


@pytest.fixture()
def client():
    """Cliente que ejecuta el ciclo de vida completo de la aplicacion."""
    with TestClient(app) as test_client:
        yield test_client
