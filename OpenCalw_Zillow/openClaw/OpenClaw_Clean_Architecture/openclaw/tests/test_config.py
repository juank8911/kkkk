"""Pruebas de la configuracion tipada."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_valores_por_defecto_seguros():
    config = Settings(_env_file=None)

    assert config.BUY_MODE == "disabled"
    assert config.FIRECRAWL_API_KEY == ""
    assert config.REDIS_PORT == 6379
    assert config.REDIS_DB == 0


def test_detecta_secretos_faltantes():
    config = Settings(_env_file=None, FIRECRAWL_API_KEY="", OPENCLAW_API_KEY="")

    assert set(config.faltantes()) == {"FIRECRAWL_API_KEY", "OPENCLAW_API_KEY"}


def test_produccion_sin_secretos_falla_al_arrancar():
    config = Settings(_env_file=None, APP_ENV="prod", FIRECRAWL_API_KEY="", OPENCLAW_API_KEY="")

    with pytest.raises(RuntimeError, match="Configuracion incompleta"):
        config.validar_arranque()


def test_dev_sin_secretos_solo_advierte():
    config = Settings(_env_file=None, APP_ENV="dev", FIRECRAWL_API_KEY="", OPENCLAW_API_KEY="")

    faltantes = config.validar_arranque()

    assert "FIRECRAWL_API_KEY" in faltantes


def test_nivel_de_log_invalido_es_rechazado():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, LOG_LEVEL="verbose")


def test_url_de_redis_se_construye_desde_el_entorno():
    config = Settings(_env_file=None, REDIS_HOST="cache", REDIS_PORT=6380, REDIS_DB=2)

    assert config.redis_url == "redis://cache:6380/2"
