"""Configuracion tipada del servicio OpenClaw.

Reglas del proyecto:
- Ningun secreto tiene un valor por defecto utilizable.
- La validacion se ejecuta al arrancar (fail-fast), no al importar el modulo,
  para que las pruebas puedan construir ``Settings`` con valores propios.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Entorno = Literal["dev", "test", "prod"]
ModoCompra = Literal["disabled", "sandbox"]

NIVELES_LOG = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})


class Settings(BaseSettings):
    """Configuracion del servicio, leida de entorno y de ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_ENV: Entorno = "dev"
    APP_NAME: str = "OpenClaw API Gateway"
    APP_VERSION: str = "0.2.0"
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True

    REDIS_HOST: str = "redis-cache"
    REDIS_PORT: int = Field(default=6379, ge=1, le=65535)
    REDIS_DB: int = Field(default=0, ge=0)
    REDIS_STREAM_SCRAPING: str = "openclaw_scraping_queue"
    REDIS_CONSUMER_GROUP: str = "scrapers"
    REDIS_CONSUMER_NAME: str = "worker-1"

    FIRECRAWL_API_KEY: str = ""
    OPENCLAW_API_KEY: str = ""

    TG_TOKEN: str = ""
    TG_CHAT_ID: str = ""

    UAO_SCALPING_API: str = "http://uao-scalping-core:8001/v1/liquidity"
    BUY_MODE: ModoCompra = "disabled"

    DATABASE_URL: str = ""

    @field_validator("LOG_LEVEL")
    @classmethod
    def _normalizar_nivel(cls, valor: str) -> str:
        nivel = valor.upper()
        if nivel not in NIVELES_LOG:
            raise ValueError(f"LOG_LEVEL invalido: {valor!r}. Valores: {sorted(NIVELES_LOG)}")
        return nivel

    @property
    def redis_url(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def secretos_obligatorios(self) -> tuple[str, ...]:
        return ("FIRECRAWL_API_KEY", "OPENCLAW_API_KEY")

    def faltantes(self) -> list[str]:
        """Devuelve los secretos obligatorios que estan vacios."""
        return [
            nombre
            for nombre in self.secretos_obligatorios
            if not str(getattr(self, nombre, "")).strip()
        ]

    def validar_arranque(self) -> list[str]:
        """Valida el entorno. En produccion falla; en dev y test solo advierte."""
        faltantes = self.faltantes()
        if faltantes and self.APP_ENV == "prod":
            detalle = ", ".join(faltantes)
            raise RuntimeError(f"Configuracion incompleta para produccion: falta {detalle}")
        return faltantes


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve la configuracion como singleton por proceso."""
    return Settings()


settings = get_settings()
