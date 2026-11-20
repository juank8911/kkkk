"""Configuracion tipada del servicio OpenClaw.

Reglas del proyecto:
- Ningun secreto tiene un valor por defecto utilizable.
- La validacion se ejecuta al arrancar (fail-fast), no al importar el modulo,
  para que las pruebas puedan construir ``Settings`` con valores propios.
"""

from __future__ import annotations

import hashlib
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
    REDIS_STREAM_DLQ: str = "openclaw_scraping_dlq"
    MAX_ENTREGAS: int = Field(default=3, ge=1, le=20)
    CLAIM_IDLE_MS: int = Field(default=60000, ge=1000)
    IDEMPOTENCIA_TTL_SEGUNDOS: int = Field(default=86400, ge=60)
    LIMITE_CONCURRENCIA_DOMINIO: int = Field(default=2, ge=1, le=10)
    GEOCODING_URL: str = "https://nominatim.openstreetmap.org/search"
    GEOCODING_USER_AGENT: str = "OpenClaw/0.2 (+dev1@openclaw.local)"
    GEOCODING_TTL_SEGUNDOS: int = Field(default=2592000, ge=3600)
    CENSUS_GEOCODING_URL: str = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"
    CENSUS_GEOCODING_BENCHMARK: str = "Public_AR_Current"
    OLLAMA_BASE_URL: str = "https://resuciter.jeikeisolutions.com/ollama"
    OLLAMA_API_KEY: str = ""
    OLLAMA_MODEL: str = "qwen2.5:3b"
    OLLAMA_TIMEOUT_SEGUNDOS: int = Field(default=90, ge=10, le=300)
    OLLAMA_MAX_MARKDOWN_CHARS: int = Field(default=50000, ge=5000, le=200000)
    REDIS_STREAM_DLQ_DECISIONES: str = "openclaw_decisiones_dlq"
    DECISION_MAX_INTENTOS_NODO: int = Field(default=2, ge=1, le=5)
    OFERTA_TRACKER_INTERVAL_SEGUNDOS: int = Field(default=43200, ge=300)
    OFERTA_TRACKER_LOCK_TTL_SEGUNDOS: int = Field(default=900, ge=60)
    OFERTA_TRACKER_MAX_OFERTAS_LOTE: int = Field(default=50, ge=1, le=500)
    REDIS_STREAM_DLQ_OFERTAS: str = "openclaw_ofertas_dlq"
    AUTO_SWEEP_ENABLED: bool = True
    AUTO_SWEEP_RUN_ON_START: bool = True
    AUTO_SWEEP_INTERVAL_SEGUNDOS: int = Field(default=86400, ge=3600)
    AUTO_SWEEP_MARKET_LIMIT: int = Field(default=1, ge=1, le=1)
    AUTO_SWEEP_PLATFORMS: str = "zillow,realtor,redfin"
    AUTO_SWEEP_LOCK_TTL_SEGUNDOS: int = Field(default=21600, ge=300)

    FIRECRAWL_API_KEY: str = ""
    OPENCLAW_API_KEY: str = ""
    OPENCLAW_API_KEY_SHA256: str = ""

    TG_TOKEN: str = ""
    TG_CHAT_ID: str = ""
    TG_WEBHOOK_SECRET: str = ""
    TG_ALLOWED_CHAT_IDS: str = ""
    TG_DEDUPE_TTL_SEGUNDOS: int = Field(default=3600, ge=60, le=86400)

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
    def api_key_hash(self) -> str:
        """Huella SHA-256 esperada de la API key tactica.

        Se prefiere ``OPENCLAW_API_KEY_SHA256`` (permite no guardar la clave en
        claro en el servidor) y, si no existe, se deriva de ``OPENCLAW_API_KEY``.
        """
        precomputada = self.OPENCLAW_API_KEY_SHA256.strip().lower()
        if precomputada:
            return precomputada
        en_claro = self.OPENCLAW_API_KEY.strip()
        if not en_claro:
            return ""
        return hashlib.sha256(en_claro.encode("utf-8")).hexdigest()

    @property
    def chats_autorizados(self) -> frozenset[int]:
        """Allowlist de chats de Telegram; vacia significa 'sin restriccion'."""
        ids: set[int] = set()
        for crudo in self.TG_ALLOWED_CHAT_IDS.replace(";", ",").split(","):
            valor = crudo.strip()
            if not valor:
                continue
            try:
                ids.add(int(valor))
            except ValueError:
                continue
        return frozenset(ids)

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
