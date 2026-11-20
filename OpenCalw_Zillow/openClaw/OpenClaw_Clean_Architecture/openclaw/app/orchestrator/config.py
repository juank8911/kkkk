"""Configuración parametrizada para los nodos del orquestador (S4-04 / S4-05 / S4-07)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FinancialConfig(BaseSettings):
    """Tasas y umbrales financieros cargables desde variables de entorno sin tocar código."""

    model_config = SettingsConfigDict(
        env_prefix="OPENCLAW_FIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    vacancy_rate: float = Field(default=0.08, ge=0.0, lt=1.0)
    maintenance_rate: float = Field(default=0.10, ge=0.0, lt=1.0)
    property_tax_rate: float = Field(default=0.02, ge=0.0, lt=1.0)
    insurance_annual_usd: float = Field(default=1200.0, ge=0.0)
    min_cap_rate_pct: float = Field(default=8.0, ge=0.0, le=100.0)
    max_price_usd: float = Field(default=150_000.0, gt=0.0)
    rules_version: str = Field(default="fin-v1.0")


class VisionConfig(BaseSettings):
    """Parámetros del nodo de visión multimodal (S4-05)."""

    model_config = SettingsConfigDict(
        env_prefix="OPENCLAW_VISION_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    model_name: str = Field(default="gpt-4o-mini")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_images: int = Field(default=6, ge=1, le=20)
    timeout_seconds: float = Field(default=20.0, gt=0.0)


class AlertConfig(BaseSettings):
    """Parámetros de plantillas, enlaces y deduplicación de alertas Telegram (S4-07)."""

    model_config = SettingsConfigDict(
        env_prefix="OPENCLAW_ALERT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    ficha_base_url: str = Field(default="https://openclaw.internal/propiedades")
    dedupe_ttl_seconds: int = Field(default=7 * 24 * 3600, ge=60)
