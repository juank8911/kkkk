"""Modelo canónico de propiedad inmobiliaria normalizada (S1-11)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatosPropiedadNormalizados(BaseModel):
    """Representación validada de una propiedad extraída desde markdown."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id_plataforma_origen: str = Field(
        ...,
        min_length=3,
        description="URL o ID único de la propiedad en la plataforma origen",
    )
    ubicacion: str = Field(
        ...,
        min_length=5,
        description="Dirección completa de la propiedad",
    )
    precio_usd: float = Field(
        ...,
        gt=0,
        description="Precio de lista en dólares (debe ser mayor a 0)",
    )
    ingreso_anual_estimado: float = Field(
        default=0.0,
        ge=0,
        description="Renta anual proyectada, si está disponible, si no 0",
    )
    imagenes: list[str] = Field(
        default_factory=list,
        description="Lista de URLs de imágenes extraídas",
    )
    condado: str = Field(
        ...,
        min_length=2,
        description="Condado al que pertenece la propiedad",
    )
    latitud: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitud: float | None = Field(default=None, ge=-180.0, le=180.0)
    descripcion_cruda: str = Field(
        ...,
        min_length=10,
        description="El texto descriptivo original del anuncio",
    )

    @field_validator("imagenes")
    @classmethod
    def _validate_imagenes(cls, urls: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for url in urls:
            candidate = url.strip()
            if not candidate.startswith(("http://", "https://")):
                raise ValueError(f"URL de imagen inválida: {candidate!r}")
            if candidate not in seen:
                seen.add(candidate)
                cleaned.append(candidate)
        return cleaned
