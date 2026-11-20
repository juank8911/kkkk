"""Deduplicación de propiedades por dirección normalizada y similitud (S3-07).

Consolida múltiples anuncios de la misma vivienda provenientes de distintas
plataformas (ej. Zillow y Realtor) en una sola propiedad canónica conservando
todas sus fuentes, imágenes combinadas y el mejor nivel de completitud.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from app.models.property import DatosPropiedadNormalizados

_STREET_ABBREVS: Final[dict[str, str]] = {
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "road": "rd",
    "lane": "ln",
    "court": "ct",
    "circle": "cir",
    "place": "pl",
    "parkway": "pkwy",
    "terrace": "ter",
    "trail": "trl",
    "highway": "hwy",
    "way": "way",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "northeast": "ne",
    "northwest": "nw",
    "southeast": "se",
    "southwest": "sw",
    "apartment": "apt",
    "suite": "ste",
    "unit": "unit",
}

_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"[^\w\s]")
_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_ZIP_RE: Final[re.Pattern[str]] = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_HOUSE_NUM_RE: Final[re.Pattern[str]] = re.compile(r"^(\d+[a-z]?)\b")


class DeduplicatedProperty(BaseModel):
    """Propiedad consolidada tras deduplicación multiplataforma."""

    model_config = ConfigDict(extra="forbid")

    dedup_key: str
    direccion_normalizada: str
    propiedad: DatosPropiedadNormalizados
    fuentes: list[str] = Field(default_factory=list)
    duplicados_fusionados: int = Field(default=0, ge=0)


def normalizar_direccion(ubicacion: str) -> str:
    """Normaliza una dirección postal a forma canónica comparable."""
    ascii_text = (
        unicodedata.normalize("NFKD", ubicacion).encode("ascii", "ignore").decode("ascii").lower()
    )
    cleaned = _PUNCT_RE.sub(" ", ascii_text)
    cleaned = _SPACE_RE.sub(" ", cleaned).strip()
    tokens = [_STREET_ABBREVS.get(tok, tok) for tok in cleaned.split()]
    return " ".join(tokens)


def generar_clave_deduplicacion(ubicacion: str, condado: str) -> str:
    """Genera una huella SHA-256 estable basada en dirección normalizada + condado."""
    norm_addr = normalizar_direccion(ubicacion)
    norm_county = normalizar_direccion(condado)
    raw = f"{norm_addr}|{norm_county}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def calcular_similitud_direcciones(addr_a: str, addr_b: str) -> float:
    """Calcula similitud [0.0, 1.0] entre dos direcciones normalizadas."""
    norm_a = normalizar_direccion(addr_a)
    norm_b = normalizar_direccion(addr_b)
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 1.0

    # Si ambos tienen número de casa y difieren, son inmuebles distintos
    num_a = _HOUSE_NUM_RE.match(norm_a)
    num_b = _HOUSE_NUM_RE.match(norm_b)
    if num_a and num_b and num_a.group(1) != num_b.group(1):
        return 0.0

    # Si ambos incluyen ZIP de 5 dígitos y difieren, no son la misma casa
    zip_a = _ZIP_RE.search(norm_a)
    zip_b = _ZIP_RE.search(norm_b)
    if zip_a and zip_b and zip_a.group(1) != zip_b.group(1):
        return 0.0

    return round(SequenceMatcher(None, norm_a, norm_b).ratio(), 4)


def son_misma_propiedad(
    a: DatosPropiedadNormalizados,
    b: DatosPropiedadNormalizados,
    *,
    umbral_similitud: float = 0.88,
    tolerancia_precio: float = 0.15,
) -> bool:
    """Determina si dos anuncios corresponden al mismo inmueble físico."""
    sim = calcular_similitud_direcciones(a.ubicacion, b.ubicacion)
    if sim < umbral_similitud:
        return False

    # Si los precios divergen dramáticamente (>15%) con similitud no exacta, evita falsos positivos
    if sim < 0.98:
        max_p = max(a.precio_usd, b.precio_usd)
        min_p = min(a.precio_usd, b.precio_usd)
        if max_p > 0 and ((max_p - min_p) / max_p) > tolerancia_precio:
            return False

    return True


def _fusionar_propiedades(
    base: DatosPropiedadNormalizados,
    entrante: DatosPropiedadNormalizados,
) -> DatosPropiedadNormalizados:
    """Fusiona dos registros del mismo inmueble reteniendo la información más completa."""
    imagenes = list(dict.fromkeys([*base.imagenes, *entrante.imagenes]))
    ingreso = (
        base.ingreso_anual_estimado
        if base.ingreso_anual_estimado > 0
        else entrante.ingreso_anual_estimado
    )
    lat = base.latitud if base.latitud is not None else entrante.latitud
    lng = base.longitud if base.longitud is not None else entrante.longitud
    descripcion = (
        base.descripcion_cruda
        if len(base.descripcion_cruda) >= len(entrante.descripcion_cruda)
        else entrante.descripcion_cruda
    )
    return base.model_copy(
        update={
            "ingreso_anual_estimado": ingreso,
            "imagenes": imagenes,
            "latitud": lat,
            "longitud": lng,
            "descripcion_cruda": descripcion,
        }
    )


def deduplicar_propiedades(
    propiedades: Sequence[DatosPropiedadNormalizados],
    *,
    umbral_similitud: float = 0.88,
) -> list[DeduplicatedProperty]:
    """Deduplica una lista de propiedades normalizadas conservando orden determinista."""
    consolidadas: list[DeduplicatedProperty] = []

    for prop in propiedades:
        matched_idx: int | None = None
        for idx, item in enumerate(consolidadas):
            if son_misma_propiedad(
                item.propiedad,
                prop,
                umbral_similitud=umbral_similitud,
            ):
                matched_idx = idx
                break

        if matched_idx is None:
            consolidadas.append(
                DeduplicatedProperty(
                    dedup_key=generar_clave_deduplicacion(prop.ubicacion, prop.condado),
                    direccion_normalizada=normalizar_direccion(prop.ubicacion),
                    propiedad=prop,
                    fuentes=[prop.id_plataforma_origen],
                    duplicados_fusionados=0,
                )
            )
        else:
            existing = consolidadas[matched_idx]
            merged = _fusionar_propiedades(existing.propiedad, prop)
            fuentes = list(dict.fromkeys([*existing.fuentes, prop.id_plataforma_origen]))
            consolidadas[matched_idx] = existing.model_copy(
                update={
                    "propiedad": merged,
                    "fuentes": fuentes,
                    "duplicados_fusionados": existing.duplicados_fusionados + 1,
                }
            )

    return consolidadas
