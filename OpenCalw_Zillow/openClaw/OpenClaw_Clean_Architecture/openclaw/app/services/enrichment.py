"""Enriquecimiento de propiedades desde markdown: precio, superficie, año y HOA (S3-05).

Extrae los cuatro atributos clave o marca explícitamente su ausencia en
`campos_ausentes` para trazabilidad en el pipeline de decisión.
"""

from __future__ import annotations

import re
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from app.services.normalizer import _extract_price_usd

_SQFT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:living\s+area|square\s+feet|sq\.?\s*ft\.?|superficie|area|size)\s*:?\s*"
    r"([\d,]+(?:\.\d+)?)\s*(?:sq\.?\s*ft\.?|sqft|sf|ft2|ft²)?|"
    r"\b([\d,]+(?:\.\d+)?)\s*(?:sq\.?\s*ft\.?|sqft|ft2|ft²)\b"
)
_SQM_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b([\d,]+(?:\.\d+)?)\s*(?:m2|m²|sq\.?\s*m\.?|sqm)\b"
)
_YEAR_LABEL_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:year\s+built|built\s+in|built|a[ñn]o(?:\s+de\s+construcci[oó]n)?|constructed)\s*:?\s*"
    r"(1[89]\d{2}|20[0-2]\d)\b"
)
_HOA_NONE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:(?:hoa(?:\s+fee|\s+dues)?|cuota\s+hoa)\s*:?\s*(?:none|no\s+hoa|n/a|\$?\s*0(?:\.00)?(?:/mo)?\b)|\b(?:no|sin)\s+hoa\b)"
)
_HOA_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:hoa(?:\s+fee|\s+dues)?|cuota\s+hoa)\s*:?\s*\$\s*([\d,]+(?:\.\d+)?)"
    r"(?:\s*/\s*(mo|month|mes|yr|year|annual|a[ñn]o|qtr|quarter))?"
)
_HOA_INLINE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\$\s*([\d,]+(?:\.\d+)?)\s*/\s*(mo|month|yr|year)\s+hoa\b"
)

_SQM_TO_SQFT: Final[float] = 10.7639


class PropertyEnrichment(BaseModel):
    """Resultado estructurado del enriquecimiento de los 4 campos de S3-05."""

    model_config = ConfigDict(extra="forbid")

    precio_usd: float | None = Field(default=None, gt=0)
    superficie_sqft: float | None = Field(default=None, gt=0)
    anio_construccion: int | None = Field(default=None, ge=1800, le=2030)
    hoa_mensual_usd: float | None = Field(default=None, ge=0)
    campos_ausentes: list[str] = Field(default_factory=list)

    @property
    def completo(self) -> bool:
        return len(self.campos_ausentes) == 0


def extraer_superficie_sqft(markdown: str) -> float | None:
    """Extrae la superficie habitable en pies cuadrados (convierte m² si aplica)."""
    if not markdown:
        return None

    for match in _SQFT_RE.finditer(markdown):
        raw = match.group(1) or match.group(2)
        if not raw:
            continue
        val = float(raw.replace(",", ""))
        if 150.0 <= val <= 200_000.0:
            return round(val, 2)

    sqm_match = _SQM_RE.search(markdown)
    if sqm_match:
        sqm = float(sqm_match.group(1).replace(",", ""))
        if 15.0 <= sqm <= 20_000.0:
            return round(sqm * _SQM_TO_SQFT, 2)

    return None


def extraer_anio_construccion(markdown: str) -> int | None:
    """Extrae el año de construcción válido [1800, 2030]."""
    if not markdown:
        return None
    match = _YEAR_LABEL_RE.search(markdown)
    if not match:
        return None
    year = int(match.group(1))
    return year if 1800 <= year <= 2030 else None


def extraer_hoa_mensual_usd(markdown: str) -> float | None:
    """Extrae la cuota HOA normalizada a USD mensuales (0.0 si declara 'No HOA')."""
    if not markdown:
        return None

    if _HOA_NONE_RE.search(markdown):
        return 0.0

    for pattern in (_HOA_RE, _HOA_INLINE_RE):
        match = pattern.search(markdown)
        if not match:
            continue
        amount = float(match.group(1).replace(",", ""))
        period = (match.group(2) or "mo").lower()
        if period in ("yr", "year", "annual", "año", "ano"):
            amount = amount / 12.0
        elif period in ("qtr", "quarter"):
            amount = amount / 3.0
        if 0.0 <= amount <= 25_000.0:
            return round(amount, 2)

    return None


def enriquecer_propiedad_desde_markdown(markdown: str) -> PropertyEnrichment:
    """Extrae precio, superficie, año de construcción y HOA, marcando los ausentes."""
    text = markdown or ""
    precio = _extract_price_usd(text)
    superficie = extraer_superficie_sqft(text)
    anio = extraer_anio_construccion(text)
    hoa = extraer_hoa_mensual_usd(text)

    ausentes: list[str] = []
    if precio is None:
        ausentes.append("precio_usd")
    if superficie is None:
        ausentes.append("superficie_sqft")
    if anio is None:
        ausentes.append("anio_construccion")
    if hoa is None:
        ausentes.append("hoa_mensual_usd")

    return PropertyEnrichment(
        precio_usd=precio,
        superficie_sqft=superficie,
        anio_construccion=anio,
        hoa_mensual_usd=hoa,
        campos_ausentes=ausentes,
    )
