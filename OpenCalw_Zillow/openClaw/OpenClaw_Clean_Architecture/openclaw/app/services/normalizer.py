"""Normalizador de markdown inmobiliario a DatosPropiedadNormalizados (S1-11).

Convierte bloques markdown extraídos de cualquiera de las 9 plataformas del catálogo
en instancias validadas de `DatosPropiedadNormalizados`, rechazando explícitamente
anuncios incompletos o con campos críticos inválidos.
"""

from __future__ import annotations

import re
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.models.property import DatosPropiedadNormalizados

_PLACEHOLDER_VALUES: Final[frozenset[str]] = frozenset(
    {
        "",
        "n/a",
        "na",
        "none",
        "null",
        "tbd",
        "unknown",
        "undisclosed",
        "address not disclosed",
        "contact for price",
        "call for price",
        "price upon request",
        "not available",
        "sin datos",
        "no disponible",
    }
)

_IGNORED_IMAGE_PATTERNS: Final[tuple[str, ...]] = (
    "favicon",
    "sprite",
    "pixel",
    "tracking",
    "/logo",
    "logo.",
    "icon",
    ".svg",
)

# Etiquetas explícitas
_LABEL_PRICE_RE = re.compile(
    r"(?im)^[\s\-*>#]*(?:list\s+price|asking\s+price|sale\s+price|price|precio)\s*:\s*(.+)$"
)
_STANDALONE_PRICE_RE = re.compile(
    r"(?m)^[\s\-*>#]*(\$\s*[\d,]+(?:\.\d+)?\s*[KkMm]?)(?:\s*(?:\||—|-|USD|\*\*|$))"
)
_CURRENCY_AMOUNT_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)\s*([KkMm])?")

_LABEL_ANNUAL_RENT_RE = re.compile(
    r"(?im)^[\s\-*>#]*(?:annual\s+(?:rent|income)|est\.?\s*annual\s+(?:rent|income)|"
    r"gross\s+(?:annual\s+)?income|ingreso\s+anual(?:\s+estimado)?)\s*:\s*(.+)$"
)
_LABEL_MONTHLY_RENT_RE = re.compile(
    r"(?im)^[\s\-*>#]*(?:est\.?\s*rent|estimated\s+rent|rent\s+zestimate|"
    r"monthly\s+rent|renta\s+mensual|rent)\s*:\s*(.+)$"
)
_INLINE_MONTHLY_RENT_RE = re.compile(
    r"(?i)(?:est\.?\s*rent|rent\s+zestimate|estimated\s+rent|monthly\s+rent)"
    r"[^\n$]{0,20}\$\s*([\d,]+(?:\.\d+)?)\s*(?:/\s*mo|per\s+month|/month)?"
)

_LABEL_ADDRESS_RE = re.compile(
    r"(?im)^[\s\-*>#]*(?:property\s+address|address|ubicaci[oó]n|location)\s*:\s*(.+)$"
)
_HEADING_RE = re.compile(r"(?m)^#{1,3}\s+(.+)$")
_US_ADDRESS_RE = re.compile(
    r"\b(\d{1,6}\s+[A-Za-z0-9.'\- ]{2,60}?"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Ct|Court|"
    r"Way|Pl|Place|Cir|Circle|Pkwy|Parkway|Ter|Terrace|Hwy|Highway|Trl|Trail|Loop)\b"
    r"[^,\n]{0,25},\s*[A-Za-z .'-]{2,40},\s*[A-Z]{2}(?:\s+\d{5}(?:-\d{4})?)?)"
)

_LABEL_COUNTY_RE = re.compile(r"(?im)^[\s\-*>#]*(?:county|condado)\s*:\s*(.+)$")
_INLINE_COUNTY_RE = re.compile(
    r"(?:\bin\s+|,\s*)([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,2})\s+County\b"
    r"(?!\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Ct|Court|Way|Pl|Place|Cir|Pkwy|Ter|Hwy|Trl)\b)"
)

_LABEL_URL_RE = re.compile(
    r"(?im)^[\s\-*>#]*(?:listing\s+url|source\s+url|url|source|id|listing\s+id)\s*:\s*(.+)$"
)
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\((https?://[^\s)]+)\)")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((https?://[^\s)]+)\)")
_RAW_IMAGE_URL_RE = re.compile(
    r"(https?://[^\s)\"'>]+\.(?:jpg|jpeg|png|webp)(?:\?[^\s)\"'>]*)?)",
    re.IGNORECASE,
)

_COORDS_PAIR_RE = re.compile(
    r"(?i)(?:coordinates|coordenadas|coords|lat/lng|geo)\s*:\s*"
    r"(-?\d{1,2}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)"
)
_LAT_RE = re.compile(r"(?i)(?:latitude|latitud|lat)\s*:\s*(-?\d{1,2}\.\d+)")
_LNG_RE = re.compile(r"(?i)(?:longitude|longitud|lng|lon)\s*:\s*(-?\d{1,3}\.\d+)")

_DESCRIPTION_SECTION_RE = re.compile(
    r"(?is)(?:^|\n)(?:#{1,3}\s*)?(?:description|descripci[oó]n|overview|about\s+this\s+home)\s*:?\s*\n+(.+)"
)

_LISTING_SPLIT_RE = re.compile(r"\n\s*---+\s*\n")


class RejectedListing(BaseModel):
    """Registro auditable de un anuncio rechazado por datos incompletos o inválidos."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., ge=0)
    reasons: list[str]
    source_hint: str | None = None
    raw_snippet: str = ""


class NormalizationBatch(BaseModel):
    """Resultado de normalizar un documento markdown con uno o varios anuncios."""

    model_config = ConfigDict(extra="forbid")

    platform: str
    valid: list[DatosPropiedadNormalizados] = Field(default_factory=list)
    rejected: list[RejectedListing] = Field(default_factory=list)


class NormalizationError(ValueError):
    """Se lanza cuando un anuncio individual no cumple los campos obligatorios."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = list(reasons)
        super().__init__(
            "Anuncio rechazado por datos incompletos o inválidos: " + "; ".join(self.reasons)
        )


class MarkdownPropertyNormalizer:
    """Extrae y valida instancias de `DatosPropiedadNormalizados` desde markdown."""

    def normalize_batch(
        self,
        markdown: str,
        *,
        platform: str,
        url_origen: str | None = None,
    ) -> NormalizationBatch:
        """Normaliza un markdown con uno o múltiples anuncios separados por `---`."""
        cleaned = (markdown or "").strip()
        if not cleaned:
            return NormalizationBatch(
                platform=platform,
                valid=[],
                rejected=[
                    RejectedListing(
                        index=0,
                        reasons=["empty_markdown"],
                        source_hint=url_origen,
                        raw_snippet="",
                    )
                ],
            )

        blocks = [block.strip() for block in _LISTING_SPLIT_RE.split(cleaned) if block.strip()]
        valid_items: list[DatosPropiedadNormalizados] = []
        rejected_items: list[RejectedListing] = []

        for idx, block in enumerate(blocks):
            prop, reasons, source_hint = self._parse_block(
                block,
                platform=platform,
                url_origen=url_origen,
            )
            if prop is not None:
                valid_items.append(prop)
            else:
                rejected_items.append(
                    RejectedListing(
                        index=idx,
                        reasons=reasons,
                        source_hint=source_hint,
                        raw_snippet=block[:240],
                    )
                )

        return NormalizationBatch(
            platform=platform,
            valid=valid_items,
            rejected=rejected_items,
        )

    def normalize_single(
        self,
        markdown: str,
        *,
        platform: str,
        url_origen: str | None = None,
    ) -> DatosPropiedadNormalizados:
        """Normaliza un único anuncio o lanza `NormalizationError` si está incompleto."""
        cleaned = (markdown or "").strip()
        if not cleaned:
            raise NormalizationError(["empty_markdown"])

        prop, reasons, _ = self._parse_block(
            cleaned,
            platform=platform,
            url_origen=url_origen,
        )
        if prop is None:
            raise NormalizationError(reasons)
        return prop

    def _parse_block(
        self,
        block: str,
        *,
        platform: str,
        url_origen: str | None,
    ) -> tuple[DatosPropiedadNormalizados | None, list[str], str | None]:
        reasons: list[str] = []

        source_id = _extract_source_id(block, platform=platform, fallback_url=url_origen)
        if not source_id:
            reasons.append("missing_source_id")

        ubicacion = _extract_address(block)
        if not ubicacion:
            reasons.append("missing_location")

        precio_usd = _extract_price_usd(block)
        if precio_usd is None or precio_usd <= 0:
            reasons.append("missing_or_invalid_price")

        condado = _extract_county(block)
        if not condado:
            reasons.append("missing_county")

        ingreso_anual = _extract_annual_income(block)
        imagenes = _extract_images(block)
        latitud, longitud = _extract_coordinates(block)
        descripcion_cruda = _extract_description(block)
        if not descripcion_cruda or len(descripcion_cruda) < 10:
            reasons.append("missing_description")

        if reasons:
            return None, reasons, source_id

        assert source_id is not None
        assert ubicacion is not None
        assert precio_usd is not None
        assert condado is not None
        assert descripcion_cruda is not None

        try:
            propiedad = DatosPropiedadNormalizados(
                id_plataforma_origen=source_id,
                ubicacion=ubicacion,
                precio_usd=precio_usd,
                ingreso_anual_estimado=ingreso_anual,
                imagenes=imagenes,
                condado=condado,
                latitud=latitud,
                longitud=longitud,
                descripcion_cruda=descripcion_cruda,
            )
            return propiedad, [], source_id
        except ValidationError as exc:
            pydantic_reasons = [
                f"validation_error:{'.'.join(str(p) for p in err['loc'])}" for err in exc.errors()
            ]
            return None, pydantic_reasons, source_id


def _clean_md_value(raw: str) -> str:
    cleaned = re.sub(r"[*_`]+", "", raw).strip()
    cleaned = cleaned.strip("|").strip()
    return cleaned


def _is_placeholder(value: str | None) -> bool:
    if not value:
        return True
    return value.strip().lower() in _PLACEHOLDER_VALUES


def _parse_money_token(text: str) -> float | None:
    cleaned = _clean_md_value(text)
    if _is_placeholder(cleaned):
        return None
    if re.search(r"(?i)\b(tbd|contact|call|auction|upon\s+request)\b", cleaned):
        return None
    if re.search(r"-\s*\$", cleaned):
        return None

    match = _CURRENCY_AMOUNT_RE.search(cleaned)
    if not match:
        return None
    raw_num = match.group(1).replace(",", "")
    suffix = (match.group(2) or "").lower()
    try:
        amount = float(raw_num)
    except ValueError:
        return None
    if suffix == "k":
        amount *= 1_000.0
    elif suffix == "m":
        amount *= 1_000_000.0
    return round(amount, 2) if amount > 0 else None


def _extract_price_usd(block: str) -> float | None:
    label_match = _LABEL_PRICE_RE.search(block)
    if label_match:
        return _parse_money_token(label_match.group(1))

    for match in _STANDALONE_PRICE_RE.finditer(block):
        line_start = block.rfind("\n", 0, match.start()) + 1
        line_end = block.find("\n", match.end())
        line = block[line_start : len(block) if line_end == -1 else line_end].lower()
        if any(tok in line for tok in ("/mo", "per month", "rent", "hoa", "tax", "fee")):
            continue
        parsed = _parse_money_token(match.group(1))
        if parsed is not None:
            return parsed
    return None


def _extract_annual_income(block: str) -> float:
    annual_match = _LABEL_ANNUAL_RENT_RE.search(block)
    if annual_match:
        val = _parse_money_token(annual_match.group(1))
        if val is not None:
            return val

    monthly_match = _LABEL_MONTHLY_RENT_RE.search(block)
    if monthly_match:
        val = _parse_money_token(monthly_match.group(1))
        if val is not None:
            raw_line = monthly_match.group(1).lower()
            if "/yr" in raw_line or "year" in raw_line or "annual" in raw_line:
                return val
            return round(val * 12.0, 2)

    inline_match = _INLINE_MONTHLY_RENT_RE.search(block)
    if inline_match:
        val = _parse_money_token(f"${inline_match.group(1)}")
        if val is not None:
            return round(val * 12.0, 2)

    return 0.0


def _extract_address(block: str) -> str | None:
    label_match = _LABEL_ADDRESS_RE.search(block)
    if label_match:
        candidate = _clean_md_value(label_match.group(1))
        if not _is_placeholder(candidate) and len(candidate) >= 5:
            return candidate
        return None

    for heading in _HEADING_RE.finditer(block):
        candidate = _clean_md_value(heading.group(1))
        if _US_ADDRESS_RE.search(candidate):
            return _US_ADDRESS_RE.search(candidate).group(1).strip()  # type: ignore[union-attr]

    us_match = _US_ADDRESS_RE.search(block)
    if us_match:
        return us_match.group(1).strip()
    return None


def _extract_county(block: str) -> str | None:
    label_match = _LABEL_COUNTY_RE.search(block)
    if label_match:
        candidate = _clean_md_value(label_match.group(1))
        candidate = re.sub(r"(?i)\s+county$", "", candidate).strip()
        if not _is_placeholder(candidate) and len(candidate) >= 2:
            return candidate
        return None

    inline_match = _INLINE_COUNTY_RE.search(block)
    if inline_match:
        candidate = inline_match.group(1).strip()
        if not _is_placeholder(candidate) and len(candidate) >= 2:
            return candidate
    return None


def _extract_source_id(block: str, *, platform: str, fallback_url: str | None) -> str | None:
    label_match = _LABEL_URL_RE.search(block)
    if label_match:
        raw_val = _clean_md_value(label_match.group(1))
        link_match = _MARKDOWN_LINK_RE.search(label_match.group(1))
        candidate = link_match.group(1).strip() if link_match else raw_val
        if not _is_placeholder(candidate) and len(candidate) >= 3:
            if candidate.startswith(("http://", "https://")) or ":" in candidate:
                return candidate
            return f"{platform}:{candidate}"

    for link_match in _MARKDOWN_LINK_RE.finditer(block):
        url = link_match.group(1).strip()
        if not _is_ignored_image(url) and not _RAW_IMAGE_URL_RE.fullmatch(url):
            return url

    if fallback_url and not _is_placeholder(fallback_url):
        return fallback_url.strip()
    return None


def _is_ignored_image(url: str) -> bool:
    lower = url.lower()
    return any(pattern in lower for pattern in _IGNORED_IMAGE_PATTERNS)


def _extract_images(block: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    for match in _MARKDOWN_IMAGE_RE.finditer(block):
        url = match.group(1).strip()
        if not _is_ignored_image(url) and url not in seen:
            seen.add(url)
            urls.append(url)

    for match in _RAW_IMAGE_URL_RE.finditer(block):
        url = match.group(1).strip()
        if not _is_ignored_image(url) and url not in seen:
            seen.add(url)
            urls.append(url)

    return urls


def _extract_coordinates(block: str) -> tuple[float | None, float | None]:
    pair_match = _COORDS_PAIR_RE.search(block)
    if pair_match:
        lat = float(pair_match.group(1))
        lng = float(pair_match.group(2))
        if -90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0:
            return lat, lng
        return None, None

    lat_match = _LAT_RE.search(block)
    lng_match = _LNG_RE.search(block)
    if lat_match and lng_match:
        lat = float(lat_match.group(1))
        lng = float(lng_match.group(1))
        if -90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0:
            return lat, lng
    return None, None


def _extract_description(block: str) -> str:
    section_match = _DESCRIPTION_SECTION_RE.search(block)
    if section_match:
        text = section_match.group(1).strip()
        text = _MARKDOWN_IMAGE_RE.sub("", text).strip()
        if len(text) >= 10:
            return text

    cleaned = _MARKDOWN_IMAGE_RE.sub("", block)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return "\n".join(lines).strip()


_default_normalizer = MarkdownPropertyNormalizer()


def normalizar_markdown_propiedades(
    markdown: str,
    *,
    plataforma: str,
    url_origen: str | None = None,
) -> NormalizationBatch:
    """Función de conveniencia para normalizar uno o varios anuncios desde markdown."""
    return _default_normalizer.normalize_batch(
        markdown,
        platform=plataforma,
        url_origen=url_origen,
    )


def normalizar_propiedad_unica(
    markdown: str,
    *,
    plataforma: str,
    url_origen: str | None = None,
) -> DatosPropiedadNormalizados:
    """Normaliza un anuncio individual o lanza `NormalizationError` si es incompleto."""
    return _default_normalizer.normalize_single(
        markdown,
        platform=plataforma,
        url_origen=url_origen,
    )
