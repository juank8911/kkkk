"""Extracción estructurada de anuncios mediante Ollama.

Ollama separa los anuncios; Pydantic y el normalizador determinista deciden
qué datos pueden persistirse. El modelo nunca debe inventar precio, dirección
ni ubicación: si no están en el contenido, devuelve null.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import settings

logger = logging.getLogger("OpenClaw.OllamaExtractor")


class OllamaListing(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_id: str | None = None
    url: str | None = None
    direccion: str | None = None
    ciudad: str | None = None
    codigo_postal: str | None = None
    condado: str | None = None
    precio_usd: float | None = Field(default=None, ge=0)
    habitaciones: float | None = Field(default=None, ge=0)
    banos: float | None = Field(default=None, ge=0)
    superficie_sqft: float | None = Field(default=None, ge=0)
    renta_mensual_usd: float | None = Field(default=None, ge=0)
    descripcion: str | None = None
    imagenes: list[str] = Field(default_factory=list)
    confianza_extraccion: float = Field(default=0, ge=0, le=1)

    @field_validator("confianza_extraccion", mode="before")
    @classmethod
    def normalizar_confianza(cls, value: Any) -> Any:
        if isinstance(value, str):
            equivalencias = {"alta": 0.9, "media": 0.6, "baja": 0.3}
            return equivalencias.get(value.strip().lower(), value)
        return value

    @field_validator("imagenes", mode="before")
    @classmethod
    def normalizar_imagenes(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value


class OllamaExtraction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    viviendas_en_venta: list[OllamaListing] = Field(default_factory=list)
    rechazadas: list[dict[str, str]] = Field(default_factory=list)


_SYSTEM_PROMPT = """Eres un extractor de anuncios inmobiliarios. Devuelve únicamente JSON válido.
Separa cada vivienda individual que esté anunciada para VENTA.
No incluy alquileres, terrenos, páginas de búsqueda, artículos ni anuncios sin evidencia.
No inventes ningún dato. Si un campo no aparece literalmente en el contenido, usa null.
Conserva source_id o URL cuando existan. La confianza solo mide extracción textual,
no la calidad de la inversión. El esquema exacto es:
{"viviendas_en_venta":[{"source_id":string|null,"url":string|null,"direccion":string|null,
"ciudad":string|null,"codigo_postal":string|null,"condado":string|null,"precio_usd":number|null,
"habitaciones":number|null,"banos":number|null,"superficie_sqft":number|null,
"renta_mensual_usd":number|null,"descripcion":string|null,"imagenes":[string],
"confianza_extraccion":number}],"rechazadas":[{"motivo":string,"fragmento":string}]}
"""


def _json_content(content: str) -> dict[str, Any]:
    texto = content.strip()
    texto = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto, flags=re.IGNORECASE | re.DOTALL).strip()
    parsed = json.loads(texto)
    if not isinstance(parsed, dict):
        raise ValueError("La respuesta de Ollama no es un objeto JSON")
    return parsed


def _listing_markdown(item: OllamaListing, *, platform: str, fallback_url: str | None) -> str:
    """Convierte salida estructurada a bloques compatibles con el normalizador existente."""
    source = item.url or item.source_id or fallback_url or "ollama:sin_source_id"
    lines = [
        f"Listing URL: {source}",
        f"Property Address: {item.direccion or 'N/A'}",
        f"City: {item.ciudad or 'N/A'}",
        f"Zip Code: {item.codigo_postal or 'N/A'}",
        f"County: {item.condado or 'N/A'}",
        f"Price: {item.precio_usd if item.precio_usd is not None else 'N/A'}",
        f"Bedrooms: {item.habitaciones if item.habitaciones is not None else 'N/A'}",
        f"Bathrooms: {item.banos if item.banos is not None else 'N/A'}",
        f"Estimated Rent: {item.renta_mensual_usd if item.renta_mensual_usd is not None else 'N/A'}",
        f"Description: {item.descripcion or 'N/A'}",
        f"Extraction Source: Ollama {settings.OLLAMA_MODEL} ({platform})",
    ]
    if item.imagenes:
        lines.extend(f"![listing image]({url})" for url in item.imagenes)
    return "\n".join(lines)


async def extraer_viviendas_ollama(
    markdown: str,
    *,
    query: str,
    platform: str,
    url_origen: str | None = None,
    cliente: httpx.AsyncClient | None = None,
) -> tuple[OllamaExtraction | None, str | None]:
    """Extrae viviendas. Devuelve `(resultado, error)` y nunca lanza por fallo remoto."""
    if not settings.OLLAMA_API_KEY.strip() or not markdown.strip():
        return None, "ollama_no_configurado"

    payload = {
        "model": settings.OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Consulta original: {query}\nPlataforma: {platform}\nURL: {url_origen}\n\n"
                    f"Contenido crudo:\n{markdown[:settings.OLLAMA_MAX_MARKDOWN_CHARS]}"
                ),
            },
        ],
    }
    headers = {"X-API-Key": settings.OLLAMA_API_KEY.strip(), "Content-Type": "application/json"}
    propio = cliente is None
    http = cliente or httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT_SEGUNDOS)
    try:
        respuesta = await http.post(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat", headers=headers, json=payload)
        respuesta.raise_for_status()
        body = respuesta.json()
        content = ((body.get("message") or {}).get("content") or "").strip()
        extraction = OllamaExtraction.model_validate(_json_content(content))
        extraction = extraction.model_copy(
            update={
                "viviendas_en_venta": [
                    item.model_copy(
                        update={
                            "url": item.url if item.url and item.url in markdown else url_origen,
                            "source_id": item.source_id if item.source_id and item.source_id in markdown else None,
                        }
                    )
                    for item in extraction.viviendas_en_venta
                ]
            }
        )
        return extraction, None
    except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
        logger.warning("ollama_extraction_error", extra={"error": str(exc), "platform": platform})
        return None, type(exc).__name__
    finally:
        if propio:
            await http.aclose()


def extraction_to_markdown(
    extraction: OllamaExtraction,
    *,
    platform: str,
    url_origen: str | None,
) -> str:
    """Serializa anuncios extraídos para reutilizar validación y persistencia existentes."""
    return "\n\n---\n\n".join(
        _listing_markdown(item, platform=platform, fallback_url=url_origen)
        for item in extraction.viviendas_en_venta
    )
