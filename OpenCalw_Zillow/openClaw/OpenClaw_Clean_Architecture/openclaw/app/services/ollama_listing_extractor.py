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
from app.services.geocoding import geocodificar_census

logger = logging.getLogger("OpenClaw.OllamaExtractor")

_CONDADO_POR_MERCADO: dict[str, str] = {
    "detroit, mi": "Wayne",
    "cleveland, oh": "Cuyahoga",
    "flint, mi": "Genesee",
    "toledo, oh": "Lucas",
    "birmingham, al": "Jefferson",
    "memphis, tn": "Shelby",
    "jackson, ms": "Hinds",
    "saginaw, mi": "Saginaw",
    "dayton, oh": "Montgomery",
    "akron, oh": "Summit",
    "st. louis, mo": "St. Louis",
    "baltimore, md": "Baltimore",
    "pittsburgh, pa": "Allegheny",
    "youngstown, oh": "Mahoning",
    "little rock, ar": "Pulaski",
    "augusta, ga": "Richmond",
    "macon, ga": "Bibb",
    "shreveport, la": "Caddo",
    "montgomery, al": "Montgomery",
    "peoria, il": "Peoria",
    "indianapolis, in": "Marion",
    "milwaukee, wi": "Milwaukee",
    "rochester, ny": "Monroe",
    "buffalo, ny": "Erie",
    "syracuse, ny": "Onondaga",
    "kansas city, mo": "Jackson",
    "tulsa, ok": "Tulsa",
    "oklahoma city, ok": "Oklahoma",
    "decatur, il": "Macon",
    "gary, in": "Lake",
}


def inferir_condado_por_mercado(ciudad: str | None, query: str) -> str | None:
    """Devuelve el condado canónico conocido para mercados objetivo cuando falta en la ficha."""
    candidatos = [query.strip().lower()]
    if ciudad:
        candidatos.insert(0, ciudad.strip().lower())
    for texto in candidatos:
        limpio = " ".join(texto.split())
        if limpio in _CONDADO_POR_MERCADO:
            return _CONDADO_POR_MERCADO[limpio]
        for clave, condado in _CONDADO_POR_MERCADO.items():
            ciudad_clave = clave.split(",")[0].strip()
            if ciudad_clave and ciudad_clave in limpio:
                return condado
    return None


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


def _listing_markdown(
    item: OllamaListing,
    *,
    platform: str,
    fallback_url: str | None,
    query: str = "",
) -> str:
    """Convierte salida estructurada a bloques compatibles con el normalizador existente."""
    source = item.url or item.source_id or fallback_url or "ollama:sin_source_id"
    precio_txt = f"${item.precio_usd:,.2f}" if item.precio_usd is not None else "N/A"
    renta_txt = (
        f"${item.renta_mensual_usd:,.2f}/mo" if item.renta_mensual_usd is not None else "N/A"
    )
    condado = item.condado or inferir_condado_por_mercado(item.ciudad, query)
    direccion = item.direccion or "N/A"
    if direccion != "N/A" and item.ciudad and item.ciudad.lower() not in direccion.lower():
        direccion = f"{direccion}, {item.ciudad}"
        if item.codigo_postal and item.codigo_postal not in direccion:
            direccion = f"{direccion} {item.codigo_postal}"
    lines = [
        f"Listing URL: {source}",
        f"Property Address: {direccion}",
        f"City: {item.ciudad or 'N/A'}",
        f"Zip Code: {item.codigo_postal or 'N/A'}",
        f"County: {condado or 'Por verificar'}",
        f"Price: {precio_txt}",
        f"Bedrooms: {item.habitaciones if item.habitaciones is not None else 'N/A'}",
        f"Bathrooms: {item.banos if item.banos is not None else 'N/A'}",
        f"Estimated Rent: {renta_txt}",
        f"Description: {item.descripcion or f'Anuncio extraído en {platform} para {direccion}'}",
        f"Extraction Source: Ollama {settings.OLLAMA_MODEL} ({platform})",
    ]
    if item.imagenes:
        lines.extend(f"![listing image]({url})" for url in item.imagenes)
    return "\n".join(lines)


async def enriquecer_ubicacion_extraccion(
    extraction: OllamaExtraction,
    *,
    query: str,
) -> OllamaExtraction:
    """Completa condado/ciudad/ZIP con Census Geocoder cuando faltan en la tarjeta."""
    actualizadas: list[OllamaListing] = []
    for item in extraction.viviendas_en_venta:
        if item.condado or not item.direccion:
            actualizadas.append(item)
            continue
        direccion_consulta = item.direccion
        if item.ciudad and item.ciudad.lower() not in direccion_consulta.lower():
            direccion_consulta = f"{direccion_consulta}, {item.ciudad}"
        elif (
            query
            and "," in query
            and query.split(",")[0].strip().lower() not in direccion_consulta.lower()
        ):
            direccion_consulta = f"{direccion_consulta}, {query}"
        if item.codigo_postal and item.codigo_postal not in direccion_consulta:
            direccion_consulta = f"{direccion_consulta} {item.codigo_postal}"

        condado_resuelto: str | None = None
        ciudad_resuelta = item.ciudad
        zip_resuelto = item.codigo_postal
        try:
            ubicacion = await geocodificar_census(direccion_consulta)
            if ubicacion is not None:
                condado_resuelto = ubicacion.condado
                ciudad_resuelta = ciudad_resuelta or ubicacion.ciudad
                zip_resuelto = zip_resuelto or ubicacion.codigo_postal
        except Exception as exc:  # noqa: BLE001 - enriquecimiento best-effort
            logger.debug("census_enrichment_skipped", extra={"error": str(exc)})

        if not condado_resuelto:
            condado_resuelto = inferir_condado_por_mercado(ciudad_resuelta, query)

        actualizadas.append(
            item.model_copy(
                update={
                    "condado": condado_resuelto,
                    "ciudad": ciudad_resuelta,
                    "codigo_postal": zip_resuelto,
                }
            )
        )
    return extraction.model_copy(update={"viviendas_en_venta": actualizadas})


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
        respuesta = await http.post(
            f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat", headers=headers, json=payload
        )
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
                            "source_id": (
                                item.source_id
                                if item.source_id and item.source_id in markdown
                                else None
                            ),
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
    query: str = "",
) -> str:
    """Serializa anuncios extraídos para reutilizar validación y persistencia existentes."""
    return "\n\n---\n\n".join(
        _listing_markdown(item, platform=platform, fallback_url=url_origen, query=query)
        for item in extraction.viviendas_en_venta
    )
