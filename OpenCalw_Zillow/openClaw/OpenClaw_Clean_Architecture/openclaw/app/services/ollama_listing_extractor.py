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
    contacto_vendedor: str | None = None
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
    if item.contacto_vendedor:
        lines.append(f"Seller Contact: {item.contacto_vendedor}")
    if item.imagenes:
        lines.extend(f"![listing image]({url})" for url in item.imagenes[:2])
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


_LISTING_URL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"https://www\.zillow\.com/homedetails/[^\s)\"'>]+", re.IGNORECASE),
    re.compile(r"https://www\.realtor\.com/realestateandhomes-detail/[^\s)\"'>]+", re.IGNORECASE),
    re.compile(r"https://www\.redfin\.com/[A-Z]{2}/[^\s)\"'>]+/home/\d+", re.IGNORECASE),
    re.compile(r"https://www\.trulia\.com/(?:home|p)/[^\s)\"'>]+", re.IGNORECASE),
    re.compile(r"https://www\.loopnet\.com/Listing/[^\s)\"'>]+", re.IGNORECASE),
)

_CARD_PRICE_RE = re.compile(r"\$\s*([\d,]{5,}(?:\.\d+)?)")
_CARD_ADDRESS_RE = re.compile(
    r"\b(\d{1,6}\s+[A-Za-z0-9.'\- ]{2,55}?"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Ct|Court|"
    r"Way|Pl|Place|Cir|Circle|Pkwy|Parkway|Ter|Terrace|Hwy|Highway|Trl|Trail|Loop|Cv|Cove)\b"
    r"[^,\n\]]{0,25},\s*[A-Za-z .'-]{2,35},\s*([A-Z]{2})(?:\s+(\d{5}))?)"
)
_CARD_STREET_ONLY_RE = re.compile(
    r"(?m)^(\d{1,6}\s+[A-Za-z0-9.'\- ]{2,55}?"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Ct|Court|"
    r"Way|Pl|Place|Cir|Circle|Pkwy|Parkway|Ter|Terrace|Hwy|Highway|Trl|Trail|Loop|Cv|Cove)\b[^\n]{0,20})$"
)
_CARD_CITY_ST_ZIP_RE = re.compile(r"(?m)^([A-Za-z .'-]{2,35},\s*([A-Z]{2})(?:\s+(\d{5}))?)$")
_CARD_BEDS_RE = re.compile(r"(?i)(?:\*\*)?(\d+(?:\.\d+)?)(?:\*\*)?\s*(?:bds?|beds?|bedrooms?)\b")
_CARD_BATHS_RE = re.compile(r"(?i)(?:\*\*)?(\d+(?:\.\d+)?)(?:\*\*)?\s*(?:ba|baths?|bathrooms?)\b")
_CARD_SQFT_RE = re.compile(r"(?i)(?:\*\*)?([\d,]{3,6})(?:\*\*)?\s*(?:sq\s*ft|sqft)\b")
_CARD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((https?://[^\s)]+)\)")
_CARD_SELLER_RE = re.compile(
    r"(?im)(?:brokered\s+by|listed\s+by|listing\s+by|agent|contact|for\s+sale\s+by\s+owner)"
    r"[:\s\-]*([^\n|]{3,100})"
)
_IGNORED_IMG_TOKENS: tuple[str, ...] = (
    "logo",
    "favicon",
    "icon",
    ".svg",
    "sprite",
    "badge",
    "equal-housing",
    "/flags/",
    "maps.gstatic",
    "app-download",
    "google-play",
    "apple-app",
)
_QUERY_STATE_RE = re.compile(r",\s*([A-Z]{2})\b")


def _recortar_bloque_anuncios(markdown: str, *, max_chars: int = 8000) -> str:
    """Elimina menús de navegación iniciales y enfoca la ventana desde el primer anuncio/precio."""
    primer_indice: int | None = None
    for pat in _LISTING_URL_PATTERNS:
        m = pat.search(markdown)
        if m is not None and (primer_indice is None or m.start() < primer_indice):
            primer_indice = m.start()
    if primer_indice is None:
        m_price = _CARD_PRICE_RE.search(markdown)
        if m_price is not None:
            primer_indice = m_price.start()
    inicio = max(0, primer_indice - 300) if primer_indice is not None else 0
    return markdown[inicio : inicio + max_chars]


def _extraer_estado_objetivo(query: str) -> str | None:
    m = _QUERY_STATE_RE.search(query.strip().upper())
    return m.group(1) if m else None


def _extraer_precio_tarjeta(fragmento: str) -> float | None:
    for m in _CARD_PRICE_RE.finditer(fragmento):
        linea_ini = fragmento.rfind("\n", 0, m.start()) + 1
        linea_fin = fragmento.find("\n", m.end())
        linea = fragmento[linea_ini : len(fragmento) if linea_fin == -1 else linea_fin].lower()
        if any(tok in linea for tok in ("/mo", "per month", "rent", "hoa", "tax", "fee")):
            continue
        try:
            valor = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if 10_000.0 <= valor <= 25_000_000.0:
            return round(valor, 2)
    return None


def extraer_viviendas_determinista(
    markdown: str,
    *,
    query: str,
    platform: str,
    url_origen: str | None = None,
    max_items: int = 25,
) -> OllamaExtraction:
    """Extrae viviendas individuales de páginas de resultados sin inventar datos.

    Se usa como respaldo rápido cuando Ollama no devuelve anuncios o agota timeout.
    """
    if not markdown.strip():
        return OllamaExtraction()

    estado_objetivo = _extraer_estado_objetivo(query)
    hits: list[tuple[int, int, str]] = []
    vistas: set[str] = set()
    for pat in _LISTING_URL_PATTERNS:
        for match in pat.finditer(markdown):
            url = match.group(0).rstrip(".,;)]")
            if url not in vistas:
                vistas.add(url)
                hits.append((match.start(), match.end(), url))

    hits.sort(key=lambda item: item[0])
    viviendas: list[OllamaListing] = []
    rechazadas: list[dict[str, str]] = []

    for idx, (pos, end_pos, url) in enumerate(hits):
        if len(viviendas) >= max_items:
            break
        prev_end = hits[idx - 1][1] if idx > 0 else 0
        inicio = max(prev_end, pos - 90)
        siguiente_pos = hits[idx + 1][0] if idx + 1 < len(hits) else len(markdown)
        limite_sig = max(end_pos, siguiente_pos - 28) if idx + 1 < len(hits) else len(markdown)
        fin = min(len(markdown), limite_sig, end_pos + 900)
        enlace_previo = markdown[max(prev_end, pos - 28) : pos]
        cuerpo_post = markdown[end_pos:fin]
        fragmento = markdown[inicio:fin]

        precio = (
            _extraer_precio_tarjeta(enlace_previo)
            or _extraer_precio_tarjeta(cuerpo_post)
            or _extraer_precio_tarjeta(fragmento)
        )
        if precio is None:
            rechazadas.append({"motivo": "sin_precio_valido", "fragmento": url[:160]})
            continue

        direccion: str | None = None
        ciudad: str | None = None
        codigo_postal: str | None = None
        estado_detectado: str | None = None

        m_addr = _CARD_ADDRESS_RE.search(fragmento)
        if m_addr:
            direccion = " ".join(m_addr.group(1).split())
            estado_detectado = m_addr.group(2).upper()
            codigo_postal = m_addr.group(3)
            partes = [p.strip() for p in direccion.split(",")]
            if len(partes) >= 2:
                ciudad = f"{partes[1]}, {estado_detectado}"
        else:
            m_street = _CARD_STREET_ONLY_RE.search(fragmento)
            m_city = _CARD_CITY_ST_ZIP_RE.search(fragmento)
            if m_street and m_city:
                calle = " ".join(m_street.group(1).split())
                estado_detectado = m_city.group(2).upper()
                codigo_postal = m_city.group(3)
                ciudad_linea = " ".join(m_city.group(1).split())
                ciudad = ciudad_linea.rsplit(" ", 1)[0] if codigo_postal else ciudad_linea
                direccion = f"{calle}, {ciudad_linea}"

        if not direccion:
            rechazadas.append({"motivo": "sin_direccion", "fragmento": url[:160]})
            continue

        if estado_objetivo and estado_detectado and estado_detectado != estado_objetivo:
            rechazadas.append(
                {
                    "motivo": f"estado_distinto:{estado_detectado}!={estado_objetivo}",
                    "fragmento": direccion[:160],
                }
            )
            continue

        m_beds = _CARD_BEDS_RE.search(fragmento)
        m_baths = _CARD_BATHS_RE.search(fragmento)
        m_sqft = _CARD_SQFT_RE.search(fragmento)
        m_seller = _CARD_SELLER_RE.search(fragmento)
        contacto_vendedor: str | None = None
        if m_seller:
            candidato = " ".join(m_seller.group(0).split()).strip(" -|*")
            if len(candidato) >= 5 and "contact agent" not in candidato.lower():
                contacto_vendedor = candidato[:120]

        imagenes: list[str] = []
        for img in _CARD_IMAGE_RE.findall(fragmento):
            low = img.lower()
            if not any(tok in low for tok in _IGNORED_IMG_TOKENS) and img not in imagenes:
                imagenes.append(img)
            if len(imagenes) >= 2:
                break

        detalles: list[str] = []
        if m_beds:
            detalles.append(f"{m_beds.group(1)} bd")
        if m_baths:
            detalles.append(f"{m_baths.group(1)} ba")
        if m_sqft:
            detalles.append(f"{m_sqft.group(1)} sqft")
        desc = (
            f"Vivienda en venta en {direccion} ({', '.join(detalles)})"
            if detalles
            else f"Vivienda en venta extraída de {platform} en {direccion}"
        )

        viviendas.append(
            OllamaListing(
                source_id=url,
                url=url,
                direccion=direccion,
                ciudad=ciudad or query,
                codigo_postal=codigo_postal,
                condado=inferir_condado_por_mercado(ciudad, query),
                precio_usd=precio,
                habitaciones=float(m_beds.group(1)) if m_beds else None,
                banos=float(m_baths.group(1)) if m_baths else None,
                superficie_sqft=float(m_sqft.group(1).replace(",", "")) if m_sqft else None,
                descripcion=desc,
                contacto_vendedor=contacto_vendedor,
                imagenes=imagenes,
                confianza_extraccion=0.85,
            )
        )

    return OllamaExtraction(viviendas_en_venta=viviendas, rechazadas=rechazadas)


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

    ventana_markdown = _recortar_bloque_anuncios(
        markdown,
        max_chars=min(settings.OLLAMA_MAX_MARKDOWN_CHARS, 8000),
    )
    payload = {
        "model": settings.OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "options": {"num_ctx": 4096, "temperature": 0},
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Consulta original: {query}\nPlataforma: {platform}\nURL: {url_origen}\n\n"
                    f"Contenido crudo:\n{ventana_markdown}"
                ),
            },
        ],
    }
    headers = {"X-API-Key": settings.OLLAMA_API_KEY.strip(), "Content-Type": "application/json"}
    propio = cliente is None
    timeout_efectivo = min(float(settings.OLLAMA_TIMEOUT_SEGUNDOS), 45.0)
    http = cliente or httpx.AsyncClient(timeout=timeout_efectivo)
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
