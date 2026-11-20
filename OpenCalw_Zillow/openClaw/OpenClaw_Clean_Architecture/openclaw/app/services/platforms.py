"""Catálogo cerrado de plataformas inmobiliarias soportadas (ADR-01 / S1-07).

Evita SSRF al no aceptar URLs arbitrarias desde el cliente: solo se admiten las
9 claves canónicas definidas en servidor, con sanitización del query y validación
del host final.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final
from urllib.parse import quote, urlparse


@dataclass(frozen=True, slots=True)
class PlatformSpec:
    """Especificación inmutable de una plataforma inmobiliaria soportada."""

    key: str
    display_name: str
    domain: str
    url_template: str
    default_rpm: int
    subdomain_query: bool = False


PLATFORM_CATALOG: Final[dict[str, PlatformSpec]] = {
    "zillow": PlatformSpec(
        key="zillow",
        display_name="Zillow",
        domain="zillow.com",
        url_template="https://www.zillow.com/homes/{query}_rb/",
        default_rpm=15,
    ),
    "realtor": PlatformSpec(
        key="realtor",
        display_name="Realtor.com",
        domain="realtor.com",
        url_template="https://www.realtor.com/realestateandhomes-search/{query}",
        default_rpm=20,
    ),
    "redfin": PlatformSpec(
        key="redfin",
        display_name="Redfin",
        domain="redfin.com",
        url_template="https://www.redfin.com/city/{query}",
        default_rpm=25,
    ),
    "trulia": PlatformSpec(
        key="trulia",
        display_name="Trulia",
        domain="trulia.com",
        url_template="https://www.trulia.com/{query}",
        default_rpm=20,
    ),
    "loopnet": PlatformSpec(
        key="loopnet",
        display_name="LoopNet",
        domain="loopnet.com",
        url_template="https://www.loopnet.com/search/commercial-real-estate/{query}/for-sale/",
        default_rpm=15,
    ),
    "craigslist": PlatformSpec(
        key="craigslist",
        display_name="Craigslist",
        domain="craigslist.org",
        url_template="https://{query}.craigslist.org/search/rea",
        default_rpm=20,
        subdomain_query=True,
    ),
    "century21": PlatformSpec(
        key="century21",
        display_name="Century 21",
        domain="century21.com",
        url_template="https://www.century21.com/real-estate/{query}/",
        default_rpm=25,
    ),
    "coldwell": PlatformSpec(
        key="coldwell",
        display_name="Coldwell Banker",
        domain="coldwellbankerhomes.com",
        url_template="https://www.coldwellbankerhomes.com/{query}/",
        default_rpm=25,
    ),
    "remax": PlatformSpec(
        key="remax",
        display_name="RE/MAX",
        domain="remax.com",
        url_template="https://www.remax.com/homes-for-sale/{query}",
        default_rpm=25,
    ),
}

VALID_PLATFORM_KEYS: Final[tuple[str, ...]] = tuple(sorted(PLATFORM_CATALOG.keys()))

_PATH_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"[^a-zA-Z0-9,\-_ ]+")
_SUBDOMAIN_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")
_MULTI_DASH_RE: Final[re.Pattern[str]] = re.compile(r"-{2,}")
_US_ZIP_QUERY_RE: Final[re.Pattern[str]] = re.compile(r"^\d{5}$")
_CITY_STATE_RE: Final[re.Pattern[str]] = re.compile(r"^([a-zA-Z .'-]+?)[,\s]+([a-zA-Z]{2})$")

# Redfin exige ID numérico en /city/{id}/{ST}/{City}; sin él devuelve 404.
_REDFIN_CITY_PATHS: Final[dict[str, str]] = {
    "memphis, tn": "11797/TN/Memphis",
    "cleveland, oh": "4243/OH/Cleveland",
    "detroit, mi": "5665/MI/Detroit",
    "flint, mi": "6683/MI/Flint",
    "jackson, ms": "9431/MS/Jackson",
    "toledo, oh": "19323/OH/Toledo",
    "birmingham, al": "1305/AL/Birmingham",
    "saginaw, mi": "16773/MI/Saginaw",
    "dayton, oh": "5049/OH/Dayton",
    "akron, oh": "361/OH/Akron",
    "youngstown, oh": "22370/OH/Youngstown",
    "baltimore, md": "1363/MD/Baltimore",
    "pittsburgh, pa": "15174/PA/Pittsburgh",
}


class InvalidPlatformError(ValueError):
    """Se lanza cuando el cliente envía una o más claves de plataforma no permitidas."""

    def __init__(self, invalid_keys: Iterable[str]) -> None:
        self.invalid_keys: tuple[str, ...] = tuple(invalid_keys)
        self.valid_keys: tuple[str, ...] = VALID_PLATFORM_KEYS
        invalid_str = ", ".join(repr(k) for k in self.invalid_keys)
        valid_str = ", ".join(self.valid_keys)
        super().__init__(
            f"Plataforma(s) no soportada(s): {invalid_str}. " f"Plataformas válidas: [{valid_str}]."
        )


class InvalidQueryError(ValueError):
    """Se lanza cuando el query no puede convertirse de forma segura en un slug."""


def get_platform_spec(platform_key: str) -> PlatformSpec:
    """Obtiene la especificación canónica de una clave de plataforma."""
    normalized = platform_key.strip().lower()
    spec = PLATFORM_CATALOG.get(normalized)
    if spec is None:
        raise InvalidPlatformError([platform_key])
    return spec


def validate_platforms(plataformas: Iterable[str]) -> list[str]:
    """Valida, normaliza a minúsculas y deduplica preservando el orden.

    Lanza InvalidPlatformError listando todas las claves válidas si alguna clave
    no pertenece al catálogo cerrado de 9 plataformas.
    """
    normalized_list: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()

    for raw in plataformas:
        key = raw.strip().lower() if isinstance(raw, str) else str(raw)
        if key not in PLATFORM_CATALOG:
            invalid.append(raw)
            continue
        if key not in seen:
            seen.add(key)
            normalized_list.append(key)

    if invalid:
        raise InvalidPlatformError(invalid)

    if not normalized_list:
        valid_str = ", ".join(VALID_PLATFORM_KEYS)
        raise ValueError(
            f"Debe especificar al menos una plataforma. Plataformas válidas: [{valid_str}]."
        )

    return normalized_list


def sanitize_query_slug(query: str, *, subdomain: bool = False) -> str:
    """Convierte el texto de búsqueda en un slug seguro contra inyección de URL/SSRF."""
    cleaned = query.strip()
    if not cleaned:
        raise InvalidQueryError("El parámetro 'query' no puede estar vacío.")

    if subdomain:
        slug = _SUBDOMAIN_SLUG_RE.sub("", cleaned.lower())
        slug = slug[:63]
        if not slug:
            raise InvalidQueryError(
                "El parámetro 'query' no produce un subdominio válido para Craigslist."
            )
        return slug

    safe = _PATH_SLUG_RE.sub(" ", cleaned)
    safe = "-".join(safe.split())
    safe = _MULTI_DASH_RE.sub("-", safe).strip("-")
    if not safe:
        raise InvalidQueryError("El parámetro 'query' no contiene caracteres de búsqueda válidos.")
    return quote(safe, safe="-_,")


def _normalizar_ciudad_estado(query: str) -> tuple[str, str] | None:
    limpio = " ".join(query.strip().split())
    match = _CITY_STATE_RE.match(limpio)
    if not match:
        return None
    ciudad = match.group(1).strip(" ,")
    estado = match.group(2).upper()
    if not ciudad:
        return None
    return ciudad, estado


def build_platform_url(platform_key: str, query: str) -> str:
    """Construye y verifica la URL objetivo para una plataforma del catálogo."""
    spec = get_platform_spec(platform_key)
    slug = sanitize_query_slug(query, subdomain=spec.subdomain_query)
    city_state = _normalizar_ciudad_estado(query)

    if spec.key == "redfin":
        limpio = query.strip()
        if _US_ZIP_QUERY_RE.match(limpio):
            url = f"https://www.redfin.com/zipcode/{limpio}"
        elif city_state is not None:
            clave = f"{city_state[0].lower()}, {city_state[1].lower()}"
            if clave in _REDFIN_CITY_PATHS:
                url = f"https://www.redfin.com/city/{_REDFIN_CITY_PATHS[clave]}"
            else:
                url = spec.url_template.format(query=slug)
        else:
            url = spec.url_template.format(query=slug)
    elif spec.key == "realtor" and city_state is not None:
        ciudad_slug = "-".join(city_state[0].split())
        url = spec.url_template.format(query=quote(f"{ciudad_slug}_{city_state[1]}", safe="-_"))
    elif spec.key == "zillow" and city_state is not None:
        ciudad_slug = "-".join(city_state[0].split())
        url = spec.url_template.format(query=quote(f"{ciudad_slug}-{city_state[1]}", safe="-_"))
    else:
        url = spec.url_template.format(query=slug)

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (host == spec.domain or host.endswith(f".{spec.domain}")):
        raise InvalidQueryError(
            f"La URL generada no corresponde al dominio autorizado '{spec.domain}'."
        )
    return url


def get_default_domain_limits() -> dict[str, int]:
    """Devuelve el mapa dominio -> límite RPM por defecto alineado con el catálogo."""
    limits = {spec.domain: spec.default_rpm for spec in PLATFORM_CATALOG.values()}
    limits["default"] = 20
    return limits
