"""Motor de extraccion con Firecrawl.

El cliente se construye de forma perezosa: importar este modulo no debe exigir
una clave configurada, porque la API y las pruebas tambien lo importan.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Any

from firecrawl import FirecrawlApp

from app.core.config import settings
from app.services.rate_limiter import adquirir_permiso_extraccion

logger = logging.getLogger("OpenClaw.Scraper")


@lru_cache(maxsize=1)
def obtener_firecrawl() -> FirecrawlApp:
    """Devuelve el cliente de Firecrawl, fallando si no hay clave."""
    if not settings.FIRECRAWL_API_KEY:
        raise RuntimeError("FIRECRAWL_API_KEY no configurada")
    return FirecrawlApp(api_key=settings.FIRECRAWL_API_KEY)


async def extraer_con_reintentos(url: str, query: str, max_intentos: int = 3) -> dict[str, Any]:
    dominio_base = url.replace("https://", "").replace("www.", "").split("/")[0]

    if not await adquirir_permiso_extraccion(dominio_base):
        return {"dominio": url, "status": "rate_limited"}

    intento = 0
    while intento < max_intentos:
        try:
            cliente = obtener_firecrawl()
            scrape_options = {"formats": ["markdown"], "waitFor": 3000, "proxy": True}
            res = await asyncio.to_thread(
                cliente.search,
                query=f"site:{url} {query}",
                scrape_options=scrape_options,
            )
            return {"dominio": url, "status": "exito", "data": res}
        except Exception as exc:  # noqa: BLE001 - se reintenta con backoff
            intento += 1
            logger.warning(
                "extraccion_fallida",
                extra={"dominio": dominio_base, "intento": intento, "error": str(exc)},
            )
            if intento >= max_intentos:
                break
            await asyncio.sleep(2**intento)

    return {"dominio": url, "status": "fallo_critico"}
