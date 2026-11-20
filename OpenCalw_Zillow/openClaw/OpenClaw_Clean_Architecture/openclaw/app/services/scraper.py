import asyncio
import logging
from firecrawl import FirecrawlApp
from app.core.config import settings
from app.services.rate_limiter import adquirir_permiso_extraccion

logger = logging.getLogger("OpenClaw.Scraper")
firecrawl_app = FirecrawlApp(api_key=settings.FIRECRAWL_API_KEY)

async def extraer_con_reintentos(url: str, query: str, max_intentos: int = 3):
    dominio_base = url.replace("https://", "").replace("www.", "").split("/")[0]
    if not await adquirir_permiso_extraccion(dominio_base):
        return {"dominio": url, "status": "rate_limited"}

    intento = 0
    while intento < max_intentos:
        try:
            scrape_options = {"formats": ["markdown"], "waitFor": 3000, "proxy": True}
            res = await asyncio.to_thread(
                firecrawl_app.search,
                query=f"site:{url} {query}",
                scrape_options=scrape_options
            )
            return {"dominio": url, "status": "exito", "data": res}
        except Exception as e:
            intento += 1
            await asyncio.sleep(2 ** intento)
    return {"dominio": url, "status": "fallo_critico"}