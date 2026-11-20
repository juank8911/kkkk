import json
import asyncio
import logging
from app.core.redis_client import redis_client
from app.services.scraper import extraer_con_reintentos

logger = logging.getLogger("OpenClaw.Worker")

async def consumir_cola_scraping():
    try:
        await redis_client.xgroup_create("openclaw_scraping_queue", "scrapers", mkstream=True)
    except Exception:
        pass
    
    while True:
        try:
            mensajes = await redis_client.xreadgroup(
                groupname="scrapers", consumername="worker-1",
                streams={"openclaw_scraping_queue": ">"}, count=1, block=5000
            )
            for stream, alertas in mensajes:
                for id_mensaje, datos in alertas:
                    payload = json.loads(datos[b'payload'].decode('utf-8'))
                    tareas = [extraer_con_reintentos(url, payload["query"]) for url in payload["plataformas"]]
                    await asyncio.gather(*tareas)
                    await redis_client.xack("openclaw_scraping_queue", "scrapers", id_mensaje)
        except Exception as e:
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(consumir_cola_scraping())