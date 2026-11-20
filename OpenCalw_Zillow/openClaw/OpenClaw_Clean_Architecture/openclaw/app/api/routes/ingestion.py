import json
from fastapi import APIRouter
from app.core.redis_client import redis_client

router = APIRouter(prefix="/v1/openclaw")

@router.post("/iniciar-barrido")
async def encolar_peticion_scraping(payload: dict):
    mensaje = {"query": payload["query"], "plataformas": payload["plataformas"]}
    await redis_client.xadd(name="openclaw_scraping_queue", fields={"payload": json.dumps(mensaje)})
    return {"status": "Encolado al broker Redis"}