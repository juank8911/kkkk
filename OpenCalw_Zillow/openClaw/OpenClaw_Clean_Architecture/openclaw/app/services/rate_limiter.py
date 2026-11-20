import logging
from app.core.redis_client import redis_client

logger = logging.getLogger("OpenClaw.RateLimiter")
LIMITES_WAF = {"zillow.com": 15, "realtor.com": 20, "redfin.com": 25, "default": 30}

async def adquirir_permiso_extraccion(dominio: str) -> bool:
    limite_rpm = LIMITES_WAF.get(dominio, LIMITES_WAF["default"])
    clave_redis = f"ratelimit:openclaw:{dominio}"
    async with redis_client.pipeline(transaction=True) as pipe:
        try:
            pipe.incr(clave_redis)
            pipe.expire(clave_redis, 60, nx=True)
            resultados = await pipe.execute()
            if resultados[0] > limite_rpm:
                return False
            return True
        except Exception:
            return True