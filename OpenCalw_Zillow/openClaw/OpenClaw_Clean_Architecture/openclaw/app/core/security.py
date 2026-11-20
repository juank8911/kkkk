"""Autenticacion por API key tactica (S2-01, ADR-04).

Reglas:
- La clave nunca se compara en claro: se guarda y se compara su SHA-256.
- La comparacion usa ``hmac.compare_digest`` para no filtrar tiempos.
- Sin clave configurada el servicio responde 503: es un fallo de configuracion,
  no una peticion sin permiso.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Annotated, Final

from fastapi import Depends, Header, HTTPException, status

from app.core.config import settings

logger = logging.getLogger("OpenClaw.Seguridad")

CABECERA_API_KEY: Final = "X-OpenClaw-Tactical-Key"
IDENTIDAD_TACTICA: Final = "tactical"


def hash_api_key(clave: str) -> str:
    """Devuelve el SHA-256 hexadecimal de una clave en claro."""
    return hashlib.sha256(clave.strip().encode("utf-8")).hexdigest()


def huella_configurada() -> str:
    """Huella esperada segun configuracion (precomputada o derivada en caliente)."""
    return settings.api_key_hash


def _rechazo(status_code: int, detalle: str, *, cabeceras: dict[str, str] | None = None) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detalle, headers=cabeceras)


async def requiere_api_key(
    x_openclaw_tactical_key: Annotated[str | None, Header(alias=CABECERA_API_KEY)] = None,
) -> str:
    """Dependencia de FastAPI que exige una API key valida en la cabecera tactica."""
    esperada = huella_configurada()
    if not esperada:
        logger.error("api_key_no_configurada")
        raise _rechazo(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "El servicio no tiene configurada la clave de acceso",
        )

    if not x_openclaw_tactical_key or not x_openclaw_tactical_key.strip():
        raise _rechazo(
            status.HTTP_401_UNAUTHORIZED,
            "Falta la cabecera X-OpenClaw-Tactical-Key",
            cabeceras={"WWW-Authenticate": CABECERA_API_KEY},
        )

    recibida = hash_api_key(x_openclaw_tactical_key)
    if not hmac.compare_digest(recibida, esperada):
        logger.warning("api_key_invalida", extra={"motivo": "huella_distinta"})
        raise _rechazo(status.HTTP_403_FORBIDDEN, "La API key no es valida")

    return IDENTIDAD_TACTICA


IdentidadTactica = Annotated[str, Depends(requiere_api_key)]
