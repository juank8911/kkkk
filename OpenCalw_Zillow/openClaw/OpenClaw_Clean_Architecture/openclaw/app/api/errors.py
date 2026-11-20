"""Manejadores de error uniformes para toda la API.

Todas las respuestas de error usan el contrato ``ErrorResponse`` definido en
``app.models.contracts``, de modo que el cliente recibe siempre la misma forma.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.models.contracts import ErrorDetail, ErrorResponse

logger = logging.getLogger("OpenClaw.Errores")

_CODIGOS: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    503: "service_unavailable",
}


def _respuesta(
    status_code: int,
    code: str,
    message: str,
    details: list[ErrorDetail] | dict[str, Any] | None = None,
) -> JSONResponse:
    cuerpo = ErrorResponse(code=code, message=message, details=details)
    return JSONResponse(status_code=status_code, content=cuerpo.model_dump(mode="json"))


async def _http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    codigo = _CODIGOS.get(exc.status_code, "error")
    return _respuesta(exc.status_code, codigo, str(exc.detail))


async def _validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    detalles = [
        ErrorDetail(
            field=".".join(str(parte) for parte in error["loc"][1:]) or None,
            reason=error["msg"],
        )
        for error in exc.errors()
    ]
    return _respuesta(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "validation_error",
        "El payload no cumple el contrato",
        detalles,
    )


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("error_no_controlado", extra={"ruta": request.url.path})
    return _respuesta(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        "Error interno del servidor",
    )


def registrar_manejadores(app: FastAPI) -> None:
    """Registra los tres manejadores en la aplicacion."""
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
