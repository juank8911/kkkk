"""Punto de entrada de la API de OpenClaw."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import registrar_manejadores
from app.api.routes import execution, health, ingestion, offers  # pyright: ignore[reportAttributeAccessIssue]
from app.core.config import settings
from app.core.logging import configure_logging
from app.frontend import router as frontend_router

logger = logging.getLogger("OpenClaw.API")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging(settings.LOG_LEVEL, settings.LOG_JSON)

    faltantes = settings.validar_arranque()
    if faltantes:
        logger.warning(
            "configuracion_incompleta",
            extra={"faltantes": ",".join(faltantes), "entorno": settings.APP_ENV},
        )

    logger.info(
        "api_inicio",
        extra={"entorno": settings.APP_ENV, "version": settings.APP_VERSION},
    )
    yield
    logger.info("api_fin")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

registrar_manejadores(app)

app.include_router(health.router)
app.include_router(ingestion.router)
app.include_router(execution.router)
app.include_router(offers.router)
app.include_router(frontend_router)
