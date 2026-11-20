"""Base declarativa y sesiones asincronas de PostgreSQL (S2-02).

El motor se crea de forma perezosa: importar la aplicacion no debe exigir una
base de datos disponible, y las pruebas pueden inyectar su propia sesion.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    """Base declarativa comun a todos los modelos."""


@lru_cache(maxsize=1)
def obtener_engine() -> AsyncEngine:
    """Crea (una sola vez por proceso) el motor asincrono."""
    if not settings.DATABASE_URL.strip():
        raise RuntimeError(
            "DATABASE_URL no esta configurada: no se puede abrir la base de datos"
        )
    return create_async_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        echo=False,
    )


@lru_cache(maxsize=1)
def obtener_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Fabrica de sesiones ligada al motor del proceso."""
    return async_sessionmaker(obtener_engine(), expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """Dependencia de FastAPI: una sesion por peticion con commit explicito."""
    fabrica = obtener_sessionmaker()
    async with fabrica() as sesion:
        try:
            yield sesion
            await sesion.commit()
        except Exception:
            await sesion.rollback()
            raise
