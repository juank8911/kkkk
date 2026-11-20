"""Unidad de trabajo por mensaje y transaccion por propiedad (S2-03).

Reglas:
- Un mensaje del worker abre una unidad de trabajo y expone los repositorios.
- Cada propiedad se confirma en su propia transaccion: si una falla, las demas
  quedan guardadas y el estado del job refleja el fallo parcial.
- Cualquier excepcion no controlada revierte la transaccion en curso.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import obtener_sessionmaker
from app.db.repositories import (
    RepositorioDecisiones,
    RepositorioEventos,
    RepositorioJobs,
    RepositorioPropiedades,
)

logger = logging.getLogger("OpenClaw.UoW")

T = TypeVar("T")
R = TypeVar("R")


class UnidadDeTrabajo:
    """Agrupa los repositorios sobre una misma sesion."""

    def __init__(self, sesion: AsyncSession) -> None:
        self.sesion = sesion
        self.jobs = RepositorioJobs(sesion)
        self.propiedades = RepositorioPropiedades(sesion)
        self.decisiones = RepositorioDecisiones(sesion)
        self.eventos = RepositorioEventos(sesion)

    async def confirmar(self) -> None:
        await self.sesion.commit()

    async def revertir(self) -> None:
        await self.sesion.rollback()

    async def cerrar(self) -> None:
        await self.sesion.close()

    async def __aenter__(self) -> "UnidadDeTrabajo":
        return self

    async def __aexit__(self, tipo: type[BaseException] | None, valor: BaseException | None, traza: Any) -> None:
        if tipo is not None:
            await self.revertir()
        await self.cerrar()


@asynccontextmanager
async def unidad_de_trabajo(sesion: AsyncSession | None = None) -> AsyncIterator[UnidadDeTrabajo]:
    """Abre una unidad de trabajo; si no se pasa sesion, usa la del proceso."""
    propia = sesion is None
    sesion_activa = sesion if sesion is not None else obtener_sessionmaker()()
    unidad = UnidadDeTrabajo(sesion_activa)
    try:
        yield unidad
        if propia:
            await unidad.confirmar()
    except Exception:
        await unidad.revertir()
        raise
    finally:
        if propia:
            await unidad.cerrar()


async def procesar_por_propiedad(
    unidad: UnidadDeTrabajo,
    propiedades: Iterable[T],
    manejador: Callable[[UnidadDeTrabajo, T], Awaitable[R]],
    *,
    job_id: uuid.UUID | None = None,
) -> tuple[list[R], list[tuple[T, str]]]:
    """Procesa cada propiedad en su propia transaccion.

    Devuelve los resultados correctos y la lista de fallos ``(elemento, motivo)``
    sin abortar el resto del lote.
    """
    resultados: list[R] = []
    fallos: list[tuple[T, str]] = []

    for elemento in propiedades:
        try:
            resultado = await manejador(unidad, elemento)
            await unidad.confirmar()
            resultados.append(resultado)
        except (SQLAlchemyError, ValueError) as exc:
            await unidad.revertir()
            motivo = f"{type(exc).__name__}: {exc}"
            fallos.append((elemento, motivo))
            logger.warning(
                "propiedad_fallida",
                extra={"job_id": str(job_id) if job_id else None, "motivo": motivo},
            )

    return resultados, fallos
