"""Semaforo de concurrencia por dominio (S3-04).

Raspar el mismo portal con demasiadas peticiones simultaneas provoca bloqueos y
baneos. Cada dominio tiene su propio semaforo, de modo que saturar zillow no
frena a redfin.

El limite es local al proceso: con varios workers en paralelo el tope efectivo se
multiplica por el numero de workers. La cuota global vive en el rate limiter de
Redis (S1), que es quien protege al proveedor.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Final

from app.core.config import settings

logger = logging.getLogger("OpenClaw.Semaforos")


class SemaforosPorDominio:
    """Registro perezoso de semaforos, uno por dominio."""

    def __init__(self, limite: int | None = None) -> None:
        self._limite = limite if limite is not None else settings.LIMITE_CONCURRENCIA_DOMINIO
        if self._limite < 1:
            raise ValueError("El limite de concurrencia debe ser >= 1")
        self._semaforos: dict[str, asyncio.Semaphore] = {}

    @property
    def limite(self) -> int:
        return self._limite

    def semaforo(self, dominio: str) -> asyncio.Semaphore:
        actual = self._semaforos.get(dominio)
        if actual is None:
            actual = asyncio.Semaphore(self._limite)
            self._semaforos[dominio] = actual
        return actual

    @asynccontextmanager
    async def turno(self, dominio: str) -> AsyncIterator[None]:
        """Bloquea hasta tener cupo en ese dominio y lo libera al salir."""
        semaforo = self.semaforo(dominio)
        await semaforo.acquire()
        try:
            yield
        finally:
            semaforo.release()

    def dominios(self) -> tuple[str, ...]:
        return tuple(sorted(self._semaforos))

    def reiniciar(self) -> None:
        self._semaforos.clear()


Semaforos: Final = SemaforosPorDominio
semaforos = SemaforosPorDominio()
