"""Pruebas de repositorios y unidad de trabajo sin base de datos real (S2-03)."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from app.db.uow import UnidadDeTrabajo, procesar_por_propiedad


class ResultadoFalso:
    """Imita lo justo de ``CursorResult`` que usan los repositorios."""

    def __init__(self, filas: list[Any] | None = None, rowcount: int = 1) -> None:
        self._filas = filas or []
        self.rowcount = rowcount

    def scalars(self) -> "ResultadoFalso":
        return self

    def all(self) -> list[Any]:
        return self._filas

    def first(self) -> Any | None:
        return self._filas[0] if self._filas else None


class SesionFalsa:
    """Sesion en memoria que registra las operaciones de transaccion."""

    def __init__(self, filas: list[Any] | None = None) -> None:
        self.agregados: list[Any] = []
        self.flushes = 0
        self.commits = 0
        self.rollbacks = 0
        self.cerrada = False
        self._filas = filas or []
        self._entidades: dict[Any, Any] = {}

    def add(self, entidad: Any) -> None:
        self.agregados.append(entidad)

    def add_all(self, entidades: list[Any]) -> None:
        self.agregados.extend(entidades)

    async def flush(self) -> None:
        self.flushes += 1

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1

    async def close(self) -> None:
        self.cerrada = True

    async def get(self, modelo: Any, clave: Any) -> Any | None:
        return self._entidades.get(clave)

    async def execute(self, _consulta: Any) -> ResultadoFalso:
        return ResultadoFalso(self._filas)


def test_repositorio_de_jobs_agrega_y_hace_flush() -> None:
    sesion = SesionFalsa()
    unidad = UnidadDeTrabajo(sesion)  # type: ignore[arg-type]

    job = object()
    import asyncio

    asyncio.run(unidad.jobs.crear(job))  # type: ignore[arg-type]

    assert sesion.agregados == [job]
    assert sesion.flushes == 1


def test_actualizar_estado_informa_si_existe_la_fila() -> None:
    sesion = SesionFalsa()
    unidad = UnidadDeTrabajo(sesion)  # type: ignore[arg-type]

    import asyncio

    encontrado = asyncio.run(unidad.jobs.actualizar_estado(uuid.uuid4(), "running"))

    assert encontrado is True


def test_unidad_de_trabajo_revierte_si_falla_el_cuerpo() -> None:
    sesion = SesionFalsa()

    import asyncio

    async def cuerpo() -> None:
        async with UnidadDeTrabajo(sesion):  # type: ignore[arg-type]
            raise RuntimeError("fallo simulado")

    with pytest.raises(RuntimeError):
        asyncio.run(cuerpo())

    assert sesion.rollbacks == 1
    assert sesion.cerrada is True


def test_unidad_de_trabajo_cierra_sin_confirmar_sola() -> None:
    sesion = SesionFalsa()

    import asyncio

    async def cuerpo() -> None:
        async with UnidadDeTrabajo(sesion):  # type: ignore[arg-type]
            pass

    asyncio.run(cuerpo())

    assert sesion.commits == 0
    assert sesion.cerrada is True


def test_procesar_por_propiedad_continua_tras_un_fallo() -> None:
    sesion = SesionFalsa()
    unidad = UnidadDeTrabajo(sesion)  # type: ignore[arg-type]

    import asyncio

    async def manejador(_unidad: UnidadDeTrabajo, elemento: int) -> int:
        if elemento == 2:
            raise ValueError("dato invalido")
        return elemento * 10

    resultados, fallos = asyncio.run(
        procesar_por_propiedad(unidad, [1, 2, 3], manejador, job_id=uuid.uuid4())
    )

    assert resultados == [10, 30]
    assert len(fallos) == 1
    assert fallos[0][0] == 2
    assert "dato invalido" in fallos[0][1]
    assert sesion.commits == 2
    assert sesion.rollbacks == 1


def test_procesar_por_propiedad_sin_fallos_confirma_cada_una() -> None:
    sesion = SesionFalsa()
    unidad = UnidadDeTrabajo(sesion)  # type: ignore[arg-type]

    import asyncio

    async def manejador(_unidad: UnidadDeTrabajo, elemento: int) -> int:
        return elemento

    resultados, fallos = asyncio.run(procesar_por_propiedad(unidad, [1, 2, 3], manejador))

    assert resultados == [1, 2, 3]
    assert fallos == []
    assert sesion.commits == 3
