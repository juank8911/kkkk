"""Pruebas del semaforo de concurrencia por dominio (S3-04)."""

from __future__ import annotations

import asyncio

import pytest

from app.services.domain_semaphore import SemaforosPorDominio


async def _carga(semaforos: SemaforosPorDominio, dominio: str, registro: list[int]) -> None:
    async with semaforos.turno(dominio):
        registro.append(1)
        await asyncio.sleep(0.01)
        registro.pop()


def test_limite_invalido() -> None:
    with pytest.raises(ValueError):
        SemaforosPorDominio(limite=0)


async def test_limite_uno_serializa() -> None:
    semaforos = SemaforosPorDominio(limite=1)
    activos: list[int] = []
    maximo = 0

    async def tarea() -> None:
        nonlocal maximo
        async with semaforos.turno("zillow.com"):
            activos.append(1)
            maximo = max(maximo, len(activos))
            await asyncio.sleep(0.01)
            activos.pop()

    await asyncio.gather(*(tarea() for _ in range(4)))
    assert maximo == 1


async def test_limite_dos_permite_paralelismo() -> None:
    semaforos = SemaforosPorDominio(limite=2)
    activos: list[int] = []
    maximo = 0

    async def tarea() -> None:
        nonlocal maximo
        async with semaforos.turno("redfin.com"):
            activos.append(1)
            maximo = max(maximo, len(activos))
            await asyncio.sleep(0.01)
            activos.pop()

    await asyncio.gather(*(tarea() for _ in range(4)))
    assert maximo == 2


async def test_dominios_distintos_no_se_bloquean() -> None:
    semaforos = SemaforosPorDominio(limite=1)
    activos: list[int] = []
    maximo = 0

    async def tarea(dominio: str) -> None:
        nonlocal maximo
        async with semaforos.turno(dominio):
            activos.append(1)
            maximo = max(maximo, len(activos))
            await asyncio.sleep(0.02)
            activos.pop()

    await asyncio.gather(tarea("zillow.com"), tarea("realtor.com"))
    assert maximo == 2


async def test_registro_de_dominios() -> None:
    semaforos = SemaforosPorDominio(limite=1)
    async with semaforos.turno("trulia.com"):
        pass
    async with semaforos.turno("loopnet.com"):
        pass
    assert semaforos.dominios() == ("loopnet.com", "trulia.com")
    semaforos.reiniciar()
    assert semaforos.dominios() == ()
