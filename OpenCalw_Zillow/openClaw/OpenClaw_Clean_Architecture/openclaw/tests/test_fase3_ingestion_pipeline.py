"""Pruebas de integración de Fase 3: enriquecimiento, deduplicación y UoW por propiedad (S3-05/07/08)."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy.exc import OperationalError

from app.db.models import Propiedad
from app.db.uow import UnidadDeTrabajo
from app.services.ingestion_pipeline import procesar_lote_extraccion_con_uow


class _InMemoryPropiedadesRepo:
    def __init__(self, *, fail_on_substring: str | None = None) -> None:
        self.rows: list[Propiedad] = []
        self._pending: list[Propiedad] = []
        self._fail_on_substring = fail_on_substring

    async def guardar(self, propiedad: Propiedad) -> Propiedad:
        if self._fail_on_substring and self._fail_on_substring in propiedad.ubicacion:
            raise OperationalError("INSERT propiedades", {}, Exception("simulated DB write error"))
        if propiedad.id is None:
            propiedad.id = uuid4()
        self._pending.append(propiedad)
        return propiedad

    async def existe_url(self, job_id: UUID, url: str) -> bool:
        return any(r.job_id == job_id and r.url_origen == url for r in self.rows)

    async def buscar_por_huella(self, huella: str) -> list[Propiedad]:
        return [r for r in self.rows if r.huella_dedupe == huella]

    async def listar_por_job(self, job_id: UUID) -> list[Propiedad]:
        return [r for r in self.rows if r.job_id == job_id]

    def commit(self) -> None:
        self.rows.extend(self._pending)
        self._pending.clear()

    def rollback(self) -> None:
        self._pending.clear()


class _InMemoryEventosRepo:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self._pending: list[dict[str, object]] = []

    async def registrar(
        self,
        tipo: str,
        mensaje: str,
        *,
        job_id: UUID | None = None,
        nivel: str = "info",
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        ev = {
            "tipo": tipo,
            "mensaje": mensaje,
            "job_id": job_id,
            "nivel": nivel,
            "payload": payload or {},
        }
        self._pending.append(ev)
        return ev

    def commit(self) -> None:
        self.events.extend(self._pending)
        self._pending.clear()

    def rollback(self) -> None:
        self._pending.clear()


class _FakeUnidadDeTrabajo(UnidadDeTrabajo):
    """UnidadDeTrabajo en memoria con semántica real de commit/rollback por propiedad."""

    def __init__(self, *, fail_on_substring: str | None = None) -> None:
        self.propiedades = _InMemoryPropiedadesRepo(fail_on_substring=fail_on_substring)  # type: ignore[assignment]
        self.eventos = _InMemoryEventosRepo()  # type: ignore[assignment]
        self.commits = 0
        self.rollbacks = 0

    async def confirmar(self) -> None:
        self.propiedades.commit()  # type: ignore[attr-defined]
        self.eventos.commit()  # type: ignore[attr-defined]
        self.commits += 1

    async def revertir(self) -> None:
        self.propiedades.rollback()  # type: ignore[attr-defined]
        self.eventos.rollback()  # type: ignore[attr-defined]
        self.rollbacks += 1


_MIXED_MARKDOWN = """
# 1420 North Maple Avenue, Cleveland, OH 44102
Price: $115,000
County: Cuyahoga County
Est. Rent: $1,450/mo
URL: https://www.zillow.com/homedetails/1420-n-maple/1001_zpid/

## Description
Brick colonial 1,420 sqft built in 1954 with No HOA and strong tenant history.

---

# 1420 N. Maple Ave., Cleveland, OH 44102
Price: $115,500
County: Cuyahoga County
Est. Rent: $1,450/mo
URL: https://www.zillow.com/homedetails/1420-n-maple/1001_dup/

## Description
Duplicate listing of 1420 N Maple Ave in Cuyahoga County.

---

# 500 Missing Price St, Cleveland, OH 44102
Price: TBD
County: Cuyahoga
Description:
Incomplete listing that must be rejected during normalization.

---

# 8842 Oakwood Dr, Detroit, MI 48219
Price: $92,500
County: Wayne County
Monthly Rent: $1,250/mo
URL: https://www.realtor.com/realestateandhomes-detail/8842-oakwood_M1

## Description
Turnkey bungalow 1,180 sqft built in 1962 with HOA: $85/mo and detached garage.
"""


async def test_fase3_pipeline_normalizes_enriches_dedupes_and_persists_idempotently() -> None:
    uow = _FakeUnidadDeTrabajo()
    job_id = uuid4()

    # Primera ejecución del lote mixto:
    # - 3 anuncios válidos normalizados (2 de la misma casa -> 1 duplicado en lote), 1 rechazado
    # - 2 propiedades únicas persistidas en BD con enriquecimiento completo
    summary_1 = await procesar_lote_extraccion_con_uow(
        uow,
        job_id=job_id,
        plataforma="zillow",
        markdown=_MIXED_MARKDOWN,
    )
    assert summary_1.normalizadas == 3
    assert summary_1.rechazadas_normalizacion == 1
    assert summary_1.duplicadas_en_lote == 1
    assert summary_1.duplicadas_en_db == 0
    assert summary_1.persistidas == 2
    assert summary_1.fallidas_persistencia == 0
    assert len(uow.propiedades.rows) == 2  # type: ignore[attr-defined]

    # Verifica enriquecimiento de los 4 campos (S3-05) en las propiedades guardadas
    enr_values = list(summary_1.enriquecimientos.values())
    assert any(
        e["superficie_sqft"] == pytest.approx(1420.0)
        and e["anio_construccion"] == 1954
        and e["hoa_mensual_usd"] == 0.0
        for e in enr_values
    )
    assert any(
        e["superficie_sqft"] == pytest.approx(1180.0)
        and e["anio_construccion"] == 1962
        and e["hoa_mensual_usd"] == pytest.approx(85.0)
        for e in enr_values
    )

    # Segunda entrega del mismo mensaje (simula at-least-once / duplicado en Redis Streams):
    # 0 propiedades nuevas duplicadas en BD
    summary_2 = await procesar_lote_extraccion_con_uow(
        uow,
        job_id=job_id,
        plataforma="zillow",
        markdown=_MIXED_MARKDOWN,
    )
    assert summary_2.persistidas == 0
    assert summary_2.duplicadas_en_db == 2
    assert len(uow.propiedades.rows) == 2  # type: ignore[attr-defined]


async def test_fase3_uow_rolls_back_only_failing_property_preserving_others() -> None:
    """Fallo transaccional en una propiedad no revierte ni bloquea el resto del lote."""
    uow = _FakeUnidadDeTrabajo(fail_on_substring="8842 Oakwood")
    job_id = uuid4()

    summary = await procesar_lote_extraccion_con_uow(
        uow,
        job_id=job_id,
        plataforma="zillow",
        markdown=_MIXED_MARKDOWN,
    )
    assert summary.persistidas == 1
    assert summary.fallidas_persistencia == 1
    assert uow.rollbacks == 1
    assert len(uow.propiedades.rows) == 1  # type: ignore[attr-defined]
    assert "1420 North Maple" in uow.propiedades.rows[0].ubicacion  # type: ignore[attr-defined]
