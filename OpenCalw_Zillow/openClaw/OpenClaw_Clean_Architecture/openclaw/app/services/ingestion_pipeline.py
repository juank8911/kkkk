"""Pipeline de dominio Fase 3: normalización, enriquecimiento, deduplicación y persistencia UoW.

Conecta las piezas de Dev 2 (`S1-11`, `S3-05`, `S3-07`) con la unidad de trabajo
transaccional por propiedad de Dev 1 (`S2-03`: `app.db.uow.UnidadDeTrabajo` y
`procesar_por_propiedad`), listo para ser invocado desde el worker de Sprint 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import Propiedad
from app.db.uow import UnidadDeTrabajo, procesar_por_propiedad
from app.models.property import DatosPropiedadNormalizados
from app.services.db_mappers import map_normalized_property_to_orm
from app.services.deduplicator import (
    DeduplicatedProperty,
    buscar_duplicado_existente,
    deduplicar_propiedades,
)
from app.services.enrichment import PropertyEnrichment, enriquecer_propiedad_desde_markdown
from app.services.normalizer import normalizar_markdown_propiedades


@dataclass(frozen=True, slots=True)
class _PersistCandidate:
    item: DeduplicatedProperty
    enrichment: PropertyEnrichment


class IngestionBatchSummary(BaseModel):
    """Resumen auditable del procesamiento de un lote extraído de una plataforma."""

    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    plataforma: str
    normalizadas: int = Field(default=0, ge=0)
    rechazadas_normalizacion: int = Field(default=0, ge=0)
    duplicadas_en_lote: int = Field(default=0, ge=0)
    duplicadas_en_db: int = Field(default=0, ge=0)
    persistidas: int = Field(default=0, ge=0)
    fallidas_persistencia: int = Field(default=0, ge=0)
    propiedad_ids: list[UUID] = Field(default_factory=list)
    enriquecimientos: dict[str, dict[str, Any]] = Field(default_factory=dict)
    errores_persistencia: list[str] = Field(default_factory=list)


async def _existe_duplicado_en_db(
    unidad: UnidadDeTrabajo,
    *,
    job_id: UUID,
    item: DeduplicatedProperty,
) -> bool:
    """Verifica duplicidad por URL exacta en el job o por huella/similitud en BD."""
    if await unidad.propiedades.existe_url(job_id, item.propiedad.id_plataforma_origen):
        return True

    existentes_huella = await unidad.propiedades.buscar_por_huella(item.dedup_key)
    if existentes_huella:
        return True

    existentes_job = await unidad.propiedades.listar_por_job(job_id)
    if existentes_job:
        candidatos_previos = [
            DatosPropiedadNormalizados(
                id_plataforma_origen=row.url_origen,
                ubicacion=row.ubicacion,
                precio_usd=float(row.precio_usd),
                ingreso_anual_estimado=float(row.ingreso_anual_estimado),
                imagenes=list(row.imagenes or []),
                condado=row.condado or "N/A",
                latitud=float(row.latitud) if row.latitud is not None else None,
                longitud=float(row.longitud) if row.longitud is not None else None,
                descripcion_cruda=(
                    row.descripcion_cruda
                    if len(row.descripcion_cruda) >= 10
                    else f"{row.descripcion_cruda} (registro previo)"
                ),
            )
            for row in existentes_job
            if float(row.precio_usd) > 0
        ]
        if buscar_duplicado_existente(item.propiedad, candidatos_previos) is not None:
            return True

    return False


async def procesar_lote_extraccion_con_uow(
    unidad: UnidadDeTrabajo,
    *,
    job_id: UUID,
    plataforma: str,
    markdown: str,
    url_origen: str | None = None,
) -> IngestionBatchSummary:
    """Normaliza, enriquece, deduplica y persiste un lote usando una transacción por propiedad."""
    batch = normalizar_markdown_propiedades(
        markdown,
        plataforma=plataforma,
        url_origen=url_origen,
    )

    if batch.rejected:
        for rej in batch.rejected:
            await unidad.eventos.registrar(
                "normalizacion.rechazada",
                f"Anuncio #{rej.index} rechazado en {plataforma}: {', '.join(rej.reasons)}",
                job_id=job_id,
                nivel="warning",
                payload=rej.model_dump(mode="json"),
            )
        await unidad.confirmar()

    deduped = deduplicar_propiedades(batch.valid)
    duplicadas_lote = sum(item.duplicados_fusionados for item in deduped)

    candidates: list[_PersistCandidate] = []
    enriquecimientos: dict[str, dict[str, Any]] = {}
    for item in deduped:
        enr = enriquecer_propiedad_desde_markdown(item.propiedad.descripcion_cruda)
        if enr.precio_usd is None:
            enr = enr.model_copy(
                update={
                    "precio_usd": item.propiedad.precio_usd,
                    "campos_ausentes": [c for c in enr.campos_ausentes if c != "precio_usd"],
                }
            )
        enriquecimientos[item.dedup_key] = enr.model_dump(mode="json")
        candidates.append(_PersistCandidate(item=item, enrichment=enr))

    duplicadas_db = 0

    async def _persistir_candidato(
        uow: UnidadDeTrabajo,
        cand: _PersistCandidate,
    ) -> Propiedad | None:
        nonlocal duplicadas_db
        if await _existe_duplicado_en_db(uow, job_id=job_id, item=cand.item):
            duplicadas_db += 1
            await uow.eventos.registrar(
                "deduplicacion.omitido",
                f"Propiedad duplicada omitida ({cand.item.dedup_key}) en {plataforma}",
                job_id=job_id,
                nivel="info",
                payload={
                    "dedup_key": cand.item.dedup_key,
                    "url_origen": cand.item.propiedad.id_plataforma_origen,
                    "plataforma": plataforma,
                },
            )
            return None

        orm_prop = map_normalized_property_to_orm(
            job_id=job_id,
            plataforma=plataforma,
            propiedad=cand.item.propiedad,
            huella_dedupe=cand.item.dedup_key,
        )
        saved = await uow.propiedades.guardar(orm_prop)
        await uow.eventos.registrar(
            "propiedad.persistida",
            f"Propiedad guardada ({cand.item.dedup_key}) desde {plataforma}",
            job_id=job_id,
            nivel="info",
            payload={
                "propiedad_id": str(saved.id),
                "dedup_key": cand.item.dedup_key,
                "fuentes": cand.item.fuentes,
                "enriquecimiento": cand.enrichment.model_dump(mode="json"),
            },
        )
        return saved

    resultados, fallos = await procesar_por_propiedad(
        unidad,
        candidates,
        _persistir_candidato,
        job_id=job_id,
    )
    guardadas = [row for row in resultados if row is not None]

    return IngestionBatchSummary(
        job_id=job_id,
        plataforma=plataforma,
        normalizadas=len(batch.valid),
        rechazadas_normalizacion=len(batch.rejected),
        duplicadas_en_lote=duplicadas_lote,
        duplicadas_en_db=duplicadas_db,
        persistidas=len(guardadas),
        fallidas_persistencia=len(fallos),
        propiedad_ids=[row.id for row in guardadas if row.id is not None],
        enriquecimientos=enriquecimientos,
        errores_persistencia=[motivo for _, motivo in fallos],
    )
