"""Pruebas de enriquecimiento (S3-05) y deduplicación de propiedades (S3-07)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.models.property import DatosPropiedadNormalizados
from app.services.deduplicator import (
    calcular_similitud_direcciones,
    deduplicar_propiedades,
    normalizar_direccion,
)
from app.services.enrichment import enriquecer_propiedad_desde_markdown

ENRICHMENT_FIXTURES: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "fixtures" / "enrichment" / "real_20_ads.json").read_text(
        encoding="utf-8"
    )
)


def test_enrichment_fixture_count_is_twenty() -> None:
    assert len(ENRICHMENT_FIXTURES) == 20


@pytest.mark.parametrize("ad", ENRICHMENT_FIXTURES, ids=lambda a: a["id"])
def test_s3_05_enrichment_extracts_four_fields_or_marks_absence(ad: dict[str, Any]) -> None:
    """Criterio S3-05: 20 anuncios reales extraen los 4 campos o marcan ausencia."""
    result = enriquecer_propiedad_desde_markdown(ad["markdown"])
    exp = ad["expected"]

    assert result.precio_usd == pytest.approx(exp["precio_usd"])
    assert result.superficie_sqft == pytest.approx(exp["superficie_sqft"])
    assert result.anio_construccion == exp["anio_construccion"]
    assert result.hoa_mensual_usd == pytest.approx(exp["hoa_mensual_usd"])
    assert result.campos_ausentes == exp["ausentes"]
    assert result.completo is (len(exp["ausentes"]) == 0)


def test_s3_07_two_listings_of_same_house_merge_into_one_property() -> None:
    """Criterio S3-07: dos anuncios de la misma casa quedan como una sola propiedad."""
    from_zillow = DatosPropiedadNormalizados(
        id_plataforma_origen="https://www.zillow.com/homedetails/1420-n-maple-ave/1001_zpid/",
        ubicacion="1420 North Maple Avenue, Cleveland, OH 44102",
        precio_usd=115_000.0,
        ingreso_anual_estimado=17_400.0,
        imagenes=["https://photos.zillowstatic.com/1001-a.jpg"],
        condado="Cuyahoga",
        latitud=41.4822,
        longitud=-81.7341,
        descripcion_cruda="Brick colonial 3 bed 1 bath.",
    )
    from_realtor = DatosPropiedadNormalizados(
        id_plataforma_origen="https://www.realtor.com/realestateandhomes-detail/1420-n-maple-ave_M1",
        ubicacion="1420 N. Maple Ave., Cleveland, OH 44102",
        precio_usd=116_000.0,
        ingreso_anual_estimado=0.0,
        imagenes=[
            "https://photos.zillowstatic.com/1001-a.jpg",
            "https://ap.rdcpix.com/1001-b.jpg",
        ],
        condado="Cuyahoga",
        latitud=None,
        longitud=None,
        descripcion_cruda=(
            "Brick colonial 3 bed 1 bath with updated electrical panel and detached garage."
        ),
    )
    different_house = DatosPropiedadNormalizados(
        id_plataforma_origen="https://www.redfin.com/OH/Cleveland/1422-N-Maple-Ave/2002",
        ubicacion="1422 N Maple Ave, Cleveland, OH 44102",
        precio_usd=118_000.0,
        ingreso_anual_estimado=16_800.0,
        imagenes=["https://ssl.cdn-redfin.com/2002.jpg"],
        condado="Cuyahoga",
        descripcion_cruda="Adjacent house with distinct street number 1422.",
    )

    assert normalizar_direccion(from_zillow.ubicacion) == normalizar_direccion(
        from_realtor.ubicacion
    )
    assert calcular_similitud_direcciones(from_zillow.ubicacion, different_house.ubicacion) == 0.0

    result = deduplicar_propiedades([from_zillow, from_realtor, different_house])
    assert len(result) == 2

    merged = result[0]
    assert merged.duplicados_fusionados == 1
    assert len(merged.fuentes) == 2
    assert len(merged.propiedad.imagenes) == 2
    assert merged.propiedad.ingreso_anual_estimado == 17_400.0
    assert merged.propiedad.latitud == pytest.approx(41.4822)
    assert "detached garage" in merged.propiedad.descripcion_cruda
