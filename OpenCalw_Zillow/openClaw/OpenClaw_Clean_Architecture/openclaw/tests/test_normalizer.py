"""Pruebas del normalizador de markdown a DatosPropiedadNormalizados (S1-11).

Verifica:
- 20 fixtures realistas cubriendo las 9 plataformas producen objetos
  `DatosPropiedadNormalizados` válidos con valores exactos.
- Anuncios incompletos o con campos críticos inválidos son rechazados
  explícitamente con motivo auditable.
- Flujo integrado cassette Firecrawl -> normalizador sin red.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.models.property import DatosPropiedadNormalizados
from app.services.normalizer import (
    NormalizationError,
    normalizar_markdown_propiedades,
    normalizar_propiedad_unica,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "normalizer"
CASSETTES_DIR = Path(__file__).parent / "cassettes" / "scraper"

VALID_FIXTURES: list[dict[str, Any]] = json.loads(
    (FIXTURES_DIR / "valid_20_fixtures.json").read_text(encoding="utf-8")
)
INCOMPLETE_FIXTURES: list[dict[str, Any]] = json.loads(
    (FIXTURES_DIR / "incomplete_fixtures.json").read_text(encoding="utf-8")
)


def test_exactly_twenty_valid_fixtures_covering_all_nine_platforms() -> None:
    assert len(VALID_FIXTURES) == 20
    covered_platforms = {item["platform"] for item in VALID_FIXTURES}
    assert covered_platforms == {
        "zillow",
        "realtor",
        "redfin",
        "trulia",
        "loopnet",
        "craigslist",
        "century21",
        "coldwell",
        "remax",
    }


@pytest.mark.parametrize("fixture", VALID_FIXTURES, ids=lambda f: f["id"])
def test_twenty_valid_fixtures_produce_valid_normalized_properties(
    fixture: dict[str, Any],
) -> None:
    """Criterio S1-11: 20 fixtures producen objetos DatosPropiedadNormalizados válidos."""
    prop = normalizar_propiedad_unica(
        fixture["markdown"],
        plataforma=fixture["platform"],
        url_origen=fixture["url_origen"],
    )

    assert isinstance(prop, DatosPropiedadNormalizados)
    expected = fixture["expected"]
    assert prop.precio_usd == pytest.approx(expected["precio_usd"])
    assert prop.condado == expected["condado"]
    assert prop.ingreso_anual_estimado == pytest.approx(expected["ingreso_anual_estimado"])
    assert len(prop.imagenes) >= expected["min_images"]
    assert len(prop.ubicacion) >= 5
    assert len(prop.id_plataforma_origen) >= 3
    assert len(prop.descripcion_cruda) >= 10

    if expected["has_coords"]:
        assert prop.latitud is not None
        assert prop.longitud is not None
    else:
        assert prop.latitud is None
        assert prop.longitud is None

    # Asegura que logos/favicons fueron filtrados
    for img in prop.imagenes:
        assert "logo" not in img.lower()
        assert "favicon" not in img.lower()


@pytest.mark.parametrize("fixture", INCOMPLETE_FIXTURES, ids=lambda f: f["id"])
def test_incomplete_fixtures_are_rejected_explicitly(fixture: dict[str, Any]) -> None:
    """Criterio S1-11: los anuncios incompletos se rechazan con motivo trazable."""
    with pytest.raises(NormalizationError) as exc_info:
        normalizar_propiedad_unica(
            fixture["markdown"],
            plataforma=fixture["platform"],
            url_origen=fixture["url_origen"],
        )
    assert fixture["expected_reason"] in exc_info.value.reasons

    batch = normalizar_markdown_propiedades(
        fixture["markdown"],
        plataforma=fixture["platform"],
        url_origen=fixture["url_origen"],
    )
    assert batch.valid == []
    assert len(batch.rejected) == 1
    assert fixture["expected_reason"] in batch.rejected[0].reasons


def test_batch_normalizer_splits_mixed_valid_and_invalid_listings() -> None:
    valid_md = VALID_FIXTURES[0]["markdown"]
    invalid_md = INCOMPLETE_FIXTURES[0]["markdown"]
    valid_md_2 = VALID_FIXTURES[1]["markdown"]

    combined = f"{valid_md}\n\n---\n\n{invalid_md}\n\n---\n\n{valid_md_2}"
    batch = normalizar_markdown_propiedades(
        combined,
        plataforma="zillow",
        url_origen="https://www.zillow.com/homes/Cleveland-OH_rb/",
    )

    assert len(batch.valid) == 2
    assert len(batch.rejected) == 1
    assert batch.rejected[0].index == 1
    assert "missing_or_invalid_price" in batch.rejected[0].reasons


def test_cassette_markdown_flows_cleanly_into_normalizer() -> None:
    cassette = json.loads((CASSETTES_DIR / "zillow_success.json").read_text(encoding="utf-8"))
    markdown = cassette["response"]["data"]["markdown"]

    prop = normalizar_propiedad_unica(
        markdown,
        plataforma="zillow",
        url_origen=cassette["response"]["data"]["metadata"]["sourceURL"],
    )
    assert prop.precio_usd == 115000.0
    assert prop.condado == "Cuyahoga"
    assert prop.ingreso_anual_estimado == 17400.0
    assert prop.latitud == pytest.approx(41.4822)
    assert prop.longitud == pytest.approx(-81.7341)
