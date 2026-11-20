"""Pruebas de URLs de plataformas (Redfin/Realtor/Zillow) y pipeline Ollama/determinista -> normalizador."""

from __future__ import annotations

from app.services.normalizer import normalizar_markdown_propiedades
from app.services.ollama_listing_extractor import (
    OllamaExtraction,
    OllamaListing,
    enriquecer_ubicacion_extraccion,
    extraction_to_markdown,
    extraer_viviendas_determinista,
)
from app.services.platforms import build_platform_url


def test_build_platform_url_resolves_redfin_realtor_and_zillow_cleanly() -> None:
    assert (
        build_platform_url("redfin", "Memphis, TN")
        == "https://www.redfin.com/city/11883/TN/Memphis"
    )
    assert (
        build_platform_url("redfin", "Cleveland, OH")
        == "https://www.redfin.com/city/4243/OH/Cleveland"
    )
    assert build_platform_url("redfin", "38104") == "https://www.redfin.com/zipcode/38104"
    assert (
        build_platform_url("realtor", "Memphis, TN")
        == "https://www.realtor.com/realestateandhomes-search/Memphis_TN"
    )
    assert (
        build_platform_url("zillow", "Memphis, TN") == "https://www.zillow.com/homes/Memphis-TN_rb/"
    )


async def test_ollama_extraction_formats_usd_price_and_resolves_county(monkeypatch) -> None:
    async def fake_census(_direccion: str):
        return None

    monkeypatch.setattr(
        "app.services.ollama_listing_extractor.geocodificar_census",
        fake_census,
    )

    extraction = OllamaExtraction(
        viviendas_en_venta=[
            OllamaListing(
                url="https://www.zillow.com/homedetails/123-Main-St-Memphis-TN-38104/12345_zpid/",
                direccion="123 Main St",
                ciudad="Memphis, TN",
                codigo_postal="38104",
                condado=None,
                precio_usd=118000.0,
                habitaciones=3,
                banos=2,
                renta_mensual_usd=1350.0,
                descripcion="Single family home updated kitchen and roof.",
            )
        ]
    )

    enriquecida = await enriquecer_ubicacion_extraccion(extraction, query="Memphis, TN")
    assert enriquecida.viviendas_en_venta[0].condado == "Shelby"

    md = extraction_to_markdown(
        enriquecida,
        platform="zillow",
        url_origen="https://www.zillow.com/homes/Memphis-TN_rb/",
        query="Memphis, TN",
    )
    assert "Price: $118,000.00" in md
    assert "County: Shelby" in md

    batch = normalizar_markdown_propiedades(
        md,
        plataforma="zillow",
        url_origen="https://www.zillow.com/homes/Memphis-TN_rb/",
    )
    assert len(batch.valid) == 1
    assert batch.rejected == []
    prop = batch.valid[0]
    assert prop.precio_usd == 118000.0
    assert prop.condado == "Shelby"
    assert prop.ingreso_anual_estimado == 16200.0


def test_deterministic_fallback_extracts_realtor_zillow_and_filters_wrong_state() -> None:
    md_realtor = """
    [688 E McKellar Ave, Memphis, TN 38106](https://www.realtor.com/realestateandhomes-detail/688-E-McKellar-Ave_Memphis_TN_38106_M87694-00926)
    House for sale
    $60,000
    - 3bed
    - 1.5bath
    - 1,309sqft

    [125 E Lake St, Rockwell City, IA 50579](https://www.redfin.com/IA/Rockwell-City/125-E-Lake-St-50579/home/116171017)
    $139,900
    3 beds 1 bath 1,231 sq ft

    - [$339,000](https://www.zillow.com/homedetails/5577-Glenwild-Ave-Memphis-TN-38119/42215378_zpid/)
      - **3** bds
      - **2** ba
      - **2,584** sqft
    House for sale
    [5577 Glenwild Ave, Memphis, TN 38119](https://www.zillow.com/homedetails/5577-Glenwild-Ave-Memphis-TN-38119/42215378_zpid/)
    """
    ext = extraer_viviendas_determinista(
        md_realtor,
        query="Memphis, TN",
        platform="realtor",
        url_origen="https://www.realtor.com/realestateandhomes-search/Memphis_TN",
    )
    assert len(ext.viviendas_en_venta) == 2
    precios = {v.precio_usd for v in ext.viviendas_en_venta}
    assert precios == {60000.0, 339000.0}
    assert all(v.condado == "Shelby" for v in ext.viviendas_en_venta)
    assert any("estado_distinto:IA!=TN" in r["motivo"] for r in ext.rechazadas)


def test_normalizer_allows_missing_county_only_when_explicitly_disabled() -> None:
    md = (
        "Listing URL: https://www.zillow.com/homedetails/99-Test-Ln/1_zpid/\n"
        "Property Address: 99 Test Ln, Unknownville, ZZ 99999\n"
        "Price: $92,500\n"
        "Description: Renovated 3 bedroom brick house ready for tenants.\n"
    )
    strict = normalizar_markdown_propiedades(md, plataforma="zillow")
    assert strict.valid == []
    assert "missing_county" in strict.rejected[0].reasons

    relaxed = normalizar_markdown_propiedades(md, plataforma="zillow", require_county=False)
    assert len(relaxed.valid) == 1
    assert relaxed.valid[0].condado == "Por verificar"
