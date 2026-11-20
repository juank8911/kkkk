from app.services.market_data import fuentes_directas_mercado


def test_catalogo_usa_fuentes_directas_sin_score_geoespacial() -> None:
    fuentes = fuentes_directas_mercado()
    ids = {fuente["id"] for fuente in fuentes}
    assert "census_geocoder" in ids
    assert "hud_fmr_safmr" in ids
    assert "comparables_renta" in ids
    assert "comparables_venta" in ids
    assert "riesgo_fisico" in ids
    assert all("score" not in fuente for fuente in fuentes)
    assert all("secret" not in str(fuente).lower() for fuente in fuentes)
