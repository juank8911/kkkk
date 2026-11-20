"""Pruebas de Fase 115: oráculo geoespacial en modo disabled, proxies y estabilidad."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.services.geospatial_oracle import (
    GeospatialDisabledError,
    GeospatialMode,
    GeospatialOracleService,
    GeospatialZoneData,
    ProhibitedProxyVariableError,
    StaticGeospatialAdapter,
    validar_variables_permitidas,
)


def _zona_ejemplo(
    zone_id: str = "OH-CLE-01",
    *,
    permits_yoy: float = 18.0,
    density: float = 120.0,
    transit: float = 65.0,
    infra: int = 8,
) -> GeospatialZoneData:
    return GeospatialZoneData(
        zone_id=zone_id,
        city="Cleveland",
        state="OH",
        data_version="2026.11",
        spatial_resolution="grid_1km_commercial",
        observed_at=datetime(2026, 11, 20, 10, 0, tzinfo=UTC),
        source="Cleveland Open Data Permits & Transit Registry",
        license="CC-BY-4.0",
        commercial_permits_yoy_pct=permits_yoy,
        commercial_density_per_km2=density,
        transit_investment_index=transit,
        infrastructure_projects_count=infra,
    )


async def test_modulo_permanece_disabled_sin_aprobacion_g2() -> None:
    adapter = StaticGeospatialAdapter({"OH-CLE-01": _zona_ejemplo()})

    svc_default = GeospatialOracleService(adapter)
    assert svc_default.mode == GeospatialMode.DISABLED
    with pytest.raises(GeospatialDisabledError):
        await svc_default.calcular_score("OH-CLE-01")

    svc_sin_g2 = GeospatialOracleService(
        adapter,
        mode=GeospatialMode.SANDBOX_RESEARCH,
        g2_legal_and_bias_approved=False,
    )
    with pytest.raises(GeospatialDisabledError):
        await svc_sin_g2.calcular_score("OH-CLE-01")


def test_prohibe_proxies_de_clases_protegidas_y_variables_no_autorizadas() -> None:
    for variable_prohibida in (
        "race",
        "redlining_grade",
        "demographic_composition",
        "income_by_race",
        "zip_demographic_proxy",
    ):
        with pytest.raises(ProhibitedProxyVariableError):
            validar_variables_permitidas({variable_prohibida: 0.5})

        with pytest.raises(ProhibitedProxyVariableError):
            GeospatialZoneData(
                zone_id="OH-CLE-BAD",
                city="Cleveland",
                state="OH",
                data_version="2026.11",
                spatial_resolution="grid_1km_commercial",
                source="Open Data",
                license="CC-BY-4.0",
                commercial_permits_yoy_pct=10.0,
                commercial_density_per_km2=80.0,
                transit_investment_index=50.0,
                infrastructure_projects_count=4,
                extra_variables={variable_prohibida: 1.0},
            )

    with pytest.raises(ProhibitedProxyVariableError):
        validar_variables_permitidas({"opaque_blackbox_feature": 42.0})


async def test_score_explicable_reproducible_y_solo_informativo_con_g2() -> None:
    zona = _zona_ejemplo()
    adapter = StaticGeospatialAdapter({"OH-CLE-01": zona})
    svc = GeospatialOracleService(
        adapter,
        mode=GeospatialMode.SANDBOX_RESEARCH,
        g2_legal_and_bias_approved=True,
    )

    r1 = await svc.calcular_score("OH-CLE-01")
    r2 = await svc.calcular_score("OH-CLE-01")

    assert r1 == r2
    assert r1.informativo_unicamente is True
    assert r1.uso_decision_automatica_prohibido is True
    assert r1.source == "Cleveland Open Data Permits & Transit Registry"
    assert r1.license == "CC-BY-4.0"
    assert r1.data_version == "2026.11"
    assert r1.spatial_resolution == "grid_1km_commercial"
    assert len(r1.contribuciones) == 4
    suma_contribuciones = round(sum(c.contribution for c in r1.contribuciones), 2)
    assert r1.score == suma_contribuciones
    assert "Fair Housing Act" in r1.disclaimer


def test_auditoria_estabilidad_y_disparidad_detecta_rangos_extremos() -> None:
    z1 = _zona_ejemplo("OH-CLE-01", permits_yoy=12.0, density=90.0, transit=55.0, infra=6)
    z2 = _zona_ejemplo("OH-CLE-02", permits_yoy=20.0, density=115.0, transit=62.0, infra=8)
    z_extrema = _zona_ejemplo("OH-CLE-99", permits_yoy=60.0, density=250.0, transit=100.0, infra=20)
    z_minima = _zona_ejemplo("OH-CLE-00", permits_yoy=-20.0, density=0.0, transit=0.0, infra=0)

    svc = GeospatialOracleService(StaticGeospatialAdapter({}))

    reporte_ok = svc.evaluar_estabilidad_y_disparidad([z1, z2])
    assert reporte_ok.estable is True
    assert reporte_ok.disparidad_dentro_umbral is True
    assert reporte_ok.aprobado_auditoria_tecnica is True

    reporte_dispar = svc.evaluar_estabilidad_y_disparidad([z_minima, z_extrema])
    assert reporte_dispar.disparidad_dentro_umbral is False
    assert reporte_dispar.aprobado_auditoria_tecnica is False
