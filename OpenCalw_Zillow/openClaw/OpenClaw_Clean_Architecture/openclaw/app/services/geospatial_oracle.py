"""Fase 115 — Oráculo geoespacial y Gentrification Score (Dev 2).

Permanece estrictamente en modo ``disabled`` por defecto hasta aprobación legal
y de sesgos (Puerta G2). Prohibe variables o proxies de clases protegidas
(Fair Housing Act / ECOA) y entrega únicamente señales informativas explicables,
nunca decisiones automáticas de vivienda o crédito.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, Self

from pydantic import BaseModel, Field, model_validator

GEOSPATIAL_DISCLAIMER = (
    "SEÑAL INFORMATIVA DE ACTIVIDAD URBANA ÚNICAMENTE. Prohibido su uso como "
    "decisión automática de vivienda, crédito, fijación discriminatoria de "
    "precios o exclusión de personas o grupos protegidos (Fair Housing Act / ECOA)."
)

PROHIBITED_PROXY_VARIABLES: frozenset[str] = frozenset(
    {
        "race",
        "ethnicity",
        "religion",
        "national_origin",
        "color",
        "sex",
        "gender",
        "disability",
        "familial_status",
        "age_group",
        "marital_status",
        "redlining_grade",
        "holc_grade",
        "income_by_race",
        "demographic_composition",
        "minority_pct",
        "language_spoken",
        "ancestry",
        "zip_demographic_proxy",
        "crime_by_demographic",
    }
)

ALLOWED_EXPLAINABLE_VARIABLES: frozenset[str] = frozenset(
    {
        "commercial_permits_yoy_pct",
        "commercial_density_per_km2",
        "transit_investment_index",
        "infrastructure_projects_count",
    }
)

DEFAULT_WEIGHTS: dict[str, float] = {
    "commercial_permits_yoy_pct": 0.30,
    "commercial_density_per_km2": 0.25,
    "transit_investment_index": 0.25,
    "infrastructure_projects_count": 0.20,
}


class GeospatialMode(StrEnum):
    """Modos permitidos del oráculo geoespacial (nunca decisión automática)."""

    DISABLED = "disabled"
    SANDBOX_RESEARCH = "sandbox_research"


class GeospatialOracleError(Exception):
    """Error base tipado del oráculo geoespacial."""


class GeospatialDisabledError(GeospatialOracleError):
    """El módulo permanece deshabilitado hasta aprobación legal y de sesgos (G2)."""


class ProhibitedProxyVariableError(GeospatialOracleError):
    """Se intentó introducir una variable prohibida o no autorizada."""


def validar_variables_permitidas(variables: dict[str, float]) -> None:
    """Rechaza cualquier variable proxy de clase protegida o fuera de la allowlist."""
    for nombre in variables:
        clave = nombre.strip().lower()
        if clave in PROHIBITED_PROXY_VARIABLES or any(
            prohibida in clave for prohibida in PROHIBITED_PROXY_VARIABLES
        ):
            raise ProhibitedProxyVariableError(
                f"Variable prohibida (Fair Housing / ECOA proxy): '{nombre}'"
            )
        if clave not in ALLOWED_EXPLAINABLE_VARIABLES:
            raise ProhibitedProxyVariableError(
                f"Variable no autorizada en el modelo explicable: '{nombre}'"
            )


class GeospatialZoneData(BaseModel):
    """Señales urbanas explicables y atribución de fuente/licencia de una zona."""

    zone_id: str = Field(..., min_length=2, max_length=64)
    city: str = Field(..., min_length=1, max_length=80)
    state: str = Field(..., min_length=2, max_length=2)
    data_version: str = Field(..., min_length=1, max_length=32)
    spatial_resolution: str = Field(..., min_length=1, max_length=64)
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str = Field(..., min_length=2, max_length=160)
    license: str = Field(..., min_length=2, max_length=120)
    commercial_permits_yoy_pct: float = Field(..., ge=-100.0, le=500.0)
    commercial_density_per_km2: float = Field(..., ge=0.0, le=10000.0)
    transit_investment_index: float = Field(..., ge=0.0, le=100.0)
    infrastructure_projects_count: int = Field(..., ge=0, le=500)
    extra_variables: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validar_proxies_y_estado(self) -> Self:
        self.state = self.state.upper()
        if self.extra_variables:
            validar_variables_permitidas(self.extra_variables)
        return self


class GeospatialDataPort(Protocol):
    """Puerto para obtener señales geoespaciales abiertas y auditables."""

    async def obtener_senales_zona(self, zone_id: str) -> GeospatialZoneData:
        """Devuelve las señales explicables de la zona solicitada."""
        ...


class StaticGeospatialAdapter:
    """Adaptador determinista en memoria para pruebas y auditoría sin red."""

    def __init__(self, zonas: dict[str, GeospatialZoneData]) -> None:
        self._zonas = dict(zonas)

    async def obtener_senales_zona(self, zone_id: str) -> GeospatialZoneData:
        if zone_id not in self._zonas:
            raise GeospatialOracleError(f"Zona geoespacial no encontrada: {zone_id}")
        return self._zonas[zone_id]


class FactorContribution(BaseModel):
    """Contribución individual explicable al índice urbano."""

    variable: str
    raw_value: float
    normalized_value: float = Field(..., ge=0.0, le=100.0)
    weight: float = Field(..., gt=0.0, le=1.0)
    contribution: float = Field(..., ge=0.0, le=100.0)


class GentrificationScoreResult(BaseModel):
    """Resultado informativo, explicable y reproducible del índice urbano."""

    zone_id: str
    score: float = Field(..., ge=0.0, le=100.0)
    banda_informativa: str
    contribuciones: list[FactorContribution]
    source: str
    license: str
    data_version: str
    spatial_resolution: str
    observed_at: datetime
    informativo_unicamente: bool = True
    uso_decision_automatica_prohibido: bool = True
    disclaimer: str = GEOSPATIAL_DISCLAIMER


class BiasStabilityReport(BaseModel):
    """Resultado de auditoría técnica de estabilidad y disparidad entre zonas."""

    total_zonas: int
    max_sensibilidad_perturbacion: float
    estable: bool
    disparidad_rango: float
    disparidad_dentro_umbral: bool
    aprobado_auditoria_tecnica: bool
    notas: list[str] = Field(default_factory=list)


def _clamp(valor: float, minimo: float = 0.0, maximo: float = 100.0) -> float:
    return max(minimo, min(maximo, valor))


def _normalizar_variables(zona: GeospatialZoneData) -> dict[str, tuple[float, float]]:
    """Normaliza las 4 variables permitidas a escala 0..100 de forma determinista."""
    permits_norm = _clamp(((zona.commercial_permits_yoy_pct + 20.0) / 70.0) * 100.0)
    density_norm = _clamp((zona.commercial_density_per_km2 / 250.0) * 100.0)
    transit_norm = _clamp(zona.transit_investment_index)
    infra_norm = _clamp((zona.infrastructure_projects_count / 20.0) * 100.0)
    return {
        "commercial_permits_yoy_pct": (zona.commercial_permits_yoy_pct, round(permits_norm, 4)),
        "commercial_density_per_km2": (zona.commercial_density_per_km2, round(density_norm, 4)),
        "transit_investment_index": (zona.transit_investment_index, round(transit_norm, 4)),
        "infrastructure_projects_count": (
            float(zona.infrastructure_projects_count),
            round(infra_norm, 4),
        ),
    }


def _clasificar_banda(score: float) -> str:
    if score >= 70.0:
        return "alta_actividad_urbana"
    if score >= 40.0:
        return "actividad_moderada"
    return "baja_actividad"


class GeospatialOracleService:
    """Servicio explicable de scoring urbano bloqueado por defecto hasta G2."""

    def __init__(
        self,
        port: GeospatialDataPort,
        *,
        mode: GeospatialMode = GeospatialMode.DISABLED,
        g2_legal_and_bias_approved: bool = False,
    ) -> None:
        self._port = port
        self._mode = mode
        self._g2_approved = g2_legal_and_bias_approved

    @property
    def mode(self) -> GeospatialMode:
        return self._mode

    def _exigir_habilitado_g2(self) -> None:
        if self._mode == GeospatialMode.DISABLED or not self._g2_approved:
            raise GeospatialDisabledError(
                "Fase 115 permanece en modo disabled hasta aprobación legal y de "
                "auditoría de sesgos (Puerta G2)."
            )

    def _calcular_desde_datos(self, zona: GeospatialZoneData) -> GentrificationScoreResult:
        if zona.extra_variables:
            validar_variables_permitidas(zona.extra_variables)

        normalizadas = _normalizar_variables(zona)
        contribuciones: list[FactorContribution] = []
        total = 0.0
        for var_name, weight in DEFAULT_WEIGHTS.items():
            raw_val, norm_val = normalizadas[var_name]
            contrib = round(norm_val * weight, 4)
            total += contrib
            contribuciones.append(
                FactorContribution(
                    variable=var_name,
                    raw_value=raw_val,
                    normalized_value=norm_val,
                    weight=weight,
                    contribution=contrib,
                )
            )

        score = round(_clamp(total), 2)
        return GentrificationScoreResult(
            zone_id=zona.zone_id,
            score=score,
            banda_informativa=_clasificar_banda(score),
            contribuciones=contribuciones,
            source=zona.source,
            license=zona.license,
            data_version=zona.data_version,
            spatial_resolution=zona.spatial_resolution,
            observed_at=zona.observed_at,
            informativo_unicamente=True,
            uso_decision_automatica_prohibido=True,
        )

    async def calcular_score(self, zone_id: str) -> GentrificationScoreResult:
        """Calcula el índice urbano explicable solo si G2 fue aprobada."""
        self._exigir_habilitado_g2()
        zona = await self._port.obtener_senales_zona(zone_id)
        return self._calcular_desde_datos(zona)

    def evaluar_estabilidad_y_disparidad(
        self,
        zonas: list[GeospatialZoneData],
        *,
        perturbacion_pct: float = 0.05,
        max_delta_permitido: float = 6.0,
        max_disparidad_permitida: float = 45.0,
    ) -> BiasStabilityReport:
        """Audita estabilidad ante pequeñas perturbaciones y disparidad entre zonas.

        Esta auditoría puede ejecutarse en modo sandbox/validación previa a G2
        para documentar el comportamiento del modelo sin emitir decisiones.
        """
        if not zonas:
            raise GeospatialOracleError("Se requiere al menos una zona para auditar estabilidad")

        scores_base: list[float] = []
        max_delta = 0.0
        for zona in zonas:
            base = self._calcular_desde_datos(zona).score
            scores_base.append(base)
            zona_perturbada = zona.model_copy(
                update={
                    "commercial_permits_yoy_pct": min(
                        500.0, zona.commercial_permits_yoy_pct * (1.0 + perturbacion_pct)
                    ),
                    "commercial_density_per_km2": min(
                        10000.0, zona.commercial_density_per_km2 * (1.0 + perturbacion_pct)
                    ),
                    "transit_investment_index": min(
                        100.0, zona.transit_investment_index * (1.0 + perturbacion_pct)
                    ),
                }
            )
            perturbado = self._calcular_desde_datos(zona_perturbada).score
            max_delta = max(max_delta, round(abs(perturbado - base), 4))

        rango = round(max(scores_base) - min(scores_base), 2)
        estable = max_delta <= max_delta_permitido
        disparidad_ok = rango <= max_disparidad_permitida
        notas: list[str] = []
        if not estable:
            notas.append(
                f"Sensibilidad {max_delta} supera el umbral permitido ({max_delta_permitido})."
            )
        if not disparidad_ok:
            notas.append(
                f"Disparidad de rango {rango} supera el umbral ({max_disparidad_permitida}); "
                "requiere revisión legal de impacto dispar."
            )
        if estable and disparidad_ok:
            notas.append("Estabilidad y disparidad dentro de umbrales técnicos de referencia.")

        return BiasStabilityReport(
            total_zonas=len(zonas),
            max_sensibilidad_perturbacion=max_delta,
            estable=estable,
            disparidad_rango=rango,
            disparidad_dentro_umbral=disparidad_ok,
            aprobado_auditoria_tecnica=estable and disparidad_ok,
            notas=notas,
        )
