"""Estratega macroeconómico de dominio para la Fase 96 (Dev 2).

Características obligatorias (PLAN_IMPLEMENTACION_FASES_FALTANTES_2DEVS.md):
- `MacroDataPort` + adaptador de fuentes permitidas (allowlist explícita: FRED, BLS, Census).
- Normalización de vacancia, tasa de interés hipotecaria, inflación y cap rate objetivo.
- Reglas de ajuste deterministas y reproducibles (sin LLM opaco).
- Deduplicación de observaciones por `(fuente, indicador, fecha_observacion)`.
- Fallback seguro: una fuente caída conserva el último valor válido y genera alerta.
- Ninguna versión se activa sin fuente, timestamp y aprobación humana explícita.
- Conversión directa de la versión activa a `FinancialConfig` para el grafo.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.orchestrator.config import FinancialConfig

ALLOWED_MACRO_SOURCES: Final[frozenset[str]] = frozenset({"fred", "bls", "census", "fhfa"})


class MacroIndicador(StrEnum):
    """Indicadores macroeconómicos soportados."""

    VACANCIA = "vacancia"
    TASA_INTERES = "tasa_interes"
    INFLACION = "inflacion"


class MacroVersionEstado(StrEnum):
    """Ciclo de vida controlado de una versión de parámetros macro."""

    PENDIENTE_APROBACION = "pendiente_aprobacion"
    ACTIVA = "activa"
    HISTORICA = "historica"
    RECHAZADA = "rechazada"


class MacroObservacion(BaseModel):
    """Observación normalizada e idempotente proveniente de una fuente permitida."""

    model_config = ConfigDict(extra="forbid")

    fuente: str = Field(..., min_length=2, max_length=40)
    indicador: MacroIndicador
    valor: float = Field(..., ge=0.0, le=1.0, description="Tasa en fracción decimal (ej. 0.065).")
    fecha_observacion: date
    capturada_en: datetime = Field(default_factory=lambda: datetime.now(UTC))
    confianza: float = Field(default=0.95, ge=0.0, le=1.0)

    def clave_dedupe(self) -> tuple[str, str, str]:
        return (self.fuente.lower(), self.indicador.value, self.fecha_observacion.isoformat())


class MacroCambioDetalle(BaseModel):
    """Trazabilidad individual por parámetro en el informe de cambio macro."""

    model_config = ConfigDict(extra="forbid")

    parametro: str
    fuente: str
    fecha_observacion: date | None = None
    valor_anterior: float
    valor_nuevo: float
    delta: float
    confianza: float = Field(..., ge=0.0, le=1.0)
    motivo: str


class MacroParametrosVersion(BaseModel):
    """Versión auditable de parámetros financieros derivados de señales macro."""

    model_config = ConfigDict(extra="forbid")

    version_id: UUID = Field(default_factory=uuid4)
    etiqueta: str
    estado: MacroVersionEstado = MacroVersionEstado.PENDIENTE_APROBACION
    vacancy_rate: float = Field(..., ge=0.0, lt=1.0)
    mortgage_rate: float = Field(..., ge=0.0, lt=1.0)
    inflation_rate: float = Field(..., ge=0.0, lt=1.0)
    maintenance_rate: float = Field(..., ge=0.0, lt=1.0)
    property_tax_rate: float = Field(..., ge=0.0, lt=1.0)
    insurance_annual_usd: float = Field(..., ge=0.0)
    target_cap_rate_pct: float = Field(..., ge=0.0, le=100.0)
    fuentes_usadas: list[str] = Field(default_factory=list)
    alertas_fuentes: list[str] = Field(default_factory=list)
    cambios: list[MacroCambioDetalle] = Field(default_factory=list)
    creada_en: datetime = Field(default_factory=lambda: datetime.now(UTC))
    aprobada_por: str | None = None
    aprobada_en: datetime | None = None

    def to_financial_config(self) -> FinancialConfig:
        """Traduce esta versión a `FinancialConfig` consumible por el grafo LangGraph."""
        return FinancialConfig(
            vacancy_rate=self.vacancy_rate,
            maintenance_rate=self.maintenance_rate,
            property_tax_rate=self.property_tax_rate,
            insurance_annual_usd=self.insurance_annual_usd,
            min_cap_rate_pct=self.target_cap_rate_pct,
            rules_version=self.etiqueta,
        )


@runtime_checkable
class MacroDataPort(Protocol):
    """Puerto de obtención de observaciones macroeconómicas desde fuentes permitidas."""

    @property
    def fuente(self) -> str: ...

    async def obtener_observaciones(self) -> list[MacroObservacion]: ...


class StaticMacroAdapter:
    """Adaptador determinista de `MacroDataPort` para fuentes permitidas y pruebas offline."""

    def __init__(
        self,
        *,
        fuente: str,
        observaciones: list[MacroObservacion] | None = None,
        error: Exception | None = None,
    ) -> None:
        normalized = fuente.strip().lower()
        if normalized not in ALLOWED_MACRO_SOURCES:
            allowed = ", ".join(sorted(ALLOWED_MACRO_SOURCES))
            raise ValueError(
                f"Fuente macro no permitida: {fuente!r}. Fuentes válidas: [{allowed}]."
            )
        self._fuente = normalized
        self._observaciones = list(observaciones or [])
        self._error = error

    @property
    def fuente(self) -> str:
        return self._fuente

    async def obtener_observaciones(self) -> list[MacroObservacion]:
        if self._error is not None:
            raise self._error
        return list(self._observaciones)


def normalizar_tasa_porcentual(valor: float) -> float:
    """Normaliza una tasa aceptando tanto porcentaje (6.5) como fracción decimal (0.065)."""
    if valor < 0:
        raise ValueError("Una tasa macroeconómica no puede ser negativa.")
    return round(valor / 100.0, 6) if valor > 1.0 else round(valor, 6)


class MacroStrategistService:
    """Gestiona observaciones deduplicadas, cálculo determinista y aprobación humana."""

    def __init__(
        self,
        *,
        version_inicial: MacroParametrosVersion | None = None,
        spread_cap_rate_pct: float = 1.75,
        min_cap_rate_floor_pct: float = 7.5,
        max_cap_rate_ceiling_pct: float = 14.0,
    ) -> None:
        self._spread = spread_cap_rate_pct
        self._floor = min_cap_rate_floor_pct
        self._ceiling = max_cap_rate_ceiling_pct
        self._observaciones: dict[tuple[str, str, str], MacroObservacion] = {}
        self._versiones: dict[UUID, MacroParametrosVersion] = {}

        base = version_inicial or MacroParametrosVersion(
            etiqueta="macro-v1.0",
            estado=MacroVersionEstado.ACTIVA,
            vacancy_rate=0.08,
            mortgage_rate=0.0625,
            inflation_rate=0.03,
            maintenance_rate=0.10,
            property_tax_rate=0.02,
            insurance_annual_usd=1200.0,
            target_cap_rate_pct=8.0,
            fuentes_usadas=["baseline"],
            aprobada_por="system_bootstrap",
            aprobada_en=datetime.now(UTC),
        )
        self._versiones[base.version_id] = base
        self._activa_id: UUID = base.version_id

    @property
    def version_activa(self) -> MacroParametrosVersion:
        return self._versiones[self._activa_id]

    @property
    def observaciones_registradas(self) -> list[MacroObservacion]:
        return list(self._observaciones.values())

    def registrar_observacion(self, obs: MacroObservacion) -> bool:
        """Inserta una observación si no existe aún; devuelve False si es duplicada."""
        fuente_norm = obs.fuente.strip().lower()
        if fuente_norm not in ALLOWED_MACRO_SOURCES:
            raise ValueError(f"Fuente macro no autorizada: {obs.fuente!r}")
        clave = obs.clave_dedupe()
        if clave in self._observaciones:
            return False
        self._observaciones[clave] = obs.model_copy(update={"fuente": fuente_norm})
        return True

    async def recolectar_y_proponer(
        self,
        adaptadores: list[MacroDataPort],
        *,
        etiqueta: str,
    ) -> MacroParametrosVersion:
        """Recolecta fuentes permitidas, conserva valores previos ante fallos y crea borrador."""
        alertas: list[str] = []
        nuevas_por_indicador: dict[MacroIndicador, MacroObservacion] = {}

        for adaptador in adaptadores:
            if adaptador.fuente.strip().lower() not in ALLOWED_MACRO_SOURCES:
                alertas.append(f"fuente_rechazada_allowlist:{adaptador.fuente}")
                continue
            try:
                lote = await adaptador.obtener_observaciones()
            except Exception as exc:
                alertas.append(f"fuente_caida:{adaptador.fuente}:{type(exc).__name__}:{exc}")
                continue

            for obs in lote:
                self.registrar_observacion(obs)
                previa = nuevas_por_indicador.get(obs.indicador)
                if previa is None or obs.fecha_observacion >= previa.fecha_observacion:
                    nuevas_por_indicador[obs.indicador] = obs

        activa = self.version_activa
        cambios: list[MacroCambioDetalle] = []
        fuentes_usadas: list[str] = []

        # 1. Vacancia (si falta fuente o cae, conserva el último valor válido)
        obs_vac = nuevas_por_indicador.get(MacroIndicador.VACANCIA)
        new_vacancy = obs_vac.valor if obs_vac is not None else activa.vacancy_rate
        if obs_vac is not None:
            fuentes_usadas.append(obs_vac.fuente)
            cambios.append(
                MacroCambioDetalle(
                    parametro="vacancy_rate",
                    fuente=obs_vac.fuente,
                    fecha_observacion=obs_vac.fecha_observacion,
                    valor_anterior=activa.vacancy_rate,
                    valor_nuevo=new_vacancy,
                    delta=round(new_vacancy - activa.vacancy_rate, 6),
                    confianza=obs_vac.confianza,
                    motivo="Actualización directa desde observación de vacancia permitida.",
                )
            )

        # 2. Tasa de interés hipotecaria
        obs_rate = nuevas_por_indicador.get(MacroIndicador.TASA_INTERES)
        new_mortgage = obs_rate.valor if obs_rate is not None else activa.mortgage_rate
        if obs_rate is not None:
            fuentes_usadas.append(obs_rate.fuente)
            cambios.append(
                MacroCambioDetalle(
                    parametro="mortgage_rate",
                    fuente=obs_rate.fuente,
                    fecha_observacion=obs_rate.fecha_observacion,
                    valor_anterior=activa.mortgage_rate,
                    valor_nuevo=new_mortgage,
                    delta=round(new_mortgage - activa.mortgage_rate, 6),
                    confianza=obs_rate.confianza,
                    motivo="Actualización desde serie de tasa hipotecaria.",
                )
            )

        # 3. Inflación -> ajusta mantenimiento de forma determinista y acotada
        obs_inf = nuevas_por_indicador.get(MacroIndicador.INFLACION)
        new_inflation = obs_inf.valor if obs_inf is not None else activa.inflation_rate
        new_maintenance = activa.maintenance_rate
        if obs_inf is not None:
            fuentes_usadas.append(obs_inf.fuente)
            delta_inf = new_inflation - activa.inflation_rate
            new_maintenance = round(
                min(0.25, max(0.06, activa.maintenance_rate + delta_inf * 0.5)),
                4,
            )
            cambios.append(
                MacroCambioDetalle(
                    parametro="maintenance_rate",
                    fuente=obs_inf.fuente,
                    fecha_observacion=obs_inf.fecha_observacion,
                    valor_anterior=activa.maintenance_rate,
                    valor_nuevo=new_maintenance,
                    delta=round(new_maintenance - activa.maintenance_rate, 6),
                    confianza=obs_inf.confianza,
                    motivo="Regla determinista: +50% del cambio de inflación acotado a [6%, 25%].",
                )
            )

        # 4. Cap rate objetivo determinista = clamp(mortgage_pct + spread + prima vacancia)
        prima_vacancia_pct = max(0.0, (new_vacancy - 0.08) * 100.0 * 0.25)
        raw_cap = (new_mortgage * 100.0) + self._spread + prima_vacancia_pct
        new_target_cap = round(min(self._ceiling, max(self._floor, raw_cap)), 2)
        cambios.append(
            MacroCambioDetalle(
                parametro="target_cap_rate_pct",
                fuente=",".join(sorted(set(fuentes_usadas))) or "fallback_ultima_activa",
                fecha_observacion=obs_rate.fecha_observacion if obs_rate else None,
                valor_anterior=activa.target_cap_rate_pct,
                valor_nuevo=new_target_cap,
                delta=round(new_target_cap - activa.target_cap_rate_pct, 4),
                confianza=min((o.confianza for o in nuevas_por_indicador.values()), default=0.5),
                motivo=(
                    f"Regla reproducible: mortgage({new_mortgage*100:.2f}%) + "
                    f"spread({self._spread:.2f}%) + prima_vacancia({prima_vacancia_pct:.2f}%)."
                ),
            )
        )

        propuesta = MacroParametrosVersion(
            etiqueta=etiqueta,
            estado=MacroVersionEstado.PENDIENTE_APROBACION,
            vacancy_rate=new_vacancy,
            mortgage_rate=new_mortgage,
            inflation_rate=new_inflation,
            maintenance_rate=new_maintenance,
            property_tax_rate=activa.property_tax_rate,
            insurance_annual_usd=activa.insurance_annual_usd,
            target_cap_rate_pct=new_target_cap,
            fuentes_usadas=sorted(set(fuentes_usadas)),
            alertas_fuentes=alertas,
            cambios=cambios,
        )
        self._versiones[propuesta.version_id] = propuesta
        return propuesta

    def aprobar_y_activar_version(
        self,
        version_id: UUID,
        *,
        aprobado_por: str,
    ) -> MacroParametrosVersion:
        """Activa una versión pendiente exigiendo fuente, timestamp y aprobación humana."""
        if not aprobado_por or not aprobado_por.strip():
            raise ValueError("Se requiere identidad humana en 'aprobado_por' para activar.")
        version = self._versiones.get(version_id)
        if version is None:
            raise KeyError(f"Versión macro no encontrada: {version_id}")
        if not version.fuentes_usadas:
            raise ValueError(
                "No se puede activar una versión macro sin al menos una fuente verificada."
            )

        anterior = self.version_activa
        self._versiones[anterior.version_id] = anterior.model_copy(
            update={"estado": MacroVersionEstado.HISTORICA}
        )
        activada = version.model_copy(
            update={
                "estado": MacroVersionEstado.ACTIVA,
                "aprobada_por": aprobado_por.strip(),
                "aprobada_en": datetime.now(UTC),
            }
        )
        self._versiones[version_id] = activada
        self._activa_id = version_id
        return activada
