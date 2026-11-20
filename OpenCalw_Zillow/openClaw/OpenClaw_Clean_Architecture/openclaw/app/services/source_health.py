"""Gestión de salud y estados de fuentes externas sin evasión WAF — Fases 110–111/117 (Dev 2).

Define los cinco estados oficiales de una fuente del catálogo cerrado:
- `disponible`: operativa para consultas normales.
- `limitada`: cuota temporalmente alcanzada (rate limit); requiere espera.
- `degradada`: fallos transitorios repetidos; se recomienda reducir ritmo.
- `bloqueada`: HTTP 403/429/WAF/CAPTCHA; el worker NO insiste hasta expirar cooldown.
- `no_autorizada`: deshabilitada por política o términos de uso; bloqueada indefinidamente.

Nunca aplica técnicas de evasión de WAF o CAPTCHA.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.services.platforms import VALID_PLATFORM_KEYS

ClockFn = Callable[[], float]


class EstadoFuente(StrEnum):
    """Estados operativos oficiales de una plataforma del catálogo."""

    DISPONIBLE = "disponible"
    LIMITADA = "limitada"
    DEGRADADA = "degradada"
    BLOQUEADA = "bloqueada"
    NO_AUTORIZADA = "no_autorizada"


class SourceHealthSnapshot(BaseModel):
    """Estado auditable y mensaje operativo de una plataforma."""

    model_config = ConfigDict(extra="forbid")

    plataforma: str
    estado: EstadoFuente
    fallos_consecutivos: int = Field(default=0, ge=0)
    cooldown_restante_seg: float = Field(default=0.0, ge=0.0)
    motivo: str = "Operativa"
    alternativas_disponibles: list[str] = Field(default_factory=list)


class PlatformSourceHealthManager:
    """Circuito de resiliencia por plataforma con cooldown y degradación a alternativas."""

    def __init__(
        self,
        *,
        degraded_threshold: int = 2,
        block_cooldown_seconds: float = 900.0,
        rate_limit_cooldown_seconds: float = 60.0,
        clock: ClockFn | None = None,
    ) -> None:
        self._degraded_threshold = degraded_threshold
        self._block_cooldown = block_cooldown_seconds
        self._rate_limit_cooldown = rate_limit_cooldown_seconds
        self._clock: ClockFn = clock if clock is not None else time.monotonic
        self._states: dict[str, EstadoFuente] = dict.fromkeys(
            VALID_PLATFORM_KEYS, EstadoFuente.DISPONIBLE
        )
        self._failures: dict[str, int] = dict.fromkeys(VALID_PLATFORM_KEYS, 0)
        self._cooldown_until: dict[str, float] = {}
        self._reasons: dict[str, str] = dict.fromkeys(VALID_PLATFORM_KEYS, "Operativa")

    def _validate_key(self, plataforma: str) -> str:
        key = plataforma.strip().lower()
        if key not in self._states:
            raise ValueError(f"Plataforma fuera del catálogo cerrado: {plataforma!r}")
        return key

    def _refresh_if_cooldown_expired(self, key: str) -> None:
        estado = self._states[key]
        if estado in (EstadoFuente.BLOQUEADA, EstadoFuente.LIMITADA):
            until = self._cooldown_until.get(key, 0.0)
            if self._clock() >= until:
                self._states[key] = EstadoFuente.DISPONIBLE
                self._failures[key] = 0
                self._cooldown_until.pop(key, None)
                self._reasons[key] = "Recuperada tras cooldown controlado"

    def puede_consultar(self, plataforma: str) -> bool:
        """True solo si la fuente no está bloqueada, limitada ni no autorizada."""
        key = self._validate_key(plataforma)
        self._refresh_if_cooldown_expired(key)
        return self._states[key] in (EstadoFuente.DISPONIBLE, EstadoFuente.DEGRADADA)

    def registrar_exito(self, plataforma: str) -> SourceHealthSnapshot:
        key = self._validate_key(plataforma)
        if self._states[key] == EstadoFuente.NO_AUTORIZADA:
            return self.obtener_estado(key)
        self._states[key] = EstadoFuente.DISPONIBLE
        self._failures[key] = 0
        self._cooldown_until.pop(key, None)
        self._reasons[key] = "Operativa"
        return self.obtener_estado(key)

    def registrar_rate_limit(
        self,
        plataforma: str,
        *,
        retry_after_seconds: float | None = None,
    ) -> SourceHealthSnapshot:
        key = self._validate_key(plataforma)
        if self._states[key] == EstadoFuente.NO_AUTORIZADA:
            return self.obtener_estado(key)
        cooldown = (
            retry_after_seconds
            if retry_after_seconds is not None and retry_after_seconds > 0
            else self._rate_limit_cooldown
        )
        self._states[key] = EstadoFuente.LIMITADA
        self._cooldown_until[key] = self._clock() + cooldown
        self._reasons[key] = f"Cuota temporal excedida; esperar {cooldown:.0f}s."
        return self.obtener_estado(key)

    def registrar_fallo_transitorio(self, plataforma: str, *, motivo: str) -> SourceHealthSnapshot:
        key = self._validate_key(plataforma)
        if self._states[key] == EstadoFuente.NO_AUTORIZADA:
            return self.obtener_estado(key)
        self._failures[key] += 1
        if self._failures[key] >= self._degraded_threshold:
            self._states[key] = EstadoFuente.DEGRADADA
            self._reasons[key] = f"Degradada tras {self._failures[key]} fallos: {motivo}"
        else:
            self._reasons[key] = f"Fallo transitorio ({self._failures[key]}): {motivo}"
        return self.obtener_estado(key)

    def registrar_bloqueo(self, plataforma: str, *, motivo: str) -> SourceHealthSnapshot:
        """Marca fuente como bloqueada (WAF/403/CAPTCHA) y detiene consultas durante cooldown."""
        key = self._validate_key(plataforma)
        self._states[key] = EstadoFuente.BLOQUEADA
        self._failures[key] += 1
        self._cooldown_until[key] = self._clock() + self._block_cooldown
        self._reasons[key] = (
            f"Bloqueada sin evasión ({motivo}); consultas suspendidas durante "
            f"{self._block_cooldown:.0f}s."
        )
        return self.obtener_estado(key)

    def marcar_no_autorizada(self, plataforma: str, *, motivo: str) -> SourceHealthSnapshot:
        key = self._validate_key(plataforma)
        self._states[key] = EstadoFuente.NO_AUTORIZADA
        self._cooldown_until.pop(key, None)
        self._reasons[key] = f"Deshabilitada por política/ToS: {motivo}"
        return self.obtener_estado(key)

    def seleccionar_alternativas_permitidas(self, plataforma_excluida: str) -> list[str]:
        """Devuelve plataformas alternativas del catálogo en estado `disponible`."""
        excluida = self._validate_key(plataforma_excluida)
        alternativas: list[str] = []
        for key in VALID_PLATFORM_KEYS:
            if key == excluida:
                continue
            self._refresh_if_cooldown_expired(key)
            if self._states[key] == EstadoFuente.DISPONIBLE:
                alternativas.append(key)
        return alternativas

    def obtener_estado(self, plataforma: str) -> SourceHealthSnapshot:
        key = self._validate_key(plataforma)
        self._refresh_if_cooldown_expired(key)
        until = self._cooldown_until.get(key)
        restante = max(0.0, round(until - self._clock(), 2)) if until is not None else 0.0
        estado = self._states[key]
        alternativas = (
            self.seleccionar_alternativas_permitidas(key)
            if estado != EstadoFuente.DISPONIBLE
            else []
        )
        return SourceHealthSnapshot(
            plataforma=key,
            estado=estado,
            fallos_consecutivos=self._failures[key],
            cooldown_restante_seg=restante,
            motivo=self._reasons[key],
            alternativas_disponibles=alternativas,
        )
