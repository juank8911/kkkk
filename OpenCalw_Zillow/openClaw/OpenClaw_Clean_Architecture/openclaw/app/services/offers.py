"""Reglas de dominio de la Fase 95 para el pipeline de ofertas."""

from __future__ import annotations

from datetime import UTC, datetime
from app.models.offers import OfertaEstado


class TransicionOfertaInvalida(ValueError):
    """La oferta no puede avanzar desde su estado actual al solicitado."""


TRANSICIONES_OFERTA: dict[OfertaEstado, frozenset[OfertaEstado]] = {
    OfertaEstado.PROPUESTA: frozenset({OfertaEstado.CONTRAOFERTA, OfertaEstado.ACEPTADA, OfertaEstado.RECHAZADA, OfertaEstado.CANCELADA}),
    OfertaEstado.CONTRAOFERTA: frozenset({OfertaEstado.PROPUESTA, OfertaEstado.ACEPTADA, OfertaEstado.RECHAZADA, OfertaEstado.CANCELADA}),
    OfertaEstado.ACEPTADA: frozenset({OfertaEstado.CERRADA, OfertaEstado.CANCELADA}),
    OfertaEstado.RECHAZADA: frozenset({OfertaEstado.PROPUESTA, OfertaEstado.CANCELADA}),
    OfertaEstado.CERRADA: frozenset(),
    OfertaEstado.CANCELADA: frozenset(),
}


def validar_transicion_oferta(actual: str, nuevo: OfertaEstado) -> None:
    try:
        estado_actual = OfertaEstado(actual)
    except ValueError as exc:
        raise TransicionOfertaInvalida(f"Estado actual desconocido: {actual}") from exc
    if nuevo not in TRANSICIONES_OFERTA[estado_actual]:
        raise TransicionOfertaInvalida(
            f"Transicion no permitida: {estado_actual.value} -> {nuevo.value}"
        )


def ahora_si_cerrada(estado: OfertaEstado) -> datetime | None:
    return datetime.now(UTC) if estado is OfertaEstado.CERRADA else None
