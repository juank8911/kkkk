"""Pruebas de las transiciones del pipeline de ofertas (Fase 95)."""

import pytest

from app.models.offers import OfertaEstado
from app.services.offers import TransicionOfertaInvalida, validar_transicion_oferta


def test_pipeline_oferta_avanza_hasta_cierre() -> None:
    validar_transicion_oferta("propuesta", OfertaEstado.CONTRAOFERTA)
    validar_transicion_oferta("contraoferta", OfertaEstado.ACEPTADA)
    validar_transicion_oferta("aceptada", OfertaEstado.CERRADA)


@pytest.mark.parametrize(
    ("actual", "nuevo"),
    [("propuesta", OfertaEstado.CERRADA), ("cerrada", OfertaEstado.PROPUESTA)],
)
def test_pipeline_rechaza_saltos_invalidos(actual: str, nuevo: OfertaEstado) -> None:
    with pytest.raises(TransicionOfertaInvalida):
        validar_transicion_oferta(actual, nuevo)
