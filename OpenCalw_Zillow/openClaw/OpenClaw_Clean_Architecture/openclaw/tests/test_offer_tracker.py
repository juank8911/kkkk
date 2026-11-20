"""Pruebas del tracker post-oferta de la Fase 95."""

import pytest

from app.models.offers import OfertaEstado
from app.services.offer_tracker import SenalOferta, interpretar_senal


def test_venta_cierra_solo_una_oferta_aceptada() -> None:
    resultado = interpretar_senal("This property is sold", OfertaEstado.ACEPTADA.value)
    assert resultado.senal is SenalOferta.VENDIDA
    assert resultado.nuevo_estado is OfertaEstado.CERRADA


def test_venta_no_cierra_una_oferta_no_aceptada() -> None:
    resultado = interpretar_senal("Status: sold", OfertaEstado.PROPUESTA.value)
    assert resultado.nuevo_estado is None
    assert resultado.motivo == "venta_sin_oferta_aceptada"


@pytest.mark.parametrize("texto", ["pending", "under contract", "texto desconocido"])
def test_senales_informativas_no_mutan_estado(texto: str) -> None:
    resultado = interpretar_senal(texto, OfertaEstado.CONTRAOFERTA.value)
    assert resultado.nuevo_estado is None
    assert resultado.senal in {
        SenalOferta.PENDIENTE,
        SenalOferta.BAJO_CONTRATO,
        SenalOferta.AMBIGUA,
    }
