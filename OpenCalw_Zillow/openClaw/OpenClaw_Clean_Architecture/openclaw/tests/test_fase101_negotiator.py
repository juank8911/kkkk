"""Pruebas de Fase 101/104 (Dev 2): negociador asistido supervisado y Fair Housing."""

from __future__ import annotations

from uuid import uuid4

from app.services.loi_generator import LoiEstado
from app.services.negotiation_assistant import (
    PropuestaNegociacionEstado,
    SupervisedNegotiationService,
    extraer_datos_comerciales_mensaje,
)


def test_fase101_ignores_protected_attributes_and_produces_explainable_alternatives() -> None:
    msg = (
        "Hello, the seller is asking $122,000 and prefers closing in 21 days as-is. "
        "Note: mention of church religion and family status should never affect pricing."
    )
    ext = extraer_datos_comerciales_mensaje(msg)
    assert ext.precio_solicitado_usd == 122_000.0
    assert ext.dias_cierre_solicitados == 21
    assert ext.menciona_as_is is True
    assert "religion" in ext.atributos_protegidos_ignorados
    assert "familial_status" in ext.atributos_protegidos_ignorados

    svc = SupervisedNegotiationService()
    prop_id = uuid4()
    propuesta = svc.analizar_y_proponer(
        propiedad_id=prop_id,
        mensaje_vendedor=msg,
        oferta_actual_usd=108_000.0,
        noi_anual_usd=12_800.0,
        capex_estimado_usd=10_000.0,
        min_cap_rate_pct=8.0,
    )
    assert propuesta.fallback_manual is False
    assert propuesta.estado == PropuestaNegociacionEstado.REVISAR
    assert propuesta.requiere_aprobacion_humana is True
    assert propuesta.envio_automatico_habilitado is False
    assert len(propuesta.alternativas) >= 2
    for alt in propuesta.alternativas:
        assert alt.cap_rate_proyectado_pct >= 8.0
        assert "cap rate proyectado" in alt.explicacion
        assert "noi_anual_usd" in alt.variables_usadas

    # Aprobación humana con corrección genera solo un borrador LOI versionado (no envío)
    aprobada = svc.resolver_revision_humana(
        propuesta.propuesta_id,
        decision=PropuestaNegociacionEstado.APROBADO,
        operador="operador-juank",
        importe_elegido_usd=115_000.0,
        notas="Ajuste manual aprobado.",
        generar_borrador_loi={
            "estado_us": "OH",
            "ubicacion": "1420 Maple Ave, Cleveland, OH 44102",
            "comprador": "OpenClaw LLC",
            "vendedor": "Seller Trust",
        },
    )
    assert aprobada.estado == PropuestaNegociacionEstado.APROBADO
    assert aprobada.loi_borrador is not None
    assert aprobada.loi_borrador.estado == LoiEstado.BORRADOR
    assert aprobada.loi_borrador.non_binding is True


def test_fase101_low_confidence_degrades_to_manual_fallback_and_allows_rejection() -> None:
    svc = SupervisedNegotiationService(min_confidence=0.65)
    propuesta = svc.analizar_y_proponer(
        propiedad_id=uuid4(),
        mensaje_vendedor="Call me next week to talk about terms.",
        oferta_actual_usd=100_000.0,
        noi_anual_usd=12_000.0,
        capex_estimado_usd=8_000.0,
    )
    assert propuesta.fallback_manual is True
    assert propuesta.alternativas == []
    assert propuesta.motivo_fallback is not None

    rechazada = svc.resolver_revision_humana(
        propuesta.propuesta_id,
        decision=PropuestaNegociacionEstado.RECHAZADO,
        operador="operador-juank",
        notas="Prefiero llamar personalmente.",
    )
    assert rechazada.estado == PropuestaNegociacionEstado.RECHAZADO
    assert rechazada.loi_borrador is None
