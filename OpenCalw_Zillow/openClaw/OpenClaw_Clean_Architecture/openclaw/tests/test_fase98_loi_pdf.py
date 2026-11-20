"""Pruebas de Fase 98/102 (Dev 2): LOI versionada, PDF determinista y aprobación humana."""

from __future__ import annotations

from datetime import date
from uuid import uuid4

import pytest

from app.services.loi_generator import (
    NON_BINDING_DISCLAIMER,
    LoiDraftInput,
    LoiDraftService,
    LoiEstado,
    StateLoiTemplate,
)


def test_fase98_generates_deterministic_non_binding_pdf_and_versions_modifications() -> None:
    prop_id = uuid4()
    datos_v1 = LoiDraftInput(
        propiedad_id=prop_id,
        estado_us="OH",
        ubicacion="1420 Maple Ave, Cleveland, OH 44102",
        comprador="OpenClaw Holdings LLC",
        vendedor="John Seller",
        importe_usd=110_000.0,
        deposito_seriedad_usd=1_500.0,
        fecha_vencimiento=date(2026, 12, 15),
        creado_por="operador-1",
    )

    svc_a = LoiDraftService()
    svc_b = LoiDraftService()

    doc_a, pdf_a = svc_a.generar_borrador(datos_v1)
    doc_b, pdf_b = svc_b.generar_borrador(datos_v1)

    # Reproducibilidad exacta del PDF y su hash SHA-256
    assert pdf_a.startswith(b"%PDF-1.4\n")
    assert pdf_a.endswith(b"%%EOF\n")
    assert pdf_a == pdf_b
    assert doc_a.pdf_sha256 == doc_b.pdf_sha256
    assert doc_a.non_binding is True
    assert NON_BINDING_DISCLAIMER in doc_a.texto_renderizado
    assert doc_a.version == 1

    # Toda modificación genera una nueva versión (v2) sin sobrescribir v1
    datos_v2 = datos_v1.model_copy(update={"importe_usd": 114_000.0})
    doc_v2, pdf_v2 = svc_a.generar_borrador(datos_v2)
    assert doc_v2.version == 2
    assert doc_v2.pdf_sha256 != doc_a.pdf_sha256
    assert len(svc_a.listar_versiones(prop_id)) == 2
    assert svc_a.obtener_pdf(doc_v2.document_id) == pdf_v2

    # No se puede marcar aprobado_para_envio sin confirmación humana explícita
    with pytest.raises(ValueError):
        svc_a.aprobar_para_envio(
            propiedad_id=prop_id,
            version=2,
            aprobado_por="abogado-1",
            confirmacion_humana=False,
        )

    aprobado = svc_a.aprobar_para_envio(
        propiedad_id=prop_id,
        version=2,
        aprobado_por="abogado-1",
        confirmacion_humana=True,
    )
    assert aprobado.estado == LoiEstado.APROBADO_PARA_ENVIO
    assert aprobado.aprobado_por == "abogado-1"
    assert not hasattr(svc_a, "enviar_automaticamente")


def test_fase98_blocks_unreviewed_or_unknown_state_templates() -> None:
    unreviewed = {
        "CA": StateLoiTemplate(
            state_code="CA",
            template_version="CA-LOI-draft",
            legal_reviewed=False,
            governing_law_clause="Draft",
            required_contingencies=("Inspection",),
        )
    }
    svc = LoiDraftService(templates=unreviewed)
    base = LoiDraftInput(
        propiedad_id=uuid4(),
        estado_us="CA",
        ubicacion="100 Sunset Blvd, Los Angeles, CA 90028",
        comprador="Buyer LLC",
        vendedor="Seller",
        importe_usd=140_000.0,
        fecha_vencimiento=date(2026, 12, 31),
        creado_por="operador-1",
    )
    with pytest.raises(ValueError, match="revisión legal"):
        svc.generar_borrador(base)

    with pytest.raises(ValueError, match="No existe plantilla"):
        svc.generar_borrador(base.model_copy(update={"estado_us": "NY"}))
