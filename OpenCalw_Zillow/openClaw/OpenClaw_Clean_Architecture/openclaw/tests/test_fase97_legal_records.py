"""Pruebas de Fase 97 (Dev 2): auditoría legal ampliada con allowlist, evidencias y 4 estados."""

from __future__ import annotations

import pytest

from app.services.legal_public_records import (
    LEGAL_DISCLAIMER,
    CountyTaxAssessorAdapter,
    EstadoAuditoriaPublica,
    ExtendedLegalAuditService,
    PublicRecordDocument,
    PublicRecordPort,
    PublicRecordSourceNotAllowedError,
)


async def test_fase97_rejects_unlisted_domains_by_allowlist() -> None:
    bad_adapter = CountyTaxAssessorAdapter(
        {
            "cuyahoga": [
                PublicRecordDocument(
                    condado="Cuyahoga",
                    url_fuente="https://unofficial-scraper.example.com/tax/123",
                    contenido_texto="No liens found",
                )
            ]
        }
    )
    assert isinstance(bad_adapter, PublicRecordPort)
    svc = ExtendedLegalAuditService(public_record_port=bad_adapter)

    with pytest.raises(PublicRecordSourceNotAllowedError):
        await svc.auditar_propiedad(condado="Cuyahoga", ubicacion="1420 Maple Ave")


async def test_fase97_separates_all_four_states_with_evidence_hash_and_disclaimer() -> None:
    adapter = CountyTaxAssessorAdapter(
        {
            "cuyahoga": [
                PublicRecordDocument(
                    condado="Cuyahoga",
                    url_fuente="https://treasurer.cuyahogacounty.gov/parcel/1001",
                    contenido_texto="Active Tax Lien certificate #8821 and delinquent taxes 2024.",
                    disponible=True,
                )
            ],
            "orange": [
                PublicRecordDocument(
                    condado="Orange",
                    url_fuente="https://ocpafl.org/parcel/2002",
                    contenido_texto="Open code violation #441 reported on porch.",
                    disponible=True,
                )
            ],
            "miami-dade": [
                PublicRecordDocument(
                    condado="Miami-Dade",
                    url_fuente="https://propertyappraiser.miamidade.gov/parcel/3003",
                    contenido_texto="Taxes paid in full through 2026. Regular warranty deed.",
                    disponible=True,
                )
            ],
            "wayne": [
                PublicRecordDocument(
                    condado="Wayne",
                    url_fuente="https://treasurer.waynecounty.com/parcel/4004",
                    contenido_texto="",
                    disponible=False,
                    error="HTTP 503",
                )
            ],
        }
    )
    svc = ExtendedLegalAuditService(public_record_port=adapter)

    # 1. requiere_abogado (tax_lien + back_taxes)
    r_abogado = await svc.auditar_propiedad(condado="Cuyahoga", ubicacion="1420 Maple Ave")
    assert r_abogado.estado == EstadoAuditoriaPublica.REQUIERE_ABOGADO
    assert r_abogado.requiere_revision_humana is True
    assert len(r_abogado.evidencias) >= 2
    assert r_abogado.evidencias[0].documento_sha256
    assert r_abogado.disclaimer == LEGAL_DISCLAIMER

    # 2. riesgo_detectado (solo code_violation media)
    r_riesgo = await svc.auditar_propiedad(condado="Orange", ubicacion="1313 Sun Blvd")
    assert r_riesgo.estado == EstadoAuditoriaPublica.RIESGO_DETECTADO
    assert r_riesgo.requiere_revision_humana is True

    # 3. sin_evidencia (no garantiza título limpio)
    r_sin = await svc.auditar_propiedad(condado="Miami-Dade", ubicacion="2210 Coral Way")
    assert r_sin.estado == EstadoAuditoriaPublica.SIN_EVIDENCIA
    assert "no garantiza título limpio" in r_sin.resumen
    assert len(r_sin.hashes_documentos) == 1

    # 4. fuente_no_disponible (fuente caída o condado sin cobertura)
    r_down = await svc.auditar_propiedad(condado="Wayne", ubicacion="8842 Oakwood Dr")
    assert r_down.estado == EstadoAuditoriaPublica.FUENTE_NO_DISPONIBLE
    assert r_down.requiere_revision_humana is True
    assert "NO implica ausencia de riesgo" in r_down.resumen
