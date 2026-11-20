"""Pruebas contractuales de Fases 112–114 (Dev 2): kill switch, G2, TCPA/DNC, OFAC y fraude."""

from __future__ import annotations

from app.services.compliance_gates import (
    CounterpartyScreeningRecord,
    G2ApprovalGate,
    GateDecisionStatus,
    SmsConsentRecord,
    evaluar_elegibilidad_sms_contractual,
    evaluar_operacion_onchain_contractual,
)


def test_fase113_sms_blocked_by_default_kill_switch_dnc_and_tcpa() -> None:
    consent_ok = SmsConsentRecord(
        destinatario_id="owner-101",
        consentimiento_expreso_escrito=True,
        en_registro_dnc=False,
        opt_out_revocado=False,
        base_licitud="consentimiento_expreso_tcpa",
    )

    # Por defecto: kill switch activo bloquea siempre
    r_default = evaluar_elegibilidad_sms_contractual(consent_ok)
    assert r_default.estado == GateDecisionStatus.BLOQUEADO_KILL_SWITCH
    assert r_default.permitido_en_produccion is False
    assert r_default.permitido_en_simulacion is False

    # Sin kill switch pero en DNC -> rechazado
    gate_sin_dictamen = G2ApprovalGate(kill_switch_activo=False, dictamen_legal_firmado=False)
    r_dnc = evaluar_elegibilidad_sms_contractual(
        consent_ok.model_copy(update={"en_registro_dnc": True}),
        gate=gate_sin_dictamen,
    )
    assert r_dnc.estado == GateDecisionStatus.RECHAZADO_DNC

    # Sin consentimiento escrito -> rechazado
    r_no_cons = evaluar_elegibilidad_sms_contractual(
        consent_ok.model_copy(update={"consentimiento_expreso_escrito": False}),
        gate=gate_sin_dictamen,
    )
    assert r_no_cons.estado == GateDecisionStatus.RECHAZADO_SIN_CONSENTIMIENTO

    # Sin dictamen G2 -> bloqueado por G2
    r_g2 = evaluar_elegibilidad_sms_contractual(consent_ok, gate=gate_sin_dictamen)
    assert r_g2.estado == GateDecisionStatus.BLOQUEADO_PUERTA_G2

    # Incluso con G2 simulada, producción permanece False (solo contrato de simulación)
    gate_sim = G2ApprovalGate(
        kill_switch_activo=False,
        dictamen_legal_firmado=True,
        referencia_dictamen="LEGAL-G2-SIM-001",
        auditoria_seguridad_externa=True,
    )
    r_sim = evaluar_elegibilidad_sms_contractual(consent_ok, gate=gate_sim)
    assert r_sim.estado == GateDecisionStatus.APTO_SOLO_SIMULACION_CONTRATO
    assert r_sim.permitido_en_produccion is False
    assert r_sim.permitido_en_simulacion is True


def test_fase112_114_onchain_blocks_sanctions_fraud_kyc_and_g2() -> None:
    gate_open_sim = G2ApprovalGate(
        kill_switch_activo=False,
        dictamen_legal_firmado=True,
        referencia_dictamen="LEGAL-G2-SIM-002",
        auditoria_seguridad_externa=True,
    )
    clean_cp = CounterpartyScreeningRecord(
        contraparte_id="cp-001",
        kyc_kyb_verificado=True,
        ofac_sancionado=False,
        jurisdiccion_bloqueada=False,
        score_fraude=0.1,
        doble_control_aprobado=True,
    )

    # Sanción OFAC -> bloqueado con reversión manual
    r_ofac = evaluar_operacion_onchain_contractual(
        clean_cp.model_copy(update={"ofac_sancionado": True}),
        gate=gate_open_sim,
    )
    assert r_ofac.estado == GateDecisionStatus.RECHAZADO_SANCIONES_OFAC
    assert r_ofac.requiere_reversion_manual is True

    # Fraude alto -> bloqueado con reversión manual
    r_fraud = evaluar_operacion_onchain_contractual(
        clean_cp.model_copy(update={"score_fraude": 0.85}),
        gate=gate_open_sim,
    )
    assert r_fraud.estado == GateDecisionStatus.RECHAZADO_FRAUDE
    assert r_fraud.requiere_reversion_manual is True

    # Falta KYC o 4-eyes -> rechazado
    r_kyc = evaluar_operacion_onchain_contractual(
        clean_cp.model_copy(update={"doble_control_aprobado": False}),
        gate=gate_open_sim,
    )
    assert r_kyc.estado == GateDecisionStatus.RECHAZADO_KYC_INCOMPLETO

    # Apto solo en simulación contractual, nunca producción real
    r_ok = evaluar_operacion_onchain_contractual(clean_cp, gate=gate_open_sim)
    assert r_ok.estado == GateDecisionStatus.APTO_SOLO_SIMULACION_CONTRATO
    assert r_ok.permitido_en_produccion is False
