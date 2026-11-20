"""Modelos contractuales de compliance y puertas G2 — Fases 112–114 (Dev 2).

BLOQUEADO HASTA DICTAMEN LEGAL G2:
- Este módulo NO conecta con Twilio, DeFi, Aave V3, wallets ni RPC blockchain.
- Modela exclusivamente contratos de consentimiento TCPA/DNC (Fase 113), KYC/KYB,
  screening OFAC, fraude/reversión y kill switch para pruebas contractuales (Fases 112 y 114).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class GateDecisionStatus(StrEnum):
    """Resultado de una evaluación contractual de cumplimiento G2."""

    BLOQUEADO_PUERTA_G2 = "bloqueado_puerta_g2"
    BLOQUEADO_KILL_SWITCH = "bloqueado_kill_switch"
    RECHAZADO_DNC = "rechazado_dnc"
    RECHAZADO_SIN_CONSENTIMIENTO = "rechazado_sin_consentimiento"
    RECHAZADO_SANCIONES_OFAC = "rechazado_sanciones_ofac"
    RECHAZADO_KYC_INCOMPLETO = "rechazado_kyc_incompleto"
    RECHAZADO_FRAUDE = "rechazado_fraude"
    APTO_SOLO_SIMULACION_CONTRATO = "apto_solo_simulacion_contrato"


class G2ApprovalGate(BaseModel):
    """Estado de la puerta legal bloqueante G2 y kill switch operativo."""

    model_config = ConfigDict(extra="forbid")

    dictamen_legal_firmado: bool = False
    referencia_dictamen: str | None = None
    auditoria_seguridad_externa: bool = False
    kill_switch_activo: bool = True

    def habilitado_para_sandbox_g2(self) -> bool:
        return (
            not self.kill_switch_activo
            and self.dictamen_legal_firmado
            and bool(self.referencia_dictamen)
            and self.auditoria_seguridad_externa
        )


class SmsConsentRecord(BaseModel):
    """Modelo contractual de consentimiento TCPA / DNC para Fase 113."""

    model_config = ConfigDict(extra="forbid")

    destinatario_id: str = Field(..., min_length=2, max_length=120)
    consentimiento_expreso_escrito: bool = False
    en_registro_dnc: bool = True
    opt_out_revocado: bool = False
    base_licitud: str = Field(default="no_establecida", min_length=2)
    registrado_en: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CounterpartyScreeningRecord(BaseModel):
    """Modelo contractual KYC/KYB, screening OFAC y control de fraude para Fases 112 y 114."""

    model_config = ConfigDict(extra="forbid")

    contraparte_id: str = Field(..., min_length=2, max_length=120)
    kyc_kyb_verificado: bool = False
    ofac_sancionado: bool = False
    jurisdiccion_bloqueada: bool = False
    score_fraude: float = Field(default=0.0, ge=0.0, le=1.0)
    doble_control_aprobado: bool = False


class ComplianceEvaluationResult(BaseModel):
    """Resultado auditable de evaluación contractual sin ejecutar acciones externas."""

    model_config = ConfigDict(extra="forbid")

    fase: str
    permitido_en_produccion: bool = False
    permitido_en_simulacion: bool = False
    estado: GateDecisionStatus
    motivo: str
    requiere_reversion_manual: bool = False


def evaluar_elegibilidad_sms_contractual(
    consent: SmsConsentRecord,
    *,
    gate: G2ApprovalGate | None = None,
) -> ComplianceEvaluationResult:
    """Evalúa requisitos TCPA/DNC y puerta G2 para Fase 113 sin enviar ningún SMS."""
    g2 = gate if gate is not None else G2ApprovalGate()

    if g2.kill_switch_activo:
        return ComplianceEvaluationResult(
            fase="fase_113_outbound_sms",
            permitido_en_produccion=False,
            permitido_en_simulacion=False,
            estado=GateDecisionStatus.BLOQUEADO_KILL_SWITCH,
            motivo="Kill switch activo: módulo outbound SMS completamente desactivado.",
        )

    if consent.en_registro_dnc or consent.opt_out_revocado:
        return ComplianceEvaluationResult(
            fase="fase_113_outbound_sms",
            permitido_en_produccion=False,
            permitido_en_simulacion=False,
            estado=GateDecisionStatus.RECHAZADO_DNC,
            motivo="Destinatario en registro DNC o con opt-out/revocación activa.",
        )

    if not consent.consentimiento_expreso_escrito:
        return ComplianceEvaluationResult(
            fase="fase_113_outbound_sms",
            permitido_en_produccion=False,
            permitido_en_simulacion=False,
            estado=GateDecisionStatus.RECHAZADO_SIN_CONSENTIMIENTO,
            motivo="Falta consentimiento previo expreso por escrito exigido por TCPA.",
        )

    if not g2.habilitado_para_sandbox_g2():
        return ComplianceEvaluationResult(
            fase="fase_113_outbound_sms",
            permitido_en_produccion=False,
            permitido_en_simulacion=False,
            estado=GateDecisionStatus.BLOQUEADO_PUERTA_G2,
            motivo="Bloqueado por puerta legal G2 pendiente de dictamen firmado y auditoría.",
        )

    return ComplianceEvaluationResult(
        fase="fase_113_outbound_sms",
        permitido_en_produccion=False,
        permitido_en_simulacion=True,
        estado=GateDecisionStatus.APTO_SOLO_SIMULACION_CONTRATO,
        motivo="Cumple contrato TCPA/DNC en entorno simulado (producción sigue bloqueada).",
    )


def evaluar_operacion_onchain_contractual(
    screening: CounterpartyScreeningRecord,
    *,
    fase: str = "fase_112_114_defi_rpc",
    gate: G2ApprovalGate | None = None,
    umbral_fraude: float = 0.7,
) -> ComplianceEvaluationResult:
    """Evalúa KYC/KYB, OFAC, fraude y puerta G2 para Fases 112 y 114 sin tocar blockchain."""
    g2 = gate if gate is not None else G2ApprovalGate()

    if g2.kill_switch_activo:
        return ComplianceEvaluationResult(
            fase=fase,
            permitido_en_produccion=False,
            permitido_en_simulacion=False,
            estado=GateDecisionStatus.BLOQUEADO_KILL_SWITCH,
            motivo="Kill switch activo: operaciones DeFi/on-chain desactivadas.",
        )

    if screening.ofac_sancionado or screening.jurisdiccion_bloqueada:
        return ComplianceEvaluationResult(
            fase=fase,
            permitido_en_produccion=False,
            permitido_en_simulacion=False,
            estado=GateDecisionStatus.RECHAZADO_SANCIONES_OFAC,
            motivo="Contraparte o dirección bloqueada por screening OFAC/sanciones.",
            requiere_reversion_manual=True,
        )

    if screening.score_fraude >= umbral_fraude:
        return ComplianceEvaluationResult(
            fase=fase,
            permitido_en_produccion=False,
            permitido_en_simulacion=False,
            estado=GateDecisionStatus.RECHAZADO_FRAUDE,
            motivo=(
                f"Score de fraude ({screening.score_fraude:.2f}) igual o superior al "
                f"umbral ({umbral_fraude:.2f}); congelar y abrir revisión manual."
            ),
            requiere_reversion_manual=True,
        )

    if not screening.kyc_kyb_verificado or not screening.doble_control_aprobado:
        return ComplianceEvaluationResult(
            fase=fase,
            permitido_en_produccion=False,
            permitido_en_simulacion=False,
            estado=GateDecisionStatus.RECHAZADO_KYC_INCOMPLETO,
            motivo="Falta verificación KYC/KYB o aprobación dual (4-eyes).",
        )

    if not g2.habilitado_para_sandbox_g2():
        return ComplianceEvaluationResult(
            fase=fase,
            permitido_en_produccion=False,
            permitido_en_simulacion=False,
            estado=GateDecisionStatus.BLOQUEADO_PUERTA_G2,
            motivo="Bloqueado por puerta legal G2 pendiente de dictamen y auditoría externa.",
        )

    return ComplianceEvaluationResult(
        fase=fase,
        permitido_en_produccion=False,
        permitido_en_simulacion=True,
        estado=GateDecisionStatus.APTO_SOLO_SIMULACION_CONTRATO,
        motivo="Apto únicamente para simulador contractual sin claves reales ni fondos.",
    )
