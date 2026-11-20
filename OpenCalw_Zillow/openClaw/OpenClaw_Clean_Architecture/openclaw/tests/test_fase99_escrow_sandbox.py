"""Pruebas de Fase 99/103 (Dev 2): escrow sandbox, doble aprobación, replay y conciliación."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.services.escrow_sandbox import (
    EscrowEstado,
    EscrowInstructionRequest,
    EscrowMode,
    EscrowReconciliationRecord,
    EscrowSandboxService,
    FakeEscrowProvider,
)


def _req(idem: str = "escrow-idem-0001", monto: float = 5_000.0) -> EscrowInstructionRequest:
    return EscrowInstructionRequest(
        propiedad_id=uuid4(),
        oferta_id=uuid4(),
        idempotency_key=idem,
        monto_usd=monto,
        entidad_escrow_regulada="Ohio Licensed Title & Escrow Co",
        cuenta_destino_sandbox="sandbox-trust-001",
        creada_por="operador-a",
    )


def test_fase99_disabled_by_default_and_no_real_mode_exists() -> None:
    svc = EscrowSandboxService()
    assert svc.mode == EscrowMode.DISABLED
    assert {m.value for m in EscrowMode} == {"disabled", "sandbox"}
    with pytest.raises(PermissionError):
        svc.crear_instruccion(_req())


async def test_fase99_dual_approval_replay_and_reconciliation_discrepancy() -> None:
    provider = FakeEscrowProvider()
    svc = EscrowSandboxService(provider=provider, mode=EscrowMode.SANDBOX)
    req = _req()

    inst = svc.crear_instruccion(req)
    assert inst.estado == EscrowEstado.PENDIENTE_SEGUNDA_APROBACION
    assert inst.replayed is False

    # Replay no crea segunda instrucción
    replay = svc.crear_instruccion(req)
    assert replay.instruction_id == inst.instruction_id
    assert replay.replayed is True

    # Misma identidad no puede aprobar (control dual 4-eyes)
    with pytest.raises(PermissionError, match="identidad distinta"):
        await svc.aprobar_y_despachar_sandbox(inst.instruction_id, segundo_aprobador="operador-a")
    assert provider.dispatches == []

    # Segunda identidad distinta aprueba y despacha solo al FakeEscrowProvider
    dispatched = await svc.aprobar_y_despachar_sandbox(
        inst.instruction_id, segundo_aprobador="controller-b"
    )
    assert dispatched.estado == EscrowEstado.DESPACHADA_SANDBOX
    assert dispatched.aprobador_2 == "controller-b"
    assert dispatched.referencia_proveedor is not None
    assert len(provider.dispatches) == 1

    # Discrepancia de monto -> requiere_revision obligatoria
    discrepant = svc.conciliar(
        inst.instruction_id,
        EscrowReconciliationRecord(
            referencia_proveedor=dispatched.referencia_proveedor,
            monto_reportado_usd=4_950.0,
            estado_proveedor="settled_sandbox",
        ),
    )
    assert discrepant.estado == EscrowEstado.REQUIERE_REVISION
    assert discrepant.motivo_revision is not None

    # Conciliación exacta -> conciliada
    ok = svc.conciliar(
        inst.instruction_id,
        EscrowReconciliationRecord(
            referencia_proveedor=dispatched.referencia_proveedor,
            monto_reportado_usd=5_000.0,
            estado_proveedor="settled_sandbox",
        ),
    )
    assert ok.estado == EscrowEstado.CONCILIADA
