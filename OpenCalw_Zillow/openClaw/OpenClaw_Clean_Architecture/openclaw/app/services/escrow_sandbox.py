"""Máquina de estados y sandbox de instrucciones de Escrow — Fase 99/103 (Dev 2).

BLOQUEADO HASTA PUERTA LEGAL G2:
- No existe ningún modo de fondos reales (`EscrowMode` solo admite `disabled` y `sandbox`).
- Exige doble aprobación por dos identidades distintas (control dual / 4-eyes).
- Idempotencia estricta por `idempotency_key` (un replay nunca crea una segunda instrucción).
- Toda discrepancia en conciliación de referencias o montos pasa a `requiere_revision`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class EscrowMode(StrEnum):
    """Solo se admiten `disabled` y `sandbox` hasta cierre escrito de puerta legal G2."""

    DISABLED = "disabled"
    SANDBOX = "sandbox"


class EscrowEstado(StrEnum):
    """Estados oficiales de una instrucción de escrow en sandbox."""

    BORRADOR = "borrador"
    PENDIENTE_SEGUNDA_APROBACION = "pendiente_segunda_aprobacion"
    APROBADA_SANDBOX = "aprobada_sandbox"
    DESPACHADA_SANDBOX = "despachada_sandbox"
    CONCILIADA = "conciliada"
    REQUIERE_REVISION = "requiere_revision"
    RECHAZADA = "rechazada"
    REVERTIDA = "revertida"


class EscrowInstructionRequest(BaseModel):
    """Solicitud parametrizada de instrucción de depósito en garantía (solo sandbox)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    propiedad_id: UUID
    oferta_id: UUID
    idempotency_key: str = Field(..., min_length=8, max_length=128)
    monto_usd: float = Field(..., gt=0, le=500_000.0)
    entidad_escrow_regulada: str = Field(..., min_length=3, max_length=120)
    cuenta_destino_sandbox: str = Field(..., min_length=4, max_length=120)
    creada_por: str = Field(..., min_length=2, max_length=120)


class EscrowInstruction(BaseModel):
    """Registro auditable de una instrucción de escrow."""

    model_config = ConfigDict(extra="forbid")

    instruction_id: UUID = Field(default_factory=uuid4)
    idempotency_key: str
    propiedad_id: UUID
    oferta_id: UUID
    monto_usd: float
    entidad_escrow_regulada: str
    cuenta_destino_sandbox: str
    estado: EscrowEstado = EscrowEstado.PENDIENTE_SEGUNDA_APROBACION
    modo: EscrowMode = EscrowMode.SANDBOX
    aprobador_1: str
    aprobador_2: str | None = None
    referencia_proveedor: str | None = None
    motivo_revision: str | None = None
    replayed: bool = False
    creada_en: datetime = Field(default_factory=lambda: datetime.now(UTC))
    actualizada_en: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EscrowReconciliationRecord(BaseModel):
    """Extracto devuelto por el proveedor sandbox para conciliar referencias y montos."""

    model_config = ConfigDict(extra="forbid")

    referencia_proveedor: str
    monto_reportado_usd: float = Field(..., ge=0.0)
    estado_proveedor: str


@runtime_checkable
class EscrowPort(Protocol):
    """Puerto del proveedor de escrow (solo fake/sandbox hasta cierre de G2)."""

    async def crear_instruccion_sandbox(self, instruction: EscrowInstruction) -> str: ...


class FakeEscrowProvider:
    """Proveedor simulado que nunca conecta cuentas reales ni mueve fondos."""

    def __init__(self) -> None:
        self.dispatches: list[UUID] = []

    async def crear_instruccion_sandbox(self, instruction: EscrowInstruction) -> str:
        self.dispatches.append(instruction.instruction_id)
        return f"fake-escrow-ref-{instruction.instruction_id.hex[:12]}"


class EscrowSandboxService:
    """Gestiona doble aprobación, idempotencia, despacho sandbox y conciliación."""

    def __init__(
        self,
        *,
        provider: EscrowPort | None = None,
        mode: EscrowMode = EscrowMode.DISABLED,
        max_instruction_usd: float = 250_000.0,
    ) -> None:
        self._provider: EscrowPort = provider if provider is not None else FakeEscrowProvider()
        self._mode = mode
        self._max_instruction_usd = max_instruction_usd
        self._by_id: dict[UUID, EscrowInstruction] = {}
        self._by_idem: dict[str, UUID] = {}

    @property
    def mode(self) -> EscrowMode:
        return self._mode

    def _require_sandbox(self) -> None:
        if self._mode != EscrowMode.SANDBOX:
            raise PermissionError(
                "Escrow deshabilitado (ESCROW_MODE=disabled). "
                "Modo real inexistente hasta cierre escrito de puerta legal G2."
            )

    def crear_instruccion(self, req: EscrowInstructionRequest) -> EscrowInstruction:
        """Crea instrucción en `pendiente_segunda_aprobacion` o devuelve replay idempotente."""
        self._require_sandbox()
        if req.monto_usd > self._max_instruction_usd:
            raise ValueError(
                f"Monto {req.monto_usd} excede el límite sandbox ({self._max_instruction_usd})."
            )

        existing_id = self._by_idem.get(req.idempotency_key)
        if existing_id is not None:
            existing = self._by_id[existing_id]
            if (
                existing.propiedad_id != req.propiedad_id
                or existing.monto_usd != req.monto_usd
                or existing.cuenta_destino_sandbox != req.cuenta_destino_sandbox
            ):
                raise ValueError(
                    f"Conflicto de idempotencia en escrow para clave '{req.idempotency_key}'."
                )
            return existing.model_copy(update={"replayed": True})

        inst = EscrowInstruction(
            idempotency_key=req.idempotency_key,
            propiedad_id=req.propiedad_id,
            oferta_id=req.oferta_id,
            monto_usd=req.monto_usd,
            entidad_escrow_regulada=req.entidad_escrow_regulada,
            cuenta_destino_sandbox=req.cuenta_destino_sandbox,
            estado=EscrowEstado.PENDIENTE_SEGUNDA_APROBACION,
            modo=self._mode,
            aprobador_1=req.creada_por,
        )
        self._by_id[inst.instruction_id] = inst
        self._by_idem[req.idempotency_key] = inst.instruction_id
        return inst

    async def aprobar_y_despachar_sandbox(
        self,
        instruction_id: UUID,
        *,
        segundo_aprobador: str,
    ) -> EscrowInstruction:
        """Exige una segunda identidad distinta antes de despachar al fake provider."""
        self._require_sandbox()
        inst = self._by_id[instruction_id]
        aprobador_2 = segundo_aprobador.strip()
        if not aprobador_2:
            raise ValueError("Se requiere identidad del segundo aprobador.")
        if aprobador_2.lower() == inst.aprobador_1.lower():
            raise PermissionError(
                "Control dual (4-eyes): el segundo aprobador debe ser una identidad distinta."
            )
        if inst.estado != EscrowEstado.PENDIENTE_SEGUNDA_APROBACION:
            raise ValueError(f"La instrucción no admite aprobación en estado '{inst.estado}'.")

        ref = await self._provider.crear_instruccion_sandbox(inst)
        updated = inst.model_copy(
            update={
                "aprobador_2": aprobador_2,
                "referencia_proveedor": ref,
                "estado": EscrowEstado.DESPACHADA_SANDBOX,
                "actualizada_en": datetime.now(UTC),
            }
        )
        self._by_id[instruction_id] = updated
        return updated

    def conciliar(
        self,
        instruction_id: UUID,
        record: EscrowReconciliationRecord,
    ) -> EscrowInstruction:
        """Concilia referencia y monto; cualquier discrepancia pasa a `requiere_revision`."""
        inst = self._by_id[instruction_id]
        discrepancias: list[str] = []
        if record.referencia_proveedor != inst.referencia_proveedor:
            discrepancias.append(
                f"referencia esperada={inst.referencia_proveedor} "
                f"vs reportada={record.referencia_proveedor}"
            )
        if round(record.monto_reportado_usd, 2) != round(inst.monto_usd, 2):
            discrepancias.append(
                f"monto esperado={inst.monto_usd:.2f} vs reportado={record.monto_reportado_usd:.2f}"
            )
        if record.estado_proveedor.lower() not in ("settled_sandbox", "confirmed_sandbox"):
            discrepancias.append(f"estado proveedor no confirmado={record.estado_proveedor}")

        if discrepancias:
            updated = inst.model_copy(
                update={
                    "estado": EscrowEstado.REQUIERE_REVISION,
                    "motivo_revision": "; ".join(discrepancias),
                    "actualizada_en": datetime.now(UTC),
                }
            )
        else:
            updated = inst.model_copy(
                update={
                    "estado": EscrowEstado.CONCILIADA,
                    "motivo_revision": None,
                    "actualizada_en": datetime.now(UTC),
                }
            )
        self._by_id[instruction_id] = updated
        return updated
