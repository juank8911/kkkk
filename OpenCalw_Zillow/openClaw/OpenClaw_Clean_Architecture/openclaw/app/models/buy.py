"""Modelos de dominio para /buy en modo simulación con límites, idempotencia y auditoría (S2-07 / ADR-05)."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

_IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9._:\-]{8,128}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._:\-]{2,128}$")


class BuyMode(StrEnum):
    """Modos permitidos para la ejecución de órdenes /buy (ADR-05)."""

    DISABLED = "disabled"
    SANDBOX = "sandbox"


class BuyOrderStatus(StrEnum):
    """Estados auditables de una solicitud de compra simulada."""

    SIMULATED_EXECUTED = "simulated_executed"
    DUPLICATE_REPLAYED = "duplicate_replayed"
    UPSTREAM_QUEUED_RETRY = "upstream_queued_retry"
    PENDING_CONFIRMATION = "pending_confirmation"
    REJECTED_DISABLED = "rejected_disabled"
    REJECTED_LIMIT = "rejected_limit"
    REJECTED_DESTINATION = "rejected_destination"
    REJECTED_CONFIRMATION = "rejected_confirmation"
    REJECTED_CONFLICT = "rejected_conflict"


class BuyOrderRequest(BaseModel):
    """Orden parametrizada y validada para /buy en modo sandbox (sin montos fijos)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    property_id: str = Field(..., min_length=2, max_length=128)
    asset: Literal["USDT", "USDC", "USD"] = "USDT"
    amount: float = Field(..., gt=0, description="Monto positivo de la operación en sandbox.")
    destination: str = Field(..., min_length=2, max_length=128)
    idempotency_key: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Clave única para garantizar ejecución exactamente una vez.",
    )
    requested_by: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Identificador del operador o chat autorizado que origina la orden.",
    )
    urgency: Literal["low", "normal", "high"] = "normal"
    confirmation_token: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
        description="Token de confirmación en dos pasos emitido para esta orden.",
    )

    @field_validator("property_id", "destination")
    @classmethod
    def _validate_safe_identifier(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError(f"Identificador inválido: {value!r}")
        return value

    @field_validator("idempotency_key")
    @classmethod
    def _validate_idempotency_key(cls, value: str) -> str:
        if not _IDEMPOTENCY_KEY_RE.fullmatch(value):
            raise ValueError(
                "idempotency_key debe tener entre 8 y 128 caracteres alfanuméricos o . _ : -"
            )
        return value

    def payload_fingerprint(self) -> str:
        """Huella determinista de los parámetros económicos de la orden (excluye token)."""
        canonical = json.dumps(
            {
                "property_id": self.property_id,
                "asset": self.asset,
                "amount": round(self.amount, 2),
                "destination": self.destination,
                "requested_by": self.requested_by,
                "urgency": self.urgency,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class BuyConfirmationChallenge(BaseModel):
    """Desafío de confirmación en dos pasos de corta duración para una orden /buy."""

    model_config = ConfigDict(extra="forbid")

    challenge_id: UUID = Field(default_factory=uuid4)
    idempotency_key: str
    confirmation_token: str
    fingerprint: str
    expires_at: datetime
    used: bool = False


class BuyAuditEvent(BaseModel):
    """Registro de auditoría inmutable para cualquier intento o ejecución de /buy."""

    model_config = ConfigDict(extra="forbid")

    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    idempotency_key: str
    property_id: str
    requested_by: str
    action: str
    status: BuyOrderStatus
    mode: BuyMode
    amount: float
    asset: str
    destination: str
    replayed: bool = False
    reason: str | None = None
    upstream_reference: str | None = None


class BuyExecutionResult(BaseModel):
    """Respuesta tipada de la evaluación o ejecución simulada de una orden /buy."""

    model_config = ConfigDict(extra="forbid")

    order_id: UUID = Field(default_factory=uuid4)
    idempotency_key: str
    status: BuyOrderStatus
    executed: bool
    replayed: bool = False
    queued_for_retry: bool = False
    mode: BuyMode
    property_id: str
    amount: float
    asset: str
    destination: str
    upstream_reference: str | None = None
    message: str
    audit_event_id: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
