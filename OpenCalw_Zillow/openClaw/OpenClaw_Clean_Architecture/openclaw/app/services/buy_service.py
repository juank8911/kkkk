"""Servicio de dominio para /buy en modo simulación con límites, idempotencia y auditoría (S2-07).

Cumple estrictamente con ADR-05:
- `BUY_MODE=disabled` por defecto: rechaza toda orden y audita el intento sin llamar a UAO.
- En modo `sandbox`, exige orden parametrizada (sin montos ni destinos fijos en código),
  verifica lista blanca de destinos, límites por operación y acumulados diarios,
  confirmación explícita en dos pasos, idempotencia estricta por `idempotency_key`
  y auditoría inmutable de cada evento.
"""

from __future__ import annotations

import hmac
import secrets
import shlex
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from app.integrations.uao_client import UaoLiquidityPort, UaoWithdrawalRequest
from app.models.buy import (
    BuyAuditEvent,
    BuyConfirmationChallenge,
    BuyExecutionResult,
    BuyMode,
    BuyOrderRequest,
    BuyOrderStatus,
)

NowFn = Callable[[], datetime]

DEFAULT_ALLOWED_DESTINATIONS: frozenset[str] = frozenset(
    {"sandbox_escrow_vault", "fiat_bridge_bank_sandbox"}
)


class BuyServiceError(Exception):
    """Excepción base del servicio /buy con código HTTP sugerido."""

    http_status: int = 400
    error_code: str = "buy_error"

    def __init__(self, message: str, *, audit_event: BuyAuditEvent | None = None) -> None:
        self.audit_event = audit_event
        super().__init__(message)


class BuyDisabledError(BuyServiceError):
    http_status = 403
    error_code = "buy_disabled"


class BuyDestinationNotAllowedError(BuyServiceError):
    http_status = 422
    error_code = "destination_not_allowed"


class BuyLimitExceededError(BuyServiceError):
    http_status = 422
    error_code = "buy_limit_exceeded"


class BuyConfirmationError(BuyServiceError):
    http_status = 409
    error_code = "confirmation_invalid"


class IdempotencyConflictError(BuyServiceError):
    http_status = 409
    error_code = "idempotency_conflict"


@runtime_checkable
class BuyAuditStorePort(Protocol):
    """Puerto de almacenamiento de eventos auditables de /buy (enchufable a PostgreSQL S2-03)."""

    async def append(self, event: BuyAuditEvent) -> None: ...

    async def list_events(self, *, idempotency_key: str | None = None) -> list[BuyAuditEvent]: ...


@runtime_checkable
class BuyIdempotencyStorePort(Protocol):
    """Puerto de persistencia para claves de idempotencia y acumulado diario."""

    async def get_record(self, idempotency_key: str) -> tuple[str, BuyExecutionResult] | None: ...

    async def save_record(
        self,
        idempotency_key: str,
        fingerprint: str,
        result: BuyExecutionResult,
    ) -> None: ...

    async def get_daily_total(self, day_iso: str) -> float: ...

    async def add_daily_total(self, day_iso: str, amount: float) -> float: ...


class InMemoryBuyAuditStore:
    """Almacén de auditoría en memoria para pruebas sin red y simulación local."""

    def __init__(self) -> None:
        self._events: list[BuyAuditEvent] = []

    async def append(self, event: BuyAuditEvent) -> None:
        self._events.append(event)

    async def list_events(self, *, idempotency_key: str | None = None) -> list[BuyAuditEvent]:
        if idempotency_key is None:
            return list(self._events)
        return [e for e in self._events if e.idempotency_key == idempotency_key]


class InMemoryBuyIdempotencyStore:
    """Almacén de idempotencia y totales diarios en memoria."""

    def __init__(self) -> None:
        self._records: dict[str, tuple[str, BuyExecutionResult]] = {}
        self._daily_totals: dict[str, float] = {}

    async def get_record(self, idempotency_key: str) -> tuple[str, BuyExecutionResult] | None:
        return self._records.get(idempotency_key)

    async def save_record(
        self,
        idempotency_key: str,
        fingerprint: str,
        result: BuyExecutionResult,
    ) -> None:
        self._records[idempotency_key] = (fingerprint, result)

    async def get_daily_total(self, day_iso: str) -> float:
        return self._daily_totals.get(day_iso, 0.0)

    async def add_daily_total(self, day_iso: str, amount: float) -> float:
        updated = round(self._daily_totals.get(day_iso, 0.0) + amount, 2)
        self._daily_totals[day_iso] = updated
        return updated


class RedisBuyIdempotencyStore:
    """Almacén de idempotencia sobre Redis (interino hasta repositorio PostgreSQL S2-03)."""

    def __init__(
        self,
        redis_client: Any,
        *,
        ttl_seconds: int = 7 * 24 * 3600,
        key_prefix: str = "openclaw:buy",
    ) -> None:
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix

    async def get_record(self, idempotency_key: str) -> tuple[str, BuyExecutionResult] | None:
        raw_fp = await self._redis.get(f"{self._key_prefix}:idem:fp:{idempotency_key}")
        raw_res = await self._redis.get(f"{self._key_prefix}:idem:res:{idempotency_key}")
        if raw_fp is None or raw_res is None:
            return None
        fp = raw_fp.decode("utf-8") if isinstance(raw_fp, bytes) else str(raw_fp)
        res_str = raw_res.decode("utf-8") if isinstance(raw_res, bytes) else str(raw_res)
        return fp, BuyExecutionResult.model_validate_json(res_str)

    async def save_record(
        self,
        idempotency_key: str,
        fingerprint: str,
        result: BuyExecutionResult,
    ) -> None:
        await self._redis.set(
            f"{self._key_prefix}:idem:fp:{idempotency_key}",
            fingerprint,
            ex=self._ttl_seconds,
        )
        await self._redis.set(
            f"{self._key_prefix}:idem:res:{idempotency_key}",
            result.model_dump_json(),
            ex=self._ttl_seconds,
        )

    async def get_daily_total(self, day_iso: str) -> float:
        raw = await self._redis.get(f"{self._key_prefix}:daily:{day_iso}")
        if raw is None:
            return 0.0
        val = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        return float(val)

    async def add_daily_total(self, day_iso: str, amount: float) -> float:
        key = f"{self._key_prefix}:daily:{day_iso}"
        new_val = float(await self._redis.incrbyfloat(key, amount))
        await self._redis.expire(key, 48 * 3600)
        return round(new_val, 2)


class BuySimulationService:
    """Orquesta validación, límites, confirmación en dos pasos, idempotencia y auditoría."""

    def __init__(
        self,
        *,
        uao_client: UaoLiquidityPort,
        mode: BuyMode = BuyMode.DISABLED,
        max_order_amount: float = 150_000.0,
        max_daily_amount: float = 300_000.0,
        allowed_destinations: Iterable[str] = DEFAULT_ALLOWED_DESTINATIONS,
        require_confirmation: bool = True,
        confirmation_ttl_seconds: int = 300,
        audit_store: BuyAuditStorePort | None = None,
        idempotency_store: BuyIdempotencyStorePort | None = None,
        now_fn: NowFn | None = None,
    ) -> None:
        if max_order_amount <= 0 or max_daily_amount <= 0:
            raise ValueError("Los límites de monto deben ser mayores que 0.")
        self._uao = uao_client
        self._mode = mode
        self._max_order_amount = max_order_amount
        self._max_daily_amount = max_daily_amount
        self._allowed_destinations = frozenset(d.strip() for d in allowed_destinations)
        self._require_confirmation = require_confirmation
        self._confirmation_ttl_seconds = confirmation_ttl_seconds
        self._audit: BuyAuditStorePort = (
            audit_store if audit_store is not None else InMemoryBuyAuditStore()
        )
        self._idem: BuyIdempotencyStorePort = (
            idempotency_store if idempotency_store is not None else InMemoryBuyIdempotencyStore()
        )
        self._now: NowFn = now_fn if now_fn is not None else lambda: datetime.now(UTC)
        self._challenges: dict[str, BuyConfirmationChallenge] = {}

    @property
    def mode(self) -> BuyMode:
        return self._mode

    @property
    def audit_store(self) -> BuyAuditStorePort:
        return self._audit

    async def request_confirmation(self, order: BuyOrderRequest) -> BuyConfirmationChallenge:
        """Paso 1 de confirmación: emite un token único de corta duración ligado a la orden."""
        await self._enforce_mode_and_policy(order, action="request_confirmation")
        now = self._now()
        token = f"cnf_{secrets.token_urlsafe(18)}"
        challenge = BuyConfirmationChallenge(
            idempotency_key=order.idempotency_key,
            confirmation_token=token,
            fingerprint=order.payload_fingerprint(),
            expires_at=now + timedelta(seconds=self._confirmation_ttl_seconds),
            used=False,
        )
        self._challenges[order.idempotency_key] = challenge
        await self._audit.append(
            BuyAuditEvent(
                timestamp=now,
                idempotency_key=order.idempotency_key,
                property_id=order.property_id,
                requested_by=order.requested_by,
                action="confirmation_issued",
                status=BuyOrderStatus.PENDING_CONFIRMATION,
                mode=self._mode,
                amount=order.amount,
                asset=order.asset,
                destination=order.destination,
                reason=f"expira_en_{self._confirmation_ttl_seconds}s",
            )
        )
        return challenge

    async def execute_buy(self, order: BuyOrderRequest) -> BuyExecutionResult:
        """Paso 2: valida idempotencia, modo, límites, confirmación y despacha en sandbox."""
        now = self._now()
        fingerprint = order.payload_fingerprint()

        # 1. Modo disabled bloquea inmediatamente antes de cualquier otra acción
        if self._mode != BuyMode.SANDBOX:
            event = await self._record_audit(
                order=order,
                action="execute_buy",
                status=BuyOrderStatus.REJECTED_DISABLED,
                reason="buy_mode_disabled",
                now=now,
            )
            raise BuyDisabledError(
                "Operación /buy deshabilitada (BUY_MODE=disabled).",
                audit_event=event,
            )

        # 2. Idempotencia: doble envío con la misma clave ejecuta una sola vez y queda auditado
        existing = await self._idem.get_record(order.idempotency_key)
        if existing is not None:
            saved_fp, saved_result = existing
            if not hmac.compare_digest(saved_fp, fingerprint):
                event = await self._record_audit(
                    order=order,
                    action="execute_buy_conflict",
                    status=BuyOrderStatus.REJECTED_CONFLICT,
                    reason="idempotency_key_reused_with_different_payload",
                    now=now,
                )
                raise IdempotencyConflictError(
                    f"La clave de idempotencia '{order.idempotency_key}' ya fue usada "
                    "con parámetros distintos.",
                    audit_event=event,
                )

            replay_event = await self._record_audit(
                order=order,
                action="execute_buy_replay",
                status=BuyOrderStatus.DUPLICATE_REPLAYED,
                reason="idempotent_replay_returned_cached_result",
                upstream_reference=saved_result.upstream_reference,
                replayed=True,
                now=now,
            )
            return saved_result.model_copy(
                update={
                    "status": BuyOrderStatus.DUPLICATE_REPLAYED,
                    "executed": False,
                    "replayed": True,
                    "audit_event_id": replay_event.event_id,
                    "message": "Orden duplicada detectada por idempotencia; no se re-ejecutó.",
                }
            )

        # 3. Validar destino permitido y límites (por orden + acumulado diario)
        await self._enforce_mode_and_policy(order, action="execute_buy")

        # 4. Verificar confirmación explícita en dos pasos si está activa
        if self._require_confirmation:
            await self._verify_confirmation(order, fingerprint=fingerprint, now=now)

        # 5. Despachar al cliente UAO en modo sandbox
        result_stub = BuyExecutionResult(
            idempotency_key=order.idempotency_key,
            status=BuyOrderStatus.SIMULATED_EXECUTED,
            executed=True,
            replayed=False,
            mode=self._mode,
            property_id=order.property_id,
            amount=order.amount,
            asset=order.asset,
            destination=order.destination,
            message="Orden simulada ejecutada.",
            audit_event_id=BuyAuditEvent(
                idempotency_key=order.idempotency_key,
                property_id=order.property_id,
                requested_by=order.requested_by,
                action="init",
                status=BuyOrderStatus.PENDING_CONFIRMATION,
                mode=self._mode,
                amount=order.amount,
                asset=order.asset,
                destination=order.destination,
            ).event_id,
            created_at=now,
        )

        uao_request = UaoWithdrawalRequest(
            order_id=str(result_stub.order_id),
            property_id=order.property_id,
            idempotency_key=order.idempotency_key,
            asset=order.asset,
            amount=order.amount,
            destination=order.destination,
            urgency=order.urgency,
            sandbox=True,
        )
        dispatch = await self._uao.request_withdrawal(uao_request)

        day_iso = now.date().isoformat()
        if dispatch.ok:
            await self._idem.add_daily_total(day_iso, order.amount)
            audit_event = await self._record_audit(
                order=order,
                action="execute_buy_sandbox",
                status=BuyOrderStatus.SIMULATED_EXECUTED,
                reason="sandbox_dispatch_accepted",
                upstream_reference=dispatch.reference_id,
                now=now,
            )
            final_result = result_stub.model_copy(
                update={
                    "status": BuyOrderStatus.SIMULATED_EXECUTED,
                    "executed": True,
                    "queued_for_retry": False,
                    "upstream_reference": dispatch.reference_id,
                    "audit_event_id": audit_event.event_id,
                    "message": "Orden simulada ejecutada en UAO sandbox.",
                }
            )
            await self._idem.save_record(order.idempotency_key, fingerprint, final_result)
            return final_result

        # Fallo del proveedor UAO: no bloquea ni pierde la orden, queda auditada e idempotente
        audit_event = await self._record_audit(
            order=order,
            action="execute_buy_deferred",
            status=BuyOrderStatus.UPSTREAM_QUEUED_RETRY,
            reason=dispatch.error_code or "uao_unavailable",
            now=now,
        )
        deferred_result = result_stub.model_copy(
            update={
                "status": BuyOrderStatus.UPSTREAM_QUEUED_RETRY,
                "executed": False,
                "queued_for_retry": dispatch.queued_for_retry,
                "upstream_reference": None,
                "audit_event_id": audit_event.event_id,
                "message": (
                    "Proveedor UAO no disponible; orden resguardada para reintento sin pérdida."
                ),
            }
        )
        if dispatch.queued_for_retry:
            await self._idem.save_record(order.idempotency_key, fingerprint, deferred_result)
        return deferred_result

    async def _enforce_mode_and_policy(self, order: BuyOrderRequest, *, action: str) -> None:
        now = self._now()
        if self._mode != BuyMode.SANDBOX:
            event = await self._record_audit(
                order=order,
                action=action,
                status=BuyOrderStatus.REJECTED_DISABLED,
                reason="buy_mode_disabled",
                now=now,
            )
            raise BuyDisabledError(
                "Operación /buy deshabilitada (BUY_MODE=disabled).",
                audit_event=event,
            )

        if order.destination not in self._allowed_destinations:
            allowed_str = ", ".join(sorted(self._allowed_destinations))
            event = await self._record_audit(
                order=order,
                action=action,
                status=BuyOrderStatus.REJECTED_DESTINATION,
                reason=f"destination_not_allowed:{order.destination}",
                now=now,
            )
            raise BuyDestinationNotAllowedError(
                f"Destino '{order.destination}' no permitido. Permitidos: [{allowed_str}].",
                audit_event=event,
            )

        if order.amount > self._max_order_amount:
            event = await self._record_audit(
                order=order,
                action=action,
                status=BuyOrderStatus.REJECTED_LIMIT,
                reason=f"exceeds_max_order_amount:{self._max_order_amount}",
                now=now,
            )
            raise BuyLimitExceededError(
                f"Monto {order.amount} excede el límite por operación ({self._max_order_amount}).",
                audit_event=event,
            )

        day_iso = now.date().isoformat()
        daily_used = await self._idem.get_daily_total(day_iso)
        if daily_used + order.amount > self._max_daily_amount:
            event = await self._record_audit(
                order=order,
                action=action,
                status=BuyOrderStatus.REJECTED_LIMIT,
                reason=f"exceeds_max_daily_amount:{self._max_daily_amount}",
                now=now,
            )
            raise BuyLimitExceededError(
                f"Monto acumulado diario ({daily_used + order.amount}) excede el límite "
                f"({self._max_daily_amount}).",
                audit_event=event,
            )

    async def _verify_confirmation(
        self,
        order: BuyOrderRequest,
        *,
        fingerprint: str,
        now: datetime,
    ) -> None:
        token = order.confirmation_token
        challenge = self._challenges.get(order.idempotency_key)
        if not token or challenge is None:
            event = await self._record_audit(
                order=order,
                action="execute_buy_confirmation_missing",
                status=BuyOrderStatus.REJECTED_CONFIRMATION,
                reason="missing_confirmation_token",
                now=now,
            )
            raise BuyConfirmationError(
                "Se requiere confirmación en dos pasos antes de ejecutar /buy.",
                audit_event=event,
            )

        if challenge.used:
            event = await self._record_audit(
                order=order,
                action="execute_buy_confirmation_reused",
                status=BuyOrderStatus.REJECTED_CONFIRMATION,
                reason="confirmation_token_already_used",
                now=now,
            )
            raise BuyConfirmationError(
                "El token de confirmación ya fue utilizado.",
                audit_event=event,
            )

        if now > challenge.expires_at:
            event = await self._record_audit(
                order=order,
                action="execute_buy_confirmation_expired",
                status=BuyOrderStatus.REJECTED_CONFIRMATION,
                reason="confirmation_token_expired",
                now=now,
            )
            raise BuyConfirmationError(
                "El token de confirmación ha expirado.",
                audit_event=event,
            )

        if not hmac.compare_digest(token, challenge.confirmation_token) or not hmac.compare_digest(
            fingerprint, challenge.fingerprint
        ):
            event = await self._record_audit(
                order=order,
                action="execute_buy_confirmation_mismatch",
                status=BuyOrderStatus.REJECTED_CONFIRMATION,
                reason="confirmation_token_or_payload_mismatch",
                now=now,
            )
            raise BuyConfirmationError(
                "El token de confirmación no coincide con los parámetros de la orden.",
                audit_event=event,
            )

        challenge.used = True

    async def _record_audit(
        self,
        *,
        order: BuyOrderRequest,
        action: str,
        status: BuyOrderStatus,
        reason: str | None = None,
        upstream_reference: str | None = None,
        replayed: bool = False,
        now: datetime | None = None,
    ) -> BuyAuditEvent:
        event = BuyAuditEvent(
            timestamp=now if now is not None else self._now(),
            idempotency_key=order.idempotency_key,
            property_id=order.property_id,
            requested_by=order.requested_by,
            action=action,
            status=status,
            mode=self._mode,
            amount=order.amount,
            asset=order.asset,
            destination=order.destination,
            replayed=replayed,
            reason=reason,
            upstream_reference=upstream_reference,
        )
        await self._audit.append(event)
        return event


def parse_telegram_buy_command(text: str, *, requested_by: str) -> BuyOrderRequest:
    """Parsea un comando Telegram parametrizado `/buy key=value ...` sin valores fijos.

    Ejemplo válido:
        /buy property_id=prop-101 amount=95000 asset=USDT destination=sandbox_escrow_vault idempotency_key=ord-2026-0001 confirm=cnf_xxx
    """
    tokens = shlex.split(text.strip())
    if not tokens or tokens[0].lower() != "/buy":
        raise ValueError("El mensaje no corresponde a un comando /buy.")

    params: dict[str, str] = {}
    for token in tokens[1:]:
        if "=" not in token:
            raise ValueError(
                f"Parámetro inválido '{token}'. Formato esperado: clave=valor "
                "(property_id, amount, destination, idempotency_key)."
            )
        k, v = token.split("=", 1)
        params[k.strip().lower()] = v.strip()

    required = ("property_id", "amount", "destination", "idempotency_key")
    missing = [key for key in required if not params.get(key)]
    if missing:
        raise ValueError(f"Faltan parámetros obligatorios en /buy: {', '.join(missing)}.")

    return BuyOrderRequest(
        property_id=params["property_id"],
        amount=float(params["amount"]),
        asset=params.get("asset", "USDT").upper(),  # type: ignore[arg-type]
        destination=params["destination"],
        idempotency_key=params["idempotency_key"],
        requested_by=requested_by,
        urgency=params.get("urgency", "normal").lower(),  # type: ignore[arg-type]
        confirmation_token=params.get("confirm") or params.get("confirmation_token"),
    )
