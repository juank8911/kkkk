"""Cliente UAO con timeout, reintentos, circuit breaker y buffer de reintento (S2-08).

Garantiza que:
- Toda llamada al proveedor de liquidez (exclusivamente en modo sandbox en este
  alcance) usa timeout estricto, cabecera `Idempotency-Key` y reintentos acotados
  con backoff exponencial solo para fallos transitorios.
- Tras alcanzar el umbral de fallos consecutivos, el circuit breaker se abre y
  rechaza rápido (`fail-fast`) sin bloquear el worker.
- Ante fallo del proveedor o circuito abierto, la orden se resguarda en un buffer
  de reintento (`OrderRetryBuffer`) para que nunca se pierda.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("OpenClaw.UaoClient")

SleepFn = Callable[[float], Awaitable[None]]
ClockFn = Callable[[], float]


class CircuitBreakerState(StrEnum):
    """Estados del circuit breaker del cliente UAO."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class UaoWithdrawalRequest(BaseModel):
    """Solicitud tipada hacia el endpoint de liquidez UAO (sandbox)."""

    model_config = ConfigDict(extra="forbid")

    order_id: str
    property_id: str
    idempotency_key: str
    asset: str
    amount: float = Field(..., gt=0)
    destination: str
    urgency: str = "normal"
    sandbox: bool = True


class UaoDispatchResult(BaseModel):
    """Resultado tipado del despacho hacia UAO."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    status: str
    reference_id: str | None = None
    attempts: int = Field(default=0, ge=0)
    queued_for_retry: bool = False
    circuit_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    error_code: str | None = None
    error_message: str | None = None


class PendingRetryEntry(BaseModel):
    """Orden preservada en buffer tras un fallo o circuito abierto de UAO."""

    model_config = ConfigDict(extra="forbid")

    request: UaoWithdrawalRequest
    reason: str
    enqueued_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


@runtime_checkable
class OrderRetryBufferPort(Protocol):
    """Puerto de resguardo de órdenes que no pudieron despacharse a UAO."""

    async def enqueue(self, entry: PendingRetryEntry) -> None: ...

    async def list_pending(self) -> list[PendingRetryEntry]: ...


class InMemoryRetryBuffer:
    """Buffer en memoria para pruebas unitarias y ejecución local sin Redis."""

    def __init__(self) -> None:
        self._entries: list[PendingRetryEntry] = []

    async def enqueue(self, entry: PendingRetryEntry) -> None:
        self._entries.append(entry)

    async def list_pending(self) -> list[PendingRetryEntry]:
        return list(self._entries)


class RedisRetryBuffer:
    """Buffer persistente en Redis Streams para órdenes pendientes de reintento UAO."""

    def __init__(self, redis_client: Any, stream_name: str = "openclaw:uao:retry_orders") -> None:
        self._redis = redis_client
        self._stream_name = stream_name

    async def enqueue(self, entry: PendingRetryEntry) -> None:
        await self._redis.xadd(
            self._stream_name,
            {
                "idempotency_key": entry.request.idempotency_key,
                "payload": entry.model_dump_json(),
            },
        )

    async def list_pending(self) -> list[PendingRetryEntry]:
        raw = await self._redis.xrange(self._stream_name, "-", "+")
        items: list[PendingRetryEntry] = []
        for _msg_id, fields in raw:
            payload = fields.get(b"payload") or fields.get("payload")
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            if payload:
                items.append(PendingRetryEntry.model_validate_json(payload))
        return items


class UaoCircuitBreaker:
    """Circuit breaker determinista para aislar fallos del proveedor UAO."""

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
        clock: ClockFn | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold debe ser al menos 1.")
        if recovery_timeout_seconds <= 0:
            raise ValueError("recovery_timeout_seconds debe ser mayor que 0.")
        self._failure_threshold = failure_threshold
        self._recovery_timeout_seconds = recovery_timeout_seconds
        self._clock: ClockFn = clock if clock is not None else time.monotonic
        self._state = CircuitBreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitBreakerState:
        if (
            self._state == CircuitBreakerState.OPEN
            and self._opened_at is not None
            and (self._clock() - self._opened_at) >= self._recovery_timeout_seconds
        ):
            self._state = CircuitBreakerState.HALF_OPEN
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def allow_request(self) -> bool:
        current = self.state
        return current in (CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN)

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None
        self._state = CircuitBreakerState.CLOSED

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if (
            self._state == CircuitBreakerState.HALF_OPEN
            or self._consecutive_failures >= self._failure_threshold
        ):
            self._state = CircuitBreakerState.OPEN
            self._opened_at = self._clock()


@runtime_checkable
class UaoLiquidityPort(Protocol):
    """Puerto del cliente de liquidez UAO consumido por BuySimulationService."""

    async def request_withdrawal(self, request: UaoWithdrawalRequest) -> UaoDispatchResult: ...


class UaoClient:
    """Cliente HTTP asíncrono hacia UAO con timeout, backoff, circuit breaker y resguardo."""

    def __init__(
        self,
        base_url: str,
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 5.0,
        max_attempts: int = 3,
        base_backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 4.0,
        circuit_breaker: UaoCircuitBreaker | None = None,
        retry_buffer: OrderRetryBufferPort | None = None,
        sleep_fn: SleepFn | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds debe ser mayor que 0.")
        if max_attempts < 1:
            raise ValueError("max_attempts debe ser al menos 1.")
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._base_backoff_seconds = base_backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._circuit_breaker = (
            circuit_breaker if circuit_breaker is not None else UaoCircuitBreaker()
        )
        self._retry_buffer: OrderRetryBufferPort = (
            retry_buffer if retry_buffer is not None else InMemoryRetryBuffer()
        )
        self._sleep: SleepFn = sleep_fn if sleep_fn is not None else asyncio.sleep

    @property
    def circuit_breaker(self) -> UaoCircuitBreaker:
        return self._circuit_breaker

    @property
    def retry_buffer(self) -> OrderRetryBufferPort:
        return self._retry_buffer

    async def request_withdrawal(self, request: UaoWithdrawalRequest) -> UaoDispatchResult:
        """Envía una orden de retiro simulada a UAO sin bloquear ni perder la orden en fallo."""
        if not self._circuit_breaker.allow_request():
            reason = "circuit_breaker_open"
            await self._retry_buffer.enqueue(PendingRetryEntry(request=request, reason=reason))
            logger.warning(
                "uao_circuito_abierto_orden_resguardada",
                extra={"idempotency_key": request.idempotency_key, "order_id": request.order_id},
            )
            return UaoDispatchResult(
                ok=False,
                status="queued_for_retry",
                attempts=0,
                queued_for_retry=True,
                circuit_state=self._circuit_breaker.state,
                error_code=reason,
                error_message="Circuito UAO abierto; orden resguardada para reintento.",
            )

        endpoint = f"{self._base_url}/request-withdrawal"
        headers = {
            "Idempotency-Key": request.idempotency_key,
            "X-OpenClaw-Mode": "sandbox" if request.sandbox else "disabled",
        }
        payload = request.model_dump(mode="json")

        last_code = "upstream_error"
        last_message = "Fallo desconocido comunicando con UAO"

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._post(endpoint, payload=payload, headers=headers)

                if 200 <= response.status_code < 300:
                    self._circuit_breaker.record_success()
                    data = _safe_json(response)
                    ref = str(
                        data.get("reference_id")
                        or data.get("tx_id")
                        or f"uao-sim-{request.order_id}"
                    )
                    return UaoDispatchResult(
                        ok=True,
                        status="accepted",
                        reference_id=ref,
                        attempts=attempt,
                        queued_for_retry=False,
                        circuit_state=self._circuit_breaker.state,
                    )

                if 400 <= response.status_code < 500 and response.status_code != 429:
                    # Error de cliente no reintentable: no abre circuito ni reintenta
                    return UaoDispatchResult(
                        ok=False,
                        status="rejected_by_provider",
                        attempts=attempt,
                        queued_for_retry=False,
                        circuit_state=self._circuit_breaker.state,
                        error_code=f"http_{response.status_code}",
                        error_message=f"UAO rechazó la orden con HTTP {response.status_code}.",
                    )

                last_code = f"http_{response.status_code}"
                last_message = f"UAO devolvió HTTP {response.status_code} (intento {attempt})."
            except (TimeoutError, httpx.TimeoutException) as exc:
                last_code = "timeout"
                last_message = f"Timeout de {self._timeout_seconds}s hacia UAO: {exc}"
            except httpx.HTTPError as exc:
                last_code = "network_error"
                last_message = f"Error de red hacia UAO: {exc}"

            self._circuit_breaker.record_failure()
            if not self._circuit_breaker.allow_request():
                break

            if attempt < self._max_attempts:
                delay = min(
                    self._base_backoff_seconds * (2 ** (attempt - 1)),
                    self._max_backoff_seconds,
                )
                await self._sleep(delay)

        await self._retry_buffer.enqueue(
            PendingRetryEntry(request=request, reason=f"{last_code}: {last_message}")
        )
        logger.error(
            "uao_fallo_orden_resguardada",
            extra={
                "idempotency_key": request.idempotency_key,
                "order_id": request.order_id,
                "error_code": last_code,
                "circuit_state": self._circuit_breaker.state.value,
            },
        )
        return UaoDispatchResult(
            ok=False,
            status="queued_for_retry",
            attempts=attempt,
            queued_for_retry=True,
            circuit_state=self._circuit_breaker.state,
            error_code=last_code,
            error_message=last_message,
        )

    async def _post(
        self,
        url: str,
        *,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        if self._http_client is not None:
            return await asyncio.wait_for(
                self._http_client.post(url, json=payload, headers=headers),
                timeout=self._timeout_seconds,
            )
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            return await asyncio.wait_for(
                client.post(url, json=payload, headers=headers),
                timeout=self._timeout_seconds,
            )


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    try:
        parsed = response.json()
        return parsed if isinstance(parsed, dict) else {}
    except ValueError:
        return {}
