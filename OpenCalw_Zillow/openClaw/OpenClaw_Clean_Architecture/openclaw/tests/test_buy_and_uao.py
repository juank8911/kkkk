"""Pruebas de /buy en modo simulación (S2-07) y cliente UAO resiliente (S2-08)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import fakeredis.aioredis
import httpx
import pytest

from app.integrations.uao_client import (
    CircuitBreakerState,
    InMemoryRetryBuffer,
    RedisRetryBuffer,
    UaoCircuitBreaker,
    UaoClient,
    UaoWithdrawalRequest,
)
from app.models.buy import BuyMode, BuyOrderRequest, BuyOrderStatus
from app.services.buy_service import (
    BuyConfirmationError,
    BuyDestinationNotAllowedError,
    BuyDisabledError,
    BuyLimitExceededError,
    BuySimulationService,
    IdempotencyConflictError,
    InMemoryBuyAuditStore,
    RedisBuyIdempotencyStore,
    parse_telegram_buy_command,
)
from tests.conftest import MutableClock


def _make_order(
    *,
    idempotency_key: str = "idem-order-0001",
    amount: float = 95_000.0,
    destination: str = "sandbox_escrow_vault",
    confirmation_token: str | None = None,
) -> BuyOrderRequest:
    return BuyOrderRequest(
        property_id="prop-cle-1001",
        asset="USDT",
        amount=amount,
        destination=destination,
        idempotency_key=idempotency_key,
        requested_by="tg-chat-9901",
        confirmation_token=confirmation_token,
    )


def _mock_uao_client(
    handler,
    *,
    failure_threshold: int = 2,
    recovery_timeout_seconds: float = 30.0,
    max_attempts: int = 3,
    clock: MutableClock | None = None,
    retry_buffer: InMemoryRetryBuffer | RedisRetryBuffer | None = None,
) -> tuple[UaoClient, list[float]]:
    delays: list[float] = []

    async def _sleep(sec: float) -> None:
        delays.append(sec)

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    breaker = UaoCircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout_seconds=recovery_timeout_seconds,
        clock=clock,
    )
    client = UaoClient(
        base_url="http://uao-sandbox.local/v1/liquidity",
        http_client=http_client,
        timeout_seconds=1.0,
        max_attempts=max_attempts,
        base_backoff_seconds=0.1,
        circuit_breaker=breaker,
        retry_buffer=retry_buffer or InMemoryRetryBuffer(),
        sleep_fn=_sleep,
    )
    return client, delays


async def test_buy_disabled_rejects_and_audits_without_calling_uao() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"reference_id": "should-not-run"})

    uao, _ = _mock_uao_client(handler)
    audit = InMemoryBuyAuditStore()
    service = BuySimulationService(uao_client=uao, mode=BuyMode.DISABLED, audit_store=audit)

    with pytest.raises(BuyDisabledError):
        await service.execute_buy(_make_order())

    assert calls == 0
    events = await audit.list_events()
    assert len(events) == 1
    assert events[0].status == BuyOrderStatus.REJECTED_DISABLED


async def test_buy_sandbox_idempotency_executes_once_and_audits_both(
    fake_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    """Criterio S2-07: doble envío con la misma clave ejecuta una sola vez y queda auditado."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers["Idempotency-Key"])
        return httpx.Response(200, json={"reference_id": "uao-sim-tx-777"})

    uao, _ = _mock_uao_client(handler)
    audit = InMemoryBuyAuditStore()
    idem_store = RedisBuyIdempotencyStore(fake_redis)
    service = BuySimulationService(
        uao_client=uao,
        mode=BuyMode.SANDBOX,
        audit_store=audit,
        idempotency_store=idem_store,
        require_confirmation=True,
    )

    base_order = _make_order(idempotency_key="idem-double-send-01")
    challenge = await service.request_confirmation(base_order)
    confirmed_order = base_order.model_copy(
        update={"confirmation_token": challenge.confirmation_token}
    )

    first = await service.execute_buy(confirmed_order)
    second = await service.execute_buy(confirmed_order)

    assert calls == ["idem-double-send-01"]
    assert first.executed is True
    assert first.replayed is False
    assert first.status == BuyOrderStatus.SIMULATED_EXECUTED
    assert first.upstream_reference == "uao-sim-tx-777"

    assert second.executed is False
    assert second.replayed is True
    assert second.status == BuyOrderStatus.DUPLICATE_REPLAYED
    assert second.order_id == first.order_id
    assert second.upstream_reference == "uao-sim-tx-777"

    events = await audit.list_events(idempotency_key="idem-double-send-01")
    assert [e.status for e in events] == [
        BuyOrderStatus.PENDING_CONFIRMATION,
        BuyOrderStatus.SIMULATED_EXECUTED,
        BuyOrderStatus.DUPLICATE_REPLAYED,
    ]


async def test_buy_sandbox_enforces_limits_destinations_and_confirmation() -> None:
    now = datetime(2026, 12, 15, 10, 0, tzinfo=UTC)

    def now_fn() -> datetime:
        return now

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"reference_id": "uao-ok"})

    uao, _ = _mock_uao_client(handler)
    service = BuySimulationService(
        uao_client=uao,
        mode=BuyMode.SANDBOX,
        max_order_amount=100_000.0,
        max_daily_amount=150_000.0,
        confirmation_ttl_seconds=60,
        now_fn=now_fn,
    )

    # Destino no permitido
    with pytest.raises(BuyDestinationNotAllowedError):
        await service.execute_buy(_make_order(destination="attacker_wallet"))

    # Excede límite por operación
    with pytest.raises(BuyLimitExceededError):
        await service.execute_buy(_make_order(amount=120_000.0))

    # Sin confirmación en dos pasos
    with pytest.raises(BuyConfirmationError):
        await service.execute_buy(_make_order(idempotency_key="idem-no-confirm", amount=80_000.0))

    # Confirmación expirada
    expiring_order = _make_order(idempotency_key="idem-expired-01", amount=80_000.0)
    ch = await service.request_confirmation(expiring_order)
    now = now + timedelta(seconds=120)
    with pytest.raises(BuyConfirmationError):
        await service.execute_buy(
            expiring_order.model_copy(update={"confirmation_token": ch.confirmation_token})
        )

    # Primera orden de 90k exitosa; segunda de 70k excede límite diario de 150k
    order_1 = _make_order(idempotency_key="idem-daily-001", amount=90_000.0)
    c1 = await service.request_confirmation(order_1)
    res_1 = await service.execute_buy(
        order_1.model_copy(update={"confirmation_token": c1.confirmation_token})
    )
    assert res_1.executed is True

    with pytest.raises(BuyLimitExceededError):
        await service.request_confirmation(
            _make_order(idempotency_key="idem-daily-002", amount=70_000.0)
        )


async def test_buy_idempotency_key_conflict_with_different_amount_raises_409() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"reference_id": "uao-1"})

    uao, _ = _mock_uao_client(handler)
    service = BuySimulationService(
        uao_client=uao,
        mode=BuyMode.SANDBOX,
        require_confirmation=False,
    )

    await service.execute_buy(_make_order(idempotency_key="idem-conflict-01", amount=50_000.0))

    with pytest.raises(IdempotencyConflictError):
        await service.execute_buy(_make_order(idempotency_key="idem-conflict-01", amount=75_000.0))


async def test_uao_client_retries_circuit_breaker_and_preserves_order(
    fake_redis: fakeredis.aioredis.FakeRedis,
    fake_clock: MutableClock,
) -> None:
    """Criterio S2-08: fallo del proveedor no bloquea el worker ni pierde la orden."""
    attempts_seen = 0
    fail_upstream = True

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts_seen
        attempts_seen += 1
        if fail_upstream:
            raise httpx.ReadTimeout("UAO timed out", request=request)
        return httpx.Response(200, json={"reference_id": "uao-recovered-99"})

    retry_buf = RedisRetryBuffer(fake_redis)
    uao, delays = _mock_uao_client(
        handler,
        failure_threshold=2,
        recovery_timeout_seconds=30.0,
        max_attempts=3,
        clock=fake_clock,
        retry_buffer=retry_buf,
    )

    req_1 = UaoWithdrawalRequest(
        order_id="ord-1",
        property_id="prop-1",
        idempotency_key="idem-uao-0001",
        asset="USDT",
        amount=50_000.0,
        destination="sandbox_escrow_vault",
    )
    res_1 = await uao.request_withdrawal(req_1)
    assert res_1.ok is False
    assert res_1.queued_for_retry is True
    assert res_1.circuit_state == CircuitBreakerState.OPEN
    assert attempts_seen == 2  # Circuito abre en el 2do fallo sin agotar 3er intento
    assert len(delays) == 1

    # Siguiente orden con circuito ABIERTO: fail-fast (0 llamadas HTTP) y orden resguardada
    req_2 = req_1.model_copy(update={"order_id": "ord-2", "idempotency_key": "idem-uao-0002"})
    res_2 = await uao.request_withdrawal(req_2)
    assert res_2.ok is False
    assert res_2.queued_for_retry is True
    assert res_2.attempts == 0
    assert res_2.error_code == "circuit_breaker_open"
    assert attempts_seen == 2

    # Ambas órdenes quedaron preservadas en RedisRetryBuffer (ninguna perdida)
    pending = await retry_buf.list_pending()
    assert [p.request.order_id for p in pending] == ["ord-1", "ord-2"]

    # Tras cooldown -> HALF_OPEN -> éxito cierra circuito
    fake_clock.advance(31.0)
    assert uao.circuit_breaker.state == CircuitBreakerState.HALF_OPEN
    fail_upstream = False

    req_3 = req_1.model_copy(update={"order_id": "ord-3", "idempotency_key": "idem-uao-0003"})
    res_3 = await uao.request_withdrawal(req_3)
    assert res_3.ok is True
    assert res_3.reference_id == "uao-recovered-99"
    assert res_3.circuit_state == CircuitBreakerState.CLOSED


def test_parse_telegram_buy_command_requires_explicit_parameters() -> None:
    parsed = parse_telegram_buy_command(
        "/buy property_id=prop-55 amount=88500 asset=USDC "
        "destination=sandbox_escrow_vault idempotency_key=tg-ord-5501 confirm=cnf_12345678",
        requested_by="chat-42",
    )
    assert parsed.property_id == "prop-55"
    assert parsed.amount == 88500.0
    assert parsed.asset == "USDC"
    assert parsed.confirmation_token == "cnf_12345678"

    with pytest.raises(ValueError):
        parse_telegram_buy_command("/buy", requested_by="chat-42")
