"""Suite dedicada de pruebas de seguridad sin red (S2-09).

Cubre:
1. Autenticación por API key con hash SHA-256 en tiempo constante (401 / 403 / 200).
2. Validación estricta de payloads (anti-SSRF, campos extra, caracteres de control, montos negativos).
3. Protección contra replay de webhooks (`update_id`) y allowlist de chats Telegram.
4. Idempotencia estricta, conflicto de payload y manipulación de tokens en `/buy`.
"""

from __future__ import annotations

import hashlib
import hmac

import fakeredis.aioredis
import httpx
import pytest
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import ValidationError

from app.integrations.uao_client import UaoClient
from app.models.buy import BuyMode, BuyOrderRequest, BuyOrderStatus
from app.models.contracts import BarridoRequest
from app.services.buy_service import (
    BuyConfirmationError,
    BuyDisabledError,
    BuySimulationService,
    IdempotencyConflictError,
    InMemoryBuyAuditStore,
)


def _sha256(secret: str) -> str:
    return hashlib.sha256(secret.strip().encode("utf-8")).hexdigest()


def _build_security_test_app(
    *,
    expected_key_hash: str,
    telegram_secret: str,
    allowed_chat_ids: set[int],
    fake_redis: fakeredis.aioredis.FakeRedis,
) -> FastAPI:
    """App FastAPI sin red que modela los contratos de seguridad ADR-04 y S2-05."""
    app = FastAPI(title="OpenClaw Security Verification Harness")

    async def verify_api_key(
        x_openclaw_tactical_key: str | None = Header(default=None, alias="X-OpenClaw-Tactical-Key"),
    ) -> str:
        if not x_openclaw_tactical_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Falta X-OpenClaw-Tactical-Key",
            )
        if not hmac.compare_digest(_sha256(x_openclaw_tactical_key), expected_key_hash):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key inválida",
            )
        return "authorized"

    @app.post("/v1/openclaw/barridos", status_code=202, dependencies=[Depends(verify_api_key)])
    async def barridos(body: BarridoRequest) -> dict[str, object]:
        return {"accepted": True, "plataformas": body.plataformas}

    @app.post("/v1/webhook/telegram-comando")
    async def telegram_webhook(
        update: dict[str, object],
        x_telegram_bot_api_secret_token: str | None = Header(
            default=None, alias="X-Telegram-Bot-Api-Secret-Token"
        ),
    ) -> dict[str, object]:
        if not x_telegram_bot_api_secret_token or not hmac.compare_digest(
            x_telegram_bot_api_secret_token, telegram_secret
        ):
            raise HTTPException(status_code=403, detail="Secret token de Telegram inválido")

        update_id = update.get("update_id")
        if not isinstance(update_id, int):
            raise HTTPException(status_code=400, detail="Falta update_id entero")

        message = update.get("message")
        chat = message.get("chat") if isinstance(message, dict) else None
        chat_id = chat.get("id") if isinstance(chat, dict) else None
        if chat_id not in allowed_chat_ids:
            raise HTTPException(status_code=403, detail="Chat no autorizado")

        dedupe_key = f"tg:update:{update_id}"
        is_new = await fake_redis.set(dedupe_key, "1", nx=True, ex=3600)
        if not is_new:
            return {"status": "ignored_replay", "update_id": update_id}

        return {"status": "processed", "update_id": update_id}

    return app


async def test_security_api_key_401_403_and_202(fake_redis: fakeredis.aioredis.FakeRedis) -> None:
    secret_key = "openclaw-tactical-secret-2026"
    app = _build_security_test_app(
        expected_key_hash=_sha256(secret_key),
        telegram_secret="tg-secret",
        allowed_chat_ids={1001},
        fake_redis=fake_redis,
    )
    payload = {"query": "Miami, FL", "plataformas": ["zillow"]}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        r_no_key = await client.post("/v1/openclaw/barridos", json=payload)
        assert r_no_key.status_code == 401

        r_bad_key = await client.post(
            "/v1/openclaw/barridos",
            json=payload,
            headers={"X-OpenClaw-Tactical-Key": "wrong-key"},
        )
        assert r_bad_key.status_code == 403

        r_ok = await client.post(
            "/v1/openclaw/barridos",
            json=payload,
            headers={"X-OpenClaw-Tactical-Key": secret_key},
        )
        assert r_ok.status_code == 202


@pytest.mark.parametrize(
    "malicious_payload",
    [
        {"query": "Miami", "plataformas": ["http://169.254.169.254/latest/meta-data"]},
        {"query": "Miami", "plataformas": ["https://internal.corp:8080/admin"]},
        {"query": "Miami\r\nX-Injected: 1", "plataformas": ["zillow"]},
        {"query": "Miami", "plataformas": ["zillow"], "admin_override": True},
        {"query": " ", "plataformas": ["zillow"]},
    ],
)
async def test_security_rejects_ssrf_and_invalid_payloads(
    fake_redis: fakeredis.aioredis.FakeRedis,
    malicious_payload: dict[str, object],
) -> None:
    secret_key = "openclaw-tactical-secret-2026"
    app = _build_security_test_app(
        expected_key_hash=_sha256(secret_key),
        telegram_secret="tg-secret",
        allowed_chat_ids={1001},
        fake_redis=fake_redis,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.post(
            "/v1/openclaw/barridos",
            json=malicious_payload,
            headers={"X-OpenClaw-Tactical-Key": secret_key},
        )
        assert resp.status_code == 422


async def test_security_telegram_secret_allowlist_and_replay_dedupe(
    fake_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    app = _build_security_test_app(
        expected_key_hash=_sha256("k"),
        telegram_secret="tg-webhook-secret-xyz",
        allowed_chat_ids={777001},
        fake_redis=fake_redis,
    )
    update = {"update_id": 50001, "message": {"chat": {"id": 777001}, "text": "/status"}}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        # Sin secret token -> 403
        r_no_sec = await client.post("/v1/webhook/telegram-comando", json=update)
        assert r_no_sec.status_code == 403

        # Chat no autorizado -> 403
        bad_chat = {"update_id": 50002, "message": {"chat": {"id": 999999}, "text": "/buy"}}
        r_bad_chat = await client.post(
            "/v1/webhook/telegram-comando",
            json=bad_chat,
            headers={"X-Telegram-Bot-Api-Secret-Token": "tg-webhook-secret-xyz"},
        )
        assert r_bad_chat.status_code == 403

        # Primer envío válido -> processed
        r_first = await client.post(
            "/v1/webhook/telegram-comando",
            json=update,
            headers={"X-Telegram-Bot-Api-Secret-Token": "tg-webhook-secret-xyz"},
        )
        assert r_first.status_code == 200
        assert r_first.json()["status"] == "processed"

        # Replay mismo update_id -> ignorado
        r_replay = await client.post(
            "/v1/webhook/telegram-comando",
            json=update,
            headers={"X-Telegram-Bot-Api-Secret-Token": "tg-webhook-secret-xyz"},
        )
        assert r_replay.status_code == 200
        assert r_replay.json()["status"] == "ignored_replay"


async def test_security_buy_tampered_confirmation_and_replay_protection() -> None:
    dispatches = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        return httpx.Response(200, json={"reference_id": f"sim-{dispatches}"})

    uao = UaoClient(
        base_url="http://uao.local",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    audit = InMemoryBuyAuditStore()

    # 1. Fail-closed en modo disabled
    disabled_svc = BuySimulationService(uao_client=uao, mode=BuyMode.DISABLED, audit_store=audit)
    order = BuyOrderRequest(
        property_id="prop-sec-1",
        asset="USDT",
        amount=45_000.0,
        destination="sandbox_escrow_vault",
        idempotency_key="sec-idem-0001",
        requested_by="chat-777001",
    )
    with pytest.raises(BuyDisabledError):
        await disabled_svc.execute_buy(order)
    assert dispatches == 0

    # 2. Manipulación del monto después de emitir confirmación -> rechazado
    sandbox_svc = BuySimulationService(uao_client=uao, mode=BuyMode.SANDBOX, audit_store=audit)
    challenge = await sandbox_svc.request_confirmation(order)
    tampered = order.model_copy(
        update={"amount": 90_000.0, "confirmation_token": challenge.confirmation_token}
    )
    with pytest.raises(BuyConfirmationError):
        await sandbox_svc.execute_buy(tampered)
    assert dispatches == 0

    # 3. Ejecución legítima + replay idempotente + intento de conflicto
    valid = order.model_copy(update={"confirmation_token": challenge.confirmation_token})
    res_1 = await sandbox_svc.execute_buy(valid)
    res_2 = await sandbox_svc.execute_buy(valid)
    assert dispatches == 1
    assert res_1.status == BuyOrderStatus.SIMULATED_EXECUTED
    assert res_2.status == BuyOrderStatus.DUPLICATE_REPLAYED

    with pytest.raises(IdempotencyConflictError):
        await sandbox_svc.execute_buy(valid.model_copy(update={"amount": 46_000.0}))

    # 4. Payload de compra con monto <= 0 o idempotency_key insegura rechazado por Pydantic
    with pytest.raises(ValidationError):
        BuyOrderRequest(
            property_id="prop-sec-2",
            asset="USDT",
            amount=-100.0,
            destination="sandbox_escrow_vault",
            idempotency_key="sec-idem-0002",
            requested_by="chat-777001",
        )
    with pytest.raises(ValidationError):
        BuyOrderRequest(
            property_id="prop-sec-2",
            asset="USDT",
            amount=1000.0,
            destination="sandbox_escrow_vault",
            idempotency_key="short",
            requested_by="chat-777001",
        )
