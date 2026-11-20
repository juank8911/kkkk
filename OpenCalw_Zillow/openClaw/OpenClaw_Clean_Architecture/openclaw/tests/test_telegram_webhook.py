"""Pruebas del webhook de Telegram: secreto, allowlist y deduplicacion (S2-05)."""

from __future__ import annotations

import pytest

from app.api.routes import execution

SECRETO = "secreto-de-prueba"


class RedisDedupe:
    """Doble de Redis con semantica de SET NX EX."""

    def __init__(self) -> None:
        self.claves: set[str] = set()
        self.falla = False

    async def set(self, clave, valor, nx=False, ex=None):  # noqa: A002 - firma de redis
        if self.falla:
            raise ConnectionError("redis caido")
        if nx and clave in self.claves:
            return None
        self.claves.add(clave)
        return True


@pytest.fixture()
def dedupe(monkeypatch: pytest.MonkeyPatch) -> RedisDedupe:
    doble = RedisDedupe()
    monkeypatch.setattr(execution.redis_client, "set", doble.set)
    return doble


@pytest.fixture()
def con_secreto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(execution.settings, "TG_WEBHOOK_SECRET", SECRETO)
    monkeypatch.setattr(execution.settings, "TG_ALLOWED_CHAT_IDS", "")


def _update(texto: str = "/status", update_id: int | None = 1, chat_id: int = 42) -> dict:
    cuerpo: dict = {"message": {"text": texto, "chat": {"id": chat_id}}}
    if update_id is not None:
        cuerpo["update_id"] = update_id
    return cuerpo


def test_secret_token_valido_procesa(anon_client, con_secreto, dedupe) -> None:
    respuesta = anon_client.post(
        "/v1/webhook/telegram-comando",
        json=_update(),
        headers={execution.CABECERA_SECRETO: SECRETO},
    )

    assert respuesta.status_code == 200
    assert respuesta.json()["status"] == "Comando ignorado"


def test_secret_token_invalido_devuelve_403(anon_client, con_secreto, dedupe) -> None:
    respuesta = anon_client.post(
        "/v1/webhook/telegram-comando",
        json=_update(),
        headers={execution.CABECERA_SECRETO: "otro-secreto"},
    )

    assert respuesta.status_code == 403


def test_sin_credenciales_con_secreto_configurado_devuelve_403(
    anon_client, con_secreto, dedupe
) -> None:
    respuesta = anon_client.post("/v1/webhook/telegram-comando", json=_update())

    assert respuesta.status_code == 403


def test_api_key_tactica_tambien_autoriza(client, con_secreto, dedupe) -> None:
    respuesta = client.post("/v1/webhook/telegram-comando", json=_update())

    assert respuesta.status_code == 200


def test_update_repetido_se_ignora(client, con_secreto, dedupe) -> None:
    primero = client.post("/v1/webhook/telegram-comando", json=_update(update_id=7))
    segundo = client.post("/v1/webhook/telegram-comando", json=_update(update_id=7))

    assert primero.status_code == 200
    assert primero.json()["status"] == "Comando ignorado"
    assert segundo.status_code == 200
    assert "duplicado" in segundo.json()["status"].lower()
    assert len(dedupe.claves) == 1


def test_update_sin_id_no_se_deduplica(client, con_secreto, dedupe) -> None:
    primero = client.post("/v1/webhook/telegram-comando", json=_update(update_id=None))
    segundo = client.post("/v1/webhook/telegram-comando", json=_update(update_id=None))

    assert primero.status_code == 200
    assert segundo.status_code == 200
    assert dedupe.claves == set()


def test_chat_fuera_de_la_allowlist_devuelve_403(
    client, con_secreto, dedupe, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(execution.settings, "TG_ALLOWED_CHAT_IDS", "100, 200")

    respuesta = client.post("/v1/webhook/telegram-comando", json=_update(chat_id=42))

    assert respuesta.status_code == 403


def test_chat_en_la_allowlist_pasa(
    client, con_secreto, dedupe, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(execution.settings, "TG_ALLOWED_CHAT_IDS", "100,42")

    respuesta = client.post("/v1/webhook/telegram-comando", json=_update(chat_id=42))

    assert respuesta.status_code == 200


def test_redis_caido_bloquea_el_webhook(client, con_secreto, dedupe) -> None:
    dedupe.falla = True

    respuesta = client.post("/v1/webhook/telegram-comando", json=_update(update_id=99))

    assert respuesta.status_code == 503
    assert respuesta.json()["code"] == "service_unavailable"


def test_allowlist_se_parsea_con_espacios_y_punto_y_coma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution.settings, "TG_ALLOWED_CHAT_IDS", " 10 ;20,30 ")

    assert execution.settings.chats_autorizados == frozenset({10, 20, 30})
