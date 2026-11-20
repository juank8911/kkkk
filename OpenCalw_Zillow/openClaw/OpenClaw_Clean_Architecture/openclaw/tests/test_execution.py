"""Pruebas de la salvaguarda del webhook de comandos."""


def test_buy_rechazado_cuando_el_modo_es_disabled(client, monkeypatch):
    from app.api.routes import execution

    monkeypatch.setattr(execution.settings, "BUY_MODE", "disabled")

    respuesta = client.post(
        "/v1/webhook/telegram-comando",
        json={"message": {"text": "/buy"}},
    )

    assert respuesta.status_code == 403
    assert respuesta.json()["code"] == "forbidden"


def test_comando_distinto_se_ignora(client):
    respuesta = client.post(
        "/v1/webhook/telegram-comando",
        json={"message": {"text": "/status"}},
    )

    assert respuesta.status_code == 200
    assert respuesta.json()["status"] == "Comando ignorado"
