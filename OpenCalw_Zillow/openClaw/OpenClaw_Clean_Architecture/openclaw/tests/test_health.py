"""Pruebas de liveness y readiness."""


def test_health_responde_ok(client):
    respuesta = client.get("/health")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["status"] == "ok"
    assert cuerpo["env"] == "test"
    assert cuerpo["version"]


def test_readyz_ok_cuando_redis_responde(client, monkeypatch):
    from app.core.redis_client import redis_client

    async def ping_ok() -> bool:
        return True

    monkeypatch.setattr(redis_client, "ping", ping_ok)

    respuesta = client.get("/readyz")

    assert respuesta.status_code == 200
    assert respuesta.json()["checks"]["redis"] == "ok"


def test_readyz_503_cuando_redis_falla(client, monkeypatch):
    from app.core.redis_client import redis_client

    async def ping_falla() -> bool:
        raise ConnectionError("sin conexion")

    monkeypatch.setattr(redis_client, "ping", ping_falla)

    respuesta = client.get("/readyz")

    assert respuesta.status_code == 503
    assert respuesta.json()["checks"]["redis"].startswith("error:")
