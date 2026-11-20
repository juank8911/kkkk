"""Pruebas de alertas Telegram con publicación, fotos, botones inline y flujo de negociación."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import fakeredis.aioredis

from app.db.models import Oferta, Propiedad
from app.orchestrator.nodes.telegram_alert import (
    InMemoryAlertDedupeStore,
    construir_botones_oportunidad,
    formatear_alerta_oportunidad,
    nodo_alerta_telegram,
)
from app.services import telegram_workflow
from app.services.normalizer import normalizar_propiedad_unica
from app.services.telegram_workflow import (
    parsear_instrucciones_negociacion,
    procesar_interaccion_telegram,
)


class _SpyTelegramBot:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.callbacks: list[tuple[str, str]] = []

    async def send_message(
        self,
        chat_id: str,
        text: str,
        *,
        reply_markup: dict[str, Any] | None = None,
        photo_urls: list[str] | None = None,
    ) -> tuple[bool, str | None]:
        self.sent.append(
            {
                "chat_id": chat_id,
                "text": text,
                "reply_markup": reply_markup,
                "photo_urls": photo_urls,
            }
        )
        return True, None

    async def answer_callback_query(
        self, callback_query_id: str, text: str = ""
    ) -> tuple[bool, str | None]:
        self.callbacks.append((callback_query_id, text))
        return True, None


def test_normalizer_preserves_underscores_in_realtor_urls_and_limits_photos() -> None:
    md = (
        "Listing URL: https://www.realtor.com/realestateandhomes-detail/462-King-Ave_Memphis_TN_38109_M81766-88066\n"
        "Property Address: 462 King Ave, Memphis, TN 38109\n"
        "County: Shelby\n"
        "Price: $89,900\n"
        "Description: Casa remodelada lista para renta.\n"
        "![f1](https://ap.rdcpix.com/foto1.webp)\n"
        "![f2](https://ap.rdcpix.com/foto2.webp)\n"
        "![f3](https://ap.rdcpix.com/foto3.webp)\n"
    )
    prop = normalizar_propiedad_unica(md, plataforma="realtor")
    assert (
        prop.id_plataforma_origen
        == "https://www.realtor.com/realestateandhomes-detail/462-King-Ave_Memphis_TN_38109_M81766-88066"
    )
    assert prop.imagenes == [
        "https://ap.rdcpix.com/foto1.webp",
        "https://ap.rdcpix.com/foto2.webp",
    ]


async def test_alerta_telegram_incluye_url_publicacion_fotos_y_botones_aprobar_rechazar() -> None:
    prop_id = str(uuid4())
    url = "https://www.zillow.com/homedetails/2055-Burnham-Ave-Memphis-TN-38127/12345_zpid/"
    state = {
        "property_id": prop_id,
        "estado": "VISION_NO_DISPONIBLE",
        "ubicacion": "2055 Burnham Ave, Memphis, TN 38127",
        "condado": "Shelby",
        "precio_usd": 54000.0,
        "noi": 10512.0,
        "cap_rate": 19.47,
        "url_origen": url,
        "imagenes": [
            "https://photos.zillowstatic.com/fp/foto1.jpg",
            "https://photos.zillowstatic.com/fp/foto2.jpg",
            "https://photos.zillowstatic.com/fp/foto3.jpg",
        ],
        "descripcion_cruda": "Casa 3bd/2ba.\nSeller Contact: Brokered by Keller Williams Realty",
    }

    texto = formatear_alerta_oportunidad(state, ficha_url=f"https://openclaw.internal/{prop_id}")
    assert f"Publicación: {url}" in texto
    assert "Foto 1: https://photos.zillowstatic.com/fp/foto1.jpg" in texto
    assert "Foto 2: https://photos.zillowstatic.com/fp/foto2.jpg" in texto
    assert "foto3.jpg" not in texto
    assert "Keller Williams Realty" in texto

    botones = construir_botones_oportunidad(state)
    assert botones is not None
    assert botones["inline_keyboard"][0][0]["callback_data"] == f"apr:{prop_id}"
    assert botones["inline_keyboard"][0][1]["callback_data"] == f"rec:{prop_id}"
    assert botones["inline_keyboard"][1][0]["url"] == url

    spy = _SpyTelegramBot()
    res = await nodo_alerta_telegram(
        state,
        notifier=spy,
        dedupe_store=InMemoryAlertDedupeStore(),
        chat_id="8503241862",
    )
    assert res["estado"] == "ALERTA_ENVIADA"
    assert len(spy.sent) == 1
    assert spy.sent[0]["reply_markup"] == botones
    assert spy.sent[0]["photo_urls"] == [
        "https://photos.zillowstatic.com/fp/foto1.jpg",
        "https://photos.zillowstatic.com/fp/foto2.jpg",
    ]


def test_parsear_instrucciones_negociacion_sin_condiciones_y_con_condiciones() -> None:
    precio, cond, sin_cond = parsear_instrucciones_negociacion(
        "ninguna", precio_publicado_usd=54000.0
    )
    assert precio == 54000.0
    assert sin_cond is True
    assert "seller financing" in cond.lower()
    assert "$5,000-$7,000" in cond
    assert "6 extra monthly payments" in cond.lower()

    precio2, cond2, sin_cond2 = parsear_instrucciones_negociacion(
        "ofrecer $48,500 al contado y cierre en 15 días",
        precio_publicado_usd=54000.0,
    )
    assert precio2 == 48500.0
    assert sin_cond2 is False
    assert "15 días" in cond2


async def test_flujo_interactivo_aprobar_instrucciones_tal_cual_y_rechazar(monkeypatch) -> None:
    fake_redis = fakeredis.aioredis.FakeRedis()
    monkeypatch.setattr(telegram_workflow, "redis_client", fake_redis)

    prop_id = uuid4()
    job_id = uuid4()
    prop = Propiedad(
        id=prop_id,
        job_id=job_id,
        plataforma="realtor",
        url_origen="https://www.realtor.com/realestateandhomes-detail/2055-Burnham-Ave_Memphis_TN_38127",
        ubicacion="2055 Burnham Ave, Memphis, TN 38127",
        condado="Shelby",
        precio_usd=54000.0,
        ingreso_anual_estimado=15600.0,
        imagenes=["https://ap.rdcpix.com/foto1.webp"],
        descripcion_cruda="Casa 3bd.\nSeller Contact: Brokered by Crye-Leike Realtors",
        created_at=datetime.now(UTC),
    )

    ofertas_creadas: list[dict[str, Any]] = []
    rechazos: list[UUID] = []

    async def fake_cargar(pid: UUID) -> Propiedad | None:
        return prop if pid == prop_id else None

    async def fake_iniciar(pid: UUID, *, chat_id: str, instrucciones: str | None):
        precio_of, cond, sin_cond = parsear_instrucciones_negociacion(
            instrucciones, precio_publicado_usd=float(prop.precio_usd)
        )
        of = Oferta(
            id=uuid4(),
            propiedad_id=pid,
            estado="propuesta",
            importe_usd=precio_of,
            notas=cond,
            actor_ultima_actualizacion=f"telegram:{chat_id}",
        )
        ofertas_creadas.append(
            {"importe_usd": precio_of, "condiciones": cond, "sin_condiciones": sin_cond}
        )
        return of, prop, cond, "Brokered by Crye-Leike Realtors", sin_cond

    async def fake_rechazar(pid: UUID, *, chat_id: str) -> Propiedad | None:
        rechazos.append(pid)
        return prop if pid == prop_id else None

    monkeypatch.setattr(telegram_workflow, "_cargar_propiedad", fake_cargar)
    monkeypatch.setattr(telegram_workflow, "_iniciar_negociacion_propiedad", fake_iniciar)
    monkeypatch.setattr(telegram_workflow, "_registrar_rechazo_humano", fake_rechazar)

    spy = _SpyTelegramBot()

    # 1. Usuario pulsa Aprobar -> pide instrucciones + botón Iniciar tal cual
    res_apr = await procesar_interaccion_telegram(
        {
            "callback_query": {
                "id": "cb-1",
                "data": f"apr:{prop_id}",
                "message": {"chat": {"id": 8503241862}},
            }
        },
        notifier=spy,  # type: ignore[arg-type]
    )
    assert res_apr["status"] == "esperando_instrucciones_negociacion"
    assert "Iniciar tal cual" in spy.sent[-1]["reply_markup"]["inline_keyboard"][0][0]["text"]
    assert "Crye-Leike Realtors" in spy.sent[-1]["text"]

    # 2. Usuario responde sin condiciones ("continuar") -> inicia tal cual con precio publicado
    res_go = await procesar_interaccion_telegram(
        {"message": {"chat": {"id": 8503241862}, "text": "continuar"}},
        notifier=spy,  # type: ignore[arg-type]
    )
    assert res_go["status"] == "negociacion_iniciada"
    assert res_go["sin_condiciones"] is True
    assert ofertas_creadas[-1]["importe_usd"] == 54000.0
    assert "NEGOCIACIÓN INICIADA (TAL CUAL · SELLER FINANCING)" in spy.sent[-1]["text"]
    assert "Seller Financing" in spy.sent[-1]["text"]
    assert "$5,000 to $7,000 USD" in spy.sent[-1]["text"]
    assert "6 extra monthly payments" in spy.sent[-1]["text"]
    assert prop.url_origen in spy.sent[-1]["text"]

    # 3. Usuario pulsa Aprobar otra vez y escribe condiciones personalizadas
    await procesar_interaccion_telegram(
        {
            "callback_query": {
                "id": "cb-2",
                "data": f"apr:{prop_id}",
                "message": {"chat": {"id": 8503241862}},
            }
        },
        notifier=spy,  # type: ignore[arg-type]
    )
    res_custom = await procesar_interaccion_telegram(
        {
            "message": {
                "chat": {"id": 8503241862},
                "text": "Ofrecer $47,000 cierre 20 dias sujeto a inspeccion",
            }
        },
        notifier=spy,  # type: ignore[arg-type]
    )
    assert res_custom["status"] == "negociacion_iniciada"
    assert res_custom["sin_condiciones"] is False
    assert ofertas_creadas[-1]["importe_usd"] == 47000.0
    assert "CON TUS INSTRUCCIONES" in spy.sent[-1]["text"]

    # 4. Usuario pulsa Rechazar -> registra rechazo humano
    res_rec = await procesar_interaccion_telegram(
        {
            "callback_query": {
                "id": "cb-3",
                "data": f"rec:{prop_id}",
                "message": {"chat": {"id": 8503241862}},
            }
        },
        notifier=spy,  # type: ignore[arg-type]
    )
    assert res_rec["status"] == "propiedad_rechazada"
    assert rechazos == [prop_id]

    await fake_redis.aclose()
