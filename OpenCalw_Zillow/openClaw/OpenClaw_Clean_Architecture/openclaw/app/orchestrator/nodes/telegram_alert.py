"""Nodo de alerta Telegram con plantilla, fotos, URL de publicación y botones inline (S4-07).

Reemplaza las llamadas síncronas bloqueantes de `requests` por `NotifierPort`
asíncrono con `httpx`, verifica la respuesta de Telegram, restringe el envío al
chat autorizado, adjunta la URL real de publicación, hasta 2 fotos y botones
interactivos de Aprobar / Negociar y Rechazar.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import re
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict

from app.core.config import settings
from app.orchestrator.config import AlertConfig

logger = logging.getLogger("OpenClaw.TelegramAlert")

_SELLER_CONTACT_RE = re.compile(
    r"(?im)^(?:seller\s+contact|brokered\s+by|listed\s+by|agent)\s*:?\s*(.+)$"
)


class AlertDispatchResult(BaseModel):
    """Resultado estructurado del envío de alerta."""

    model_config = ConfigDict(extra="forbid")

    sent: bool
    deduplicated: bool = False
    dedupe_key: str
    ficha_url: str
    status: str
    message_text: str = ""
    error: str | None = None


@runtime_checkable
class NotifierPort(Protocol):
    """Puerto de notificación (sección 3 de PLAN_IMPLEMENTACION_2DEVS.md)."""

    async def send_message(self, chat_id: str, text: str) -> tuple[bool, str | None]: ...


@runtime_checkable
class AlertDedupeStorePort(Protocol):
    """Puerto de deduplicación de alertas enviadas."""

    async def claim_if_new(self, dedupe_key: str, *, ttl_seconds: int) -> bool: ...


class InMemoryAlertDedupeStore:
    """Almacén de deduplicación de alertas en memoria."""

    def __init__(self) -> None:
        self._keys: set[str] = set()

    async def claim_if_new(self, dedupe_key: str, *, ttl_seconds: int) -> bool:
        if dedupe_key in self._keys:
            return False
        self._keys.add(dedupe_key)
        return True


class RedisAlertDedupeStore:
    """Almacén de deduplicación de alertas respaldado por Redis (`SET NX EX`)."""

    def __init__(self, redis_client: Any, *, key_prefix: str = "openclaw:alert:dedupe") -> None:
        self._redis = redis_client
        self._key_prefix = key_prefix

    async def claim_if_new(self, dedupe_key: str, *, ttl_seconds: int) -> bool:
        claimed = await self._redis.set(
            f"{self._key_prefix}:{dedupe_key}",
            "1",
            nx=True,
            ex=ttl_seconds,
        )
        return bool(claimed)


def extraer_contacto_vendedor(descripcion: str | None) -> str | None:
    """Extrae el contacto o corredor del vendedor si aparece en el texto del anuncio."""
    if not descripcion:
        return None
    m = _SELLER_CONTACT_RE.search(descripcion)
    if not m:
        return None
    valor = " ".join(m.group(1).split()).strip(" -|*")
    return valor[:140] if len(valor) >= 3 else None


def construir_botones_oportunidad(state: dict[str, Any]) -> dict[str, Any] | None:
    """Construye el teclado inline de Telegram (Aprobar/Rechazar + enlace publicación)."""
    prop_id = str(state.get("property_id") or "").strip()
    url_origen = str(state.get("url_origen") or "").strip()
    filas: list[list[dict[str, str]]] = []
    if prop_id:
        filas.append(
            [
                {"text": "✅ Aprobar / Negociar", "callback_data": f"apr:{prop_id}"},
                {"text": "❌ Rechazar", "callback_data": f"rec:{prop_id}"},
            ]
        )
    if url_origen.startswith(("https://", "http://")):
        filas.append([{"text": "🔗 Ver publicación", "url": url_origen}])
    return {"inline_keyboard": filas} if filas else None


class TelegramNotifierAdapter:
    """Implementación asíncrona de NotifierPort sobre la API Bot de Telegram."""

    def __init__(
        self,
        *,
        bot_token: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._bot_token = bot_token if bot_token is not None else getattr(settings, "TG_TOKEN", "")
        self._http_client = http_client
        self._timeout_seconds = timeout_seconds

    async def _post_telegram(self, method: str, payload: dict[str, Any]) -> tuple[bool, str | None]:
        url = f"https://api.telegram.org/bot{self._bot_token}/{method}"
        try:
            if self._http_client is not None:
                resp = await self._http_client.post(url, json=payload)
            else:
                async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                    resp = await client.post(url, json=payload)

            if resp.status_code != 200:
                return False, f"telegram_http_{resp.status_code}"
            body = resp.json()
            if not isinstance(body, dict) or not body.get("ok", False):
                return False, f"telegram_api_error:{body}"
            return True, None
        except httpx.HTTPError as exc:
            logger.error("telegram_network_error", extra={"error": str(exc)})
            return False, f"telegram_network_error:{type(exc).__name__}"

    async def send_message(
        self,
        chat_id: str,
        text: str,
        *,
        reply_markup: dict[str, Any] | None = None,
        photo_urls: list[str] | None = None,
    ) -> tuple[bool, str | None]:
        if not self._bot_token:
            return False, "tg_token_not_configured"
        if not chat_id:
            return False, "tg_chat_id_not_configured"

        fotos_validas = [
            u.strip()
            for u in (photo_urls or [])
            if isinstance(u, str) and u.strip().startswith(("https://", "http://"))
        ][:2]

        # Si hay foto válida y el texto entra en el límite de caption (1024 chars), enviamos sendPhoto
        if fotos_validas and len(text) <= 1020:
            photo_payload: dict[str, Any] = {
                "chat_id": chat_id,
                "photo": fotos_validas[0],
                "caption": text,
            }
            if reply_markup is not None:
                photo_payload["reply_markup"] = reply_markup
            ok_photo, _err_photo = await self._post_telegram("sendPhoto", photo_payload)
            if ok_photo:
                return True, None

        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": False,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        return await self._post_telegram("sendMessage", payload)

    async def answer_callback_query(
        self,
        callback_query_id: str,
        text: str = "",
    ) -> tuple[bool, str | None]:
        if not self._bot_token or not callback_query_id:
            return False, "callback_invalid"
        return await self._post_telegram(
            "answerCallbackQuery",
            {"callback_query_id": callback_query_id, "text": text[:180]},
        )


def construir_ficha_url(state: dict[str, Any], *, base_url: str) -> str:
    """Construye el enlace directo a la ficha auditable de la propiedad."""
    prop_id = str(
        state.get("property_id")
        or state.get("dedup_key")
        or state.get("id_plataforma_origen")
        or ""
    ).strip()
    if not prop_id or "://" in prop_id:
        prop_id = hashlib.sha256(
            f"{state.get('ubicacion', '')}|{state.get('condado', '')}".encode()
        ).hexdigest()[:16]
    return f"{base_url.rstrip('/')}/{prop_id}"


def generar_clave_dedupe_alerta(state: dict[str, Any]) -> str:
    """Genera una clave determinista de deduplicación por oportunidad."""
    explicit = str(state.get("dedup_key") or state.get("property_id") or "").strip()
    if explicit:
        return explicit
    raw = f"{state.get('ubicacion', '').strip().lower()}|{state.get('condado', '').strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def formatear_alerta_oportunidad(state: dict[str, Any], *, ficha_url: str) -> str:
    """Plantilla estructurada de alerta con publicación, 1–2 fotos, contacto y aviso legal."""
    ubicacion = state.get("ubicacion", "Ubicación no especificada")
    condado = state.get("condado", "N/A")
    precio = float(state.get("precio_usd") or 0.0)
    noi = float(state.get("noi") or 0.0)
    cap_rate = state.get("cap_rate_ajustado") or state.get("cap_rate") or 0.0
    capex = state.get("capex")
    vision_txt = (
        f"${float(capex):,.0f} (confianza {float(state.get('vision_confianza') or 0.0):.0%})"
        if capex is not None
        else "No disponible"
    )
    legal_flag = bool(state.get("requiere_revision_humana", False))
    legal_txt = (
        f"ALERTA ({state.get('riesgo_legal', 'medio')}) — Revisión humana obligatoria"
        if legal_flag
        else "Sin señales de riesgo en texto"
    )

    lineas = [
        "OPORTUNIDAD OPENCLAW DETECTADA",
        f"Propiedad: {ubicacion} ({condado})",
        f"Precio: ${precio:,.0f} | NOI: ${noi:,.0f} | Cap Rate: {float(cap_rate):.2f}%",
        f"CAPEX visual: {vision_txt}",
        f"Auditoría legal: {legal_txt}",
    ]

    url_origen = str(state.get("url_origen") or "").strip()
    if url_origen.startswith(("https://", "http://")):
        lineas.append(f"Publicación: {url_origen}")

    contacto = extraer_contacto_vendedor(str(state.get("descripcion_cruda") or ""))
    if contacto:
        lineas.append(f"Vendedor/Agente: {contacto}")

    fotos = [
        str(u).strip()
        for u in (state.get("imagenes") or [])
        if str(u).strip().startswith(("https://", "http://"))
    ][:2]
    for idx, foto in enumerate(fotos, start=1):
        lineas.append(f"Foto {idx}: {foto}")

    lineas.append(f"Ficha: {ficha_url}")
    return "\n".join(lineas)


async def _despachar_notificacion(
    notifier: NotifierPort,
    *,
    chat_id: str,
    text: str,
    reply_markup: dict[str, Any] | None,
    photo_urls: list[str],
) -> tuple[bool, str | None]:
    """Invoca ``notifier.send_message`` preservando compatibilidad con dobles simples de prueba."""
    sig = inspect.signature(notifier.send_message)
    kwargs: dict[str, Any] = {}
    if "reply_markup" in sig.parameters:
        kwargs["reply_markup"] = reply_markup
    if "photo_urls" in sig.parameters:
        kwargs["photo_urls"] = photo_urls
    return await notifier.send_message(chat_id, text, **kwargs)


async def nodo_alerta_telegram(
    state: dict[str, Any],
    *,
    notifier: NotifierPort | None = None,
    dedupe_store: AlertDedupeStorePort | None = None,
    alert_config: AlertConfig | None = None,
    chat_id: str | None = None,
) -> dict[str, Any]:
    """Envía una única alerta deduplicada para oportunidades elegibles con botones y publicación."""
    cfg = alert_config if alert_config is not None else AlertConfig()
    ficha_url = construir_ficha_url(state, base_url=cfg.ficha_base_url)
    dedupe_key = generar_clave_dedupe_alerta(state)

    estado_actual = state.get("estado")
    if estado_actual not in ("INSPECCION_VISUAL_OK", "VISION_NO_DISPONIBLE"):
        result = AlertDispatchResult(
            sent=False,
            deduplicated=False,
            dedupe_key=dedupe_key,
            ficha_url=ficha_url,
            status="ALERTA_OMITIDA",
        )
        return {"estado": "ALERTA_OMITIDA", "alerta": result.model_dump(), "ficha_url": ficha_url}

    store = dedupe_store if dedupe_store is not None else InMemoryAlertDedupeStore()
    is_new = await store.claim_if_new(dedupe_key, ttl_seconds=cfg.dedupe_ttl_seconds)
    if not is_new:
        result = AlertDispatchResult(
            sent=False,
            deduplicated=True,
            dedupe_key=dedupe_key,
            ficha_url=ficha_url,
            status="ALERTA_DUPLICADA_OMITIDA",
        )
        return {
            "estado": "ALERTA_DUPLICADA_OMITIDA",
            "alerta": result.model_dump(),
            "ficha_url": ficha_url,
        }

    mensaje = formatear_alerta_oportunidad(state, ficha_url=ficha_url)
    botones = construir_botones_oportunidad(state)
    fotos = [
        str(u).strip()
        for u in (state.get("imagenes") or [])
        if str(u).strip().startswith(("https://", "http://"))
    ][:2]

    target_chat = chat_id if chat_id is not None else str(getattr(settings, "TG_CHAT_ID", "") or "")
    active_notifier = notifier if notifier is not None else TelegramNotifierAdapter()

    ok, err = await _despachar_notificacion(
        active_notifier,
        chat_id=target_chat,
        text=mensaje,
        reply_markup=botones,
        photo_urls=fotos,
    )
    status_str = "ALERTA_ENVIADA" if ok else "ALERTA_ERROR"
    result = AlertDispatchResult(
        sent=ok,
        deduplicated=False,
        dedupe_key=dedupe_key,
        ficha_url=ficha_url,
        status=status_str,
        message_text=mensaje,
        error=err,
    )
    return {"estado": status_str, "alerta": result.model_dump(), "ficha_url": ficha_url}
