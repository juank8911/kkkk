"""Flujo interactivo de aprobación, rechazo y negociación supervisada en Telegram.

Reglas de cumplimiento:
- Toda aprobación, rechazo e inicio de negociación requiere acción humana explícita.
- No se envían correos, SMS ni ofertas vinculantes automáticamente a terceros:
  al iniciar la negociación se registra la `Oferta` (estado `propuesta`) +
  `OfertaEvento` en PostgreSQL y se devuelve el borrador listo para revisión y
  envío humano con el precio, condiciones, URL de publicación y contacto/vendedor.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from uuid import UUID

from app.core.redis_client import redis_client
from app.db.models import Decision, Oferta, Propiedad
from app.db.offers_repository import RepositorioOfertas
from app.db.uow import unidad_de_trabajo
from app.models.offers import OfertaEstado
from app.orchestrator.nodes.telegram_alert import (
    TelegramNotifierAdapter,
    extraer_contacto_vendedor,
)

logger = logging.getLogger("OpenClaw.TelegramWorkflow")

PENDING_NEG_PREFIX = "openclaw:tg:pending_neg:"
PENDING_NEG_TTL_SEGUNDOS = 1800

_SIN_CONDICIONES_FRASES: frozenset[str] = frozenset(
    {
        "",
        "ninguna",
        "ninguno",
        "sin condiciones",
        "sin condicion",
        "como esta",
        "como está",
        "tal cual",
        "continuar",
        "iniciar",
        "ok",
        "si",
        "sí",
        "no",
        "-",
    }
)

_PRECIO_INSTRUCCION_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)\s*([Kk])?|\b(\d{2,3})\s*[Kk]\b")


def parsear_instrucciones_negociacion(
    texto: str | None,
    *,
    precio_publicado_usd: float,
) -> tuple[float, str, bool]:
    """Interpreta las instrucciones del usuario devolviendo `(precio_oferta, condiciones, sin_condiciones)`.

    Si el usuario no escribe ninguna condición (o escribe `ninguna`/`tal cual`/`continuar`),
    mantiene exactamente `precio_publicado_usd`.
    """
    limpio = " ".join((texto or "").strip().split())
    if limpio.lower() in _SIN_CONDICIONES_FRASES:
        return (
            round(precio_publicado_usd, 2),
            "Sin condiciones adicionales; iniciar con el precio publicado de la vivienda.",
            True,
        )

    precio_objetivo = precio_publicado_usd
    m = _PRECIO_INSTRUCCION_RE.search(limpio)
    if m:
        if m.group(1):
            val = float(m.group(1).replace(",", ""))
            if (m.group(2) or "").lower() == "k":
                val *= 1_000.0
            if 5_000.0 <= val <= 25_000_000.0:
                precio_objetivo = val
        elif m.group(3):
            val = float(m.group(3)) * 1_000.0
            if 5_000.0 <= val <= 25_000_000.0:
                precio_objetivo = val

    return round(precio_objetivo, 2), limpio[:600], False


def construir_borrador_negociacion(
    *,
    ubicacion: str,
    url_origen: str,
    precio_publicado_usd: float,
    precio_oferta_usd: float,
    condiciones: str,
    contacto_vendedor: str,
) -> str:
    """Genera un borrador no vinculante listo para enviar al vendedor o agente."""
    return (
        f"Hola ({contacto_vendedor}),\n"
        f"Estoy interesado en la propiedad ubicada en {ubicacion} ({url_origen}).\n"
        f"Precio publicado: ${precio_publicado_usd:,.0f} | Propuesta inicial: ${precio_oferta_usd:,.0f}.\n"
        f"Condiciones: {condiciones}\n"
        "Quedo atento para confirmar disponibilidad, divulgaciones y siguientes pasos.\n"
        "[Borrador no vinculante generado por OpenClaw — requiere envío y firma humana]"
    )


async def _cargar_propiedad(propiedad_id: UUID) -> Propiedad | None:
    async with unidad_de_trabajo() as unidad:
        return await unidad.sesion.get(Propiedad, propiedad_id)


async def _guardar_pendiente(chat_id: str, datos: dict[str, Any]) -> None:
    await redis_client.set(
        f"{PENDING_NEG_PREFIX}{chat_id}",
        json.dumps(datos),
        ex=PENDING_NEG_TTL_SEGUNDOS,
    )


async def _leer_pendiente(chat_id: str) -> dict[str, Any] | None:
    crudo = await redis_client.get(f"{PENDING_NEG_PREFIX}{chat_id}")
    if crudo is None:
        return None
    texto = crudo.decode("utf-8") if isinstance(crudo, bytes) else str(crudo)
    try:
        data = json.loads(texto)
        return data if isinstance(data, dict) else None
    except ValueError:
        return None


async def _limpiar_pendiente(chat_id: str) -> None:
    await redis_client.delete(f"{PENDING_NEG_PREFIX}{chat_id}")


async def _registrar_rechazo_humano(propiedad_id: UUID, *, chat_id: str) -> Propiedad | None:
    async with unidad_de_trabajo() as unidad:
        prop = await unidad.sesion.get(Propiedad, propiedad_id)
        if prop is None:
            return None
        decision = Decision(
            propiedad_id=prop.id,
            veredicto="descartar",
            motivo="RECHAZADO_HUMANO_TELEGRAM",
            detalle={
                "estado": "RECHAZADO_HUMANO",
                "actor": f"telegram:{chat_id}",
                "origen": "boton_telegram",
            },
        )
        await unidad.decisiones.registrar(decision)
        await unidad.eventos.registrar(
            "propiedad.rechazada_telegram",
            f"Propiedad {prop.ubicacion} rechazada por operador en Telegram",
            job_id=prop.job_id,
            nivel="info",
            payload={"propiedad_id": str(prop.id), "chat_id": chat_id},
        )
        await unidad.confirmar()
        return prop


async def _iniciar_negociacion_propiedad(
    propiedad_id: UUID,
    *,
    chat_id: str,
    instrucciones: str | None,
) -> tuple[Oferta, Propiedad, str, str, bool] | None:
    async with unidad_de_trabajo() as unidad:
        prop = await unidad.sesion.get(Propiedad, propiedad_id)
        if prop is None:
            return None

        precio_pub = float(prop.precio_usd)
        precio_oferta, condiciones, sin_condiciones = parsear_instrucciones_negociacion(
            instrucciones,
            precio_publicado_usd=precio_pub,
        )
        contacto = extraer_contacto_vendedor(prop.descripcion_cruda) or (
            f"Vendedor / agente en publicación ({prop.plataforma})"
        )

        repo_ofertas = RepositorioOfertas(unidad.sesion)
        oferta = Oferta(
            propiedad_id=prop.id,
            estado=OfertaEstado.PROPUESTA.value,
            importe_usd=precio_oferta,
            notas=f"Contacto: {contacto} | Condiciones: {condiciones}",
            actor_ultima_actualizacion=f"telegram:{chat_id}",
        )
        guardada = await repo_ofertas.crear(
            oferta,
            notas_evento=(
                "Negociación iniciada desde Telegram "
                f"({'tal cual' if sin_condiciones else 'con instrucciones'}): {condiciones}"
            ),
        )
        await unidad.eventos.registrar(
            "negociacion.iniciada_telegram",
            f"Negociación iniciada para {prop.ubicacion} por ${precio_oferta:,.0f}",
            job_id=prop.job_id,
            nivel="info",
            payload={
                "oferta_id": str(guardada.id),
                "propiedad_id": str(prop.id),
                "precio_publicado_usd": precio_pub,
                "precio_oferta_usd": precio_oferta,
                "sin_condiciones": sin_condiciones,
                "contacto_vendedor": contacto,
                "url_origen": prop.url_origen,
                "condiciones": condiciones,
            },
        )
        await unidad.confirmar()
        return guardada, prop, condiciones, contacto, sin_condiciones


async def procesar_interaccion_telegram(
    update: dict[str, Any],
    *,
    notifier: TelegramNotifierAdapter | None = None,
) -> dict[str, Any]:
    """Procesa botones inline (`callback_query`) y respuestas de instrucciones de negociación."""
    bot = notifier if notifier is not None else TelegramNotifierAdapter()

    # 1. Botones inline (callback_query)
    callback = update.get("callback_query")
    if isinstance(callback, dict):
        cb_id = str(callback.get("id") or "")
        data = str(callback.get("data") or "").strip()
        msg = callback.get("message") or {}
        chat_id = str((msg.get("chat") or {}).get("id") or "")
        if not chat_id or ":" not in data:
            await bot.answer_callback_query(cb_id, "Acción no reconocida")
            return {"status": "callback_invalido"}

        accion, raw_id = data.split(":", 1)
        try:
            prop_id = UUID(raw_id.strip())
        except ValueError:
            await bot.answer_callback_query(cb_id, "ID de propiedad inválido")
            return {"status": "propiedad_id_invalido"}

        if accion == "apr":
            prop = await _cargar_propiedad(prop_id)
            if prop is None:
                await bot.answer_callback_query(cb_id, "Propiedad no encontrada")
                return {"status": "propiedad_no_encontrada"}

            contacto = extraer_contacto_vendedor(prop.descripcion_cruda) or (
                f"Vendedor / agente en publicación ({prop.plataforma})"
            )
            precio_pub = float(prop.precio_usd)
            await _guardar_pendiente(
                chat_id,
                {
                    "propiedad_id": str(prop.id),
                    "ubicacion": prop.ubicacion,
                    "precio_usd": precio_pub,
                    "url_origen": prop.url_origen,
                    "contacto_vendedor": contacto,
                },
            )
            await bot.answer_callback_query(cb_id, "Aprobada: indica condiciones o inicia tal cual")
            prompt = (
                "✅ PROPIEDAD APROBADA PARA NEGOCIAR\n"
                f"Propiedad: {prop.ubicacion}\n"
                f"Precio publicado: ${precio_pub:,.0f}\n"
                f"Contacto/Vendedor: {contacto}\n"
                f"Publicación: {prop.url_origen}\n\n"
                "✍️ Escribe ahora tus instrucciones o condiciones para negociar "
                "(ej: ofrecer $48,000, cierre en 21 días, inspección).\n"
                "Si no escribes ninguna condición, pulsa 'Iniciar tal cual' o escribe "
                "'continuar' para iniciar con el precio publicado y el contacto del vendedor."
            )
            botones_prompt = {
                "inline_keyboard": [
                    [
                        {
                            "text": f"▶️ Iniciar tal cual (${precio_pub:,.0f})",
                            "callback_data": f"neg_go:{prop.id}",
                        }
                    ],
                    [{"text": "✖️ Cancelar", "callback_data": f"neg_cancel:{prop.id}"}],
                ]
            }
            await bot.send_message(chat_id, prompt, reply_markup=botones_prompt)
            return {"status": "esperando_instrucciones_negociacion", "propiedad_id": str(prop.id)}

        if accion == "rec":
            prop = await _registrar_rechazo_humano(prop_id, chat_id=chat_id)
            await _limpiar_pendiente(chat_id)
            if prop is None:
                await bot.answer_callback_query(cb_id, "Propiedad no encontrada")
                return {"status": "propiedad_no_encontrada"}
            await bot.answer_callback_query(cb_id, "Propiedad rechazada")
            await bot.send_message(
                chat_id,
                f"❌ Propiedad rechazada y registrada en el panel:\n{prop.ubicacion}\n{prop.url_origen}",
            )
            return {"status": "propiedad_rechazada", "propiedad_id": str(prop.id)}

        if accion == "neg_go":
            resultado = await _iniciar_negociacion_propiedad(
                prop_id,
                chat_id=chat_id,
                instrucciones="",
            )
            await _limpiar_pendiente(chat_id)
            if resultado is None:
                await bot.answer_callback_query(cb_id, "Propiedad no encontrada")
                return {"status": "propiedad_no_encontrada"}
            oferta, prop, condiciones, contacto, _sin_cond = resultado
            borrador = construir_borrador_negociacion(
                ubicacion=prop.ubicacion,
                url_origen=prop.url_origen,
                precio_publicado_usd=float(prop.precio_usd),
                precio_oferta_usd=float(oferta.importe_usd),
                condiciones=condiciones,
                contacto_vendedor=contacto,
            )
            await bot.answer_callback_query(cb_id, "Negociación iniciada tal cual")
            await bot.send_message(
                chat_id,
                "🤝 NEGOCIACIÓN INICIADA (TAL CUAL)\n"
                f"Oferta ID: {oferta.id}\n"
                f"Propiedad: {prop.ubicacion}\n"
                f"Precio inicio: ${float(oferta.importe_usd):,.0f}\n"
                f"Contacto/Vendedor: {contacto}\n"
                f"Publicación: {prop.url_origen}\n\n"
                f"Borrador listo para enviar:\n{borrador}",
            )
            return {"status": "negociacion_iniciada", "oferta_id": str(oferta.id)}

        if accion == "neg_cancel":
            await _limpiar_pendiente(chat_id)
            await bot.answer_callback_query(cb_id, "Negociación cancelada")
            await bot.send_message(chat_id, "✖️ Inicio de negociación cancelado.")
            return {"status": "negociacion_cancelada"}

        await bot.answer_callback_query(cb_id, "Acción desconocida")
        return {"status": "callback_ignorado"}

    # 2. Mensaje de texto cuando hay una negociación pendiente esperando instrucciones
    mensaje = update.get("message") or {}
    chat_id = str((mensaje.get("chat") or {}).get("id") or "")
    texto = str(mensaje.get("text") or "").strip()
    if not chat_id or not texto or texto.startswith("/"):
        return {"status": "sin_accion"}

    pendiente = await _leer_pendiente(chat_id)
    if pendiente is None:
        return {"status": "sin_negociacion_pendiente"}

    try:
        prop_id = UUID(str(pendiente.get("propiedad_id") or ""))
    except ValueError:
        await _limpiar_pendiente(chat_id)
        return {"status": "pendiente_invalido"}

    resultado = await _iniciar_negociacion_propiedad(
        prop_id,
        chat_id=chat_id,
        instrucciones=texto,
    )
    await _limpiar_pendiente(chat_id)
    if resultado is None:
        await bot.send_message(chat_id, "No se encontró la propiedad para iniciar la negociación.")
        return {"status": "propiedad_no_encontrada"}

    oferta, prop, condiciones, contacto, sin_cond = resultado
    borrador = construir_borrador_negociacion(
        ubicacion=prop.ubicacion,
        url_origen=prop.url_origen,
        precio_publicado_usd=float(prop.precio_usd),
        precio_oferta_usd=float(oferta.importe_usd),
        condiciones=condiciones,
        contacto_vendedor=contacto,
    )
    modo_txt = "TAL CUAL" if sin_cond else "CON TUS INSTRUCCIONES"
    await bot.send_message(
        chat_id,
        f"🤝 NEGOCIACIÓN INICIADA ({modo_txt})\n"
        f"Oferta ID: {oferta.id}\n"
        f"Propiedad: {prop.ubicacion}\n"
        f"Precio propuesto: ${float(oferta.importe_usd):,.0f}\n"
        f"Contacto/Vendedor: {contacto}\n"
        f"Publicación: {prop.url_origen}\n\n"
        f"Borrador listo para enviar:\n{borrador}",
    )
    return {
        "status": "negociacion_iniciada",
        "oferta_id": str(oferta.id),
        "sin_condiciones": sin_cond,
    }
