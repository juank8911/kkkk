"""Flujo interactivo de aprobación, rechazo y negociación supervisada en Telegram.

Reglas de cumplimiento y estrategia:
- Toda aprobación, rechazo e inicio de negociación requiere acción humana explícita.
- Estrategia por defecto en EE. UU.: Seller Financing con cuota inicial baja de
  $5,000 a $7,000 USD y bono adicional de 6 cuotas mensuales al finalizar el pago
  del crédito. Todos los borradores dirigidos al vendedor se redactan en inglés.
- Monitoreo de mensajes del vendedor vía `/msg <id> <texto>` (o `/vendedor`) y
  reporte de estado bajo demanda vía `/reporte` (o `/negociaciones`), además del
  reporte diario automático a las 11:00 AM.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.core.config import settings
from app.core.redis_client import redis_client
from app.db.models import Decision, Oferta, Propiedad
from app.db.offers_repository import RepositorioOfertas
from app.db.uow import unidad_de_trabajo
from app.models.offers import OfertaEstado
from app.orchestrator.nodes.telegram_alert import (
    TelegramNotifierAdapter,
    extraer_contacto_vendedor,
)
from app.services.negotiation_reporter import (
    construir_mensaje_ingles_vendedor,
    construir_respuesta_ingles_vendedor,
    consultar_negociaciones_y_generar_reporte,
    estrategia_seller_financing_resumen,
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
    """Interpreta instrucciones devolviendo `(precio_oferta, condiciones_extra, sin_condiciones)`.

    Si el usuario no escribe ninguna condición (o escribe `ninguna`/`tal cual`/`continuar`),
    mantiene `precio_publicado_usd` y aplica la estrategia Seller Financing por defecto
    ($5,000–$7,000 de cuota inicial + bono de 6 cuotas adicionales tras liquidar el crédito).
    """
    limpio = " ".join((texto or "").strip().split())
    dp_min = settings.NEGOTIATION_DOWN_PAYMENT_MIN_USD
    dp_max = settings.NEGOTIATION_DOWN_PAYMENT_MAX_USD
    bonus_m = settings.NEGOTIATION_BONUS_EXTRA_MONTHS
    estrategia_base = (
        f"Seller Financing (low down payment target ${dp_min:,.0f}-${dp_max:,.0f} USD, "
        f"bonus of {bonus_m} extra monthly payments after full loan payoff; English outreach)."
    )

    if limpio.lower() in _SIN_CONDICIONES_FRASES:
        return round(precio_publicado_usd, 2), estrategia_base, True

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
    sin_condiciones: bool = True,
) -> str:
    """Genera el borrador inicial en inglés (Seller Financing $5k–$7k + bono 6 cuotas)."""
    return construir_mensaje_ingles_vendedor(
        ubicacion=ubicacion,
        url_origen=url_origen,
        precio_publicado_usd=precio_publicado_usd,
        precio_oferta_usd=precio_oferta_usd,
        condiciones_extra=None if sin_condiciones else condiciones,
        contacto_vendedor=contacto_vendedor,
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
            f"Seller / listing agent ({prop.plataforma})"
        )

        repo_ofertas = RepositorioOfertas(unidad.sesion)
        oferta = Oferta(
            propiedad_id=prop.id,
            estado=OfertaEstado.PROPUESTA.value,
            importe_usd=precio_oferta,
            notas=(
                f"Strategy: Seller Financing (${settings.NEGOTIATION_DOWN_PAYMENT_MIN_USD:,.0f}-"
                f"${settings.NEGOTIATION_DOWN_PAYMENT_MAX_USD:,.0f} down + "
                f"{settings.NEGOTIATION_BONUS_EXTRA_MONTHS} extra monthly payments bonus) | "
                f"Contact: {contacto} | Terms: {condiciones}"
            ),
            actor_ultima_actualizacion=f"telegram:{chat_id}",
        )
        guardada = await repo_ofertas.crear(
            oferta,
            notas_evento=(
                "Negociación Seller Financing (EN) iniciada desde Telegram "
                f"({'tal cual' if sin_condiciones else 'con instrucciones'}): {condiciones}"
            ),
        )
        await unidad.eventos.registrar(
            "negociacion.iniciada_telegram",
            f"Negociación Seller Financing iniciada para {prop.ubicacion} por ${precio_oferta:,.0f}",
            job_id=prop.job_id,
            nivel="info",
            payload={
                "oferta_id": str(guardada.id),
                "propiedad_id": str(prop.id),
                "precio_publicado_usd": precio_pub,
                "precio_oferta_usd": precio_oferta,
                "sin_condiciones": sin_condiciones,
                "idioma_vendedor": settings.NEGOTIATION_LANGUAGE,
                "down_payment_min_usd": settings.NEGOTIATION_DOWN_PAYMENT_MIN_USD,
                "down_payment_max_usd": settings.NEGOTIATION_DOWN_PAYMENT_MAX_USD,
                "bonus_extra_months": settings.NEGOTIATION_BONUS_EXTRA_MONTHS,
                "contacto_vendedor": contacto,
                "url_origen": prop.url_origen,
                "condiciones": condiciones,
            },
        )
        await unidad.confirmar()
        return guardada, prop, condiciones, contacto, sin_condiciones


async def registrar_comunicacion_vendedor(
    oferta_ref: str,
    mensaje_vendedor: str,
    *,
    chat_id: str,
) -> tuple[Oferta, Propiedad, str, dict[str, object]] | None:
    """Registra una comunicación del vendedor, actualiza contraoferta si aplica y redacta respuesta EN."""
    ref_limpia = oferta_ref.strip().lower()
    if not ref_limpia or not mensaje_vendedor.strip():
        return None

    async with unidad_de_trabajo() as unidad:
        res = await unidad.sesion.execute(
            select(Oferta, Propiedad)
            .join(Propiedad, Propiedad.id == Oferta.propiedad_id)
            .order_by(Oferta.actualizada_en.desc())
            .limit(100)
        )
        encontrada: tuple[Oferta, Propiedad] | None = None
        for oferta, prop in res.all():
            oid = str(oferta.id).lower()
            if oid == ref_limpia or oid.startswith(ref_limpia):
                encontrada = (oferta, prop)
                break
        if encontrada is None:
            return None

        oferta, prop = encontrada
        borrador_en, analisis = construir_respuesta_ingles_vendedor(
            ubicacion=prop.ubicacion,
            mensaje_vendedor=mensaje_vendedor,
            precio_oferta_actual_usd=float(oferta.importe_usd),
        )
        repo_ofertas = RepositorioOfertas(unidad.sesion)
        precio_vendedor = analisis.get("precio_solicitado_usd")
        if (
            isinstance(precio_vendedor, int | float)
            and oferta.estado == OfertaEstado.PROPUESTA.value
        ):
            await repo_ofertas.cambiar_estado(
                oferta,
                OfertaEstado.CONTRAOFERTA,
                actor="vendedor",
                notas=f"mensaje_vendedor: {mensaje_vendedor[:500]}",
                importe_contraoferta_usd=float(precio_vendedor),
            )
        else:
            from app.db.models import OfertaEvento

            unidad.sesion.add(
                OfertaEvento(
                    oferta_id=oferta.id,
                    estado_anterior=oferta.estado,
                    estado_nuevo=oferta.estado,
                    actor="vendedor",
                    notas=f"mensaje_vendedor: {mensaje_vendedor[:500]}",
                )
            )
            await unidad.sesion.flush()

        await unidad.eventos.registrar(
            "negociacion.mensaje_vendedor",
            f"Comunicación del vendedor registrada en oferta {str(oferta.id)[:8]}",
            job_id=prop.job_id,
            nivel="info",
            payload={
                "oferta_id": str(oferta.id),
                "propiedad_id": str(prop.id),
                "chat_id": chat_id,
                "analisis": analisis,
            },
        )
        await unidad.confirmar()
        return oferta, prop, borrador_en, analisis


async def _manejar_comando_texto(
    chat_id: str,
    texto: str,
    *,
    bot: TelegramNotifierAdapter,
) -> dict[str, Any] | None:
    """Procesa comandos de monitoreo: `/reporte`, `/negociaciones`, `/msg` y `/vendedor`."""
    partes = texto.split(maxsplit=2)
    cmd = partes[0].lower().split("@", 1)[0]

    if cmd in ("/reporte", "/negociaciones"):
        reporte = await consultar_negociaciones_y_generar_reporte()
        await bot.send_message(chat_id, reporte)
        return {"status": "reporte_enviado"}

    if cmd in ("/msg", "/vendedor"):
        if len(partes) < 3:
            await bot.send_message(
                chat_id,
                "Uso: /msg <ID_oferta_8_chars> <mensaje del vendedor>\n"
                "Ejemplo: /msg a1b2c3d4 Seller asks $10,000 down and $54,000 price.",
            )
            return {"status": "msg_uso_invalido"}

        oferta_ref, msg_vendedor = partes[1], partes[2]
        reg = await registrar_comunicacion_vendedor(oferta_ref, msg_vendedor, chat_id=chat_id)
        if reg is None:
            await bot.send_message(
                chat_id,
                f"No encontré ninguna oferta activa con referencia '{oferta_ref}'. Usa /reporte para ver los IDs.",
            )
            return {"status": "oferta_no_encontrada"}

        oferta, prop, borrador_en, analisis = reg
        precio_det = analisis.get("precio_solicitado_usd")
        resumen_det = (
            f"Precio mencionado: ${float(precio_det):,.0f}"
            if isinstance(precio_det, int | float)
            else "Sin cambio explícito de precio"
        )
        await bot.send_message(
            chat_id,
            "📩 COMUNICACIÓN DEL VENDEDOR REGISTRADA\n"
            f"Oferta: {str(oferta.id)[:8]} ({oferta.estado.upper()})\n"
            f"Propiedad: {prop.ubicacion}\n"
            f"Análisis: {resumen_det} | Seller Financing mencionado: "
            f"{'Sí' if analisis.get('menciona_seller_financing') else 'No'}\n\n"
            f"Borrador de respuesta en inglés (Seller Financing $5k–$7k + bono 6 cuotas):\n\n"
            f"{borrador_en}",
        )
        return {"status": "comunicacion_vendedor_registrada", "oferta_id": str(oferta.id)}

    return None


async def procesar_interaccion_telegram(
    update: dict[str, Any],
    *,
    notifier: TelegramNotifierAdapter | None = None,
) -> dict[str, Any]:
    """Procesa botones inline (`callback_query`), comandos de monitoreo y respuestas de negociación."""
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
                f"Seller / listing agent ({prop.plataforma})"
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
                f"Publicación: {prop.url_origen}\n"
                f"Estrategia por defecto: {estrategia_seller_financing_resumen()}\n\n"
                "✍️ Escribe condiciones adicionales si deseas ajustar algo "
                "(ej: precio objetivo, días de cierre, reparaciones).\n"
                "Si no escribes ninguna condición, pulsa 'Iniciar tal cual' o escribe "
                "'continuar' para generar el primer mensaje en inglés con Seller Financing "
                "($5,000–$7,000 inicial + bono de 6 cuotas extra)."
            )
            botones_prompt = {
                "inline_keyboard": [
                    [
                        {
                            "text": f"▶️ Iniciar tal cual (${precio_pub:,.0f} · Seller Financing)",
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
            oferta, prop, condiciones, contacto, sin_cond = resultado
            borrador = construir_borrador_negociacion(
                ubicacion=prop.ubicacion,
                url_origen=prop.url_origen,
                precio_publicado_usd=float(prop.precio_usd),
                precio_oferta_usd=float(oferta.importe_usd),
                condiciones=condiciones,
                contacto_vendedor=contacto,
                sin_condiciones=sin_cond,
            )
            await bot.answer_callback_query(cb_id, "Negociación Seller Financing iniciada (EN)")
            await bot.send_message(
                chat_id,
                "🤝 NEGOCIACIÓN INICIADA (SELLER FINANCING · INGLÉS)\n"
                f"Oferta ID: {str(oferta.id)[:8]} ({oferta.id})\n"
                f"Propiedad: {prop.ubicacion}\n"
                f"Precio inicio: ${float(oferta.importe_usd):,.0f}\n"
                f"Estrategia: {estrategia_seller_financing_resumen()}\n"
                f"Contacto/Vendedor: {contacto}\n"
                f"Publicación: {prop.url_origen}\n\n"
                f"Borrador en inglés listo para enviar al vendedor:\n\n{borrador}\n\n"
                f"Cuando el vendedor responda, regístralo con:\n/msg {str(oferta.id)[:8]} <mensaje del vendedor>",
            )
            return {"status": "negociacion_iniciada", "oferta_id": str(oferta.id)}

        if accion == "neg_cancel":
            await _limpiar_pendiente(chat_id)
            await bot.answer_callback_query(cb_id, "Negociación cancelada")
            await bot.send_message(chat_id, "✖️ Inicio de negociación cancelado.")
            return {"status": "negociacion_cancelada"}

        await bot.answer_callback_query(cb_id, "Acción desconocida")
        return {"status": "callback_ignorado"}

    # 2. Mensajes de texto: comandos de monitoreo o instrucciones para negociación pendiente
    mensaje = update.get("message") or {}
    chat_id = str((mensaje.get("chat") or {}).get("id") or "")
    texto = str(mensaje.get("text") or "").strip()
    if not chat_id or not texto:
        return {"status": "sin_accion"}

    if texto.startswith("/"):
        res_cmd = await _manejar_comando_texto(chat_id, texto, bot=bot)
        return res_cmd if res_cmd is not None else {"status": "sin_accion"}

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
        sin_condiciones=sin_cond,
    )
    modo_txt = (
        "TAL CUAL · SELLER FINANCING" if sin_cond else "CON TUS INSTRUCCIONES · SELLER FINANCING"
    )
    await bot.send_message(
        chat_id,
        f"🤝 NEGOCIACIÓN INICIADA ({modo_txt})\n"
        f"Oferta ID: {str(oferta.id)[:8]} ({oferta.id})\n"
        f"Propiedad: {prop.ubicacion}\n"
        f"Precio propuesto: ${float(oferta.importe_usd):,.0f}\n"
        f"Estrategia: {estrategia_seller_financing_resumen()}\n"
        f"Contacto/Vendedor: {contacto}\n"
        f"Publicación: {prop.url_origen}\n\n"
        f"Borrador en inglés listo para enviar al vendedor:\n\n{borrador}\n\n"
        f"Cuando el vendedor responda, regístralo con:\n/msg {str(oferta.id)[:8]} <mensaje del vendedor>",
    )
    return {
        "status": "negociacion_iniciada",
        "oferta_id": str(oferta.id),
        "sin_condiciones": sin_cond,
    }
