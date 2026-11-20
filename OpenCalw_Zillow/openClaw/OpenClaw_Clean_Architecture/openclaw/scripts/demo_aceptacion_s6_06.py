"""Demo ejecutable de aceptación contra los 8 criterios del Release Candidate (S6-06).

Demuestra en vivo y de forma reproducible (sin dependencias externas) los 8 criterios
de aceptación del plan `PLAN_IMPLEMENTACION_2DEVS.md` integrando los módulos de
Dev 1 (Plataforma) y Dev 2 (Dominio e IA).

Uso:
    python scripts/demo_aceptacion_s6_06.py
    python scripts/demo_aceptacion_s6_06.py --write-doc docs/DEMO_ACEPTACION_S6_06.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fakeredis.aioredis  # noqa: E402
import httpx  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from app.api.main import app as openclaw_app  # noqa: E402
from app.api.routes import ingestion as ingestion_route  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.core.security import CABECERA_API_KEY, hash_api_key  # noqa: E402
from app.db.models import Decision  # noqa: E402
from app.db.uow import UnidadDeTrabajo  # noqa: E402
from app.frontend import router as panel_router  # noqa: E402
from app.integrations.firecrawl_adapter import FirecrawlAdapter  # noqa: E402
from app.integrations.uao_client import UaoClient  # noqa: E402
from app.models.buy import BuyMode, BuyOrderRequest, BuyOrderStatus  # noqa: E402
from app.models.contracts import JobStatus  # noqa: E402
from app.orchestrator.config import AlertConfig  # noqa: E402
from app.orchestrator.nodes.telegram_alert import RedisAlertDedupeStore  # noqa: E402
from app.orchestrator.nodes.vision import VisionAssessment  # noqa: E402
from app.orchestrator.workflow import evaluar_y_persistir_propiedad_con_uow  # noqa: E402
from app.services import idempotency as idempotency_module  # noqa: E402
from app.services import job_store as job_store_module  # noqa: E402
from app.services.buy_service import BuySimulationService, InMemoryBuyAuditStore  # noqa: E402
from app.services.deduplicator import deduplicar_propiedades  # noqa: E402
from app.services.enrichment import enriquecer_propiedad_desde_markdown  # noqa: E402
from app.services.normalizer import normalizar_markdown_propiedades  # noqa: E402
from app.services.rate_limiter import SlidingWindowRateLimiter  # noqa: E402
from app.workers.state_machine import TransicionInvalida, aplicar_transicion  # noqa: E402

CASSETTE_PATH = PROJECT_ROOT / "tests" / "cassettes" / "scraper" / "zillow_success.json"


class _DemoCassetteClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    async def scrape_url(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._response


class _DemoValuator:
    async def evaluate_images(
        self,
        image_urls: list[str],
        *,
        context: dict[str, Any],
    ) -> VisionAssessment:
        return VisionAssessment(
            capex_usd=10_500.0,
            confianza=0.91,
            motivo="Techo y fachada sólidos; requiere pintura interior estándar.",
            severidad="baja",
            imagenes_evaluadas=len(image_urls),
        )


class _DemoNotifier:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    async def send_message(self, chat_id: str, text: str) -> tuple[bool, str | None]:
        self.messages.append((chat_id, text))
        return True, None


class _DemoDecisionesRepo:
    def __init__(self) -> None:
        self.rows: list[Decision] = []

    async def registrar(self, decision: Decision) -> Decision:
        self.rows.append(decision)
        return decision


class _DemoEventosRepo:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def registrar(self, tipo: str, mensaje: str, **kwargs: Any) -> dict[str, Any]:
        ev = {"tipo": tipo, "mensaje": mensaje, **kwargs}
        self.events.append(ev)
        return ev


class _DemoUoW(UnidadDeTrabajo):
    def __init__(self) -> None:
        self.decisiones = _DemoDecisionesRepo()  # type: ignore[assignment]
        self.eventos = _DemoEventosRepo()  # type: ignore[assignment]


async def ejecutar_demo_8_criterios() -> dict[str, Any]:
    """Ejecuta y valida los 8 criterios de aceptación en vivo."""
    fake_redis = fakeredis.aioredis.FakeRedis()
    orig_ing_redis = ingestion_route.redis_client
    orig_store_redis = job_store_module.redis_client
    orig_idem_redis = idempotency_module.redis_client
    orig_key = settings.OPENCLAW_API_KEY
    orig_hash = settings.OPENCLAW_API_KEY_SHA256
    orig_buy = settings.BUY_MODE

    demo_key = "demo-rc-tactical-key-2026"
    ingestion_route.redis_client = fake_redis
    job_store_module.redis_client = fake_redis
    idempotency_module.redis_client = fake_redis
    settings.OPENCLAW_API_KEY = ""
    settings.OPENCLAW_API_KEY_SHA256 = hash_api_key(demo_key)
    settings.BUY_MODE = "disabled"
    auth_headers = {CABECERA_API_KEY: demo_key}

    criterios: dict[str, dict[str, Any]] = {}

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=openclaw_app),
            base_url="http://testserver",
        ) as client:
            # Criterio 1: Autenticación SHA-256 y validación anti-SSRF del catálogo
            r_401 = await client.post(
                "/v1/openclaw/barridos",
                json={"query": "Cleveland, OH", "plataformas": ["zillow"]},
            )
            r_403 = await client.post(
                "/v1/openclaw/barridos",
                json={"query": "Cleveland, OH", "plataformas": ["zillow"]},
                headers={CABECERA_API_KEY: "clave-invalida"},
            )
            r_422 = await client.post(
                "/v1/openclaw/barridos",
                json={"query": "Cleveland, OH", "plataformas": ["http://169.254.169.254"]},
                headers=auth_headers,
            )
            c1_ok = (
                r_401.status_code == 401 and r_403.status_code == 403 and r_422.status_code == 422
            )
            criterios["C1_auth_y_validacion_catalogo"] = {
                "ok": c1_ok,
                "status_sin_clave": r_401.status_code,
                "status_clave_invalida": r_403.status_code,
                "status_ssrf_bloqueado": r_422.status_code,
            }

            # Criterio 2: Ciclo de vida del job y máquina de estados (queued -> running -> completed)
            r_create = await client.post(
                "/v1/openclaw/barridos",
                json={"query": "Cleveland, OH", "plataformas": ["zillow"]},
                headers=auth_headers,
            )
            job_id = UUID(r_create.json()["job_id"])
            r_queued = await client.get(f"/v1/openclaw/barridos/{job_id}", headers=auth_headers)
            await aplicar_transicion(job_id, JobStatus.RUNNING)
            await aplicar_transicion(job_id, JobStatus.COMPLETED, progress={"zillow": "exito"})
            reopen_blocked = False
            try:
                await aplicar_transicion(job_id, JobStatus.RUNNING)
            except TransicionInvalida:
                reopen_blocked = True
            r_done = await client.get(f"/v1/openclaw/barridos/{job_id}", headers=auth_headers)
            criterios["C2_ciclo_vida_job_y_estados"] = {
                "ok": (
                    r_create.status_code == 202
                    and r_queued.json()["status"] == "queued"
                    and r_done.json()["status"] == "completed"
                    and reopen_blocked
                ),
                "job_id": str(job_id),
                "estado_final": r_done.json()["status"],
                "reapertura_bloqueada": reopen_blocked,
            }

            # Criterio 3: ScraperPort + rate limiter fail-closed + normalizador + enriquecimiento + dedupe
            cassette = json.loads(CASSETTE_PATH.read_text(encoding="utf-8"))
            limiter = SlidingWindowRateLimiter(redis_client=fake_redis)
            adapter = FirecrawlAdapter(
                client=_DemoCassetteClient(cassette["response"]),
                rate_limiter=limiter,
            )
            scraped = await adapter.scrape("zillow", "Cleveland OH")
            batch = normalizar_markdown_propiedades(
                scraped.markdown,
                plataforma="zillow",
                url_origen=scraped.url,
            )
            prop = batch.valid[0]
            enrichment = enriquecer_propiedad_desde_markdown(scraped.markdown)
            deduped = deduplicar_propiedades([prop, prop])
            criterios["C3_extraccion_normalizacion_dedupe"] = {
                "ok": (
                    scraped.ok
                    and len(batch.valid) == 1
                    and enrichment.precio_usd == prop.precio_usd
                    and len(deduped) == 1
                    and deduped[0].duplicados_fusionados == 1
                ),
                "precio_usd": prop.precio_usd,
                "condado": prop.condado,
                "duplicados_fusionados": deduped[0].duplicados_fusionados,
            }

            # Criterio 4: Idempotencia por (job_id, plataforma) y cola Redis Streams
            first_claim = await idempotency_module.reclamar(job_id, "zillow")
            second_claim = await idempotency_module.reclamar(job_id, "zillow")
            stream_len = await fake_redis.xlen(settings.REDIS_STREAM_SCRAPING)
            criterios["C4_cola_redis_e_idempotencia_worker"] = {
                "ok": first_claim is True and second_claim is False and stream_len >= 1,
                "primera_reclamacion": first_claim,
                "duplicado_omitido": not second_claim,
                "mensajes_en_stream": stream_len,
            }

            # Criterio 5 y 6: Orquestador completo (Financiero -> Visión -> Legal -> Alerta dedup) + Decision UoW
            notifier = _DemoNotifier()
            dedupe_alerts = RedisAlertDedupeStore(fake_redis)
            uow = _DemoUoW()
            prop_uuid = uuid4()
            state_1, decision_1 = await evaluar_y_persistir_propiedad_con_uow(
                uow,
                propiedad_id=prop_uuid,
                job_id=job_id,
                initial_state={
                    "dedup_key": deduped[0].dedup_key,
                    **prop.model_dump(),
                    "descripcion_cruda": (
                        f"{prop.descripcion_cruda} Note: property has back taxes pending review."
                    ),
                },
                valuator=_DemoValuator(),
                notifier=notifier,
                dedupe_store=dedupe_alerts,
                alert_config=AlertConfig(ficha_base_url="https://openclaw.local/propiedades"),
                chat_id="chat-autorizado",
            )
            # Re-ejecutar sobre la misma oportunidad no duplica la alerta
            state_2, _ = await evaluar_y_persistir_propiedad_con_uow(
                uow,
                propiedad_id=prop_uuid,
                job_id=job_id,
                initial_state={
                    "dedup_key": deduped[0].dedup_key,
                    **prop.model_dump(),
                },
                valuator=_DemoValuator(),
                notifier=notifier,
                dedupe_store=dedupe_alerts,
                alert_config=AlertConfig(ficha_base_url="https://openclaw.local/propiedades"),
                chat_id="chat-autorizado",
            )
            criterios["C5_orquestacion_y_persistencia_decisiones"] = {
                "ok": (
                    decision_1.veredicto == "revisar"
                    and state_1["requiere_revision_humana"] is True
                    and len(uow.decisiones.rows) == 2
                ),
                "veredicto_con_alerta_legal": decision_1.veredicto,
                "requiere_revision_humana": state_1["requiere_revision_humana"],
                "capex_estimado": float(decision_1.capex_estimado or 0.0),
            }
            criterios["C6_alerta_telegram_deduplicada_con_ficha"] = {
                "ok": (
                    state_1["estado"] == "ALERTA_ENVIADA"
                    and state_2["estado"] == "ALERTA_DUPLICADA_OMITIDA"
                    and len(notifier.messages) == 1
                    and "https://openclaw.local/propiedades/" in notifier.messages[0][1]
                ),
                "alertas_enviadas": len(notifier.messages),
                "ficha_url": state_1["ficha_url"],
            }

            # Criterio 7: /buy desactivado por defecto y sandbox idempotente + auditado
            r_buy_disabled = await client.post(
                "/v1/webhook/telegram-comando",
                json={"update_id": 90001, "message": {"chat": {"id": 1}, "text": "/buy"}},
                headers=auth_headers,
            )
            uao_calls = 0

            def _uao_handler(req: httpx.Request) -> httpx.Response:
                nonlocal uao_calls
                uao_calls += 1
                return httpx.Response(200, json={"reference_id": "uao-demo-tx-1"})

            uao = UaoClient(
                base_url="http://uao-sandbox.local",
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(_uao_handler)),
            )
            audit = InMemoryBuyAuditStore()
            buy_svc = BuySimulationService(uao_client=uao, mode=BuyMode.SANDBOX, audit_store=audit)
            order = BuyOrderRequest(
                property_id=str(prop_uuid),
                asset="USDT",
                amount=95_000.0,
                destination="sandbox_escrow_vault",
                idempotency_key="demo-buy-idem-0001",
                requested_by="chat-autorizado",
            )
            ch = await buy_svc.request_confirmation(order)
            confirmed = order.model_copy(update={"confirmation_token": ch.confirmation_token})
            b1 = await buy_svc.execute_buy(confirmed)
            b2 = await buy_svc.execute_buy(confirmed)
            criterios["C7_buy_disabled_y_sandbox_idempotente"] = {
                "ok": (
                    r_buy_disabled.status_code in (403, 503)
                    and uao_calls == 1
                    and b1.status == BuyOrderStatus.SIMULATED_EXECUTED
                    and b2.status == BuyOrderStatus.DUPLICATE_REPLAYED
                ),
                "http_buy_disabled": r_buy_disabled.status_code,
                "llamadas_uao_tras_doble_envio": uao_calls,
                "eventos_auditados": len(await audit.list_events()),
            }

        # Criterio 8: Panel web mínimo y health operativo
        panel_app = FastAPI()
        panel_app.include_router(panel_router)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=panel_app),
            base_url="http://testserver",
        ) as pclient:
            r_panel = await pclient.get("/panel")
        criterios["C8_panel_y_operacion_rc"] = {
            "ok": r_panel.status_code == 200
            and "OpenClaw — Panel de Jobs y Telemetría" in r_panel.text,
            "panel_status_code": r_panel.status_code,
        }
    finally:
        ingestion_route.redis_client = orig_ing_redis
        job_store_module.redis_client = orig_store_redis
        idempotency_module.redis_client = orig_idem_redis
        settings.OPENCLAW_API_KEY = orig_key
        settings.OPENCLAW_API_KEY_SHA256 = orig_hash
        settings.BUY_MODE = orig_buy
        await fake_redis.aclose()

    todos_ok = all(item["ok"] for item in criterios.values())
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "total_criterios": len(criterios),
        "criterios_aprobados": sum(1 for item in criterios.values() if item["ok"]),
        "aceptacion_global": todos_ok,
        "criterios": criterios,
    }


def formatear_markdown(resultado: dict[str, Any]) -> str:
    filas = [
        f"| `{k}` | {'PASS' if v['ok'] else 'FAIL'} | `{json.dumps(v, ensure_ascii=False)}` |"
        for k, v in resultado["criterios"].items()
    ]
    tabla = "\n".join(filas)
    return (
        "# Demo de Aceptación contra los Ocho Criterios — OpenClaw (S6-06)\n\n"
        f"- **Fecha (UTC):** `{resultado['timestamp_utc']}`\n"
        f"- **Script ejecutable:** `scripts/demo_aceptacion_s6_06.py`\n"
        f"- **Resultado global:** `{resultado['criterios_aprobados']}/{resultado['total_criterios']} PASS`\n\n"
        "| Criterio | Estado | Evidencia |\n"
        "|---|---|---|\n"
        f"{tabla}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo de aceptación de 8 criterios (S6-06).")
    parser.add_argument("--write-doc", type=Path, default=None)
    args = parser.parse_args()

    res = asyncio.run(ejecutar_demo_8_criterios())
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.write_doc is not None:
        args.write_doc.parent.mkdir(parents=True, exist_ok=True)
        args.write_doc.write_text(formatear_markdown(res), encoding="utf-8")
    return 0 if res["aceptacion_global"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
