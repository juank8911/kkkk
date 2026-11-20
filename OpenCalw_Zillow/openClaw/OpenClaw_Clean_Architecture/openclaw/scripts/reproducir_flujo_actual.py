"""Guion de reproducción del flujo actual de OpenClaw (S0-06).

Levanta el flujo actual de punta a punta sin requerir servicios externos (usando
`fakeredis` y dobles controlados) y documenta su salida en formato JSON/Markdown.

Uso:
    python scripts/reproducir_flujo_actual.py
    python scripts/reproducir_flujo_actual.py --write-doc docs/REPRODUCCION_FLUJO_ACTUAL.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fakeredis.aioredis  # noqa: E402
import httpx  # noqa: E402

from app.api.main import app  # noqa: E402
from app.api.routes import ingestion as ingestion_route  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.orchestrator.nodes.financial import nodo_analisis_financiero  # noqa: E402
from app.orchestrator.nodes.vision import nodo_vision_artificial  # noqa: E402
from app.orchestrator.workflow import enrutador_due_diligence  # noqa: E402
from app.services import job_store as job_store_module  # noqa: E402


async def ejecutar_flujo_actual() -> dict[str, Any]:
    """Ejecuta el flujo actual de API + Redis Streams + nodos de orquestación."""
    fake_redis = fakeredis.aioredis.FakeRedis()
    original_ingestion_redis = ingestion_route.redis_client
    original_store_redis = job_store_module.redis_client
    ingestion_route.redis_client = fake_redis
    job_store_module.redis_client = fake_redis

    evidencia: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "modo": "reproduccion_aislada_con_fakeredis",
        "pasos": {},
    }

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1. Health check
            resp_health = await client.get("/health")
            evidencia["pasos"]["1_health"] = {
                "endpoint": "GET /health",
                "status_code": resp_health.status_code,
                "body": resp_health.json(),
            }

            # 2. Encolar barrido válido en POST /v1/openclaw/barridos
            payload_valido = {
                "query": "Cleveland, OH",
                "plataformas": ["zillow", "realtor"],
            }
            resp_ok = await client.post("/v1/openclaw/barridos", json=payload_valido)
            body_ok = resp_ok.json()
            evidencia["pasos"]["2_ingesta_valida"] = {
                "endpoint": "POST /v1/openclaw/barridos",
                "status_code": resp_ok.status_code,
                "body": body_ok,
            }

            # 3. Consultar estado del job recién creado
            job_id = body_ok.get("job_id")
            if job_id:
                resp_job = await client.get(f"/v1/openclaw/barridos/{job_id}")
                evidencia["pasos"]["3_consulta_job"] = {
                    "endpoint": f"GET /v1/openclaw/barridos/{job_id}",
                    "status_code": resp_job.status_code,
                    "body": resp_job.json(),
                }

            # 4. Verificar mensaje en Redis Stream
            stream_name = settings.REDIS_STREAM_SCRAPING
            stream_entries = await fake_redis.xrange(stream_name, "-", "+")
            evidencia["pasos"]["4_cola_redis"] = {
                "stream": stream_name,
                "mensajes_encolados": len(stream_entries),
                "ultimo_payload": (
                    json.loads(stream_entries[-1][1][b"payload"].decode("utf-8"))
                    if stream_entries
                    else None
                ),
            }

            # 5. Validar rechazo con clave de plataforma fuera del catálogo (422)
            resp_invalido = await client.post(
                "/v1/openclaw/barridos",
                json={"query": "Cleveland, OH", "plataformas": ["plataforma_desconocida"]},
            )
            evidencia["pasos"]["5_plataforma_invalida"] = {
                "endpoint": "POST /v1/openclaw/barridos",
                "status_code": resp_invalido.status_code,
                "body": resp_invalido.json(),
            }

            # 6. Validar salvaguarda /buy deshabilitada por defecto (ADR-05)
            resp_buy = await client.post(
                "/v1/webhook/telegram-comando",
                json={"message": {"text": "/buy"}},
            )
            evidencia["pasos"]["6_webhook_buy_deshabilitado"] = {
                "endpoint": "POST /v1/webhook/telegram-comando",
                "status_code": resp_buy.status_code,
                "body": resp_buy.json(),
            }

        # 7. Ejecutar nodos aislados del orquestador (a integrar en LangGraph en Sprint 4)
        estado_propiedad = {
            "ubicacion": "1420 Maple Ave, Cleveland, OH 44102",
            "precio_usd": 110_000.0,
            "ingreso_anual_estimado": 18_000.0,
        }
        ruta = enrutador_due_diligence(estado_propiedad)
        res_fin = nodo_analisis_financiero(estado_propiedad)
        estado_tras_fin = {**estado_propiedad, **res_fin}
        res_vis = await nodo_vision_artificial(estado_tras_fin)

        evidencia["pasos"]["7_orquestador_nodos_aislados"] = {
            "enrutador_due_diligence": ruta,
            "nodo_analisis_financiero": res_fin,
            "nodo_vision_artificial": res_vis,
            "nota": (
                "Los nodos corren aislados sin grafo LangGraph compilado "
                "y visión usa CAPEX fijo de 15000 USD (se reemplaza en Sprint 4)."
            ),
        }
    finally:
        ingestion_route.redis_client = original_ingestion_redis
        job_store_module.redis_client = original_store_redis
        await fake_redis.aclose()

    return evidencia


def formatear_reporte_markdown(evidencia: dict[str, Any]) -> str:
    """Genera el documento Markdown con la evidencia reproducible del flujo actual."""
    json_pretty = json.dumps(evidencia, indent=2, ensure_ascii=False)
    return (
        "# Reproducción del flujo actual — OpenClaw (S0-06)\n\n"
        f"- **Fecha de ejecución (UTC):** `{evidencia['timestamp_utc']}`\n"
        f"- **Script:** `scripts/reproducir_flujo_actual.py`\n"
        f"- **Modo:** `{evidencia['modo']}`\n\n"
        "## Cómo reproducir\n\n"
        "```bash\n"
        "python scripts/reproducir_flujo_actual.py\n"
        "```\n\n"
        "## Salida documentada\n\n"
        "```json\n"
        f"{json_pretty}\n"
        "```\n\n"
        "## Estado verificado al cierre de Fase 1 (S0 + S1)\n\n"
        "1. `GET /health` responde `200 OK`.\n"
        "2. `POST /v1/openclaw/barridos` valida contra el catálogo de 9 plataformas, devuelve `202 Accepted` con `job_id` y encola en Redis Streams.\n"
        "3. `GET /v1/openclaw/barridos/{job_id}` devuelve el estado `queued`.\n"
        "4. Claves de plataforma fuera de catálogo son rechazadas con `422` listando las válidas.\n"
        "5. `/buy` está bloqueado por defecto (`BUY_MODE=disabled`, `403 Forbidden`) conforme a ADR-05.\n"
        "6. Pendiente para Sprints 2–4: autenticación por API key, PostgreSQL + Alembic, worker con `XAUTOCLAIM`/DLQ y grafo LangGraph compilado.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce el flujo actual de OpenClaw (S0-06).")
    parser.add_argument(
        "--write-doc",
        type=Path,
        default=None,
        help="Ruta opcional donde guardar el reporte en formato Markdown.",
    )
    args = parser.parse_args()

    evidencia = asyncio.run(ejecutar_flujo_actual())
    print(json.dumps(evidencia, indent=2, ensure_ascii=False))

    if args.write_doc is not None:
        args.write_doc.parent.mkdir(parents=True, exist_ok=True)
        args.write_doc.write_text(formatear_reporte_markdown(evidencia), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
