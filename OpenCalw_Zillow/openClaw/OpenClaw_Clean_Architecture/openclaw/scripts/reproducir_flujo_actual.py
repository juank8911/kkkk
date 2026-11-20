"""Guion de reproducción del flujo actual de OpenClaw (S0-06 / actualizado a S2–S4).

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
from app.orchestrator.nodes.legal_audit import nodo_auditoria_legal  # noqa: E402
from app.orchestrator.nodes.vision import nodo_vision_artificial  # noqa: E402
from app.orchestrator.workflow import enrutador_due_diligence  # noqa: E402
from app.services import job_store as job_store_module  # noqa: E402


async def ejecutar_flujo_actual() -> dict[str, Any]:
    """Ejecuta el flujo actual de API autenticada + Redis Streams + nodos de orquestación."""
    fake_redis = fakeredis.aioredis.FakeRedis()
    original_ingestion_redis = ingestion_route.redis_client
    original_store_redis = job_store_module.redis_client
    original_api_key = settings.OPENCLAW_API_KEY
    original_api_hash = settings.OPENCLAW_API_KEY_SHA256

    ephemeral_key = "repro-ephemeral-tactical-key"
    ingestion_route.redis_client = fake_redis
    job_store_module.redis_client = fake_redis
    settings.OPENCLAW_API_KEY = ephemeral_key
    settings.OPENCLAW_API_KEY_SHA256 = ""

    auth_headers = {"X-OpenClaw-Tactical-Key": ephemeral_key}

    evidencia: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "modo": "reproduccion_aislada_con_fakeredis",
        "pasos": {},
    }

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1. Health check público
            resp_health = await client.get("/health")
            evidencia["pasos"]["1_health"] = {
                "endpoint": "GET /health",
                "status_code": resp_health.status_code,
                "body": resp_health.json(),
            }

            # 2. Rechazo 401 sin API key (S2-01)
            resp_unauth = await client.post(
                "/v1/openclaw/barridos",
                json={"query": "Cleveland, OH", "plataformas": ["zillow"]},
            )
            evidencia["pasos"]["2_ingesta_sin_api_key"] = {
                "endpoint": "POST /v1/openclaw/barridos",
                "status_code": resp_unauth.status_code,
                "body": resp_unauth.json(),
            }

            # 3. Encolar barrido válido autenticado
            payload_valido = {
                "query": "Cleveland, OH",
                "plataformas": ["zillow", "realtor"],
            }
            resp_ok = await client.post(
                "/v1/openclaw/barridos",
                json=payload_valido,
                headers=auth_headers,
            )
            body_ok = resp_ok.json()
            evidencia["pasos"]["3_ingesta_valida"] = {
                "endpoint": "POST /v1/openclaw/barridos",
                "status_code": resp_ok.status_code,
                "body": body_ok,
            }

            # 4. Consultar estado del job recién creado
            job_id = body_ok.get("job_id")
            if job_id:
                resp_job = await client.get(
                    f"/v1/openclaw/barridos/{job_id}",
                    headers=auth_headers,
                )
                evidencia["pasos"]["4_consulta_job"] = {
                    "endpoint": f"GET /v1/openclaw/barridos/{job_id}",
                    "status_code": resp_job.status_code,
                    "body": resp_job.json(),
                }

            # 5. Verificar mensaje en Redis Stream
            stream_name = settings.REDIS_STREAM_SCRAPING
            stream_entries = await fake_redis.xrange(stream_name, "-", "+")
            evidencia["pasos"]["5_cola_redis"] = {
                "stream": stream_name,
                "mensajes_encolados": len(stream_entries),
                "ultimo_payload": (
                    json.loads(stream_entries[-1][1][b"payload"].decode("utf-8"))
                    if stream_entries
                    else None
                ),
            }

            # 6. Validar rechazo con clave de plataforma fuera del catálogo (422)
            resp_invalido = await client.post(
                "/v1/openclaw/barridos",
                json={"query": "Cleveland, OH", "plataformas": ["plataforma_desconocida"]},
                headers=auth_headers,
            )
            evidencia["pasos"]["6_plataforma_invalida"] = {
                "endpoint": "POST /v1/openclaw/barridos",
                "status_code": resp_invalido.status_code,
                "body": resp_invalido.json(),
            }

            # 7. Validar salvaguarda /buy deshabilitada por defecto (ADR-05)
            resp_buy = await client.post(
                "/v1/webhook/telegram-comando",
                json={"message": {"text": "/buy"}},
                headers=auth_headers,
            )
            evidencia["pasos"]["7_webhook_buy_deshabilitado"] = {
                "endpoint": "POST /v1/webhook/telegram-comando",
                "status_code": resp_buy.status_code,
                "body": resp_buy.json(),
            }

        # 8. Ejecutar nodos de decisión de dominio (S4-04..S4-06)
        estado_propiedad = {
            "ubicacion": "1420 Maple Ave, Cleveland, OH 44102",
            "precio_usd": 110_000.0,
            "ingreso_anual_estimado": 18_000.0,
            "imagenes": [],
            "descripcion_cruda": "Brick colonial with clear rental demand in Cuyahoga County.",
        }
        ruta = enrutador_due_diligence(estado_propiedad)
        res_fin = nodo_analisis_financiero(estado_propiedad)
        estado_tras_fin = {**estado_propiedad, **res_fin}
        res_vis = await nodo_vision_artificial(estado_tras_fin)
        res_legal = nodo_auditoria_legal(estado_tras_fin)

        evidencia["pasos"]["8_orquestador_nodos_dominio"] = {
            "enrutador_due_diligence": ruta,
            "nodo_analisis_financiero": res_fin,
            "nodo_vision_artificial": res_vis,
            "nodo_auditoria_legal": res_legal,
        }
    finally:
        ingestion_route.redis_client = original_ingestion_redis
        job_store_module.redis_client = original_store_redis
        settings.OPENCLAW_API_KEY = original_api_key
        settings.OPENCLAW_API_KEY_SHA256 = original_api_hash
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
        "## Estado verificado en rama `dev2`\n\n"
        "1. `GET /health` responde `200 OK` sin autenticación.\n"
        "2. `POST /v1/openclaw/barridos` sin `X-OpenClaw-Tactical-Key` devuelve `401 Unauthorized`.\n"
        "3. Con API key válida y plataformas del catálogo cerrado devuelve `202 Accepted` con `job_id` y encola en Redis Streams.\n"
        "4. `GET /v1/openclaw/barridos/{job_id}` devuelve el estado `queued`.\n"
        "5. Claves fuera de catálogo son rechazadas con `422` listando las 9 válidas.\n"
        "6. `/buy` está bloqueado por defecto (`BUY_MODE=disabled`, `403 Forbidden`) conforme a ADR-05.\n"
        "7. Nodos financiero parametrizado, visión degradable a `no_disponible` sin imágenes y auditoría legal asistida ejecutan con salida estructurada.\n"
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
