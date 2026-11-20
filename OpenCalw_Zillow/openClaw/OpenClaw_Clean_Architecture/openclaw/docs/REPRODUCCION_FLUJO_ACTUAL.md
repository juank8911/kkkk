# Reproducción del flujo actual — OpenClaw (S0-06)

- **Fecha de ejecución (UTC):** `2026-11-20T08:54:01.073286+00:00`
- **Script:** `scripts/reproducir_flujo_actual.py`
- **Modo:** `reproduccion_aislada_con_fakeredis`

## Cómo reproducir

```bash
python scripts/reproducir_flujo_actual.py
```

## Salida documentada

```json
{
  "timestamp_utc": "2026-11-20T08:54:01.073286+00:00",
  "modo": "reproduccion_aislada_con_fakeredis",
  "pasos": {
    "1_health": {
      "endpoint": "GET /health",
      "status_code": 200,
      "body": {
        "status": "ok",
        "service": "OpenClaw API Gateway",
        "version": "0.2.0",
        "env": "dev"
      }
    },
    "2_ingesta_valida": {
      "endpoint": "POST /v1/openclaw/barridos",
      "status_code": 202,
      "body": {
        "job_id": "fe092703-50e0-483b-a821-ed2441e45ebd",
        "status": "queued",
        "query": "Cleveland, OH",
        "plataformas": [
          "zillow",
          "realtor"
        ],
        "created_at": "2026-11-20T08:54:01.087295Z",
        "updated_at": null,
        "trace_id": null,
        "progress": {},
        "error": null
      }
    },
    "3_consulta_job": {
      "endpoint": "GET /v1/openclaw/barridos/fe092703-50e0-483b-a821-ed2441e45ebd",
      "status_code": 200,
      "body": {
        "job_id": "fe092703-50e0-483b-a821-ed2441e45ebd",
        "status": "queued",
        "query": "Cleveland, OH",
        "plataformas": [
          "zillow",
          "realtor"
        ],
        "created_at": "2026-11-20T08:54:01.087295Z",
        "updated_at": null,
        "trace_id": null,
        "progress": {},
        "error": null
      }
    },
    "4_cola_redis": {
      "stream": "openclaw_scraping_queue",
      "mensajes_encolados": 1,
      "ultimo_payload": {
        "job_id": "fe092703-50e0-483b-a821-ed2441e45ebd",
        "query": "Cleveland, OH",
        "plataformas": [
          "zillow",
          "realtor"
        ],
        "max_resultados": 20,
        "encolado_en": "2026-11-20T08:54:01.088028+00:00"
      }
    },
    "5_plataforma_invalida": {
      "endpoint": "POST /v1/openclaw/barridos",
      "status_code": 422,
      "body": {
        "code": "validation_error",
        "message": "El payload no cumple el contrato",
        "details": [
          {
            "field": "plataformas",
            "reason": "Value error, Plataforma(s) no soportada(s): 'plataforma_desconocida'. Plataformas válidas: [century21, coldwell, craigslist, loopnet, realtor, redfin, remax, trulia, zillow].",
            "valid_values": null
          }
        ],
        "trace_id": null,
        "timestamp": "2026-11-20T08:54:01.093587Z"
      }
    },
    "6_webhook_buy_deshabilitado": {
      "endpoint": "POST /v1/webhook/telegram-comando",
      "status_code": 403,
      "body": {
        "code": "forbidden",
        "message": "Compra deshabilitada: BUY_MODE no esta en modo sandbox",
        "details": null,
        "trace_id": null,
        "timestamp": "2026-11-20T08:54:01.101225Z"
      }
    },
    "7_orquestador_nodos_aislados": {
      "enrutador_due_diligence": "agente_financiero",
      "nodo_analisis_financiero": {
        "estado": "ANALISIS_FINANCIERO_OK",
        "cap_rate": 10.327272727272726,
        "noi": 11360.0
      },
      "nodo_vision_artificial": {
        "estado": "INSPECCION_VISUAL_OK",
        "capex": 15000.0
      },
      "nota": "Los nodos corren aislados sin grafo LangGraph compilado y visión usa CAPEX fijo de 15000 USD (se reemplaza en Sprint 4)."
    }
  }
}
```

## Estado verificado al cierre de Fase 1 (S0 + S1)

1. `GET /health` responde `200 OK`.
2. `POST /v1/openclaw/barridos` valida contra el catálogo de 9 plataformas, devuelve `202 Accepted` con `job_id` y encola en Redis Streams.
3. `GET /v1/openclaw/barridos/{job_id}` devuelve el estado `queued`.
4. Claves de plataforma fuera de catálogo son rechazadas con `422` listando las válidas.
5. `/buy` está bloqueado por defecto (`BUY_MODE=disabled`, `403 Forbidden`) conforme a ADR-05.
6. Pendiente para Sprints 2–4: autenticación por API key, PostgreSQL + Alembic, worker con `XAUTOCLAIM`/DLQ y grafo LangGraph compilado.
