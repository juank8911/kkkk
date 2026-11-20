# Reproducción del flujo actual — OpenClaw (S0-06)

- **Fecha de ejecución (UTC):** `2026-11-20T09:17:13.223316+00:00`
- **Script:** `scripts/reproducir_flujo_actual.py`
- **Modo:** `reproduccion_aislada_con_fakeredis`

## Cómo reproducir

```bash
python scripts/reproducir_flujo_actual.py
```

## Salida documentada

```json
{
  "timestamp_utc": "2026-11-20T09:17:13.223316+00:00",
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
    "2_ingesta_sin_api_key": {
      "endpoint": "POST /v1/openclaw/barridos",
      "status_code": 401,
      "body": {
        "code": "unauthorized",
        "message": "Falta la cabecera X-OpenClaw-Tactical-Key",
        "details": null,
        "trace_id": null,
        "timestamp": "2026-11-20T09:17:13.241206Z"
      }
    },
    "3_ingesta_valida": {
      "endpoint": "POST /v1/openclaw/barridos",
      "status_code": 202,
      "body": {
        "job_id": "069904f8-c8b0-446d-94a9-bb37e5cc120b",
        "status": "queued",
        "query": "Cleveland, OH",
        "plataformas": [
          "zillow",
          "realtor"
        ],
        "created_at": "2026-11-20T09:17:13.246484Z",
        "updated_at": null,
        "trace_id": null,
        "progress": {},
        "error": null
      }
    },
    "4_consulta_job": {
      "endpoint": "GET /v1/openclaw/barridos/069904f8-c8b0-446d-94a9-bb37e5cc120b",
      "status_code": 200,
      "body": {
        "job_id": "069904f8-c8b0-446d-94a9-bb37e5cc120b",
        "status": "queued",
        "query": "Cleveland, OH",
        "plataformas": [
          "zillow",
          "realtor"
        ],
        "created_at": "2026-11-20T09:17:13.246484Z",
        "updated_at": null,
        "trace_id": null,
        "progress": {},
        "error": null
      }
    },
    "5_cola_redis": {
      "stream": "openclaw_scraping_queue",
      "mensajes_encolados": 1,
      "ultimo_payload": {
        "job_id": "069904f8-c8b0-446d-94a9-bb37e5cc120b",
        "query": "Cleveland, OH",
        "plataformas": [
          "zillow",
          "realtor"
        ],
        "max_resultados": 20,
        "encolado_en": "2026-11-20T09:17:13.247747+00:00"
      }
    },
    "6_plataforma_invalida": {
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
        "timestamp": "2026-11-20T09:17:13.255460Z"
      }
    },
    "7_webhook_buy_deshabilitado": {
      "endpoint": "POST /v1/webhook/telegram-comando",
      "status_code": 403,
      "body": {
        "code": "forbidden",
        "message": "Compra deshabilitada: BUY_MODE no esta en modo sandbox",
        "details": null,
        "trace_id": null,
        "timestamp": "2026-11-20T09:17:13.263389Z"
      }
    },
    "8_orquestador_nodos_dominio": {
      "enrutador_due_diligence": "agente_financiero",
      "nodo_analisis_financiero": {
        "estado": "ANALISIS_FINANCIERO_OK",
        "cap_rate": 10.3273,
        "noi": 11360.0,
        "desglose_financiero": {
          "vacancia_usd": 1440.0,
          "mantenimiento_usd": 1800.0,
          "impuestos_usd": 2200.0,
          "seguro_usd": 1200.0,
          "hoa_anual_usd": 0.0,
          "gastos_totales_usd": 6640.0,
          "min_cap_rate_pct": 8.0,
          "rules_version": "fin-v1.0"
        },
        "reglas_financieras_version": "fin-v1.0"
      },
      "nodo_vision_artificial": {
        "estado": "VISION_NO_DISPONIBLE",
        "vision_disponible": false,
        "capex": null,
        "vision_confianza": 0.0,
        "vision_motivo": "No hay imágenes disponibles para inspección visual.",
        "vision_evaluacion": {
          "disponible": false,
          "estado_vision": "no_disponible",
          "capex_usd": null,
          "confianza": 0.0,
          "motivo": "No hay imágenes disponibles para inspección visual.",
          "severidad": "desconocida",
          "imagenes_evaluadas": 0
        }
      },
      "nodo_auditoria_legal": {
        "estado_legal": "AUDITORIA_LEGAL_OK",
        "riesgo_legal": "bajo",
        "requiere_revision_humana": false,
        "senales_legales": [],
        "auditoria_legal": {
          "estado": "AUDITORIA_LEGAL_OK",
          "riesgo": "bajo",
          "senales_detectadas": [],
          "requiere_revision_humana": false,
          "resumen": "Sin señales explícitas de liens, back taxes o conflictos legales en el texto."
        }
      }
    }
  }
}
```

## Estado verificado en rama `dev2`

1. `GET /health` responde `200 OK` sin autenticación.
2. `POST /v1/openclaw/barridos` sin `X-OpenClaw-Tactical-Key` devuelve `401 Unauthorized`.
3. Con API key válida y plataformas del catálogo cerrado devuelve `202 Accepted` con `job_id` y encola en Redis Streams.
4. `GET /v1/openclaw/barridos/{job_id}` devuelve el estado `queued`.
5. Claves fuera de catálogo son rechazadas con `422` listando las 9 válidas.
6. `/buy` está bloqueado por defecto (`BUY_MODE=disabled`, `403 Forbidden`) conforme a ADR-05.
7. Nodos financiero parametrizado, visión degradable a `no_disponible` sin imágenes y auditoría legal asistida ejecutan con salida estructurada.
