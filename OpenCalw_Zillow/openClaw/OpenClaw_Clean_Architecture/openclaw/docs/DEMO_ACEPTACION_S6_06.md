# Demo de Aceptación contra los Ocho Criterios — OpenClaw (S6-06)

- **Fecha (UTC):** `2026-11-20T09:49:15.949360+00:00`
- **Script ejecutable:** `scripts/demo_aceptacion_s6_06.py`
- **Resultado global:** `8/8 PASS`

| Criterio | Estado | Evidencia |
|---|---|---|
| `C1_auth_y_validacion_catalogo` | PASS | `{"ok": true, "status_sin_clave": 401, "status_clave_invalida": 403, "status_ssrf_bloqueado": 422}` |
| `C2_ciclo_vida_job_y_estados` | PASS | `{"ok": true, "job_id": "b895447a-52f9-4dfd-9707-a55db34b35c4", "estado_final": "completed", "reapertura_bloqueada": true}` |
| `C3_extraccion_normalizacion_dedupe` | PASS | `{"ok": true, "precio_usd": 115000.0, "condado": "Cuyahoga", "duplicados_fusionados": 1}` |
| `C4_cola_redis_e_idempotencia_worker` | PASS | `{"ok": true, "primera_reclamacion": true, "duplicado_omitido": true, "mensajes_en_stream": 1}` |
| `C5_orquestacion_y_persistencia_decisiones` | PASS | `{"ok": true, "veredicto_con_alerta_legal": "revisar", "requiere_revision_humana": true, "capex_estimado": 10500.0}` |
| `C6_alerta_telegram_deduplicada_con_ficha` | PASS | `{"ok": true, "alertas_enviadas": 1, "ficha_url": "https://openclaw.local/propiedades/91b70784-d729-4ab5-94ad-fe7b2bc16144"}` |
| `C7_buy_disabled_y_sandbox_idempotente` | PASS | `{"ok": true, "http_buy_disabled": 503, "llamadas_uao_tras_doble_envio": 1, "eventos_auditados": 3}` |
| `C8_panel_y_operacion_rc` | PASS | `{"ok": true, "panel_status_code": 200}` |
