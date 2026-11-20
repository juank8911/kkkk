# SLOs y Umbrales Operativos — OpenClaw (Fase 100 · Dev 2)

## Objetivos de Nivel de Servicio (SLOs)

| Indicador (SLI) | Objetivo (SLO) | Ventana | Alerta asociada |
|---|---|---|---|
| Disponibilidad API (`/health` + `/readyz`) | ≥ 99,5 % | 30 días | `OpenClawApiUnavailable` |
| Latencia p95 de procesamiento por etapa | ≤ 15 s | 5 min | `OpenClawHighPipelineLatencyP95` |
| Tasa de decisiones `veredicto="error"` | ≤ 5 % | 10 min | `OpenClawDecisionErrorRateHigh` |
| Profundidad de DLQ (scraping y decisiones) | 0 mensajes sostenidos (> 2 min) | 2 min | `OpenClawDlqDepthNonZero` |
| Tasa de fallos por fuente externa | ≤ 0,2 fallos/s por plataforma | 5 min | `OpenClawSourceFailureSpike` |
| Cobertura de alertas deduplicadas | 100 % sin duplicados por oportunidad | Continuo | Auditoría `alerta_duplicada_omitida` |

## Artefactos de observabilidad
- Dashboard Grafana: `deploy/observability/grafana_dashboard_openclaw.json`
- Reglas Prometheus: `deploy/observability/prometheus_alerts_openclaw.yml`
- Evaluador determinista de umbrales para pruebas E2E: `app/services/slo_evaluator.py`
