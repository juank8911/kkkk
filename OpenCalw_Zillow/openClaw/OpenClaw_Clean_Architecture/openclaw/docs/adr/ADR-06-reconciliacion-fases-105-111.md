# ADR-06: Reconciliación de numeración Fases 105–109 y 110–111 / 116–117

- **Estado:** Propuesto / Aprobado por Dev 2 (pendiente firma Dev 1)
- **Fecha:** 20 de noviembre de 2026
- **Referencia:** `PLAN_IMPLEMENTACION_FASES_FALTANTES_2DEVS.md` (Sección 6: Fases 105–111)

## Contexto

Los documentos históricos del proyecto presentan dos vacíos de numeración:
1. **Fases 105–109:** no cuentan con especificación funcional consolidada en los documentos de actualización revisados (`0–94`, `95–104`, `110–117`).
2. **Fases 110–111 vs 116–117:** los documentos mezclan la numeración de gateway de ingesta, circuito WAF y feed WebSocket de telemetría.

## Decisión sobre Fases 105–109 (Descubrimiento formal — sin código productivo inventado)

| Fase | Estado documental | Propuesta formal | Riesgo legal / de negocio | Acción acordada |
|---|---|---|---|---|
| **105** | Sin especificación en docs | **Absorber** en `S2-09` / `S6-02` (hardening API y pruebas de seguridad) | Bajo | No crear módulo separado; trazado a `tests/test_security_suite.py` y `OWASP_API_TOP10_REVIEW.md`. |
| **106** | Sin especificación en docs | **Absorber** en Fase 100 (dashboards de negocio y SLOs operativos) | Bajo | Trazado a `deploy/observability/` y `docs/SLOS_Y_ALERTAS_F100.md`. |
| **107** | Sin especificación en docs | **Descartar / Congelar** hasta que exista caso de uso de negocio aprobado | Medio (posible duplicidad con Fase 96 macro) | Bloqueada: no se escribe código sin ficha G0 firmada. |
| **108** | Sin especificación en docs | **Absorber** en Fase 98/102 (plantillas documentales versionadas no vinculantes) | Medio (revisión legal por estado) | Cubierta por `app/services/loi_generator.py`. |
| **109** | Sin especificación en docs | **Absorber** en Fase 101/104 (revisión humana y trazabilidad de contraofertas) | Alto (Fair Housing / brokerage) | Cubierta por `app/services/negotiation_assistant.py`. |

## Decisión sobre Fases 110–111 y 116–117 (Núcleo común de resiliencia)

Se reconcilia la numeración en dos bloques sin duplicar funcionalidades ya cubiertas por Sprints 1–5:
- **Fases 110–111 (Resiliencia de fuentes y gateway):**
  - Gateway protegido por `X-OpenClaw-Tactical-Key` (`S2-01`), validación de payload (`S1-02`/`S1-07`) y catálogo cerrado anti-SSRF (`ADR-01`).
  - Salud y circuito por fuente (`app/services/source_health.py`) con cinco estados explícitos: `disponible`, `limitada`, `degradada`, `bloqueada` y `no_autorizada`, cooldown controlado y degradación a fuentes alternativas permitidas.
  - **Política estricta:** nunca se implementan técnicas de evasión de WAF, CAPTCHA o controles de acceso.
- **Fases 116–117 (Telemetría WebSocket autenticada + métricas de circuito):**
  - El feed WebSocket autenticado y métricas Prometheus se construyen sobre la arquitectura existente de Redis Pub/Sub y `PlatformSourceHealthManager`.

## Consecuencias

- Se elimina el riesgo de implementar funcionalidades inventadas para 105–109.
- Toda plataforma bloqueada entra en cooldown explícito sin reintentos agresivos y degrada hacia fuentes permitidas disponibles.
