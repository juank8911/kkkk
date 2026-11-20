# Architecture Decision Records (ADRs) — OpenClaw

Este directorio contiene las decisiones arquitectónicas fundacionales congeladas en el **Sprint 0 (tarea S0-03)** del plan de implementación (`PLAN_IMPLEMENTACION_2DEVS.md`).

| ID | Título | Estado | Fecha | Firmantes |
|---|---|---|---|---|
| [ADR-01](./ADR-01-catalogo-plataformas.md) | Contrato de plataformas con catálogo cerrado en servidor | Aprobado | 2026-11-20 | Dev B (autor), Dev A (revisor) |
| [ADR-02](./ADR-02-patron-cola-redis.md) | Patrón de cola en Redis Streams con Consumer Group, `XAUTOCLAIM` y DLQ | Aprobado | 2026-11-20 | Dev B (autor), Dev A (revisor) |
| [ADR-03](./ADR-03-persistencia-postgresql.md) | Persistencia principal en PostgreSQL 16 con SQLAlchemy 2 y Alembic | Aprobado | 2026-11-20 | Dev B (autor), Dev A (revisor) |
| [ADR-04](./ADR-04-autenticacion-api-key.md) | Autenticación de ingesta mediante API key con hash en `X-OpenClaw-Tactical-Key` | Aprobado | 2026-11-20 | Dev B (autor), Dev A (revisor) |
| [ADR-05](./ADR-05-alcance-buy-sandbox.md) | Alcance seguro del webhook `/buy` (`BUY_MODE=disabled` por defecto y modo `sandbox`) | Aprobado | 2026-11-20 | Dev B (autor), Dev A (revisor) |
| [ADR-06](./ADR-06-reconciliacion-fases-105-111.md) | Reconciliación de numeración Fases 105–109 y 110–111 / 116–117 | Propuesto / Dev 2 | 2026-11-20 | Dev B (autor) |

## Política de cambios

Estas cinco decisiones quedan congeladas para los Sprints 1 a 6. Cualquier modificación posterior requiere un nuevo ADR que sustituya formalmente al anterior y aprobación conjunta de Dev A y Dev B.
