# ADR-03: Persistencia principal en PostgreSQL 16 con SQLAlchemy 2 y Alembic

- **Estado:** Aprobado
- **Fecha:** 2026-11-20
- **Autores / Firmantes:** Dev B (Dominio e IA) · Dev A (Plataforma)
- **Tarea backlog:** `S0-03` (habilita `S0-02`, `S1-04`, `S2-02`, `S2-03`, `S4-01`)

## Contexto

El prototipo actual no persiste jobs, propiedades normalizadas, decisiones del grafo ni auditoría financiera. Almacenar el estado de negocio únicamente en Redis compromete la durabilidad, dificulta consultas relacionales, impide auditoría histórica de decisiones y complica el checkpointing estructurado de LangGraph.

## Decisión

Se separa explícitamente la capa de estado persistente de la capa de mensajería y caché:

1. **PostgreSQL 16** es la fuente de verdad para:
   - `jobs`: ciclo de vida, parámetros del barrido, `trace_id`, timestamps y estado terminal.
   - `propiedades`: entidades normalizadas (`DatosPropiedadNormalizados`), plataforma origen, clave de deduplicación y payload crudo.
   - `decisiones`: resultados por nodo del grafo (financiero, visión, auditoría legal, alerta), versión de reglas, entradas y justificación.
   - `eventos`: bitácora auditable de ejecuciones, alertas enviadas y operaciones `/buy` en simulación.
   - Checkpoints del grafo LangGraph (`S4-01`).
2. **ORM y migraciones:** SQLAlchemy 2 (modo asíncrono) + Alembic. Toda tabla o alteración entra mediante migración versionada verificada con `alembic upgrade head` desde cero.
3. **Rol de Redis:** exclusivamente cola de mensajes (Streams), caché efímera (geocodificación), semáforos de concurrencia, rate limiting de ventana deslizante y pub/sub de telemetría.

## Alternativa descartada

- **Guardar todo en Redis:** descartada por carecer de garantías relacionales, consultas analíticas ad hoc y trazabilidad auditable a largo plazo.

## Consecuencias

- **Positivas:** historial consultable, integridad transaccional por propiedad (unidad de trabajo en `S2-03`) y base sólida para auditoría.
- **Costes:** requiere gestionar migraciones Alembic y contenedor PostgreSQL en desarrollo y producción (`S0-02`, `S5-03`).
