# ADR-02: Patrón de cola en Redis Streams con Consumer Group, `XAUTOCLAIM` y DLQ

- **Estado:** Aprobado
- **Fecha:** 2026-11-20
- **Autores / Firmantes:** Dev B (Dominio e IA) · Dev A (Plataforma)
- **Tarea backlog:** `S0-03` (habilita `S1-03`, `S3-01`, `S3-02`, `S3-08`)

## Contexto

En los documentos del proyecto convivían dos estrategias incompatibles de consumo de Redis Streams: `XREADGROUP` + `XACK` en el prototipo y `XREAD` desde `0-0` + `XDEL` en el borrador de fases 110–117. Con `XREAD` + `XDEL` no existe reparto entre múltiples workers, un fallo del proceso a mitad de ejecución provoca pérdida o reprocesamiento descontrolado y no hay mecanismo de dead-letter para mensajes venenosos.

## Decisión

Se estandariza el procesamiento asíncrono sobre **Redis Streams con Consumer Groups**:

1. **Stream principal:** `openclaw:streams:barridos` (manteniendo compatibilidad controlada durante la transición de S1).
2. **Consumer Group:** `openclaw_scrapers`, con consumidores identificados de forma única (`worker-<host>-<pid>`).
3. **Confirmación (`XACK`):** solo se ejecuta después de persistir el resultado o transferir el mensaje a la DLQ.
4. **Recuperación de pendientes (`XAUTOCLAIM`):** el worker reclama periódicamente mensajes ociosos (`idle >= min_idle_ms`) pertenecientes a consumidores caídos.
5. **Reintentos y Dead-Letter Queue (DLQ):** tras superar `max_deliveries`, el mensaje se mueve al stream `openclaw:streams:barridos:dlq` junto con `job_id`, plataforma, conteo de intentos, último error y marca de tiempo, y se confirma con `XACK` en el stream principal.
6. **Idempotencia:** todo consumidor debe ser idempotente por `(job_id, plataforma)` dado que el modelo garantiza entrega *al menos una vez*.

## Alternativa descartada

- **`XREAD` desde `0-0` con `XDEL`:** descartada porque impide escalado horizontal de workers, pierde trazabilidad de fallos y no ofrece recuperación determinista tras caídas.

## Consecuencias

- **Positivas:** garantía *at-least-once*, recuperación automática ante caídas del worker, observabilidad de profundidad de DLQ (`S5-01`) y escalado horizontal seguro.
- **Costes:** obliga a diseñar la persistencia e ingesta con claves de idempotencia explícitas (`S3-02`).
