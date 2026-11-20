# ADR-05: Alcance seguro del webhook `/buy` (`BUY_MODE=disabled` por defecto y modo `sandbox`)

- **Estado:** Aprobado
- **Fecha:** 2026-11-20
- **Autores / Firmantes:** Dev B (Dominio e IA) · Dev A (Plataforma)
- **Tarea backlog:** `S0-03` (habilita `S2-07`, `S2-08`, `S2-09`)

## Contexto

El prototipo en `app/api/routes/execution.py` dispara una transferencia fija de `125000.0 USDT` hacia `fiat_bridge_bank` ante cualquier mensaje de Telegram que comience por `/buy`, sin verificar origen, autorización, saldo, límites ni idempotencia. Mantener ese comportamiento supone un riesgo financiero crítico.

## Decisión

Se rediseña el alcance de `/buy` bajo principio de denegación por defecto:

1. **Interruptor de modo (`BUY_MODE`):**
   - Valor por defecto obligatorio: `BUY_MODE=disabled`. En este estado, cualquier invocación a `/buy` devuelve error controlado (`403` / `503`) y registra el intento en auditoría sin llamar a proveedores externos.
   - Único modo habilitable durante el alcance de este plan: `BUY_MODE=sandbox` (simulación). La ejecución con fondos reales queda fuera de alcance hasta revisión legal y de seguridad completa.
2. **Requisitos obligatorios en modo `sandbox` (`S2-07`):**
   - Orden parametrizada validada: `property_id`, `asset`, `amount`, `destination` (contra lista permitida) e `idempotency_key`. Se eliminan por completo los montos y destinos fijos en código.
   - Límites máximos configurables por operación y diarios.
   - Confirmación explícita en dos pasos (token de confirmación con expiración corta).
   - Idempotencia estricta: reenvíos con la misma `idempotency_key` devuelven el resultado original sin re-ejecutar.
   - Registro inmutable en tabla de auditoría (`eventos`).
3. **Resiliencia del cliente UAO (`S2-08`):** llamadas a UAO exclusivamente en `sandbox` con timeout estricto, reintentos acotados y circuit breaker para no bloquear el worker ni perder trazabilidad.

## Alternativa descartada

- **Webhook activo con monto y destino fijos en código:** descartada por representar una vulnerabilidad crítica de ejecución financiera no autorizada y duplicable.

## Consecuencias

- **Positivas:** elimina completamente el riesgo de retiros accidentales o no autorizados; permite probar el flujo completo de punta a punta de forma segura y auditable.
- **Costes:** exige definir flujos de confirmación e idempotencia antes de habilitar el entorno `sandbox`.
