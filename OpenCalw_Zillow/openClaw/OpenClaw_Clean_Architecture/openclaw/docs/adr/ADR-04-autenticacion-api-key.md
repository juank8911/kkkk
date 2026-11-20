# ADR-04: Autenticación de ingesta mediante API key con hash en `X-OpenClaw-Tactical-Key`

- **Estado:** Aprobado
- **Fecha:** 2026-11-20
- **Autores / Firmantes:** Dev B (Dominio e IA) · Dev A (Plataforma)
- **Tarea backlog:** `S0-03` (habilita `S1-01`, `S2-01`, `S2-05`, `S2-09`)

## Contexto

El endpoint actual `POST /v1/openclaw/iniciar-barrido` no realiza autenticación ni autorización, y `app/core/config.py` incluye valores por defecto inseguros. Se necesita un mecanismo de protección inmediato, auditable y proporcionado para un servicio interno sin bloquear el calendario con infraestructura PKI/mTLS compleja.

## Decisión

1. **Cabecera estándar:** toda ruta protegida de la API exige la cabecera `X-OpenClaw-Tactical-Key`.
2. **Verificación por hash en tiempo constante:** el servidor almacena únicamente el hash criptográfico (SHA-256 / HMAC-SHA256) de las claves autorizadas, nunca secretos hardcodeados en código. La comparación se realiza con `hmac.compare_digest`.
3. **Códigos HTTP explícitos:**
   - Cabecera ausente → `401 Unauthorized`.
   - Clave inválida o revocada → `403 Forbidden`.
4. **Fallo rápido al arranque (`S1-01`):** si falta la configuración de clave en el entorno, la aplicación rehúsa iniciar con un error claro.
5. **Rate limit por clave:** las solicitudes autenticadas se limitan por identificador de clave para contener abuso o bucles de clientes.
6. **Webhook Telegram (`S2-05`):** se protege adicionalmente con `X-Telegram-Bot-Api-Secret-Token`, lista blanca de `chat_id` autorizados y deduplicación por `update_id`.

## Alternativa descartada

- **Zero-Trust completo con mTLS y emisión de certificados cliente:** pospuesto para una fase posterior porque exige PKI, rotación automatizada de certificados y terminación TLS avanzada fuera del alcance de estas 13 semanas.

## Consecuencias

- **Positivas:** implementación rápida, auditable, compatible con proxies reversos estándar y sin secretos en el repositorio.
- **Costes:** requiere rotación periódica operativa de API keys documentada en el runbook (`S6-03`).
