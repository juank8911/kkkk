# Backlog Priorizado y Estimado — Fases 95–104 y Puerta Legal (S6-05)

- **Responsable:** Dev B (Dominio e IA)
- **Fecha:** 2026-11-20
- **Contexto:** Conforme a las secciones 1 y 10 de `PLAN_IMPLEMENTACION_2DEVS.md`, estos módulos quedan fuera del release candidate de las semanas 1–13 y se entregan priorizados, estimados y con sus dependencias técnicas y requisitos legales identificados.

## 1. Módulos de Fases 95–104

| Prioridad | Fase | Módulo | Tamaño | Est. (días-persona) | Dependencias técnicas | Requisito legal / de cumplimiento |
|---|---|---|---|---|---|---|
| **P1** | Fase 95 | Seguimiento post-oferta de propiedades (pipeline de estados de oferta, contraoferta, cierre) | M | 5 | PostgreSQL (`S2-02`), repositorio de propiedades (`S2-03`), panel de jobs (`S5-04`) | Retención y trazabilidad de registros comerciales; políticas de privacidad para datos de vendedores/agentes. |
| **P1** | Fase 96 | Estratega macro (ajuste periódico de tasas de vacancia, interés y cap rate objetivo desde fuentes macro) | M | 4 | Nodo financiero parametrizado (`S4-04`), scheduler/cron seguro | Licencia de uso comercial de fuentes de datos macroeconómicos (FRED, MLS, censos); auditoría de versiones de parámetros. |
| **P1** | Fase 97 | Auditoría legal ampliada con fuentes de registros públicos (condados / tax assessor) | L | 8 | Nodo de auditoría legal asistida (`S4-06`), caché Redis (`S3-06`) | Términos de uso de portales públicos del condado; aviso legal obligatorio indicando que el análisis es asistido y **no sustituye opinión jurídica ni title insurance**. |
| **P1** | Fase 100 | Matriz de despliegue productivo extendida (PostgreSQL + pgvector, Prometheus, Grafana, alertas) | M | 5 | `docker-compose.prod` (`S5-03`), métricas Prometheus (`S5-01`) | Cumplimiento de cifrado en tránsito/reposo y política de copias de seguridad para registros financieros y de auditoría. |
| **P2** | Fase 98 | Generador de cartas de intención (LOI) en PDF con plantillas versionadas | M | 5 | Seguimiento post-oferta (Fase 95), repositorio auditable | **Revisión legal obligatoria** por abogado inmobiliario por estado (EE. UU.): cláusula explícita de no vinculación (*non-binding LOI*), disclaimers y firma humana previa al envío. |
| **P2** | Fase 101 | Negociador asistido de contraofertas mediante NLP (borradores supervisados *human-in-the-loop*) | L | 8 | Generador LOI (Fase 98), seguimiento post-oferta (Fase 95) | Prohibición de envío autónomo vinculante sin aprobación humana; cumplimiento de normativa de corretaje inmobiliario (licencia de broker cuando aplique) y Fair Housing Act. |
| **P3** | Fase 99 | Puente de escrow bancario (instrucciones de depósito en garantía y conciliación) | XL | 12 | `/buy` sandbox auditado (`S2-07`), cliente UAO resiliente (`S2-08`), revisión OWASP (`S6-02`) | **Puerta legal bloqueante:** KYC/KYB, AML/BSA, acuerdos con entidad regulada de escrow/custodia, segregación de fondos y control dual (4-eyes) para transferencias. |

*(Nota: Telemetría WebSocket, seguridad por API key y métricas Prometheus —correspondientes a los bloques operativos de fases 95–104— ya fueron absorbidos en los Sprints 2 y 5 del plan actual).*

## 2. Módulos de Fases 112–115 sujetos a Puerta Legal Previa

| Prioridad | Fase | Módulo | Tamaño | Est. (días-persona) | Dependencias técnicas | Requisito legal / de cumplimiento (Puerta bloqueante) |
|---|---|---|---|---|---|---|
| **P3** | Fase 115 | Oráculo geoespacial y `Gentrification Score` (permisos comerciales y densidad de consumo) | L | 8 | Geocodificación con caché (`S3-06`), PostgreSQL (`S2-02`) | Revisión contra sesgos discriminatorios (*Fair Housing Act* / *ECOA*) y licencias de APIs geoespaciales comerciales. |
| **P4 (Bloqueado)** | Fase 113 | Adquisición outbound off-market (skip tracing + SMS vía Twilio) | XL | 12 | Seguimiento post-oferta (Fase 95), auditoría de consentimiento | **Puerta legal bloqueante:** TCPA (consentimiento previo expreso para SMS), DNC Registry (Do Not Call), leyes estatales de privacidad (CCPA/CPRA) y base de licitud para tratamiento de datos personales de propietarios. |
| **P4 (Bloqueado)** | Fase 112 | Tesorería DeFi (barrido de capital ocioso hacia USDC / Aave V3) | XL | 15 | Puente de escrow (Fase 99), auditoría de smart contracts | **Puerta legal bloqueante:** tratamiento contable, fiscal y regulatorio de activos digitales y yield en custodia, políticas de riesgo de contraparte y aprobación de compliance financiero. |
| **P4 (Bloqueado)** | Fase 114 | Endpoint AGNET RPC con verificación EIP-712 y liquidación USDC en Base Mainnet | XL | 14 | Tesorería DeFi (Fase 112), seguridad mTLS / HSM | **Puerta legal bloqueante:** normativa de transmisión de dinero (FinCEN/MSB), sanciones OFAC (screening de direcciones on-chain) y auditoría criptográfica externa. |

## Resumen de esfuerzo estimado futuro
- **Bloque P1–P2 (operable tras revisión legal estándar):** 35 días-persona.
- **Bloque P3–P4 (requiere dictamen legal/regulatorio previo antes de iniciar desarrollo):** 61 días-persona.
