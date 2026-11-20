# Plan de implementación de fases faltantes de OpenClaw

**Versión:** 1.0  
**Fecha:** 20 de noviembre de 2026  
**Equipo:** Dev 1 — Plataforma, seguridad y datos; Dev 2 — Dominio, IA e integraciones  
**Repositorio:** `F:\zillow_actualizado\OpenCalw_Zillow\openClaw`  
**Ramas:** `dev1`, `dev2`, `main`  
**Base actual:** `main` en `29ae0e1`

---

## 1. Objetivo

Completar las capacidades que todavía faltan después de los Sprints 0–6 y de la implementación inicial de la Fase 95, sin romper la arquitectura limpia actual, la trazabilidad ni los controles de seguridad.

El resultado esperado es un sistema capaz de:

1. Mantener el seguimiento auditable de ofertas y contraofertas.
2. Ajustar parámetros macroeconómicos con versiones, fuentes y aprobación humana.
3. Ampliar la auditoría legal con fuentes públicas permitidas.
4. Generar borradores de LOI no vinculantes en PDF.
5. Proponer contraofertas mediante NLP supervisado, sin envío autónomo.
6. Disponer de un despliegue productivo observable y recuperable.
7. Exponer telemetría autenticada y un circuito de resiliencia WAF seguro.
8. Dejar diseñadas, pero bloqueadas, las capacidades de escrow, DeFi, SMS outbound, AGNET RPC y `Gentrification Score` hasta superar sus puertas legales.

Este plan **no autoriza transferencias de dinero, envío de SMS, liquidación on-chain ni decisiones inmobiliarias autónomas**.

---

## 2. Estado de partida

| Área | Estado actual |
|---|---|
| Fases 0–94 | Implementadas mediante Sprints 0–6 y fusionadas en `main` |
| Fase 95 | Implementada como modelo y API auditable de ofertas; falta el tracker automático descrito en el documento original |
| Fases 96–104 | Pendientes salvo capacidades absorbidas por Sprints 2 y 5 |
| Fases 105–109 | No tienen una especificación funcional consolidada en los documentos revisados; requieren definición antes de programar |
| Fases 110–111 | Requieren reconciliar la numeración entre los documentos de gateway, WAF y telemetría |
| Fases 112–114 | Bloqueadas por revisión legal/regulatoria y controles financieros |
| Fase 115 | Pendiente y sujeta a revisión de sesgos, Fair Housing/ECOA y licencias de datos |
| Fases 116–117 | Telemetría WebSocket y circuito WAF descritos en el documento 110–117; deben implementarse como hardening después de cerrar contratos |

### 2.1 Capacidades ya absorbidas

No se deben volver a desarrollar como duplicados:

- API key con hash y dependencia FastAPI.
- Redis Streams, consumer group, `XACK`, `XAUTOCLAIM` y DLQ.
- Worker con reintentos, idempotencia y semáforo por dominio.
- PostgreSQL, Alembic, repositorios y unidad de trabajo.
- Normalización, enriquecimiento y deduplicación.
- Grafo LangGraph, decisiones y auditoría.
- Panel de jobs, E2E, carga controlada y revisión OWASP inicial.
- Geocodificación con caché Redis.

Toda fase nueva debe extender estos componentes mediante puertos y servicios, no crear segundos modelos paralelos.

---

## 3. Reglas de trabajo para los dos desarrolladores

### Dev 1 — Plataforma, seguridad y operación

Responsable de:

- `app/api`, `app/core`, `app/workers`.
- `app/db`, migraciones Alembic y repositorios.
- Docker, despliegue productivo, observabilidad y backups.
- Autenticación, autorización, rate limiting, auditoría de seguridad.
- Integraciones externas desde el punto de vista de timeouts, secretos, circuit breakers y contratos.

### Dev 2 — Dominio, IA y experiencia inmobiliaria

Responsable de:

- `app/services`, `app/orchestrator` e integraciones de dominio.
- Reglas de ofertas, macroeconomía, auditoría legal, LOI y NLP.
- Evaluación de calidad de datos, sesgos y revisión humana.
- Fixtures de propiedades, contratos de dominio y pruebas de decisión.

### Trabajo conjunto

- Contratos Pydantic y ADR antes de implementar.
- Revisión cruzada obligatoria de cada PR.
- Pruebas E2E y demo de aceptación.
- Actualización de `ESTADO_IMPLEMENTACION.md` al cerrar cada fase.
- Un commit por fase y push inmediato de la rama terminada.
- Integración a `main` únicamente mediante merge `--no-ff`.

---

## 4. Puertas obligatorias antes de programar

| Puerta | Responsable | Criterio de salida |
|---|---|---|
| G0 — Requisitos | Ambos | Fase, contrato, estados y datos de entrada documentados |
| G1 — Legal estándar | Dev 2 + responsable legal | Avisos, retención, fuentes y revisión humana documentados |
| G2 — Legal bloqueante | Usuario/responsable legal | Dictamen escrito para dinero, SMS, cripto o scoring sensible |
| G3 — Seguridad | Dev 1 | Threat model, secretos, permisos, logs y rate limits aprobados |
| G4 — Datos | Dev 1 | Migración reversible, índices, retención y backup definidos |
| G5 — Aceptación | Ambos | Pruebas automatizadas y demo reproducible |

Las fases 99/103, 112, 113 y 114 no pueden pasar de diseño a producción sin G2. La Fase 115 necesita G2 específica de sesgo y vivienda justa.

---

## 5. Orden recomendado por oleadas

### Oleada A — Fundaciones y cierre de Fase 95

1. Cerrar el tracker post-oferta de la Fase 95.
2. Definir contratos de macroparámetros y fuentes.
3. Definir modelo legal ampliado y política de evidencias.
4. Crear ADR de numeración para 105–111.

### Oleada B — Capacidades operables sin custodia de fondos

5. Fase 96 — Estratega macro.
6. Fase 97 — Auditoría legal ampliada.
7. Fase 98/102 — LOI versionada y generación PDF.
8. Fase 100 — Despliegue productivo.
9. Fase 101/104 — Negociador asistido con aprobación humana.
10. Fase 116/117 — Telemetría y resiliencia WAF.

### Oleada C — Diseño condicionado por legal

11. Fases 99/103 — Escrow en sandbox, únicamente después de G2.
12. Fases 112–114 — Diseño técnico no operativo y luego implementación gradual tras G2.
13. Fase 115 — Oráculo geoespacial con evaluación de sesgos y aprobación de fuentes.

---

## 6. Plan fase por fase

### Fase 95 — Tracker y seguimiento post-oferta

**Estado:** base implementada en `a9e64d6`; falta el tracker automático del documento original.

**Dev 1**

- Añadir scheduler controlado y registro de ejecuciones.
- Implementar locks Redis por oferta para evitar doble rastreo.
- Añadir timeout, reintentos y DLQ de actualizaciones de estado.
- Exponer métricas de ofertas activas, errores y última comprobación.

**Dev 2**

- Definir `OfertaTrackerPort` y adaptador Firecrawl simulado.
- Mapear señales `pending`, `sold`, `under contract`, `withdrawn` y equivalentes.
- Convertir cada señal a una transición válida y exigir revisión humana cuando sea ambigua.
- Añadir plantilla de aviso Telegram sin realizar acciones comerciales automáticas.

**Aceptación**

- Una oferta activa se revisa como máximo una vez por periodo configurado.
- Dos workers no procesan la misma oferta simultáneamente.
- Una señal ambigua no cierra ni cancela una oferta.
- Cada actualización genera `oferta_evento` auditable.
- No se envía ninguna oferta ni comunicación vinculante automáticamente.

**Estimación:** 4 días-persona.

---

### Fase 96 — Estratega macroeconómico

**Dev 1**

- Crear tabla `macro_fuentes`, `macro_observaciones` y `macro_parametros_version`.
- Implementar scheduler seguro e idempotente.
- Cifrar o proteger credenciales de fuentes externas mediante variables de entorno.
- Añadir endpoint de consulta y activación de una versión aprobada.

**Dev 2**

- Crear `MacroDataPort` y adaptadores para fuentes permitidas.
- Normalizar vacancia, tasas de interés, inflación y cap rate objetivo.
- Implementar reglas de ajuste reproducibles antes de usar cualquier LLM.
- Crear informe de cambio con fuente, fecha, valor anterior, valor nuevo y confianza.

**Aceptación**

- No se activa un parámetro sin fuente, timestamp y aprobación humana.
- Repetir la misma observación no duplica datos.
- El grafo utiliza únicamente la versión marcada como activa.
- Una fuente caída conserva el último valor válido y genera alerta.
- Se puede reconstruir por qué cambió cada parámetro.

**Estimación:** 6 días-persona.

---

### Fase 97 — Auditoría legal ampliada

**Dev 1**

- Crear `PublicRecordPort` con allowlist de dominios permitidos.
- Añadir caché, rate limit, trazabilidad de URL y hash del documento consultado.
- Guardar evidencia, fecha de consulta y estado de disponibilidad.
- Añadir control de retención y eliminación de datos.

**Dev 2**

- Implementar adaptadores por condado/tax assessor sin scraping fuera de términos permitidos.
- Extraer señales de liens, tax delinquency, foreclosure y titularidad.
- Separar `sin_evidencia`, `riesgo_detectado`, `fuente_no_disponible` y `requiere_abogado`.
- Añadir disclaimer: análisis asistido, no opinión jurídica ni title insurance.

**Aceptación**

- Toda señal legal tiene fuente, fecha y evidencia asociada.
- La ausencia de datos no se interpreta como ausencia de riesgo.
- Cualquier riesgo legal fuerza revisión humana.
- Las fuentes no permitidas son rechazadas por allowlist.
- El resultado aparece en decisión y auditoría.

**Estimación:** 10 días-persona.

---

### Fase 98/102 — LOI versionada y PDF no vinculante

**Dev 1**

- Crear almacenamiento de plantillas versionadas y documentos generados.
- Añadir hash del contenido, usuario creador y estado de aprobación.
- Proteger descarga y acceso por propiedad/job.
- Añadir antivirus o validación de formato si se incorpora almacenamiento externo.

**Dev 2**

- Crear plantillas por estado de EE. UU. solo después de revisión legal.
- Incluir siempre `non-binding`, disclaimers y campos de revisión.
- Generar PDF determinista con importe, contingencias, vencimiento y datos de la propiedad.
- Impedir la marca `aprobado_para_envio` sin firma o confirmación humana.

**Aceptación**

- El PDF se puede reproducir desde la plantilla y versión registrada.
- El documento contiene disclaimer no vinculante.
- No existe endpoint de envío automático.
- Toda modificación genera una nueva versión.
- Un abogado/operador puede revisar y aprobar el borrador.

**Estimación:** 7 días-persona.

---

### Fase 99/103 — Escrow bancario en sandbox

**Estado:** bloqueado hasta G2.

**Dev 1**

- Diseñar doble aprobación, límites, idempotency key y segregación de estados.
- Implementar únicamente un `EscrowPort` con fake provider.
- Registrar intentos, aprobaciones, rechazos, conciliaciones y reversos.
- No conectar cuentas reales ni ejecutar transferencias.

**Dev 2**

- Definir máquina de estados de instrucciones de escrow.
- Diseñar conciliación de referencias y discrepancias.
- Documentar KYC/KYB, AML/BSA y responsabilidades de la entidad regulada.
- Crear casos de prueba de doble aprobación y replay.

**Aceptación de sandbox**

- Ningún flujo puede mover fondos reales.
- Dos identidades distintas son necesarias para aprobar.
- Un replay no crea una segunda instrucción.
- Toda discrepancia queda en estado `requiere_revision`.
- El modo real no existe mientras no se cierre G2.

**Estimación:** 12 días-persona después de G2.

---

### Fase 100 — Despliegue productivo extendido

**Dev 1**

- Crear `docker-compose.prod.yml` y perfiles separados.
- Incorporar PostgreSQL, Redis, Prometheus, Grafana y alertas.
- Añadir backup, restore probado, rotación de logs y healthchecks.
- Documentar TLS, secretos, mínimos privilegios y recuperación ante desastre.

**Dev 2**

- Definir dashboards de negocio: jobs, propiedades, decisiones, ofertas y errores.
- Crear alertas de latencia, DLQ, fallos de fuente y decisiones `error`.
- Validar métricas contra escenarios E2E.
- Documentar SLOs y umbrales operativos.

**Aceptación**

- Despliegue reproducible desde una máquina limpia.
- Backup y restore se prueban en un entorno aislado.
- No hay secretos en imágenes ni repositorio.
- Alertas se disparan en pruebas controladas.
- API, worker, Redis y PostgreSQL tienen healthchecks.

**Estimación:** 7 días-persona.

---

### Fase 101/104 — Negociador asistido y observabilidad de decisiones

**Dev 1**

- Crear API de borrador con autenticación, rate limit y auditoría.
- Añadir cola de revisión humana y estados `borrador`, `revisar`, `aprobado`, `rechazado`.
- Impedir envío, aceptación o firma desde el servicio automático.
- Exponer métricas de tiempos de revisión y propuestas rechazadas.

**Dev 2**

- Implementar extracción estructurada de correo/mensaje sin inferir atributos protegidos.
- Generar alternativas de contraoferta y explicación de variables usadas.
- Añadir umbrales de confianza y fallback manual.
- Integrar con LOI solo como borrador versionado.

**Aceptación**

- La IA solo propone; una persona decide.
- No se usan atributos protegidos ni proxies discriminatorios.
- Cada sugerencia incluye explicación y datos de entrada.
- No se envía ningún correo o SMS automáticamente.
- Un operador puede rechazar y corregir la propuesta.

**Estimación:** 10 días-persona.

---

### Fases 105–109 — Definición y cierre de huecos

Los documentos revisados no contienen una especificación funcional consolidada para estas fases. Se crea una etapa explícita de descubrimiento, no una implementación inventada.

**Dev 1**

- Inventariar referencias de código, endpoints y tablas asociadas.
- Crear matriz de trazabilidad requisito → módulo → prueba.
- Detectar duplicados entre fases 95–104 y 110–117.

**Dev 2**

- Reunir ejemplos de negocio, usuarios, entradas y salidas esperadas.
- Definir criterios de aceptación y riesgos legales por fase.
- Proponer si cada fase se absorbe, se renombra o se cancela.

**Aceptación**

- Cada fase 105–109 tiene una ficha aprobada, o queda formalmente descartada.
- No se escribe código productivo sin contrato aprobado.
- La numeración se reconcilia en un ADR.

**Estimación:** 4 días-persona.

---

### Fases 110–111 — Gateway, WAF y contratos de resiliencia

La numeración está inconsistente entre los documentos. Primero se debe aprobar ADR-110-111. Mientras tanto se implementa solo el núcleo común.

**Dev 1**

- Consolidar middleware de acceso, rate limit, correlation ID y límites de payload.
- Crear circuito por plataforma con cooldown, métricas y DLQ sin evasión de controles.
- Revisar SSRF, allowlist y términos de uso.

**Dev 2**

- Definir estados de fuente: disponible, limitada, degradada, bloqueada y no autorizada.
- Probar que el worker no insiste contra una fuente bloqueada.
- Documentar mensajes operativos y degradación a fuentes alternativas permitidas.

**Aceptación**

- Un circuito abierto reduce llamadas y se recupera por cooldown controlado.
- Nunca se intenta eludir CAPTCHA, WAF o controles de acceso.
- Cada bloqueo tiene causa y métrica.
- Las rutas no autorizadas devuelven error tipado.

**Estimación:** 5 días-persona.

---

### Fase 115 — Oráculo geoespacial y Gentrification Score

**Estado:** requiere revisión legal y de sesgos antes de activar cualquier scoring.

**Dev 1**

- Crear `GeospatialDataPort`, caché y límites por fuente.
- Guardar versión de datos, resolución espacial y fecha.
- Añadir controles de acceso y borrado de datos sensibles.

**Dev 2**

- Definir variables explicables: permisos, densidad comercial y tendencias temporales.
- Prohibir variables que funcionen como proxies de clases protegidas.
- Validar estabilidad y disparidad del score por zona.
- Mostrar el resultado como señal informativa, nunca como decisión automática de vivienda o crédito.

**Aceptación**

- El score es explicable y reproducible.
- Se documentan fuentes y licencias.
- Se ejecutan pruebas de sesgo y revisión legal.
- El score no bloquea ni prioriza automáticamente personas o grupos protegidos.
- Sin aprobación legal, el módulo permanece en modo `disabled`.

**Estimación:** 10 días-persona después de G2.

---

### Fases 112–114 — DeFi, outbound SMS y AGNET RPC

**Estado:** bloqueadas. Solo se permite diseño, contratos fake y documentación hasta G2.

**Dev 1**

- Diseñar interfaces aisladas y políticas de autorización.
- Preparar simuladores sin claves reales ni fondos.
- Definir mTLS/HSM, doble control, límites y auditoría.

**Dev 2**

- Preparar modelos de consentimiento, DNC, KYC/KYB y screening.
- Definir casos de fraude, reversión y sanciones.
- Proponer pruebas contractuales sin ejecutar acciones externas.

**Aceptación previa a cualquier implementación real**

- Dictamen legal y regulatorio firmado.
- Entidad regulada y responsabilidades identificadas.
- Pruebas externas de seguridad/compliance.
- Sandbox separado de producción.
- Kill switch y control dual probado.

**Estimación posterior a G2:** 41 días-persona para las tres fases, sin incluir auditorías externas.

---

## 7. Calendario sugerido

| Sprint | Semanas | Dev 1 | Dev 2 | Salida |
|---|---:|---|---|---|
| F7 | 1–2 | Cierre tracker F95, contratos, locks | Tracker y señales de oferta | F95 completa |
| F8 | 3–4 | Persistencia/scheduler macro | Adaptadores y reglas macro | F96 |
| F9 | 5–7 | Evidencias, caché y retención | Fuentes públicas y análisis legal | F97 |
| F10 | 8–9 | Plantillas, almacenamiento y despliegue base | LOI, PDF y revisión humana | F98/102 |
| F11 | 10–11 | Compose productivo, backups y métricas | Dashboards y alertas | F100 |
| F12 | 12–13 | API de revisión y seguridad | NLP supervisado | F101/104 |
| F13 | 14 | ADR 105–111 y hardening WAF | Definición funcional y pruebas | F105–111 |
| F14 | 15–16 | Diseño sandbox escrow/DeFi/RPC | Modelos legales y simuladores | G2 preparada |
| F15 | 17–18 | Infraestructura geoespacial | Score explicable y sesgos | F115 en modo disabled |
| F16+ | Después de G2 | Implementación gradual de 99/103/112–114 | Integraciones y compliance | Solo si se aprueba legalmente |

La duración es orientativa. Las fases bloqueadas no deben consumirse como compromiso de producción antes de recibir aprobación legal.

---

## 8. Flujo Git obligatorio por fase

Para cada fase:

```powershell
# Dev 1 o Dev 2 actualiza referencias y revisa cambios del otro dev
git fetch kkk
git log --oneline dev2..dev1
git log --oneline dev1..dev2

# Desarrollo en la rama propia
git switch dev1   # o dev2
git status
git add .
git commit -m "feat(faseXX): descripcion"
git push kkk dev1:dev1  # o dev2:dev2

# Integración únicamente cuando la fase está validada
git switch main
git merge --no-ff dev2 -m "merge: integra fase XX de dev2 en main"
git merge --no-ff dev1 -m "merge: integra fase XX de dev1 en main"
git push kkk main:main
```

Antes de cada fase nueva se debe:

1. Obtener y revisar los cambios de `dev2`.
2. Comparar el diff con `dev1`.
3. Fusionar solo después de resolver conflictos y ejecutar validaciones.
4. No iniciar la siguiente fase hasta que la anterior esté subida a remoto.

---

## 9. Definición de terminado

Una fase está terminada solo si cumple todo lo siguiente:

- Código en `dev1` o `dev2` y posteriormente en `main`.
- Commit identificable y push confirmado.
- Pruebas unitarias y de integración relevantes.
- `pyright` sin errores nuevos.
- Migración Alembic si se agregan tablas o columnas.
- Docker Compose arranca los servicios necesarios.
- Logs sin secretos ni datos personales innecesarios.
- Documentación de operación y rollback.
- Revisión del otro desarrollador.
- Criterios de aceptación demostrados.
- Requisitos legales documentados y puerta correspondiente cerrada.

Si Python, PostgreSQL o los proveedores externos no están disponibles en la máquina de desarrollo, se debe ejecutar la validación equivalente en CI o Docker y registrar la limitación, nunca declarar la fase como completamente validada solo por análisis estático.

---

## 10. Riesgos y controles

| Riesgo | Control |
|---|---|
| Cambios de fuentes externas | Ports/adapters, contratos versionados y fixtures offline |
| Datos personales de vendedores | Minimización, retención, acceso auditado y borrado |
| Decisión legal incorrecta | Evidencia, disclaimers y revisión humana obligatoria |
| Automatización de una oferta vinculante | No existe endpoint de envío automático |
| Movimiento accidental de fondos | Solo fake provider hasta G2; doble control y kill switch |
| SMS sin consentimiento | Módulo desactivado hasta TCPA/DNC y base de licitud aprobada |
| Sesgo geoespacial | Variables explicables, pruebas de disparidad y modo disabled |
| Evasión WAF | No se implementan técnicas de bypass; solo backoff, cooldown y respeto de términos |
| Divergencia Git | Fetch y revisión de `dev2` antes de cada fase; merges `--no-ff` |

---

## 11. Prioridad de ejecución

1. Completar Fase 95.
2. Implementar Fase 96.
3. Implementar Fase 97.
4. Implementar Fase 98/102.
5. Implementar Fase 100.
6. Implementar Fase 101/104.
7. Reconciliar y definir Fases 105–111.
8. Implementar hardening 110–111/116–117 según ADR aprobado.
9. Preparar diseños sandbox de 99/103 y 112–114.
10. Implementar Fase 115 únicamente con aprobación de sesgos y licencias.
11. Activar fases financieras, SMS y blockchain solo después de G2 y auditoría externa.

**Próximo corte recomendado:** cerrar el tracker automático de la Fase 95, actualizar el estado y subir el commit antes de iniciar la Fase 96.
