# Estado de la implementación OpenClaw

Actualizado: 20 de noviembre de 2026 · Responsable de este corte: Dev 1

## Ramas y flujo de trabajo

| Rama | Uso | Estado |
|---|---|---|
| `main` | Integración estable | Recibe fusiones `--no-ff` de `dev1` y `dev2` |
| `dev1` | Trabajo de Dev 1 (núcleo, API, seguridad, datos) | Activa |
| `dev2` | Trabajo de Dev 2 (ingesta, contratos, normalizador) | Activa, en `wt-dev2` |

Flujo acordado:

1. Dev 1 y Dev 2 trabajan cada uno en su rama.
2. Cuando Dev 2 termina un bloque, Dev 1 cambia a `dev2`, revisa el código y lo fusiona en `dev1`.
3. `dev1` se fusiona en `main` con `--no-ff` para conservar la traza de cada entrega.

## Sprint 0 — Cimientos (completo)

- Toolchain reproducible (`pyproject.toml`, `requirements*.txt`, `pytest.ini`).
- Imagen multi-target `api` / `worker` y `docker-compose.dev.yml` con Redis y PostgreSQL.
- CI de GitHub Actions con lint y pruebas; `Makefile` con los atajos del día a día.
- Configuración con `pydantic-settings` y fallo rápido en producción.
- Logging JSON con `job_id` por contexto, cliente Redis y `/health` + `/readyz`.
- Reproducción del flujo heredado (`scripts/reproducir_flujo_actual.py`).

## Sprint 1 — Núcleo e ingesta (completo)

| ID | Tarea | Estado |
|---|---|---|
| S1-01 | Settings validados al arranque | Hecho |
| S1-02 | Contratos `BarridoRequest`, `JobResponse`, `ErrorResponse` | Hecho |
| S1-03 | `POST /v1/openclaw/barridos` con 202 y `job_id` | Hecho |
| S1-04 | `GET /v1/openclaw/barridos/{job_id}` con 404 tipado | Hecho |
| S1-05 | Entrypoints API y worker separados | Hecho |
| S1-06 | Logging estructurado con `job_id` | Hecho |
| S1-07 | Catálogo de 9 plataformas | Hecho |
| S1-08 | `ScraperPort` y `FirecrawlAdapter` con backoff | Hecho |
| S1-09 | Rate limiter de ventana deslizante | Hecho |
| S1-10 | Pruebas de contrato con cassettes | Hecho |
| S1-11 | Normalizador de markdown | Hecho |

## Sprint 2 — Seguridad y datos (en curso)

| ID | Tarea | Estado |
|---|---|---|
| S2-01 | Autenticación por API key con hash | Hecho |
| S2-02 | Esquema PostgreSQL y migraciones Alembic | Hecho |
| S2-04 | Semilla del catálogo de plataformas | Hecho |
| S2-03 | Repositorios y unidad de trabajo | Pendiente |
| S2-05 | Webhook de Telegram con `secret_token` y dedupe | Pendiente |
| S2-06 a S2-09 | Bloque de Dev 2 (normalizador, `/buy` sandbox, cliente UAO, suite seguridad) | Hecho en `dev2` |

## Sprints 3 a 6 — Bloque Dev 2 (completado en `dev2`)

| Sprint | ID | Tarea | Estado |
|---|---|---|---|
| S3 | S3-05 | Enriquecimiento (precio, superficie, año, HOA) con 20 anuncios | Hecho en `dev2` |
| S3 | S3-07 | Deduplicación por dirección normalizada y similitud | Hecho en `dev2` |
| S3 | S3-08 | Pruebas de resiliencia del worker (pérdida, reintento, DLQ, duplicados) | Hecho en `dev2` |
| S4 | S4-04 | Nodo financiero parametrizado desde configuración (`OPENCLAW_FIN_*`) | Hecho en `dev2` |
| S4 | S4-05 | Nodo de visión multimodal con `ValuationPort` y degradación a `no_disponible` | Hecho en `dev2` |
| S4 | S4-06 | Nodo de auditoría legal asistida (liens, back taxes, revisión humana obligatoria) | Hecho en `dev2` |
| S4 | S4-07 | Nodo de alerta Telegram con plantilla, dedupe y enlace a ficha | Hecho en `dev2` |
| S5 | S5-04 | Panel mínimo de jobs y telemetría (`app/frontend/panel.html`) | Hecho en `dev2` |
| S5 | S5-05 | Suite E2E con Firecrawl simulado + prueba real nocturna opcional | Hecho en `dev2` |
| S5 | S5-06 | Prueba de carga controlada (100 jobs sin errores no controlados) | Hecho en `dev2` |
| S6 | S6-02 | Revisión de seguridad OWASP API Top 10 (`docs/security/OWASP_API_TOP10_REVIEW.md`) | Hecho en `dev2` |
| S6 | S6-05 | Backlog priorizado y estimado de fases 95–104 (`docs/BACKLOG_FASES_95_104.md`) | Hecho en `dev2` |

### S2-01 · Autenticación

- Módulo `app/core/security.py`: `hash_api_key`, `requiere_api_key`.
- La clave viaja en `X-OpenClaw-Tactical-Key` y solo se compara su SHA-256 con `hmac.compare_digest`.
- Respuestas: 401 sin cabecera, 403 con clave inválida, 503 si el servicio no tiene clave configurada.
- `/health` y `/readyz` quedan públicos para las sondas.
- `OPENCLAW_API_KEY_SHA256` permite no guardar la clave en claro en el servidor.
- Pruebas: `tests/test_security.py` (8 casos) y clientes autenticados en `conftest.py`.

### S2-02 y S2-04 · Datos

- `app/db/models.py`: `jobs`, `propiedades`, `decisiones`, `eventos`, `plataformas`.
- `app/db/base.py`: motor asíncrono perezoso y sesión por petición con commit/rollback.
- `alembic/versions/0001_esquema_inicial.py`: esquema completo con índices y restricciones.
- `alembic/versions/0002_semilla_plataformas.py`: semilla idempotente de las 9 plataformas.

### S2-03 · Repositorios y unidad de trabajo

- `app/db/repositories.py`: `RepositorioJobs`, `RepositorioPropiedades`,
  `RepositorioDecisiones` y `RepositorioEventos`. Ninguno gestiona transacciones.
- `app/db/uow.py`: `UnidadDeTrabajo` (contexto con commit o rollback) y
  `procesar_por_propiedad`, que confirma cada propiedad en su propia transacción
  y devuelve la lista de fallos sin abortar el lote.
- Pruebas: `tests/test_repositories.py` (6 casos con sesión falsa).

### S2-05 · Webhook de Telegram

- Autenticación por `X-Telegram-Bot-Api-Secret-Token` o, en su defecto, la API
  key táctica; sin `secret_token` configurado en producción responde 503.
- Allowlist de chats en `TG_ALLOWED_CHAT_IDS` (CSV); vacía significa sin restricción.
- Deduplicación de `update_id` con `SET NX EX` en Redis y TTL configurable.
  Si Redis no responde el webhook falla cerrado (503), porque un `/buy`
  duplicado cuesta más que un reintento.
- Pruebas: `tests/test_telegram_webhook.py` (10 casos).

## Verificación

| Comprobación | Resultado |
|---|---|
| `pyright` (modo basic) sobre `app`, `tests`, `scripts` | 0 errores, 68 archivos |
| Bloque de Dev 2 (S2-06 a S6-05) | Revisado y fusionado en `dev1` |
| Importación de contratos y catálogo | Consistente entre `contracts.py` y `platforms.py` |
| Migraciones | Pendientes de ejecutar: en este equipo no hay Python utilizable |

Limitación del entorno: el intérprete de Python del equipo está bloqueado
(`Acceso denegado` en `Python313`) y el `.venv` del proyecto no arranca, por lo
que `pytest` y `alembic` no se pudieron ejecutar aquí. La validación se apoya en
análisis estático y el CI queda como puerta de calidad.

## Sprint 3 (Dev 1)

### S3-01 · Consumer group con `XAUTOCLAIM`, reintentos y DLQ
- `app/workers/ingestion_worker.py` recupera los mensajes pendientes de
  consumidores caídos con `XAUTOCLAIM` (inactividad mínima `CLAIM_IDLE_MS`) antes
  de leer nuevos, de modo que una caída a mitad de proceso no pierde trabajo.
- Cada mensaje acumula un contador de entregas en Redis
  (`openclaw:entregas:<id>`). Mientras queden intentos, el mensaje **no** se
  confirma: sigue pendiente y `XAUTOCLAIM` lo reentrega.
- Al alcanzar `MAX_ENTREGAS` (3 por defecto) el mensaje se copia a la
  dead-letter queue `openclaw_scraping_dlq` con el payload original, el motivo y
  el número de entregas, y solo entonces se confirma para no atascar el grupo.
- El bucle tolera caídas de Redis (`error_lectura_cola`) y crea el grupo con
  `mkstream` tratando `BUSYGROUP` como caso normal.

### S3-02 · Idempotencia por `job_id` y plataforma
- `app/services/idempotency.py` reclama `openclaw:idem:<job_id>:<plataforma>` con
  `SET NX EX` antes de raspar cada dominio. Si la clave ya existía, el trabajo se
  marca como `omitido_idempotente` y no se vuelve a llamar al proveedor.
- Reprocesar el mismo mensaje (por reentrega o por reintento) no duplica
  propiedades ni consume cuota de Firecrawl.
- `liberar()` permite un reintento legítimo tras un fallo controlado.

### S3-03 · Máquina de estados del job
- `app/workers/state_machine.py` declara las transiciones válidas:
  `queued → running|failed`, `running → completed|partial|failed`,
  `partial → running`, `failed → queued|running`; `completed` es terminal.
- `aplicar_transicion()` valida antes de escribir y lanza `TransicionInvalida`
  ante un salto ilegal, sin tocar el estado almacenado.
- El worker ya no reabre jobs cerrados: si el mensaje llega con el job en estado
  terminal se registra `job_ya_cerrado` y se descarta el trabajo.

### S3-04 · Semáforo de concurrencia por dominio
- `app/services/domain_semaphore.py` mantiene un `asyncio.Semaphore` por dominio
  (`LIMITE_CONCURRENCIA_DOMINIO`, 2 por defecto), así que saturar Zillow no frena
  a Redfin.
- El worker envuelve la extracción en `async with semaforos.turno(spec.domain)`.
- La cuota global contra el proveedor sigue en el rate limiter de Redis; el
  semáforo solo evita ráfagas locales por dominio.

### S3-06 · Geocodificación con caché
- `app/services/geocoding.py` resuelve direcciones contra Nominatim con
  `User-Agent` identificable y caché en Redis (`openclaw:geo:<sha256>` de la
  dirección normalizada, TTL `GEOCODING_TTL_SEGUNDOS`).
- Los negativos también se cachean para no repetir consultas inútiles; los fallos
  transitorios de red **no** se cachean.
- `geocodificar()` acepta un `httpx.AsyncClient` inyectado, lo que permite
  probarla sin red.

### Pruebas nuevas
| Archivo | Cobertura |
|---|---|
| `tests/test_worker_queue.py` | reintentos sin confirmar, envío a DLQ, job cerrado, idempotencia por plataforma y recuperación de pendientes |
| `tests/test_state_machine.py` | transiciones válidas, ilegales, job inexistente y reintento tras fallo |
| `tests/test_idempotency.py` | primera reclamación, duplicados, plataformas distintas, liberar y TTL |
| `tests/test_domain_semaphore.py` | límite 1 serializa, límite 2 paraleliza, dominios independientes |
| `tests/test_geocoding.py` | caché positiva y negativa, error HTTP sin cachear, respuesta inválida |

## Verificación del corte
| Comprobación | Resultado |
|---|---|
| `pyright` (modo basic) sobre `app`, `tests`, `scripts` | 0 errores, 4 avisos (`redis.exceptions` sin resolver por no haber paquetes instalados) |
| Rama `dev1` | `7f3de30` más el commit de S3-04/S3-06 |
| `pytest` y `alembic` | No ejecutables en este equipo (intérprete de Python bloqueado) |

## Sprint 4 (Dev 1)

### S4-01 · Grafo tipado con aristas condicionales y checkpointer
- `app/orchestrator/graph.py` compila un `StateGraph` real sobre `EstadoGrafo`
  (hereda el estado de los nodos y añade `ruta`, `nodo_fallido`, `motivo_fallo`).
- Recorrido: `clasificar → financiero → vision → legal → alerta`, con dos salidas
  tempranas: precio fuera de rango termina en `DESCARTADO_PRECIO` sin llamar al
  proveedor, y un nodo agotado termina el grafo en `FALLO_NODO`.
- El checkpointer es `MemorySaver` por defecto e inyectable; `thread_id` =
  `propiedad:<uuid>` para poder reanudar una evaluación.
- Si `langgraph` no está instalado, `evaluar_propiedad()` cae al pipeline
  secuencial de `workflow.py` en lugar de romper el worker.

### S4-02 · Cada propiedad evaluada deja una fila en `decisiones`
- `app/orchestrator/decisiones.py` traduce el estado final a `Decision`:
  veredicto (`comprar`, `revisar`, `descartar`, `error`), cap rate normalizado a
  fracción para `Numeric(6,4)`, CAPEX, motivo y un `detalle` con visión, legal,
  alerta y ficha.
- `estado_inicial_desde_orm()` adapta la fila de `propiedades` al estado del
  grafo, así el worker no tiene que construir el diccionario a mano.
- `evaluar_y_registrar_decision()` guarda la decisión y el evento
  `decision.registrada` en la misma transacción por propiedad.

### S4-03 · Reintento por nodo y dead-letter de decisiones
- `con_reintentos(nombre, nodo, intentos)` reintenta el nodo
  (`DECISION_MAX_INTENTOS_NODO`, 2 por defecto) y, al agotarlos, escribe en
  `openclaw_decisiones_dlq` el nodo, el motivo, el número de intentos, la
  propiedad y el job.
- La DLQ es best-effort: si Redis no responde se registra el error pero el grafo
  termina igual con veredicto `error`, que exige revisión humana.

### Pruebas nuevas
| Archivo | Cobertura |
|---|---|
| `tests/test_decisiones_orquestador.py` | veredictos, normalización del cap rate, mapeo a `Decision`, reintentos con éxito y con DLQ, Redis caído y aristas condicionales |

## Verificación del corte
| Comprobación | Resultado |
|---|---|
| `pyright` (modo basic) sobre `app`, `tests`, `scripts` | 0 errores, 4 avisos (`redis.exceptions` sin resolver) |
| `dev1` → `main` | Fusión `--no-ff` sin conflictos |
| `pytest` | No ejecutable en este equipo (intérprete de Python bloqueado) |

## Fase 95 — Seguimiento post-oferta (Dev 1)

- Tabla `ofertas` y tabla append-only `oferta_eventos`, con migración
  `0003_seguimiento_ofertas.py` y trazabilidad por propiedad.
- Pipeline explícito `propuesta → contraoferta → aceptada → cerrada`, además de
  rechazo y cancelación; los saltos inválidos responden con `409`.
- API protegida por API key: crear, listar y cambiar el estado de una oferta en
  `/v1/openclaw/propiedades/{propiedad_id}/ofertas`.
- No se envían ofertas automáticamente: el actor humano y sus notas quedan
  registrados en cada transición.
- Pruebas: `tests/test_offers.py` cubre el flujo válido y los saltos ilegales.

### Tracker automático post-oferta

- `app/services/offer_tracker.py` define `OfertaTrackerPort`, interpreta señales
  explícitas (`sold`, `withdrawn`) y deja las señales ambiguas como eventos sin
  mutar el estado de negocio.
- Cada consulta toma un lock Redis por oferta con TTL; el token se verifica al
  liberarlo para evitar que una ejecución tardía borre el lock de otra.
- La señal `sold` solo cierra una oferta previamente aceptada; una ficha retirada
  cancela una oferta activa. `pending` y `under contract` son informativas.
- `app/workers/offer_tracker.py` procesa lotes configurables y registra fallos
  por oferta sin detener las restantes. `docker-compose.dev.yml` lo ejecuta como
  servicio separado `offer-tracker`.
- Configuración: `OFERTA_TRACKER_INTERVAL_SEGUNDOS`,
  `OFERTA_TRACKER_LOCK_TTL_SEGUNDOS` y `OFERTA_TRACKER_MAX_OFERTAS_LOTE`.

## Plan de fases faltantes 95–117 — Bloque Dev 2 (completado en `dev2`)

Todas las entregas de dominio, IA, observabilidad, documentación legal y contratos
sandbox asignadas a Dev 2 en `PLAN_IMPLEMENTACION_FASES_FALTANTES_2DEVS.md` quedaron
implementadas, probadas y publicadas en `dev2`:

| Fase | Entregables Dev 2 | Estado | Pruebas |
|---|---|---|---|
| **95** | `app/services/offer_tracker.py` (`OfertaTrackerPort`, `FirecrawlOfertaTrackerAdapter`, lock Redis con token verificable, señales explícitas vs ambiguas) | Hecho en `dev2` | `tests/test_fase95_offer_tracker.py` (10 casos) |
| **96** | `app/services/macro_strategist.py` (`MacroDataPort`, `StaticMacroAdapter`, allowlist `fred`/`bls`/`census`/`fhfa`, reglas deterministas, fallback, aprobación humana obligatoria) | Hecho en `dev2` | `tests/test_fase96_macro_strategist.py` (3 casos) |
| **97** | `app/services/legal_public_records.py` (`PublicRecordPort`, `CountyTaxAssessorAdapter`, allowlist HTTPS, hash SHA-256, 4 estados y `LEGAL_DISCLAIMER`) | Hecho en `dev2` | `tests/test_fase97_legal_records.py` (2 casos) |
| **98/102** | `app/services/loi_generator.py` (plantillas OH/FL/MI/TX con `legal_reviewed`, PDF 1.4 determinista, versionado inmutable, `NON_BINDING_DISCLAIMER`, sin envío automático) | Hecho en `dev2` | `tests/test_fase98_loi_pdf.py` (2 casos) |
| **100** | `deploy/observability/grafana_dashboard_openclaw.json`, `deploy/observability/prometheus_alerts_openclaw.yml`, `docs/SLOS_Y_ALERTAS_F100.md`, `app/services/slo_evaluator.py` | Hecho en `dev2` | `tests/test_fase100_observability_slos.py` (2 casos) |
| **101/104** | `app/services/negotiation_assistant.py` (extracción comercial sin atributos protegidos Fair Housing, alternativas explicables, umbral de confianza + fallback manual, integración con borrador LOI) | Hecho en `dev2` | `tests/test_fase101_negotiator.py` (2 casos) |
| **105–111 / 116–117** | `docs/adr/ADR-06-reconciliacion-fases-105-111.md`, `app/services/source_health.py` (`PlatformSourceHealthManager`, cooldown, fuentes alternativas, cero evasión WAF/CAPTCHA) | Hecho en `dev2` | `tests/test_fase105_111_source_health.py` (2 casos) |
| **99/103** | `app/services/escrow_sandbox.py` (solo `disabled`/`sandbox`, `FakeEscrowProvider`, doble aprobación 4-eyes, idempotencia, conciliación), `docs/legal/ESCROW_COMPLIANCE_F99_103.md` | Sandbox/G2 en `dev2` | `tests/test_fase99_escrow_sandbox.py` (2 casos) |
| **112–114** | `app/services/compliance_gates.py` (contratos TCPA/DNC, KYC/KYB/OFAC, fraude/reversión, kill switch y puerta G2 sin acciones externas), `docs/legal/PUERTA_G2_FASES_112_114.md` | Bloqueado/G2 en `dev2` | `tests/test_fase112_114_compliance_gates.py` (2 casos) |
| **115** | `app/services/geospatial_oracle.py` (modo `disabled` por defecto hasta G2, prohibición estricta de proxies Fair Housing/ECOA, score explicable informativo y auditoría de estabilidad/disparidad), `docs/legal/F115_SESGOS_Y_LICENCIAS.md` | Disabled/G2 en `dev2` | `tests/test_fase115_geospatial_oracle.py` (4 casos) |

### Verificación completa en `dev2`

- `pytest --cov=app -q`: **256 passed, 3 skipped** (pruebas en vivo opcionales), **89 % de cobertura** sobre 4 303 sentencias.
- `ruff check`, `black --check` y `mypy --follow-imports=silent --explicit-package-bases`: **0 errores** en todos los módulos y pruebas de Dev 2.

## Siguiente corte
1. Dev 1: integración de endpoints/persistencia/scheduler para fases 96, 97, 98/102, 100, 101/104, 110–111 y 115.
2. Puerta legal G2: dictamen firmado y auditoría externa antes de activar cualquier módulo financiero real, SMS/outreach o scoring geoespacial (fases 99/103, 112–114 y 115).
