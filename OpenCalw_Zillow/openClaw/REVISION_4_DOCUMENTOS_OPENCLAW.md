# Revisión de los cuatro documentos de actualización — OpenClaw

**Fecha:** 20 de noviembre de 2026  
**Código de referencia:** `OpenCalw_Zillow/openClaw/OpenClaw_Clean_Architecture/openclaw`  
**Documentos revisados:** los cuatro `.docx` ubicados en `OpenCalw_Zillow/openClaw`.

## 1. Confirmación de lectura

Los cuatro archivos se abrieron y se extrajo su texto completo desde `word/document.xml`. Ninguno contiene tablas, encabezados, pies de página, notas ni texto en cuadros; todo el contenido está en el cuerpo del documento. Por eso la extracción cubre el 100 % del texto.

| Documento | Párrafos con texto | Caracteres de texto | Funciones `def` | Contenido |
|---|---:|---:|---:|---|
| `OpenClaw_Fases_0_94.md.docx` | 160 | 6.972 | 9 | Base consolidada: esquema, scraper, worker, gateway y orquestador |
| `OpenClaw_Fases_95_104.md.docx` | 118 | 5.389 | 8 | Diez módulos nuevos: seguimiento, legal, seguridad, métricas, contratos, escrow, NLP |
| `OpenClaw_Fases_110_117_Actualizacion.md.docx` | 215 | 8.509 | 6 | Ingesta endurecida, circuit breaker, worker de 9 plataformas, telemetría y DeFi |
| `OpenClaw_Actualizacion_Ruta_OpenCalw_Zillow_openClaw.md.docx` | 215 | 8.524 | 6 | Mismo contenido que el anterior, con título distinto |

## 2. Dos hallazgos sobre los propios documentos

**Documentos duplicados.** `OpenClaw_Actualizacion_Ruta_OpenCalw_Zillow_openClaw.md.docx` y `OpenClaw_Fases_110_117_Actualizacion.md.docx` son el mismo documento. Tienen idéntica estructura, idénticas cinco secciones y el mismo código; la única diferencia está en la primera línea del título (uno dice “Actualización en Ruta (Mi unidad > OpenCalw_Zillow > openClaw)” y el otro “Actualización en Google Drive (Fases 110 a 117)”). La diferencia total es de 15 caracteres y una palabra. En la práctica son **tres documentos distintos, no cuatro**.

**Código sin indentación.** Los bloques de código de los tres documentos perdieron toda la sangría: hay cero líneas con espacios iniciales en los cuatro archivos. El código se puede leer, pero **no se puede copiar y ejecutar directamente**, porque Python lanzaría `IndentationError` en la primera estructura de control. Cualquier implementación debe reindentar y volver a probar cada bloque.

## 3. Trazabilidad del documento de fases 0–94

Este documento define la arquitectura base. Es el único cuyos módulos existen, aunque con diferencias de nombre, de firma y de comportamiento.

| Elemento del documento | Fase | Archivo actual | Estado |
|---|---:|---|---|
| `models/propiedad_schema.py` | 83 | `app/models/property.py` | Existe con otro nombre; no se usa para validar |
| Scraper Firecrawl con reintentos | 84, 85, 89 | `app/services/scraper.py` | Implementado; usa `search` en lugar de `scrape_url` |
| Rate limiter por dominio | 85 | `app/services/rate_limiter.py` | Implementado; continúa si Redis falla |
| Worker Redis Streams con `XREADGROUP`/`XACK` | 88 | `app/workers/ingestion_worker.py` | Implementado; sin arranque documentado |
| Gateway API y webhook | 86, 94 | `app/api/main.py`, `routes/` | Parcial; sin health ni autenticación |
| Nodo financiero | 90–92 | `app/orchestrator/nodes/financial.py` | Implementado y aislado |
| Nodo de visión | 90–92 | `app/orchestrator/nodes/vision.py` | Placeholder con CAPEX fijo |
| Nodo de alerta Telegram | 90–92 | `app/orchestrator/nodes/telegram_alert.py` | Implementado, sin manejo de errores |
| Enrutador de due diligence | 93 | `app/orchestrator/workflow.py` | Solo la función de ruteo |
| Grafo LangGraph compilado | 90–93 | — | No existe |

## 4. Trazabilidad del documento de fases 95–104

Ninguno de los diez módulos existe en el árbol actual. La carpeta `app/api/middleware` tampoco existe.

| Fase | Módulo solicitado | Función descrita | Estado |
|---:|---|---|---|
| 95 | `app/workers/portfolio_tracker.py` | Seguimiento post-oferta y detección de `pending`/`sold` | No existe |
| 96 | `app/orchestrator/strategist.py` | Calibración macro con LLM | No existe |
| 97 | `app/orchestrator/nodes/legal.py` | Auditoría legal de liens y back taxes | No existe |
| 98 | `app/api/routes/websockets.py` | WebSocket sobre `openclaw:eventos_asimov` | No existe |
| 99 | `app/api/middleware/security.py` | API key `X-OpenClaw-Tactical-Key` | No existe |
| 100 | `docker-compose.prod.yml` | Redis, pgvector, API, workers, Prometheus, Grafana | No existe (solo comentario en el documento) |
| 101 | `app/api/middleware/metrics.py` | Contador Prometheus por estado final | No existe |
| 102 | `app/services/contract_generator.py` | LOI en PDF con FPDF | No existe |
| 103 | `app/services/escrow_bridge.py` | Transferencia de earnest money | No existe |
| 104 | `app/workers/mail_negotiator.py` | Evaluación de contraofertas | No existe |

Además, este documento requiere dependencias que no están declaradas en `requirements.txt`: `prometheus_client`, `fpdf` y `langchain_core`.

## 5. Trazabilidad del documento de fases 110–117

Este documento describe la versión más avanzada del sistema. Tampoco tiene implementación, y su worker entra en conflicto con el worker actual.

| Fase | Componente | Estado | Observación |
|---:|---|---|---|
| 110–111 | Gateway de ingesta con validación y errores controlados | No implementado | La ruta actual falla con `KeyError` si falta un campo |
| 99/117 | Seguridad Zero-Trust en la ingesta | No implementado | La ruta activa no tiene dependencia de autenticación |
| 117 | `app/core/waf_circuit_breaker.py` | No implementado | Define umbrales 2 y 4 fallos, cooldown de 900 s |
| 117 | Worker con mapa de 9 plataformas | No implementado | El worker actual recibe URLs, no claves de plataforma |
| 116 | `app/api/routes/telemetry.py` | No implementado | Canal Pub/Sub `openclaw_live_telemetry` |
| 112 | `app/workers/treasury_manager.py` | No implementado | Barrido de capital hacia UAO_Grid |
| 113 | `app/workers/outbound_acquisition.py` | No implementado | Skip tracing y SMS con Twilio |
| 114 | `app/api/routes/agnet_rpc.py` | No implementado | EIP-712 y liquidación USDC en Base |
| 115 | `app/workers/geospatial_oracle.py` | No implementado | `Gentrification Score` por código postal |

El documento también propone un patrón de consumo distinto: el worker documentado usa `XREAD` sobre `0-0` y borra con `XDEL`, mientras el worker actual usa consumer group con `XACK`. El primero relee el historial completo del stream en cada ciclo; el segundo confirma mensajes. No son intercambiables y hay que escoger uno.

## 6. Conclusión de la revisión

Los cuatro archivos están efectivamente revisados y su contenido está identificado. El resultado es que **existe un solo documento base implementado parcialmente (fases 0–94)** y **dos documentos de hoja de ruta sin implementación (95–104 y 110–117)**, uno de ellos duplicado. El informe anterior, `REVISION_SISTEMA_OPENCLAW.md`, describe la arquitectura y los riesgos; este documento añade la trazabilidad fase por fase que permite decidir qué construir primero.
