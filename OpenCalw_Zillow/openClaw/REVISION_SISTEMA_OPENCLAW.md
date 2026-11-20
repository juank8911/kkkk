# Revisión del sistema OpenClaw

**Fecha de revisión:** 20 de noviembre de 2026  
**Código revisado:** `OpenClaw_Clean_Architecture/openclaw`  
**Documentos de actualización revisados:** fases 0–94, fases 95–104 y fases 110–117.

## 1. Resumen ejecutivo

OpenClaw es una base de sistema para búsqueda y evaluación automatizada de oportunidades inmobiliarias. La implementación actual está construida alrededor de FastAPI, Redis Streams, Firecrawl y workers asíncronos. Recibe una consulta y una lista de plataformas, coloca la solicitud en Redis y el worker intenta extraer información en paralelo. También contiene un modelo de propiedad normalizado, un cálculo financiero básico, un cálculo preliminar de CAPEX/retorno visual y un webhook de Telegram conectado a un servicio externo de liquidez.

El proyecto actual **no contiene todavía la implementación completa descrita por las actualizaciones**. Las fases documentadas 95–117 deben tratarse como especificación o diseño pendiente, no como funcionalidades ya desplegadas. En el árbol revisado no existen los módulos de seguridad Zero-Trust, circuit breaker WAF, WebSocket de telemetría, persistencia de resultados, seguimiento de cartera, auditoría legal, métricas Prometheus, contratos, escrow, negociador NLP, tesorería DeFi, adquisición off-market, AGNET RPC ni oráculo geoespacial.

**Estado general:** prototipo funcional parcial; no está listo para producción ni para ejecutar operaciones financieras reales.

## 2. Arquitectura actualmente presente

### API Gateway

`app/api/main.py` crea una aplicación FastAPI llamada `OpenClaw API Gateway` y registra dos routers:

- `app/api/routes/ingestion.py`
- `app/api/routes/execution.py`

La ruta `POST /v1/openclaw/iniciar-barrido` recibe un diccionario, toma `query` y `plataformas`, y publica el mensaje en el Redis Stream `openclaw_scraping_queue`.

La ruta `POST /v1/webhook/telegram-comando` reconoce mensajes cuyo texto comienza por `/buy`. Actualmente realiza una llamada fija a `UAO_SCALPING_API/request-withdrawal` con **125000 USDT**, urgencia alta y destino `fiat_bridge_bank`. Esto es una operación sensible y no debe permanecer hardcodeada ni ejecutarse sin autenticación, autorización, confirmación explícita, idempotencia y auditoría.

### Cola y procesamiento asíncrono

`app/workers/ingestion_worker.py` crea o utiliza el grupo Redis `scrapers`, lee mensajes mediante `XREADGROUP`, ejecuta `extraer_con_reintentos` en paralelo para cada elemento de `plataformas` y confirma el mensaje con `XACK`.

El worker no está registrado en el ciclo de vida de FastAPI ni hay en el árbol revisado un archivo de despliegue que demuestre cómo se inicia como proceso independiente. Debe definirse claramente el modo de ejecución de API y worker.

### Extracción

`app/services/scraper.py` usa `FirecrawlApp`, hace búsquedas con reintentos de hasta tres intentos y espera exponencial. `app/services/rate_limiter.py` aplica límites por dominio mediante Redis durante una ventana de 60 segundos.

La implementación actual usa `plataformas` como URLs, mientras que la actualización de fases 110–117 propone nombres de plataformas (`zillow`, `realtor`, `redfin`, etc.) y un mapa de nueve plataformas. Debe elegirse un contrato único y validarlo.

El limitador devuelve permiso cuando Redis falla (`fail-open`). Esa decisión puede provocar exceso de solicitudes y debe revisarse para producción, preferiblemente con una política explícita de degradación segura.

### Modelo de datos

`app/models/property.py` define `DatosPropiedadNormalizados` con origen, ubicación, precio, ingreso anual, imágenes, condado, coordenadas y descripción original. Actualmente el modelo no se utiliza en las rutas o en el worker para validar y normalizar el resultado extraído.

### Orquestación y análisis

`app/orchestrator/workflow.py` solo contiene el enrutador `enrutador_due_diligence` y un comentario indicando que falta compilar/integrar LangGraph.

`app/orchestrator/nodes/financial.py` calcula gastos aproximados, NOI y cap rate; rechaza propiedades con datos inválidos o cap rate menor de 8 %. `app/orchestrator/nodes/vision.py` usa un CAPEX fijo de 15000 USD; no realiza análisis de imágenes. `telegram_alert.py` envía una alerta cuando el estado es `INSPECCION_VISUAL_OK`.

En consecuencia, el flujo financiero, visual y de alerta existe como piezas aisladas, pero no existe un pipeline ejecutable que conecte extracción -> normalización -> due diligence -> análisis financiero -> visión -> alerta.

## 3. Qué indican los archivos de actualización

### Fases 0–94

Definen la base conceptual del sistema: esquema de propiedad, scraper con rate limiting, worker Redis Streams, gateway FastAPI, análisis financiero, visión, alerta Telegram y enrutamiento de due diligence.

La mayor parte de esa base está representada parcialmente en el código actual, pero el orquestador LangGraph y la conexión entre nodos permanecen incompletos.

### Fases 95–104

Solicitan añadir:

1. Seguimiento post-oferta de propiedades.
2. Estratega macro para ajustar parámetros de mercado.
3. Auditoría legal.
4. WebSocket de telemetría.
5. Seguridad Zero-Trust con API key.
6. Matriz de despliegue Docker/Redis/pgvector/API/workers/Prometheus/Grafana.
7. Métricas Prometheus.
8. Generación de cartas de intención en PDF.
9. Puente de escrow bancario.
10. Negociador de contraofertas mediante NLP.

Ninguno de los módulos específicos de estas fases aparece en el árbol actual revisado.

### Fases 110–117

Solicitan una ingesta endurecida y ampliada:

1. Validación de payload y manejo correcto de errores en el gateway.
2. Autenticación Zero-Trust mediante `X-OpenClaw-Tactical-Key`.
3. Circuit breaker adaptativo por plataforma para fallos WAF.
4. Worker para nueve plataformas con telemetría Redis Pub/Sub.
5. Feed WebSocket para visualizar el scraping en tiempo real.
6. Tesorería DeFi.
7. Adquisición outbound de activos off-market y mensajería SMS.
8. Endpoint AGNET RPC con verificación EIP-712 y liquidación USDC.
9. Oráculo geoespacial y `Gentrification Score`.

Estos componentes tampoco están presentes actualmente. Además, la implementación propuesta en esos documentos debe revisarse técnicamente, legalmente y desde el punto de vista de cumplimiento antes de incorporar capacidades de evasión WAF, skip tracing, SMS, DeFi o liquidación on-chain.

## 4. Diferencias y riesgos prioritarios

### Bloqueantes técnicos

- No existe el grafo LangGraph compilado ni una función de entrada para procesar una propiedad completa.
- No hay persistencia de propiedades, resultados de extracción, estados, errores o auditoría.
- La respuesta del worker no se devuelve al solicitante ni se publica en un almacén consultable.
- La ruta de ingesta usa `payload["query"]` y `payload["plataformas"]`; un payload incompleto produce error no controlado en vez de `400`.
- No existe validación Pydantic del request ni de las plataformas permitidas.
- `main.py` no registra rutas de telemetría ni mecanismos de health/readiness.
- El worker actual y el worker propuesto en los documentos utilizan contratos de Redis distintos; debe definirse una sola estrategia de consumer groups, reintentos, mensajes pendientes y dead-letter queue.
- La dependencia `langchain-openai` aparece declarada, pero no hay integración activa con el LLM; los documentos también usan `langchain_core` y otras dependencias que no están declaradas explícitamente.

### Seguridad y operaciones financieras

- La ingesta activa no tiene autenticación Zero-Trust.
- El webhook `/buy` no verifica que el comando provenga de un chat autorizado ni valida firma, nonce, monto, saldo, usuario o contexto de la oportunidad.
- El monto y destino de retiro están fijos en el código.
- No hay timeout, validación de respuesta, reintento seguro ni idempotency key para la llamada de liquidez.
- Los secretos tienen valores por defecto inseguros, incluido `FIRECRAWL_API_KEY="demo"`.
- El limitador de tasa permite continuar si Redis falla.
- La lógica de scraping debe operar respetando términos de servicio, robots, permisos, privacidad y legislación aplicable; no se debe implementar evasión de controles de acceso como objetivo del sistema.

### Calidad y mantenibilidad

- No hay pruebas unitarias, de integración, contrato de API o pruebas de worker visibles.
- No hay `Dockerfile`, `docker-compose`, migraciones, configuración de logging estructurado ni documentación de ejecución en el árbol revisado.
- El análisis visual es un placeholder: CAPEX fijo de 15000 USD.
- La alerta Telegram no comprueba el código de respuesta de Telegram ni maneja excepciones.
- El uso de solicitudes síncronas en `telegram_alert.py` dentro de un flujo potencialmente asíncrono puede bloquear el worker.

## 5. Plan de trabajo recomendado

### Fase A — Estabilizar el núcleo

1. Definir contratos Pydantic para request de barrido, respuesta, propiedad normalizada y estado de job.
2. Implementar validación de payload y errores HTTP consistentes.
3. Definir si `plataformas` contiene URLs o claves de un catálogo permitido; se recomienda catálogo validado.
4. Separar claramente API, worker y orquestador como procesos ejecutables.
5. Añadir health checks de API y Redis.
6. Crear almacenamiento de jobs, resultados, errores y estados.
7. Implementar reintentos controlados, dead-letter queue y recuperación de mensajes pendientes.

### Fase B — Seguridad antes de cualquier producción

1. Incorporar API key o mecanismo de identidad para la ingesta.
2. Restringir el webhook de Telegram a `TG_CHAT_ID` autorizado y verificar el origen.
3. Eliminar montos y destinos fijos; exigir una orden parametrizada, validada y autorizada.
4. Añadir confirmación de dos pasos para cualquier acción financiera.
5. Añadir idempotencia, límites de monto, timeouts, auditoría y modo simulación.
6. Mover secretos a variables seguras y eliminar valores de demostración.

### Fase C — Completar el flujo de análisis

1. Compilar el grafo LangGraph con estados tipados y transiciones explícitas.
2. Conectar extracción, normalización, análisis financiero, visión, auditoría legal y notificación.
3. Reemplazar el CAPEX fijo por una estimación trazable o marcarlo como dato no disponible.
4. Persistir cada decisión con sus entradas, fórmula, versión y resultado.
5. Implementar el seguimiento post-oferta y alertas solo después de contar con persistencia y pruebas.

### Fase D — Observabilidad y despliegue

1. Incorporar métricas Prometheus, logs estructurados, correlation/job IDs y trazas.
2. Añadir WebSocket o SSE de telemetría con autenticación y control de conexiones.
3. Crear Dockerfile, composición de servicios, configuración por entorno y readiness probes.
4. Añadir pruebas automatizadas y una prueba de carga controlada.

### Fase E — Integraciones avanzadas

Implementar únicamente tras revisión legal, de seguridad y de cumplimiento: contratos, escrow, servicios off-market, SMS, DeFi, EIP-712 y oráculos geoespaciales. Cada integración debe comenzar en sandbox, con permisos mínimos, límites y un interruptor de desactivación.

## 6. Criterios mínimos para considerar el sistema listo para pruebas integradas

- `POST /v1/openclaw/iniciar-barrido` rechaza solicitudes incompletas con `400` y requiere autenticación.
- Un job recibe un identificador, puede consultarse y termina en estado explícito: `queued`, `running`, `completed` o `failed`.
- Las propiedades extraídas se validan contra `DatosPropiedadNormalizados` y se guardan junto con el origen.
- El worker procesa mensajes sin pérdida, con ACK, reintentos y dead-letter queue.
- El grafo ejecuta todas las etapas previstas y guarda las decisiones.
- Telegram solo notifica a un chat autorizado y registra errores.
- `/buy` está desactivado por defecto en producción o funciona exclusivamente en sandbox con confirmación e idempotencia.
- Redis, Firecrawl y las variables de entorno tienen pruebas de conectividad y configuración documentada.
- Existen pruebas automatizadas para autenticación, validación, rate limiting, worker, análisis financiero y manejo de errores.

## 7. Conclusión

El proyecto actual es un esqueleto bien orientado para un pipeline inmobiliario asíncrono, pero todavía está entre la etapa de prototipo y la de integración. Las actualizaciones en los documentos describen una hoja de ruta amplia, no código ya instalado. La prioridad correcta es consolidar el núcleo, proteger el webhook financiero, implementar persistencia y orquestación, y después añadir telemetría y módulos avanzados de forma controlada.
