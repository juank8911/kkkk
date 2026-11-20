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
| S2-06 a S2-09 | Bloque de Dev 2 | En curso por Dev 2 |

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

## Verificación

| Comprobación | Resultado |
|---|---|
| `pyright` (modo basic) sobre `app`, `tests`, `scripts` | 0 errores, 49 archivos |
| Importación de contratos y catálogo | Consistente entre `contracts.py` y `platforms.py` |
| Migraciones | Pendientes de ejecutar: en este equipo no hay Python utilizable |

Limitación del entorno: el intérprete de Python del equipo está bloqueado
(`Acceso denegado` en `Python313`) y el `.venv` del proyecto no arranca, por lo
que `pytest` y `alembic` no se pudieron ejecutar aquí. La validación se apoya en
análisis estático y el CI queda como puerta de calidad.

## Siguiente corte

1. S2-03: repositorios y unidad de trabajo por mensaje.
2. S2-05: webhook de Telegram con `secret_token`, allowlist y dedupe de `update_id`.
3. Revisión del bloque de Dev 2 (S2-06 a S2-09) y fusión en `dev1`.
