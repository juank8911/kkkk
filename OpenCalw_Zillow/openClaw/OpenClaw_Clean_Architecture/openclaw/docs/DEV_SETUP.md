# Guia de desarrollo

## Requisitos

| Herramienta | Version minima | Nota |
|---|---|---|
| Python | 3.12 | El proyecto usa `X \| None` y `match` |
| Docker Desktop | 24 | Para Redis, PostgreSQL y los contenedores de API y worker |
| Make | 4 | Opcional en Windows; los comandos estan en el `Makefile` |

## Puesta en marcha

1. Copiar `.env.example` a `.env` y completar `FIRECRAWL_API_KEY` y `OPENCLAW_API_KEY`.
2. Ejecutar `make up`. El primer arranque construye las imagenes y tarda unos minutos.
3. Verificar con `curl http://localhost:8000/health` y `curl http://localhost:8000/readyz`.
4. Enviar un barrido de prueba:

```bash
curl -X POST http://localhost:8000/v1/openclaw/barridos \
  -H "Content-Type: application/json" \
  -H "X-OpenClaw-Tactical-Key: $OPENCLAW_API_KEY" \
  -d '{"query": "Miami, FL", "plataformas": ["zillow", "redfin"]}'
```

5. Aplicar las migraciones de base de datos:

```bash
alembic upgrade head
```

## Autenticacion

Todo endpoint de negocio exige la cabecera `X-OpenClaw-Tactical-Key`. El
servicio nunca compara la clave en claro: guarda y contrasta su SHA-256 con
`hmac.compare_digest`. Respuestas posibles:

| Situacion | Codigo | `code` del cuerpo |
|---|---|---|
| Falta la cabecera | 401 | `unauthorized` |
| La clave no coincide | 403 | `forbidden` |
| El servicio no tiene clave configurada | 503 | `service_unavailable` |

En produccion se recomienda informar `OPENCLAW_API_KEY_SHA256` (huella
precomputada) en lugar de la clave en claro. `/health` y `/readyz` siguen
siendo publicos para las sondas del orquestador.

## Base de datos

El esquema vive en `app/db/models.py` y se versiona con Alembic:

| Revision | Contenido |
|---|---|
| `0001_esquema_inicial` | `jobs`, `propiedades`, `decisiones`, `eventos`, `plataformas` |
| `0002_semilla_plataformas` | Semilla idempotente de las 9 plataformas del catalogo |

```bash
alembic upgrade head     # aplicar
alembic downgrade -1     # revertir la ultima
alembic current          # revision aplicada
```

## Endpoints

| Metodo | Ruta | Proposito | Estado |
|---|---|---|---|
| GET | `/health` | Liveness | Disponible |
| GET | `/readyz` | Readiness con chequeo de Redis | Disponible |
| POST | `/v1/openclaw/barridos` | Valida y encola un barrido | Disponible (S1-03) |
| GET | `/v1/openclaw/barridos/{job_id}` | Estado y progreso del barrido | Disponible (S1-04) |
| POST | `/v1/webhook/telegram-comando` | Comando `/buy` | 403 salvo `BUY_MODE=sandbox` |
| GET | `/v1/openclaw/barridos` | (pendiente) listado paginado de jobs | Sprint 3 |

Salvo `/health` y `/readyz`, todas las rutas exigen la cabecera
`X-OpenClaw-Tactical-Key`.

Las plataformas se envian como claves del catalogo cerrado (`zillow`, `realtor`,
`redfin`, `trulia`, `loopnet`, `craigslist`, `century21`, `coldwell`, `remax`),
nunca como URL. Todos los errores usan el contrato `ErrorResponse` con los
campos `code`, `message`, `details` y `timestamp`.

## Variables de entorno

Todas se leen con `pydantic-settings` desde el entorno y desde `.env`. Los
secretos no tienen valor por defecto utilizable: en `APP_ENV=prod` el servicio
no arranca si falta alguno.

## Pruebas

```bash
make test          # suite completa con cobertura
pytest -k health   # solo un grupo
```

Las pruebas no necesitan Redis real: el ping se sustituye con `monkeypatch`.
Cuando el sprint 3 agregue el worker, se usara `fakeredis`.

## Problemas frecuentes

| Sintoma | Causa probable | Solucion |
|---|---|---|
| `Configuracion incompleta` al arrancar | Falta una clave en `.env` | Completar `FIRECRAWL_API_KEY` y `OPENCLAW_API_KEY` |
| `/readyz` devuelve 503 | Redis no responde | `docker compose -f docker-compose.dev.yml ps` y revisar logs |
| El worker no procesa nada | No hay mensajes en el stream | Enviar un barrido por la API |
| `ModuleNotFoundError: app` | Se ejecuta desde otra carpeta | Situarse en la raiz del proyecto |
