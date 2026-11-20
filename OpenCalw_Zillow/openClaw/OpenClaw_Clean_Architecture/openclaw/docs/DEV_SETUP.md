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
  -d '{"query": "Miami, FL", "plataformas": ["zillow", "redfin"]}'
```

## Endpoints

| Metodo | Ruta | Proposito | Estado |
|---|---|---|---|
| GET | `/health` | Liveness | Disponible |
| GET | `/readyz` | Readiness con chequeo de Redis | Disponible |
| POST | `/v1/openclaw/barridos` | Valida y encola un barrido | Disponible (S1-03) |
| GET | `/v1/openclaw/barridos/{job_id}` | Estado y progreso del barrido | Disponible (S1-04) |
| POST | `/v1/webhook/telegram-comando` | Comando `/buy` | 403 salvo `BUY_MODE=sandbox` |

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
