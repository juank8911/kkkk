# OpenClaw

API de ingesta y orquestacion para deteccion de oportunidades inmobiliarias.
Recibe una orden de barrido, la encola en Redis, extrae anuncios con Firecrawl y
los evalua con un grafo de decision.

## Estado

Sprint 0 cerrado: repositorio, CI, entorno Docker, configuracion tipada con
validacion al arranque, logging estructurado y endpoints de salud.
El sprint 1 (contratos y nucleo) empieza con S1-01 ya resuelto.

| Capacidad | Estado |
|---|---|
| API con liveness y readiness | Listo |
| Configuracion validada y sin secretos por defecto | Listo |
| Logging JSON con `job_id` | Listo |
| Docker multi-target (api y worker) | Listo |
| Contratos de ingesta validados | Sprint 1 |
| Persistencia PostgreSQL | Sprint 2 |
| Grafo LangGraph | Sprint 4 |

## Arranque rapido

```bash
cp .env.example .env          # completar FIRECRAWL_API_KEY y OPENCLAW_API_KEY
make up                       # Redis, PostgreSQL, API y worker
curl http://localhost:8000/health
curl http://localhost:8000/readyz
```

Sin Docker:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
uvicorn app.api.main:app --reload
```

## Comandos

| Comando | Que hace |
|---|---|
| `make up` / `make down` | Levanta o detiene el entorno completo |
| `make logs` | Sigue los logs de API y worker |
| `make test` | Suite de pruebas con cobertura |
| `make lint` / `make fmt` | Revisa o corrige estilo |
| `make typecheck` | `mypy` sobre el nucleo |

## Estructura

```
app/
  api/            Gateway FastAPI y rutas
  core/           Configuracion, logging y cliente Redis
  models/         Esquema de propiedad
  orchestrator/   Grafo y nodos de decision
  services/       Scraper y rate limiter
  workers/        Consumo del stream
docs/             Guia de desarrollo y ADR
tests/            Suite de pruebas
```

La guia detallada esta en `docs/DEV_SETUP.md` y las decisiones de arquitectura
en `docs/adr/`.
