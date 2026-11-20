.PHONY: help up down logs ps test lint fmt typecheck build api worker

help: ## Muestra los comandos disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

up: ## Levanta Redis, PostgreSQL, API y worker en segundo plano
	docker compose -f docker-compose.dev.yml up -d --build

down: ## Detiene el entorno y conserva los volumenes
	docker compose -f docker-compose.dev.yml down

logs: ## Sigue los logs de API y worker
	docker compose -f docker-compose.dev.yml logs -f api worker

ps: ## Muestra el estado de los servicios
	docker compose -f docker-compose.dev.yml ps

build: ## Construye las imagenes de API y worker
	docker build --target api -t openclaw-api:dev .
	docker build --target worker -t openclaw-worker:dev .

test: ## Ejecuta la suite de pruebas con cobertura
	pytest --cov=app --cov-report=term-missing

lint: ## Revisa estilo y errores sin modificar archivos
	ruff check app tests
	black --check app tests

fmt: ## Formatea y corrige lo que sea seguro corregir
	ruff check --fix app tests
	black app tests

typecheck: ## Comprueba tipos en el nucleo
	mypy

api: ## Arranca solo la API en local (requiere entorno virtual)
	uvicorn app.api.main:app --reload --port 8000

worker: ## Arranca solo el worker en local
	python -m app.workers.ingestion_worker
