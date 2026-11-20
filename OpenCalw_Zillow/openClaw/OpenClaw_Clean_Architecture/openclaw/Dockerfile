# Imagen unica con objetivos: migrate, api, worker y offer-tracker.
# Uso: docker build --target api -t openclaw-api . | docker build --target migrate -t openclaw-migrate .

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home /app app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app

USER app

EXPOSE 8000

FROM base AS migrate

HEALTHCHECK NONE

CMD ["alembic", "upgrade", "head"]

FROM base AS api

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS worker

HEALTHCHECK NONE

CMD ["python", "-m", "app.workers.ingestion_worker"]

FROM base AS offer-tracker

HEALTHCHECK NONE

CMD ["python", "-m", "app.workers.offer_tracker"]
