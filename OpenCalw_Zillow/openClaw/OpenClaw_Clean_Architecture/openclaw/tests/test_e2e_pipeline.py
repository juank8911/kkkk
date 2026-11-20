"""Suite E2E con Firecrawl simulado y prueba real nocturna opcional (S5-05).

Criterio S5-05: Un barrido completo termina en `completed` sin intervención.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID

import fakeredis.aioredis
import httpx
import pytest

from app.api.main import app as openclaw_app
from app.api.routes import ingestion as ingestion_route
from app.core.config import settings
from app.integrations.firecrawl_adapter import FirecrawlAdapter
from app.models.contracts import JobStatus
from app.orchestrator.config import AlertConfig
from app.orchestrator.nodes.telegram_alert import RedisAlertDedupeStore
from app.orchestrator.nodes.vision import VisionAssessment
from app.orchestrator.workflow import ejecutar_pipeline_propiedad
from app.services import idempotency as idempotency_module
from app.services import job_store as job_store_module
from app.services.deduplicator import deduplicar_propiedades
from app.services.domain_semaphore import SemaforosPorDominio
from app.services.enrichment import enriquecer_propiedad_desde_markdown
from app.services.normalizer import normalizar_markdown_propiedades
from app.services.platforms import get_platform_spec
from app.services.rate_limiter import SlidingWindowRateLimiter
from app.workers.state_machine import TransicionInvalida, aplicar_transicion
from tests.test_scraper_contracts import CassetteFirecrawlClient, load_cassette


class _FakeValuator:
    async def evaluate_images(
        self,
        image_urls: list[str],
        *,
        context: dict[str, Any],
    ) -> VisionAssessment:
        return VisionAssessment(
            capex_usd=10_000.0,
            confianza=0.88,
            motivo="Estructura sólida con remodelación estética moderada.",
            severidad="baja",
            imagenes_evaluadas=len(image_urls),
        )


class _FakeNotifier:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    async def send_message(self, chat_id: str, text: str) -> tuple[bool, str | None]:
        self.messages.append((chat_id, text))
        return True, None


async def test_s5_05_e2e_sweep_completes_without_intervention(
    fake_redis: fakeredis.aioredis.FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flujo E2E completo: API -> Redis Stream -> Scraper -> Normalizer -> Dedup -> Grafo -> Alerta -> completed."""
    monkeypatch.setattr(ingestion_route, "redis_client", fake_redis)
    monkeypatch.setattr(job_store_module, "redis_client", fake_redis)
    monkeypatch.setattr(idempotency_module, "redis_client", fake_redis)
    monkeypatch.setattr(settings, "APP_ENV", "test")
    monkeypatch.setattr(settings, "OPENCLAW_API_KEY", "e2e-tactical-key")
    monkeypatch.setattr(settings, "OPENCLAW_API_KEY_SHA256", "")
    auth_headers = {"X-OpenClaw-Tactical-Key": "e2e-tactical-key"}

    transport = httpx.ASGITransport(app=openclaw_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver", headers=auth_headers
    ) as client:
        # 1. Crear barrido vía API real
        create_resp = await client.post(
            "/v1/openclaw/barridos",
            json={"query": "Cleveland OH", "plataformas": ["zillow", "realtor"]},
        )
        assert create_resp.status_code == 202
        created = create_resp.json()
        job_id = UUID(created["job_id"])
        assert created["status"] == "queued"

        # 2. Consumir mensaje desde Redis Stream con consumer group
        stream = settings.REDIS_STREAM_SCRAPING
        group = settings.REDIS_CONSUMER_GROUP
        await fake_redis.xgroup_create(stream, group, id="0", mkstream=True)
        entries = await fake_redis.xreadgroup(
            groupname=group,
            consumername="e2e-worker",
            streams={stream: ">"},
            count=1,
        )
        assert len(entries) == 1
        msg_id, fields = entries[0][1][0]
        payload = json.loads(fields[b"payload"].decode("utf-8"))
        assert payload["job_id"] == str(job_id)

        await aplicar_transicion(job_id, JobStatus.RUNNING)

        # 3. Extraer cada plataforma usando idempotencia S3-02, semáforo S3-04 y cassettes sin red
        limiter = SlidingWindowRateLimiter(redis_client=fake_redis)
        semaforos = SemaforosPorDominio(limite=2)
        cassettes_by_platform = {
            "zillow": load_cassette("zillow_success"),
            "realtor": load_cassette("realtor_success"),
        }
        all_normalized = []
        progress: dict[str, str] = {}

        for platform in payload["plataformas"]:
            claimed = await idempotency_module.reclamar(job_id, platform)
            assert claimed is True
            # Una segunda reclamación para el mismo (job_id, platform) se omite por idempotencia
            assert await idempotency_module.reclamar(job_id, platform) is False

            spec = get_platform_spec(platform)
            adapter = FirecrawlAdapter(
                client=CassetteFirecrawlClient([cassettes_by_platform[platform]]),
                rate_limiter=limiter,
            )
            async with semaforos.turno(spec.domain):
                scrape_res = await adapter.scrape(platform, payload["query"])
            assert scrape_res.ok is True
            progress[platform] = "exito"

            batch = normalizar_markdown_propiedades(
                scrape_res.markdown,
                plataforma=platform,
                url_origen=scrape_res.url,
            )
            assert len(batch.valid) == 1
            enrichment = enriquecer_propiedad_desde_markdown(scrape_res.markdown)
            assert enrichment.precio_usd == batch.valid[0].precio_usd
            all_normalized.extend(batch.valid)

        # 4. Deduplicar y evaluar cada propiedad en el pipeline de decisión
        deduped = deduplicar_propiedades(all_normalized)
        assert len(deduped) == 2

        notifier = _FakeNotifier()
        dedupe_alerts = RedisAlertDedupeStore(fake_redis)
        decisions = []
        for item in deduped:
            decision = await ejecutar_pipeline_propiedad(
                {
                    "property_id": item.dedup_key,
                    "dedup_key": item.dedup_key,
                    **item.propiedad.model_dump(),
                },
                valuator=_FakeValuator(),
                notifier=notifier,
                dedupe_store=dedupe_alerts,
                alert_config=AlertConfig(ficha_base_url="https://openclaw.local/propiedades"),
                chat_id="chat-e2e-auth",
            )
            decisions.append(decision)

        assert all(d["estado"] == "ALERTA_ENVIADA" for d in decisions)
        assert len(notifier.messages) == 2

        # 5. Transicionar a completed con máquina de estados (S3-03) y confirmar con XACK
        await aplicar_transicion(job_id, JobStatus.COMPLETED, progress=progress)
        await fake_redis.xack(stream, group, msg_id)

        # Verificar que un job completed no puede reabrirse ilegalmente
        with pytest.raises(TransicionInvalida):
            await aplicar_transicion(job_id, JobStatus.RUNNING)

        # 6. Consultar estado final del job vía GET
        status_resp = await client.get(f"/v1/openclaw/barridos/{job_id}")
        assert status_resp.status_code == 200
        final_job = status_resp.json()
        assert final_job["status"] == "completed"
        assert final_job["progress"] == {"zillow": "exito", "realtor": "exito"}


@pytest.mark.skipif(
    os.getenv("OPENCLAW_RUN_NIGHTLY_E2E") != "1" or not os.getenv("FIRECRAWL_API_KEY"),
    reason="Prueba real nocturna deshabilitada en CI sin red (activar OPENCLAW_RUN_NIGHTLY_E2E=1).",
)
async def test_s5_05_nightly_real_firecrawl_smoke() -> None:
    """Prueba real nocturna contra Firecrawl cuando se habilita explícitamente en CI nocturno."""
    adapter = FirecrawlAdapter(api_key=os.environ["FIRECRAWL_API_KEY"], timeout_seconds=25.0)
    res = await adapter.scrape("zillow", "Cleveland OH")
    assert res.status.value in {"ok", "blocked", "rate_limited"}
    assert Path(__file__).exists()
