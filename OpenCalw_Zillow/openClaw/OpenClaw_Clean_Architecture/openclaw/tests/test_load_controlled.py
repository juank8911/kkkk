"""Prueba de carga controlada de 100 jobs sin errores no controlados (S5-06).

Criterio S5-06: 100 jobs sin errores no controlados; latencia reportada.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from typing import Any
from uuid import uuid4

import fakeredis.aioredis

from app.integrations.firecrawl_adapter import FirecrawlAdapter
from app.models.contracts import BarridoRequest, JobResponse, JobStatus
from app.orchestrator.nodes.telegram_alert import InMemoryAlertDedupeStore
from app.orchestrator.nodes.vision import VisionAssessment
from app.orchestrator.workflow import ejecutar_pipeline_propiedad
from app.services.normalizer import normalizar_propiedad_unica
from app.services.rate_limiter import SlidingWindowRateLimiter
from tests.test_scraper_contracts import CassetteFirecrawlClient, load_cassette


class _FastValuator:
    async def evaluate_images(
        self,
        image_urls: list[str],
        *,
        context: dict[str, Any],
    ) -> VisionAssessment:
        return VisionAssessment(
            capex_usd=9_000.0,
            confianza=0.9,
            motivo="Evaluación de carga controlada.",
            severidad="baja",
            imagenes_evaluadas=len(image_urls),
        )


class _SilentNotifier:
    async def send_message(self, chat_id: str, text: str) -> tuple[bool, str | None]:
        return True, None


async def test_s5_06_controlled_load_100_jobs_zero_unhandled_errors(
    fake_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    """Ejecuta 100 jobs concurrentes controlados y reporta latencias p50/p95/p99."""
    total_jobs = 100
    concurrency_sem = asyncio.Semaphore(20)
    cassette = load_cassette("zillow_success")
    limiter = SlidingWindowRateLimiter(
        redis_client=fake_redis,
        limits={"zillow.com": 500, "default": 500},
    )
    valuator = _FastValuator()
    notifier = _SilentNotifier()
    dedupe = InMemoryAlertDedupeStore()

    latencies_ms: list[float] = []
    completed_jobs: list[JobResponse] = []

    async def run_one_job(idx: int) -> None:
        async with concurrency_sem:
            started = time.perf_counter()
            req = BarridoRequest(query=f"Cleveland OH {idx}", plataformas=["zillow"])
            job = JobResponse(
                job_id=uuid4(),
                status=JobStatus.QUEUED,
                query=req.query,
                plataformas=req.plataformas,
            )

            adapter = FirecrawlAdapter(
                client=CassetteFirecrawlClient([cassette]),
                rate_limiter=limiter,
            )
            scraped = await adapter.scrape("zillow", req.query)
            assert scraped.ok is True

            prop = normalizar_propiedad_unica(
                scraped.markdown,
                plataforma="zillow",
                url_origen=scraped.url,
            )
            decision = await ejecutar_pipeline_propiedad(
                {
                    "property_id": f"load-prop-{idx}",
                    "dedup_key": f"load-prop-{idx}",
                    **prop.model_dump(),
                },
                valuator=valuator,
                notifier=notifier,
                dedupe_store=dedupe,
                chat_id="chat-load",
            )
            assert decision["estado"] == "ALERTA_ENVIADA"

            done_job = job.model_copy(
                update={"status": JobStatus.COMPLETED, "progress": {"zillow": "exito"}}
            )
            completed_jobs.append(done_job)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)

    t0 = time.perf_counter()
    await asyncio.gather(*(run_one_job(i) for i in range(total_jobs)))
    total_sec = max(0.001, time.perf_counter() - t0)

    assert len(completed_jobs) == total_jobs
    assert all(j.status == JobStatus.COMPLETED for j in completed_jobs)

    ordered = sorted(latencies_ms)
    p50 = statistics.median(ordered)
    p95 = ordered[int(0.95 * len(ordered)) - 1]
    p99 = ordered[int(0.99 * len(ordered)) - 1]
    throughput = total_jobs / total_sec

    report = {
        "total_jobs": total_jobs,
        "unhandled_errors": 0,
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "max_ms": round(ordered[-1], 2),
        "throughput_jobs_per_sec": round(throughput, 2),
    }
    print(f"LOAD_TEST_REPORT={report}")
    assert report["unhandled_errors"] == 0
    assert report["p99_ms"] > 0
