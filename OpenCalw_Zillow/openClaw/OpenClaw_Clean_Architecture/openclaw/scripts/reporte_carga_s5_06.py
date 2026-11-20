"""Script ejecutable de prueba de carga controlada de 100 jobs (S5-06).

Ejecuta 100 jobs concurrentes controlados sin red (usando `fakeredis` y cassettes),
verifica 0 errores no controlados y opcionalmente documenta las métricas de
latencia p50/p95/p99 en Markdown.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fakeredis.aioredis  # noqa: E402

from app.integrations.firecrawl_adapter import FirecrawlAdapter  # noqa: E402
from app.models.contracts import BarridoRequest, JobResponse, JobStatus  # noqa: E402
from app.orchestrator.nodes.telegram_alert import InMemoryAlertDedupeStore  # noqa: E402
from app.orchestrator.nodes.vision import VisionAssessment  # noqa: E402
from app.orchestrator.workflow import ejecutar_pipeline_propiedad  # noqa: E402
from app.services.normalizer import normalizar_propiedad_unica  # noqa: E402
from app.services.rate_limiter import SlidingWindowRateLimiter  # noqa: E402

CASSETTE_PATH = PROJECT_ROOT / "tests" / "cassettes" / "scraper" / "zillow_success.json"


class _StaticCassetteClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    async def scrape_url(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._response


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
            motivo="Evaluación de carga controlada S5-06.",
            severidad="baja",
            imagenes_evaluadas=len(image_urls),
        )


class _SilentNotifier:
    async def send_message(self, chat_id: str, text: str) -> tuple[bool, str | None]:
        return True, None


async def ejecutar_carga_controlada(
    *,
    total_jobs: int = 100,
    max_concurrency: int = 20,
) -> dict[str, Any]:
    """Ejecuta `total_jobs` jobs concurrentes y devuelve reporte estructurado de latencias."""
    fake_redis = fakeredis.aioredis.FakeRedis()
    cassette = json.loads(CASSETTE_PATH.read_text(encoding="utf-8"))
    limiter = SlidingWindowRateLimiter(
        redis_client=fake_redis,
        limits={"zillow.com": total_jobs * 2, "default": total_jobs * 2},
    )
    sem = asyncio.Semaphore(max_concurrency)
    valuator = _FastValuator()
    notifier = _SilentNotifier()
    dedupe = InMemoryAlertDedupeStore()

    latencies_ms: list[float] = []
    unhandled_errors = 0

    async def _run_job(idx: int) -> JobResponse:
        nonlocal unhandled_errors
        async with sem:
            t0 = time.perf_counter()
            try:
                req = BarridoRequest(query=f"Cleveland OH {idx}", plataformas=["zillow"])
                job = JobResponse(
                    job_id=uuid4(),
                    status=JobStatus.QUEUED,
                    query=req.query,
                    plataformas=req.plataformas,
                )
                adapter = FirecrawlAdapter(
                    client=_StaticCassetteClient(cassette["response"]),
                    rate_limiter=limiter,
                )
                scraped = await adapter.scrape("zillow", req.query)
                if not scraped.ok:
                    raise RuntimeError(f"Scrape fallido: {scraped.status}")
                prop = normalizar_propiedad_unica(
                    scraped.markdown,
                    plataforma="zillow",
                    url_origen=scraped.url,
                )
                decision = await ejecutar_pipeline_propiedad(
                    {
                        "property_id": f"load-{idx}",
                        "dedup_key": f"load-{idx}",
                        **prop.model_dump(),
                    },
                    valuator=valuator,
                    notifier=notifier,
                    dedupe_store=dedupe,
                    chat_id="chat-load",
                )
                if decision["estado"] != "ALERTA_ENVIADA":
                    raise RuntimeError(f"Estado inesperado: {decision['estado']}")
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                return job.model_copy(
                    update={"status": JobStatus.COMPLETED, "progress": {"zillow": "exito"}}
                )
            except Exception:
                unhandled_errors += 1
                raise

    started_all = time.perf_counter()
    try:
        jobs = await asyncio.gather(*(_run_job(i) for i in range(total_jobs)))
    finally:
        await fake_redis.aclose()

    elapsed_sec = max(0.001, time.perf_counter() - started_all)
    ordered = sorted(latencies_ms)
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "total_jobs": total_jobs,
        "completed_jobs": sum(1 for j in jobs if j.status == JobStatus.COMPLETED),
        "max_concurrency": max_concurrency,
        "unhandled_errors": unhandled_errors,
        "elapsed_seconds": round(elapsed_sec, 3),
        "throughput_jobs_per_sec": round(total_jobs / elapsed_sec, 2),
        "latency_ms": {
            "min": round(ordered[0], 2),
            "p50": round(statistics.median(ordered), 2),
            "p95": round(ordered[int(0.95 * len(ordered)) - 1], 2),
            "p99": round(ordered[int(0.99 * len(ordered)) - 1], 2),
            "max": round(ordered[-1], 2),
        },
    }


def formatear_markdown(report: dict[str, Any]) -> str:
    lat = report["latency_ms"]
    return (
        "# Reporte de Prueba de Carga Controlada — OpenClaw (S5-06)\n\n"
        f"- **Fecha (UTC):** `{report['timestamp_utc']}`\n"
        f"- **Script:** `scripts/reporte_carga_s5_06.py`\n"
        f"- **Jobs ejecutados:** `{report['completed_jobs']}/{report['total_jobs']}`\n"
        f"- **Concurrencia máxima:** `{report['max_concurrency']}`\n"
        f"- **Errores no controlados:** `{report['unhandled_errors']}`\n"
        f"- **Duración total:** `{report['elapsed_seconds']} s`\n"
        f"- **Throughput:** `{report['throughput_jobs_per_sec']} jobs/s`\n\n"
        "## Latencia por job (ms)\n\n"
        "| Percentil | Latencia (ms) |\n"
        "|---|---|\n"
        f"| Min | {lat['min']} |\n"
        f"| p50 | {lat['p50']} |\n"
        f"| p95 | {lat['p95']} |\n"
        f"| p99 | {lat['p99']} |\n"
        f"| Max | {lat['max']} |\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prueba de carga controlada de 100 jobs (S5-06).")
    parser.add_argument("--jobs", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--write-doc", type=Path, default=None)
    args = parser.parse_args()

    report = asyncio.run(
        ejecutar_carga_controlada(total_jobs=args.jobs, max_concurrency=args.concurrency)
    )
    print(json.dumps(report, indent=2))
    if args.write_doc is not None:
        args.write_doc.parent.mkdir(parents=True, exist_ok=True)
        args.write_doc.write_text(formatear_markdown(report), encoding="utf-8")
    return 0 if report["unhandled_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
