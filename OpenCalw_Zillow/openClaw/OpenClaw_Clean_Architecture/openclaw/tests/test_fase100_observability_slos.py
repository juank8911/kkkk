"""Pruebas de Fase 100 (Dev 2): dashboards Grafana, reglas Prometheus y disparo de alertas SLO."""

from __future__ import annotations

import json
from pathlib import Path

from app.services.slo_evaluator import OperationalMetricsSnapshot, evaluar_alertas_slo

OBS_DIR = Path(__file__).resolve().parents[1] / "deploy" / "observability"


def test_fase100_grafana_dashboard_and_prometheus_alerts_are_valid() -> None:
    dashboard_path = OBS_DIR / "grafana_dashboard_openclaw.json"
    alerts_path = OBS_DIR / "prometheus_alerts_openclaw.yml"

    assert dashboard_path.exists()
    assert alerts_path.exists()

    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    assert dashboard["uid"] == "openclaw-business-ops-f100"
    titles = " ".join(p["title"].lower() for p in dashboard["panels"])
    for expected_topic in ("jobs", "propiedades", "decisiones", "ofertas", "dlq", "errores"):
        assert expected_topic in titles

    alerts_text = alerts_path.read_text(encoding="utf-8")
    for alert_name in (
        "OpenClawHighPipelineLatencyP95",
        "OpenClawDlqDepthNonZero",
        "OpenClawSourceFailureSpike",
        "OpenClawDecisionErrorRateHigh",
    ):
        assert alert_name in alerts_text


def test_fase100_slo_alerts_trigger_in_controlled_fault_scenarios() -> None:
    healthy = OperationalMetricsSnapshot(
        p95_stage_latency_seconds=1.2,
        dlq_depth=0,
        source_failure_rate_per_sec=0.0,
        decisions_total=100,
        decisions_error=1,
    )
    assert evaluar_alertas_slo(healthy) == []

    degraded = OperationalMetricsSnapshot(
        p95_stage_latency_seconds=22.5,
        dlq_depth=3,
        source_failure_rate_per_sec=0.45,
        decisions_total=100,
        decisions_error=8,
    )
    fired = evaluar_alertas_slo(degraded)
    assert {a.name for a in fired} == {
        "OpenClawHighPipelineLatencyP95",
        "OpenClawDlqDepthNonZero",
        "OpenClawSourceFailureSpike",
        "OpenClawDecisionErrorRateHigh",
    }
