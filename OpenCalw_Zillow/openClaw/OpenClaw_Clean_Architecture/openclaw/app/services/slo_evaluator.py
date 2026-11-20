"""Evaluador determinista de SLOs y alertas operativas para pruebas E2E — Fase 100 (Dev 2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OperationalMetricsSnapshot(BaseModel):
    """Instantánea de métricas de negocio y operación para evaluación de SLOs."""

    model_config = ConfigDict(extra="forbid")

    p95_stage_latency_seconds: float = Field(..., ge=0.0)
    dlq_depth: int = Field(..., ge=0)
    source_failure_rate_per_sec: float = Field(..., ge=0.0)
    decisions_total: int = Field(..., ge=0)
    decisions_error: int = Field(..., ge=0)


class TriggeredAlert(BaseModel):
    """Alerta disparada cuando se incumple un umbral operativo de Fase 100."""

    model_config = ConfigDict(extra="forbid")

    name: str
    severity: Literal["warning", "critical"]
    observed_value: float
    threshold: float
    message: str


def evaluar_alertas_slo(
    snapshot: OperationalMetricsSnapshot,
    *,
    max_p95_latency_seconds: float = 15.0,
    max_source_failure_rate: float = 0.2,
    max_decision_error_ratio: float = 0.05,
) -> list[TriggeredAlert]:
    """Evalúa los 4 umbrales definidos en `prometheus_alerts_openclaw.yml`."""
    alerts: list[TriggeredAlert] = []

    if snapshot.p95_stage_latency_seconds > max_p95_latency_seconds:
        alerts.append(
            TriggeredAlert(
                name="OpenClawHighPipelineLatencyP95",
                severity="warning",
                observed_value=snapshot.p95_stage_latency_seconds,
                threshold=max_p95_latency_seconds,
                message="Latencia p95 superior al SLO operativo.",
            )
        )

    if snapshot.dlq_depth > 0:
        alerts.append(
            TriggeredAlert(
                name="OpenClawDlqDepthNonZero",
                severity="critical",
                observed_value=float(snapshot.dlq_depth),
                threshold=0.0,
                message="Profundidad de DLQ mayor que cero.",
            )
        )

    if snapshot.source_failure_rate_per_sec > max_source_failure_rate:
        alerts.append(
            TriggeredAlert(
                name="OpenClawSourceFailureSpike",
                severity="warning",
                observed_value=snapshot.source_failure_rate_per_sec,
                threshold=max_source_failure_rate,
                message="Tasa de fallos por fuente externa por encima del umbral.",
            )
        )

    if snapshot.decisions_total > 0:
        ratio = snapshot.decisions_error / snapshot.decisions_total
        if ratio > max_decision_error_ratio:
            alerts.append(
                TriggeredAlert(
                    name="OpenClawDecisionErrorRateHigh",
                    severity="critical",
                    observed_value=round(ratio, 4),
                    threshold=max_decision_error_ratio,
                    message="Tasa de decisiones con veredicto error por encima del 5%.",
                )
            )

    return alerts
