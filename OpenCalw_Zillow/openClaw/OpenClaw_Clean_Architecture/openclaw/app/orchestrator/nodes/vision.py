"""Nodo de visión multimodal con salida estructurada y degradación segura (S4-05).

Reemplaza el CAPEX fijo de 15 000 USD por una evaluación trazable a través de
`ValuationPort`. Devuelve CAPEX estimado, nivel de confianza y motivo; cuando no
existen imágenes, el proveedor falla o la confianza es inferior al umbral,
degrada explícitamente a `no_disponible`.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from app.orchestrator.config import FinancialConfig, VisionConfig

logger = logging.getLogger("OpenClaw.Vision")


class VisionAssessment(BaseModel):
    """Salida estructurada exigida al LLM multimodal (o doble en pruebas)."""

    model_config = ConfigDict(extra="forbid")

    disponible: bool = True
    estado_vision: Literal["evaluado", "no_disponible"] = "evaluado"
    capex_usd: float | None = Field(default=None, ge=0.0)
    confianza: float = Field(..., ge=0.0, le=1.0)
    motivo: str = Field(..., min_length=3)
    severidad: Literal["baja", "media", "alta", "desconocida"] = "desconocida"
    imagenes_evaluadas: int = Field(default=0, ge=0)


@runtime_checkable
class ValuationPort(Protocol):
    """Puerto de valuación visual/CAPEX (sección 3 de PLAN_IMPLEMENTACION_2DEVS.md)."""

    async def evaluate_images(
        self,
        image_urls: list[str],
        *,
        context: dict[str, Any],
    ) -> VisionAssessment: ...


class MultimodalLLMValuationAdapter:
    """Adaptador de ValuationPort sobre LangChain OpenAI multimodal con salida estructurada."""

    def __init__(self, *, model_name: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self._model_name = model_name
        self._api_key = api_key

    async def evaluate_images(
        self,
        image_urls: list[str],
        *,
        context: dict[str, Any],
    ) -> VisionAssessment:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=self._model_name, api_key=self._api_key, temperature=0)
        structured_llm = llm.with_structured_output(VisionAssessment)
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Evalúa el estado físico de la propiedad inmobiliaria a partir de las "
                    "imágenes y estima el CAPEX en USD, tu nivel de confianza (0 a 1), "
                    "severidad (baja/media/alta) y motivo conciso.\n"
                    f"Contexto: ubicación={context.get('ubicacion')}, "
                    f"precio_usd={context.get('precio_usd')}."
                ),
            },
            *[{"type": "image_url", "image_url": {"url": url}} for url in image_urls],
        ]
        result = await structured_llm.ainvoke([{"role": "user", "content": content}])
        if isinstance(result, VisionAssessment):
            return result.model_copy(update={"imagenes_evaluadas": len(image_urls)})
        return VisionAssessment.model_validate(
            {**dict(result), "imagenes_evaluadas": len(image_urls)}
        )


def _no_disponible(motivo: str, *, imagenes_evaluadas: int = 0) -> dict[str, Any]:
    assessment = VisionAssessment(
        disponible=False,
        estado_vision="no_disponible",
        capex_usd=None,
        confianza=0.0,
        motivo=motivo,
        severidad="desconocida",
        imagenes_evaluadas=imagenes_evaluadas,
    )
    return {
        "estado": "VISION_NO_DISPONIBLE",
        "vision_disponible": False,
        "capex": None,
        "vision_confianza": 0.0,
        "vision_motivo": motivo,
        "vision_evaluacion": assessment.model_dump(),
    }


async def nodo_vision_artificial(
    state: dict[str, Any],
    *,
    valuator: ValuationPort | None = None,
    vision_config: VisionConfig | None = None,
    financial_config: FinancialConfig | None = None,
) -> dict[str, Any]:
    """Evalúa imágenes con LLM multimodal y ajusta el cap rate con el CAPEX estimado."""
    vcfg = vision_config if vision_config is not None else VisionConfig()
    fcfg = financial_config if financial_config is not None else FinancialConfig()

    raw_images = state.get("imagenes") or []
    images = [
        str(u).strip() for u in raw_images if str(u).strip().startswith(("http://", "https://"))
    ]
    if not images:
        return _no_disponible("No hay imágenes disponibles para inspección visual.")

    if valuator is None:
        return _no_disponible(
            "Proveedor de visión multimodal no configurado; marcado como no disponible.",
            imagenes_evaluadas=0,
        )

    selected_images = images[: vcfg.max_images]
    try:
        assessment = await valuator.evaluate_images(selected_images, context=state)
    except Exception as exc:
        logger.warning("vision_error_degradado", extra={"error": str(exc)})
        return _no_disponible(
            f"Error al consultar el evaluador visual ({type(exc).__name__}); degradado a no disponible.",
            imagenes_evaluadas=len(selected_images),
        )

    if (
        not assessment.disponible
        or assessment.estado_vision == "no_disponible"
        or assessment.capex_usd is None
        or assessment.confianza < vcfg.min_confidence
    ):
        motivo = (
            assessment.motivo
            if assessment.confianza >= vcfg.min_confidence
            else (
                f"Confianza visual insuficiente ({assessment.confianza:.2f} < "
                f"{vcfg.min_confidence:.2f}): {assessment.motivo}"
            )
        )
        return _no_disponible(motivo, imagenes_evaluadas=len(selected_images))

    precio = float(state.get("precio_usd") or 0.0)
    noi = float(state.get("noi") or 0.0)
    inversion_total = precio + assessment.capex_usd
    if inversion_total <= 0.0:
        return _no_disponible("Inversión total inválida para calcular cap rate ajustado.")

    cap_rate_ajustado = round((noi / inversion_total) * 100.0, 4)
    normalized_assessment = assessment.model_copy(
        update={
            "disponible": True,
            "estado_vision": "evaluado",
            "imagenes_evaluadas": assessment.imagenes_evaluadas or len(selected_images),
        }
    )

    if cap_rate_ajustado < fcfg.min_cap_rate_pct:
        return {
            "estado": "RECHAZADO",
            "vision_disponible": True,
            "capex": assessment.capex_usd,
            "cap_rate_ajustado": cap_rate_ajustado,
            "vision_confianza": assessment.confianza,
            "vision_motivo": assessment.motivo,
            "vision_evaluacion": normalized_assessment.model_dump(),
        }

    return {
        "estado": "INSPECCION_VISUAL_OK",
        "vision_disponible": True,
        "capex": assessment.capex_usd,
        "cap_rate_ajustado": cap_rate_ajustado,
        "vision_confianza": assessment.confianza,
        "vision_motivo": assessment.motivo,
        "vision_evaluacion": normalized_assessment.model_dump(),
    }
