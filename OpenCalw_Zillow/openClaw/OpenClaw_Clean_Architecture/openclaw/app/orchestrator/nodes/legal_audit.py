"""Nodo de auditoría legal asistida (S4-06).

Detecta señales tempranas de gravámenes (liens), impuestos atrasados (back taxes),
ejecuciones hipotecarias (foreclosure / lis pendens), violaciones de código y
conflictos de título. Toda alerta legal generada exige revisión humana obligatoria.
"""

from __future__ import annotations

import re
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SignalCategory = Literal[
    "tax_lien",
    "hoa_lien",
    "mechanic_lien",
    "general_lien",
    "back_taxes",
    "foreclosure_lis_pendens",
    "code_violation",
    "title_cloud_probate",
]

_SIGNAL_RULES: Final[
    tuple[tuple[SignalCategory, Literal["media", "alta"], re.Pattern[str]], ...]
] = (
    (
        "tax_lien",
        "alta",
        re.compile(r"(?i)\b(?:tax\s+lien|tax\s+certificate|gravamen\s+fiscal)\b"),
    ),
    (
        "hoa_lien",
        "alta",
        re.compile(r"(?i)\b(?:hoa\s+lien|super[- ]priority\s+hoa|condo\s+lien)\b"),
    ),
    (
        "mechanic_lien",
        "alta",
        re.compile(r"(?i)\b(?:mechanic'?s?\s+lien|contractor\s+lien|construction\s+lien)\b"),
    ),
    (
        "general_lien",
        "alta",
        re.compile(
            r"(?i)\b(?:judgment\s+lien|municipal\s+lien|utility\s+lien|open\s+liens?|liens?\s+attached|embargo|gravamen)\b"
        ),
    ),
    (
        "back_taxes",
        "alta",
        re.compile(
            r"(?i)\b(?:back\s+taxes?|delinquent\s+taxes?|unpaid\s+taxes?|tax\s+default|"
            r"past[- ]due\s+taxes?|impuestos?\s+atrasados?|morosidad\s+fiscal)\b"
        ),
    ),
    (
        "foreclosure_lis_pendens",
        "alta",
        re.compile(
            r"(?i)\b(?:lis\s+pendens|pre[- ]?foreclosure|foreclosure\s+auction|"
            r"sheriff'?s?\s+sale|notice\s+of\s+default|ejecuci[oó]n\s+hipotecaria)\b"
        ),
    ),
    (
        "code_violation",
        "media",
        re.compile(r"(?i)\b(?:code\s+violations?|condemned|unpermitted\s+work|open\s+permits?)\b"),
    ),
    (
        "title_cloud_probate",
        "media",
        re.compile(
            r"(?i)\b(?:cloud\s+on\s+title|quitclaim\s+only|probate\s+sale|heir\s+dispute)\b"
        ),
    ),
)


class LegalSignal(BaseModel):
    """Señal individual de riesgo legal encontrada en la documentación del anuncio."""

    model_config = ConfigDict(extra="forbid")

    categoria: SignalCategory
    severidad: Literal["media", "alta"]
    evidencia: str = Field(..., min_length=2)


class LegalAuditResult(BaseModel):
    """Resultado estructurado de la auditoría legal asistida."""

    model_config = ConfigDict(extra="forbid")

    estado: Literal["AUDITORIA_LEGAL_OK", "AUDITORIA_LEGAL_ALERTA"]
    riesgo: Literal["bajo", "medio", "alto"]
    senales_detectadas: list[LegalSignal] = Field(default_factory=list)
    requiere_revision_humana: bool = False
    resumen: str

    @model_validator(mode="after")
    def _enforce_human_review_on_alerts(self) -> LegalAuditResult:
        if (self.estado == "AUDITORIA_LEGAL_ALERTA" or self.senales_detectadas) and (
            not self.requiere_revision_humana
        ):
            raise ValueError("Toda alerta legal debe exigir revisión humana obligatoria.")
        return self


def auditar_riesgo_legal(texto_fuente: str) -> LegalAuditResult:
    """Analiza texto libre del anuncio/notas en busca de señales legales de riesgo."""
    texto = (texto_fuente or "").strip()
    senales: list[LegalSignal] = []
    seen: set[SignalCategory] = set()

    for categoria, severidad, pattern in _SIGNAL_RULES:
        match = pattern.search(texto)
        if match and categoria not in seen:
            seen.add(categoria)
            start = max(0, match.start() - 30)
            end = min(len(texto), match.end() + 40)
            snippet = texto[start:end].replace("\n", " ").strip()
            senales.append(
                LegalSignal(
                    categoria=categoria,
                    severidad=severidad,
                    evidencia=snippet,
                )
            )

    if not senales:
        return LegalAuditResult(
            estado="AUDITORIA_LEGAL_OK",
            riesgo="bajo",
            senales_detectadas=[],
            requiere_revision_humana=False,
            resumen="Sin señales explícitas de liens, back taxes o conflictos legales en el texto.",
        )

    tiene_alta = any(s.severidad == "alta" for s in senales)
    riesgo: Literal["medio", "alto"] = "alto" if tiene_alta else "medio"
    categorias_str = ", ".join(s.categoria for s in senales)
    return LegalAuditResult(
        estado="AUDITORIA_LEGAL_ALERTA",
        riesgo=riesgo,
        senales_detectadas=senales,
        requiere_revision_humana=True,
        resumen=(
            f"Alerta legal asistida ({riesgo}): señales detectadas [{categorias_str}]. "
            "Requiere revisión humana y verificación de título antes de ofertar."
        ),
    )


def nodo_auditoria_legal(state: dict[str, Any]) -> dict[str, Any]:
    """Nodo del grafo que ejecuta la auditoría legal asistida sobre el estado."""
    partes = [
        str(state.get("descripcion_cruda") or ""),
        str(state.get("notas_legales") or ""),
        str(state.get("impuestos_texto") or ""),
    ]
    resultado = auditar_riesgo_legal("\n".join(p for p in partes if p.strip()))
    return {
        "estado_legal": resultado.estado,
        "riesgo_legal": resultado.riesgo,
        "requiere_revision_humana": resultado.requiere_revision_humana,
        "senales_legales": [s.model_dump() for s in resultado.senales_detectadas],
        "auditoria_legal": resultado.model_dump(),
    }
