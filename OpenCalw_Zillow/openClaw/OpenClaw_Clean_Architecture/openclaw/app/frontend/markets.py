"""Catálogo y puntuación de atractivo (%) de los 30 mercados objetivo para el panel.

El porcentaje de atractivo (0 a 100 %) es una métrica financiera y operativa
explicable y determinista basada únicamente en precio de compra, renta mensual
estimada, rendimiento bruto anual y factores operativos/fiscales explícitos.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class _MercadoRaw:
    puesto_lista: int
    categoria: str
    ciudad: str
    estado: str
    query: str
    precio_min_usd: int
    precio_max_usd: int
    renta_min_usd: int
    renta_max_usd: int
    factor_operativo: int
    notas: str


class MercadoAtractivo(BaseModel):
    """Mercado objetivo con métricas derivadas y % de atractivo calculado."""

    ranking: int = Field(..., ge=1)
    puesto_lista: int = Field(..., ge=1)
    categoria: str
    ciudad: str
    estado: str
    nombre_completo: str
    query: str
    precio_min_usd: int = Field(..., gt=0)
    precio_max_usd: int = Field(..., gt=0)
    precio_medio_usd: float = Field(..., gt=0)
    renta_min_usd: int = Field(..., gt=0)
    renta_max_usd: int = Field(..., gt=0)
    renta_media_usd: float = Field(..., gt=0)
    ratio_renta_precio_pct: float = Field(..., gt=0)
    yield_bruto_anual_pct: float = Field(..., gt=0)
    atractivo_pct: float = Field(..., ge=0.0, le=100.0)
    notas: str


CAT_1 = "Puestos 1 al 10: Máximo flujo y menor costo"
CAT_2 = "Puestos 11 al 20: Equilibrio entre flujo y estabilidad"
CAT_3 = "Puestos 21 al 30: Mercados secundarios con potencial"

_MERCADOS_BASE: tuple[_MercadoRaw, ...] = (
    # Puestos 1 al 10: Máximo flujo y menor costo
    _MercadoRaw(
        1,
        CAT_1,
        "Detroit",
        "Míchigan",
        "Detroit, MI",
        76_000,
        88_000,
        1_350,
        1_350,
        -4,
        "Unos $76k a $88k de compra y renta cerca de $1,350; alto flujo bruto, pero exige filtro estricto de calle y reforma.",
    ),
    _MercadoRaw(
        2,
        CAT_1,
        "Cleveland",
        "Ohio",
        "Cleveland, OH",
        95_000,
        121_000,
        1_280,
        1_550,
        4,
        "Unos $95k a $121k de compra y renta de $1,280 a $1,550; alta demanda de vales y buen inventario de casas o dúplex.",
    ),
    _MercadoRaw(
        3,
        CAT_1,
        "Flint",
        "Míchigan",
        "Flint, MI",
        65_000,
        65_000,
        1_120,
        1_120,
        2,
        "Cerca de $65k de compra y $1,120 de renta; entrada mínima, ideal para comprar al contado y evitar intereses.",
    ),
    _MercadoRaw(
        4,
        CAT_1,
        "Toledo",
        "Ohio",
        "Toledo, OH",
        85_000,
        85_000,
        1_145,
        1_145,
        4,
        "Cerca de $85k de compra y $1,145 de renta; buen uso de vales y costos operativos contenidos.",
    ),
    _MercadoRaw(
        5,
        CAT_1,
        "Birmingham",
        "Alabama",
        "Birmingham, AL",
        95_000,
        95_000,
        1_266,
        1_266,
        5,
        "Cerca de $95k de compra y $1,266 de renta; impuestos prediales bajos y fuerte presencia de Sección 8.",
    ),
    _MercadoRaw(
        6,
        CAT_1,
        "Memphis",
        "Tennessee",
        "Memphis, TN",
        110_000,
        145_000,
        1_300,
        1_300,
        5,
        "Entre $110k y $145k de compra y $1,300 de renta; sin impuesto estatal sobre la renta y mucha demanda subsidiada.",
    ),
    _MercadoRaw(
        7,
        CAT_1,
        "Jackson",
        "Misisipi",
        "Jackson, MS",
        80_000,
        80_000,
        1_289,
        1_289,
        4,
        "Cerca de $80k de compra y $1,289 de renta; excelente relación renta a precio y miles de vales activos.",
    ),
    _MercadoRaw(
        8,
        CAT_1,
        "Saginaw",
        "Míchigan",
        "Saginaw, MI",
        70_000,
        70_000,
        1_080,
        1_080,
        2,
        "Cerca de $70k de compra y $1,080 de renta; opción barata en Míchigan con menos competencia.",
    ),
    _MercadoRaw(
        9,
        CAT_1,
        "Dayton",
        "Ohio",
        "Dayton, OH",
        95_000,
        95_000,
        1_180,
        1_180,
        3,
        "Cerca de $95k de compra y $1,180 de renta; flujo estable apoyado en empleo aeronáutico y salud.",
    ),
    _MercadoRaw(
        10,
        CAT_1,
        "Akron",
        "Ohio",
        "Akron, OH",
        100_000,
        100_000,
        1_200,
        1_200,
        3,
        "Cerca de $100k de compra y $1,200 de renta; mercado cercano a Cleveland con casas sólidas y buena renta.",
    ),
    # Puestos 11 al 20: Equilibrio entre flujo y estabilidad
    _MercadoRaw(
        11,
        CAT_2,
        "St. Louis",
        "Misuri",
        "St. Louis, MO",
        115_000,
        180_000,
        1_280,
        1_280,
        -1,
        "Entre $115k y $180k de compra y $1,280 de renta; conviene elegir códigos postales con renta SAFMR alta.",
    ),
    _MercadoRaw(
        12,
        CAT_2,
        "Baltimore",
        "Maryland",
        "Baltimore, MD",
        120_000,
        120_000,
        1_650,
        1_650,
        -4,
        "Cerca de $120k de compra y $1,650 de renta; rentas altas frente al precio, pero vigila impuestos y reglas locales.",
    ),
    _MercadoRaw(
        13,
        CAT_2,
        "Pittsburgh",
        "Pensilvania",
        "Pittsburgh, PA",
        115_000,
        115_000,
        1_250,
        1_250,
        3,
        "Cerca de $115k de compra y $1,250 de renta; plaza estable con demanda sólida de dos y tres habitaciones.",
    ),
    _MercadoRaw(
        14,
        CAT_2,
        "Youngstown",
        "Ohio",
        "Youngstown, OH",
        75_000,
        75_000,
        1_050,
        1_050,
        2,
        "Cerca de $75k de compra y $1,050 de renta; precio de entrada bajo para buscar flujo neto rápido.",
    ),
    _MercadoRaw(
        15,
        CAT_2,
        "Little Rock",
        "Arkansas",
        "Little Rock, AR",
        115_000,
        115_000,
        1_180,
        1_180,
        4,
        "Cerca de $115k de compra y $1,180 de renta; marco legal ágil para propietarios y desalojos más rápidos.",
    ),
    _MercadoRaw(
        16,
        CAT_2,
        "Augusta",
        "Georgia",
        "Augusta, GA",
        120_000,
        120_000,
        1_329,
        1_329,
        4,
        "Cerca de $120k de compra y $1,329 de renta; ocupación firme impulsada por sectores militar y médico.",
    ),
    _MercadoRaw(
        17,
        CAT_2,
        "Macon",
        "Georgia",
        "Macon, GA",
        95_000,
        95_000,
        1_150,
        1_150,
        3,
        "Cerca de $95k de compra y $1,150 de renta; precios bajos y buena cobertura de subsidio.",
    ),
    _MercadoRaw(
        18,
        CAT_2,
        "Shreveport",
        "Luisiana",
        "Shreveport, LA",
        95_000,
        95_000,
        1_120,
        1_120,
        -3,
        "Cerca de $95k de compra y $1,120 de renta; entrada económica, pero revisa con cuidado el seguro de propiedad.",
    ),
    _MercadoRaw(
        19,
        CAT_2,
        "Montgomery",
        "Alabama",
        "Montgomery, AL",
        100_000,
        100_000,
        1_160,
        1_160,
        4,
        "Cerca de $100k de compra y $1,160 de renta; impuestos prediales bajos y demanda familiar constante.",
    ),
    _MercadoRaw(
        20,
        CAT_2,
        "Peoria",
        "Illinois",
        "Peoria, IL",
        85_000,
        85_000,
        1_100,
        1_100,
        -4,
        "Cerca de $85k de compra y $1,100 de renta; buen ratio inicial, aunque debes descontar impuestos locales.",
    ),
    # Puestos 21 al 30: Mercados secundarios con potencial
    _MercadoRaw(
        21,
        CAT_3,
        "Indianápolis",
        "Indiana",
        "Indianapolis, IN",
        145_000,
        145_000,
        1_473,
        1_473,
        4,
        "Cerca de $145k de compra y $1,473 de renta; mayor liquidez y apreciación sin perder flujo positivo.",
    ),
    _MercadoRaw(
        22,
        CAT_3,
        "Milwaukee",
        "Wisconsin",
        "Milwaukee, WI",
        135_000,
        135_000,
        1_338,
        1_338,
        3,
        "Cerca de $135k de compra y $1,338 de renta; atractivo para dúplex con rentas sólidas.",
    ),
    _MercadoRaw(
        23,
        CAT_3,
        "Rochester",
        "Nueva York",
        "Rochester, NY",
        125_000,
        125_000,
        1_350,
        1_350,
        -5,
        "Cerca de $125k de compra y $1,350 de renta; buen flujo bruto, pero los impuestos prediales pesan bastante.",
    ),
    _MercadoRaw(
        24,
        CAT_3,
        "Búfalo",
        "Nueva York",
        "Buffalo, NY",
        135_000,
        135_000,
        1_320,
        1_320,
        2,
        "Cerca de $135k de compra y $1,320 de renta; demanda firme de alquiler y precios más accesibles que en otras zonas del estado.",
    ),
    _MercadoRaw(
        25,
        CAT_3,
        "Siracusa",
        "Nueva York",
        "Syracuse, NY",
        125_000,
        125_000,
        1_300,
        1_300,
        2,
        "Cerca de $125k de compra y $1,300 de renta; buen rendimiento bruto en multifamiliares pequeños.",
    ),
    _MercadoRaw(
        26,
        CAT_3,
        "Kansas City",
        "Misuri",
        "Kansas City, MO",
        145_000,
        145_000,
        1_358,
        1_358,
        3,
        "Cerca de $145k de compra y $1,358 de renta; mezcla equilibrada entre flujo inmediato y plusvalía media.",
    ),
    _MercadoRaw(
        27,
        CAT_3,
        "Tulsa",
        "Oklahoma",
        "Tulsa, OK",
        130_000,
        130_000,
        1_220,
        1_220,
        3,
        "Cerca de $130k de compra y $1,220 de renta; costos operativos contenidos y buena demanda en tres habitaciones.",
    ),
    _MercadoRaw(
        28,
        CAT_3,
        "Oklahoma City",
        "Oklahoma",
        "Oklahoma City, OK",
        140_000,
        140_000,
        1_253,
        1_253,
        3,
        "Cerca de $140k de compra y $1,253 de renta; mercado en crecimiento con relación sana entre compra y renta.",
    ),
    _MercadoRaw(
        29,
        CAT_3,
        "Decatur",
        "Illinois",
        "Decatur, IL",
        75_000,
        75_000,
        1_040,
        1_040,
        1,
        "Cerca de $75k de compra y $1,040 de renta; compra económica útil para estrategias de reforma y refinanciación.",
    ),
    _MercadoRaw(
        30,
        CAT_3,
        "Gary y Hammond",
        "Indiana",
        "Gary, IN",
        95_000,
        95_000,
        1_290,
        1_290,
        4,
        "Cerca de $95k de compra y $1,290 de renta; impuestos menores que en la zona de Chicago con rentas competitivas.",
    ),
)


def _clamp(valor: float, minimo: float = 0.0, maximo: float = 100.0) -> float:
    return max(minimo, min(maximo, valor))


def calcular_atractivo_pct(
    *,
    precio_medio_usd: float,
    renta_media_usd: float,
    factor_operativo: int,
) -> tuple[float, float, float]:
    """Devuelve ``(ratio_renta_precio_pct, yield_bruto_anual_pct, atractivo_pct)``.

    Ponderación explicable:
    - 50 % Rendimiento bruto anual (escala 8 % a 22 %)
    - 25 % Accesibilidad de compra (escala $160k a $60k)
    - 15 % Renta mensual bruta (escala $950 a $1,700)
    - 10 % Factor operativo/fiscal/demanda explícito en la ficha
    """
    ratio_mensual = (renta_media_usd / precio_medio_usd) * 100.0
    yield_anual = ratio_mensual * 12.0

    score_yield = _clamp(((yield_anual - 8.0) / (22.0 - 8.0)) * 100.0)
    score_precio = _clamp(((160_000.0 - precio_medio_usd) / (160_000.0 - 60_000.0)) * 100.0)
    score_renta = _clamp(((renta_media_usd - 950.0) / (1_700.0 - 950.0)) * 100.0)
    score_operativo = _clamp(55.0 + (factor_operativo * 8.0))

    atractivo = (
        0.50 * score_yield + 0.25 * score_precio + 0.15 * score_renta + 0.10 * score_operativo
    )
    return round(ratio_mensual, 2), round(yield_anual, 2), round(_clamp(atractivo), 1)


def listar_mercados_ordenados() -> list[MercadoAtractivo]:
    """Retorna los 30 mercados ordenados de mayor a menor % de atractivo."""
    calculados: list[tuple[float, float, float, MercadoAtractivo]] = []
    for raw in _MERCADOS_BASE:
        precio_medio = (raw.precio_min_usd + raw.precio_max_usd) / 2.0
        renta_media = (raw.renta_min_usd + raw.renta_max_usd) / 2.0
        ratio_pct, yield_pct, atractivo = calcular_atractivo_pct(
            precio_medio_usd=precio_medio,
            renta_media_usd=renta_media,
            factor_operativo=raw.factor_operativo,
        )
        mercado = MercadoAtractivo(
            ranking=1,
            puesto_lista=raw.puesto_lista,
            categoria=raw.categoria,
            ciudad=raw.ciudad,
            estado=raw.estado,
            nombre_completo=f"{raw.ciudad}, {raw.estado}",
            query=raw.query,
            precio_min_usd=raw.precio_min_usd,
            precio_max_usd=raw.precio_max_usd,
            precio_medio_usd=round(precio_medio, 2),
            renta_min_usd=raw.renta_min_usd,
            renta_max_usd=raw.renta_max_usd,
            renta_media_usd=round(renta_media, 2),
            ratio_renta_precio_pct=ratio_pct,
            yield_bruto_anual_pct=yield_pct,
            atractivo_pct=atractivo,
            notas=raw.notas,
        )
        calculados.append((atractivo, yield_pct, -precio_medio, mercado))

    calculados.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [
        item[3].model_copy(update={"ranking": idx}) for idx, item in enumerate(calculados, start=1)
    ]
