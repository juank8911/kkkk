"""Catálogo de fuentes directas para enriquecer propiedades.

Este módulo no calcula un score geoespacial agregado ni usa variables demográficas.
Los adaptadores deben guardar junto al resultado fuente, fecha, cobertura,
metodología, versión y confianza.
"""

from __future__ import annotations

from typing import Any


_DIRECT_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "id": "census_geocoder",
        "tipo": "ubicacion",
        "estado": "activo",
        "url": "https://geocoding.geo.census.gov/geocoder/Geocoding_Services_API.html",
        "cobertura": "Dirección, coordenadas, ciudad, ZIP y condado cuando el geocoder los devuelve.",
        "metodologia": "Consulta de la dirección del anuncio; se conserva la respuesta cruda y su fecha.",
        "confianza": "alta cuando la dirección coincide; media si solo hay coincidencia parcial.",
    },
    {
        "id": "hud_fmr_safmr",
        "tipo": "renta_referencia",
        "estado": "pendiente_de_conectar",
        "url": "https://www.huduser.gov/portal/datasets/fmr.html",
        "cobertura": "FMR por área y SAFMR por ZIP, según disponibilidad oficial de HUD.",
        "metodologia": "Consulta por ZIP y número de dormitorios; es referencia de estándar de pago, no renta garantizada.",
        "confianza": "alta para el dato publicado; no equivale a una tasación ni a ingreso asegurado.",
    },
    {
        "id": "comparables_renta",
        "tipo": "comparable_renta",
        "estado": "configurable",
        "url": None,
        "cobertura": "Proveedor autorizado configurado por el operador.",
        "metodologia": "Mediana y rango de anuncios comparables con fecha, muestra y filtros conservados.",
        "confianza": "depende de cobertura, tamaño de muestra y frescura del proveedor.",
    },
    {
        "id": "comparables_venta",
        "tipo": "comparable_venta",
        "estado": "configurable",
        "url": None,
        "cobertura": "Proveedor autorizado configurado por el operador.",
        "metodologia": "Rango de ventas comparables por zona y características, sin scraping no auditado.",
        "confianza": "depende de cobertura, tamaño de muestra y frescura del proveedor.",
    },
    {
        "id": "riesgo_fisico",
        "tipo": "mantenimiento_seguro",
        "estado": "configurable",
        "url": None,
        "cobertura": "Señal física acotada a mantenimiento/seguro y documentada por fuente.",
        "metodologia": "No usa demografía ni sirve como proxy de grupos protegidos; se guarda fuente y fecha.",
        "confianza": "requiere metodología y fecha del proveedor.",
    },
)


def fuentes_directas_mercado() -> list[dict[str, Any]]:
    """Devuelve metadatos serializables sin secretos ni credenciales."""
    return [dict(fuente) for fuente in _DIRECT_SOURCES]
