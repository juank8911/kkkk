"""Configuracion tipada del servicio OpenClaw.

Reglas del proyecto:
- Ningun secreto tiene un valor por defecto utilizable.
- La validacion se ejecuta al arrancar (fail-fast), no al importar el modulo,
  para que las pruebas puedan construir ``Settings`` con valores propios.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_vali