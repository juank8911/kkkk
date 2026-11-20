"""Renombra señales heredadas a datos directos de mercado.

Revision ID: 0006_datos_mercado_propiedades
Revises: 0005_datos_oraculo_propiedades
Create Date: 2026-11-20
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0006_datos_mercado_propiedades"
down_revision: str | None = "0005_datos_oraculo_propiedades"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column("propiedades", "datos_oraculo", new_column_name="datos_mercado")


def downgrade() -> None:
    op.alter_column("propiedades", "datos_mercado", new_column_name="datos_oraculo")
