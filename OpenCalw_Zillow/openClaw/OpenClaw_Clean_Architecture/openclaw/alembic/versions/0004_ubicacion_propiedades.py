"""Añade ubicación validable y trazabilidad a propiedades (Fase ubicación).

Revision ID: 0004_ubicacion_propiedades
Revises: 0003_seguimiento_ofertas
Create Date: 2026-11-20
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_ubicacion_propiedades"
down_revision: str | None = "0003_seguimiento_ofertas"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("propiedades", sa.Column("direccion_normalizada", sa.Text(), nullable=True))
    op.add_column("propiedades", sa.Column("ciudad", sa.String(length=120), nullable=True))
    op.add_column("propiedades", sa.Column("codigo_postal", sa.String(length=10), nullable=True))
    op.add_column(
        "propiedades",
        sa.Column("fuente_ubicacion", sa.String(length=80), nullable=False, server_default="anuncio"),
    )
    op.add_column(
        "propiedades",
        sa.Column("confianza_ubicacion", sa.String(length=30), nullable=False, server_default="anuncio"),
    )
    op.create_index("ix_propiedades_ciudad", "propiedades", ["ciudad"])
    op.create_index("ix_propiedades_codigo_postal", "propiedades", ["codigo_postal"])


def downgrade() -> None:
    op.drop_index("ix_propiedades_codigo_postal", table_name="propiedades")
    op.drop_index("ix_propiedades_ciudad", table_name="propiedades")
    op.drop_column("propiedades", "confianza_ubicacion")
    op.drop_column("propiedades", "fuente_ubicacion")
    op.drop_column("propiedades", "codigo_postal")
    op.drop_column("propiedades", "ciudad")
    op.drop_column("propiedades", "direccion_normalizada")
