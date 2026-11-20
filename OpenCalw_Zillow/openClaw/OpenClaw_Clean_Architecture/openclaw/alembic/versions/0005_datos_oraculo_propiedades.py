"""Guarda señales informativas del Oráculo por propiedad.

Revision ID: 0005_datos_oraculo_propiedades
Revises: 0004_ubicacion_propiedades
Create Date: 2026-11-20
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_datos_oraculo_propiedades"
down_revision: str | None = "0004_ubicacion_propiedades"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "propiedades",
        sa.Column(
            "datos_oraculo",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
    )


def downgrade() -> None:
    op.drop_column("propiedades", "datos_oraculo")
