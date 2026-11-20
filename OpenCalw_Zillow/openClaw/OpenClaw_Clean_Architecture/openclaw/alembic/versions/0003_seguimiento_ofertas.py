"""Seguimiento post-oferta y trazabilidad comercial (Fase 95).

Revision ID: 0003_seguimiento_ofertas
Revises: 0002_semilla_plataformas
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_seguimiento_ofertas"
down_revision: str | None = "0002_semilla_plataformas"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ofertas",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("propiedad_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("estado", sa.String(length=20), nullable=False, server_default="propuesta"),
        sa.Column("importe_usd", sa.Numeric(14, 2), nullable=False),
        sa.Column("importe_contraoferta_usd", sa.Numeric(14, 2), nullable=True),
        sa.Column("notas", sa.Text(), nullable=True),
        sa.Column("actor_ultima_actualizacion", sa.String(length=120), nullable=False),
        sa.Column("creada_en", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("actualizada_en", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("cerrada_en", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["propiedad_id"], ["propiedades.id"], ondelete="CASCADE"),
        sa.CheckConstraint(
            "estado IN ('propuesta','contraoferta','aceptada','rechazada','cerrada','cancelada')",
            name="ck_ofertas_estado_valido",
        ),
        sa.CheckConstraint("importe_usd > 0", name="ck_ofertas_importe_positivo"),
        sa.CheckConstraint(
            "importe_contraoferta_usd IS NULL OR importe_contraoferta_usd > 0",
            name="ck_ofertas_contraoferta_positiva",
        ),
    )
    op.create_index("ix_ofertas_propiedad_id", "ofertas", ["propiedad_id"])
    op.create_index("ix_ofertas_estado", "ofertas", ["estado"])
    op.create_table(
        "oferta_eventos",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("oferta_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("estado_anterior", sa.String(length=20), nullable=True),
        sa.Column("estado_nuevo", sa.String(length=20), nullable=False),
        sa.Column("actor", sa.String(length=120), nullable=False),
        sa.Column("notas", sa.Text(), nullable=True),
        sa.Column("ocurrido_en", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["oferta_id"], ["ofertas.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_oferta_eventos_oferta_id", "oferta_eventos", ["oferta_id"])


def downgrade() -> None:
    op.drop_table("oferta_eventos")
    op.drop_table("ofertas")
