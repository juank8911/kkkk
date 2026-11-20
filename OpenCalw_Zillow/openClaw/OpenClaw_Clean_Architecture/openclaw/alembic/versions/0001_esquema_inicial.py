"""Esquema inicial: jobs, propiedades, decisiones, eventos y plataformas (S2-02).

Revision ID: 0001_esquema_inicial
Revises:
Create Date: 2026-11-20
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_esquema_inicial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("query", sa.String(length=200), nullable=False),
        sa.Column("plataformas", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "status IN ('queued','running','completed','partial','failed')",
            name="ck_jobs_status_valido",
        ),
    )
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_trace_id", "jobs", ["trace_id"])
    op.create_index("ix_jobs_created_at", "jobs", ["created_at"])

    op.create_table(
        "plataformas",
        sa.Column("clave", sa.String(length=40), primary_key=True),
        sa.Column("nombre", sa.String(length=80), nullable=False),
        sa.Column("dominio", sa.String(length=120), nullable=False, unique=True),
        sa.Column("plantilla_url", sa.Text(), nullable=False),
        sa.Column("activa", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("creada_en", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "propiedades",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("jobs.job_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("plataforma", sa.String(length=40), nullable=False),
        sa.Column("url_origen", sa.Text(), nullable=False),
        sa.Column("ubicacion", sa.Text(), nullable=False),
        sa.Column("condado", sa.String(length=120), nullable=True),
        sa.Column("precio_usd", sa.Numeric(14, 2), nullable=False),
        sa.Column("ingreso_anual_estimado", sa.Numeric(14, 2), nullable=False, server_default="0"),
        sa.Column("imagenes", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="[]"),
        sa.Column("latitud", sa.Numeric(9, 6), nullable=True),
        sa.Column("longitud", sa.Numeric(9, 6), nullable=True),
        sa.Column("descripcion_cruda", sa.Text(), nullable=False),
        sa.Column("huella_dedupe", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("job_id", "url_origen", name="uq_propiedades_job_url"),
        sa.CheckConstraint("precio_usd >= 0", name="ck_propiedades_precio_positivo"),
    )
    op.create_index("ix_propiedades_condado", "propiedades", ["condado"])
    op.create_index("ix_propiedades_huella_dedupe", "propiedades", ["huella_dedupe"])

    op.create_table(
        "decisiones",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "propiedad_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("propiedades.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("veredicto", sa.String(length=30), nullable=False),
        sa.Column("cap_rate", sa.Numeric(6, 4), nullable=True),
        sa.Column("roi_estimado", sa.Numeric(6, 4), nullable=True),
        sa.Column("capex_estimado", sa.Numeric(14, 2), nullable=True),
        sa.Column("motivo", sa.Text(), nullable=True),
        sa.Column("detalle", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "veredicto IN ('comprar','revisar','descartar','error')",
            name="ck_decisiones_veredicto_valido",
        ),
    )
    op.create_index("ix_decisiones_propiedad_id", "decisiones", ["propiedad_id"])

    op.create_table(
        "eventos",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("jobs.job_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("tipo", sa.String(length=60), nullable=False),
        sa.Column("nivel", sa.String(length=10), nullable=False, server_default="info"),
        sa.Column("mensaje", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("ocurrido_en", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "nivel IN ('debug','info','warning','error','critical')",
            name="ck_eventos_nivel_valido",
        ),
    )
    op.create_index("ix_eventos_tipo", "eventos", ["tipo"])
    op.create_index("ix_eventos_ocurrido_en", "eventos", ["ocurrido_en"])


def downgrade() -> None:
    op.drop_table("eventos")
    op.drop_table("decisiones")
    op.drop_table("propiedades")
    op.drop_table("plataformas")
    op.drop_table("jobs")
