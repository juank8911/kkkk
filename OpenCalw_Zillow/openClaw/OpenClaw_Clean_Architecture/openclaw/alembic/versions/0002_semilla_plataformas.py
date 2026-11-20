"""Datos semilla del catalogo de plataformas, idempotente (S2-04 / ADR-01).

Revision ID: 0002_semilla_plataformas
Revises: 0001_esquema_inicial
Create Date: 2026-11-20
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_semilla_plataformas"
down_revision: str | None = "0001_esquema_inicial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

PLATAFORMAS: tuple[tuple[str, str, str, str], ...] = (
    ("zillow", "Zillow", "zillow.com", "https://www.zillow.com/homes/{query}_rb/"),
    ("realtor", "Realtor.com", "realtor.com", "https://www.realtor.com/realestateandhomes-search/{query}"),
    ("redfin", "Redfin", "redfin.com", "https://www.redfin.com/city/{query}"),
    ("trulia", "Trulia", "trulia.com", "https://www.trulia.com/{query}"),
    ("loopnet", "LoopNet", "loopnet.com", "https://www.loopnet.com/search/commercial-real-estate/{query}/for-sale/"),
    ("craigslist", "Craigslist", "craigslist.org", "https://{query}.craigslist.org/search/rea"),
    ("century21", "Century 21", "century21.com", "https://www.century21.com/real-estate/{query}/"),
    ("coldwell", "Coldwell Banker", "coldwellbankerhomes.com", "https://www.coldwellbankerhomes.com/{query}/"),
    ("remax", "RE/MAX", "remax.com", "https://www.remax.com/homes-for-sale/{query}"),
)

TABLA = sa.table(
    "plataformas",
    sa.column("clave", sa.String),
    sa.column("nombre", sa.String),
    sa.column("dominio", sa.String),
    sa.column("plantilla_url", sa.Text),
    sa.column("activa", sa.Boolean),
)


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            INSERT INTO plataformas (clave, nombre, dominio, plantilla_url, activa)
            VALUES (:clave, :nombre, :dominio, :plantilla_url, true)
            ON CONFLICT (clave) DO UPDATE
               SET nombre = EXCLUDED.nombre,
                   dominio = EXCLUDED.dominio,
                   plantilla_url = EXCLUDED.plantilla_url,
                   activa = true
            """
        ).bindparams(
            sa.bindparam("clave"),
            sa.bindparam("nombre"),
            sa.bindparam("dominio"),
            sa.bindparam("plantilla_url"),
        ),
        [
            {
                "clave": clave,
                "nombre": nombre,
                "dominio": dominio,
                "plantilla_url": plantilla,
            }
            for clave, nombre, dominio, plantilla in PLATAFORMAS
        ],
    )


def downgrade() -> None:
    claves = [fila[0] for fila in PLATAFORMAS]
    op.execute(
        sa.text("DELETE FROM plataformas WHERE clave = ANY(:claves)").bindparams(
            sa.bindparam("claves", expanding=False)
        ),
        {"claves": claves},
    )
