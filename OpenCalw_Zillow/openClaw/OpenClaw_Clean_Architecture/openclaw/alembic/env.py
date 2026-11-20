"""Entorno de Alembic para el esquema de OpenClaw (S2-02).

Usa el motor asincrono de SQLAlchemy con la misma ``DATABASE_URL`` que la API,
de modo que migraciones y aplicacion hablan con el mismo esquema.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.core.config import settings
from app.db.base import Base
from app.db import models  # noqa: F401 - registra las tablas en el metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Genera el SQL sin conectarse a la base de datos."""
    context.configure(
        url=settings.DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    conectable = config.get_section(config.config_ini_section, {})
    motor = async_engine_from_config(
        conectable,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with motor.connect() as conexion:
        await conexion.run_sync(do_run_migrations)
    await motor.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
