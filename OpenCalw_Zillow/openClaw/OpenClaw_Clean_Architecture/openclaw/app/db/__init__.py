"""Capa de persistencia PostgreSQL (S2-02 / ADR-03)."""

from app.db.base import Base, get_session, obtener_engine, obtener_sessionmaker

__all__ = ["Base", "get_session", "obtener_engine", "obtener_sessionmaker"]
