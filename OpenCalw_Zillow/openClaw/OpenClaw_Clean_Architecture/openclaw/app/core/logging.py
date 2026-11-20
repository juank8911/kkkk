"""Logging estructurado en JSON con contexto de job.

Cada linea de log de un proceso lleva el ``job_id`` y, si existe, el ``trace_id``.
El contexto se propaga con ``ContextVar``, asi que sobrevive a ``await`` sin
necesidad de pasarlo como argumento por toda la cadena de llamadas.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

job_id_ctx: ContextVar[str | None] = ContextVar("job_id", default=None)
trace_id_ctx: ContextVar[str | None] = ContextVar("trace_id", default=None)

_CAMPOS_ESTANDAR = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Serializa cada registro como una linea JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        for clave, ctx in (("job_id", job_id_ctx), ("trace_id", trace_id_ctx)):
            valor = ctx.get()
            if valor:
                payload[clave] = valor

        for clave, valor in record.__dict__.items():
            if clave in _CAMPOS_ESTANDAR or clave in payload:
                continue
            if isinstance(valor, (str, int, float, bool)):
                payload[clave] = valor

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Formato legible para desarrollo local."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-8s %(name)s | %(message)s")


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configura el logger raiz una sola vez por proceso."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else TextFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


@contextmanager
def job_context(job_id: str, trace_id: str | None = None) -> Iterator[None]:
    """Vincula un job al contexto actual y lo libera al salir."""
    token_job = job_id_ctx.set(job_id)
    token_trace = trace_id_ctx.set(trace_id) if trace_id else None
    try:
        yield
    finally:
        job_id_ctx.reset(token_job)
        if token_trace is not None:
            trace_id_ctx.reset(token_trace)
