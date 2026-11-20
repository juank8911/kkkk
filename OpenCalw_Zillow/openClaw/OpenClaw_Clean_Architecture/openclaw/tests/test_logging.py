"""Pruebas del formato de log y del contexto de job."""

import json
import logging

from app.core.logging import JsonFormatter, job_context


def _registro(mensaje: str) -> logging.LogRecord:
    return logging.LogRecord("test", logging.INFO, __file__, 10, mensaje, None, None)


def test_json_incluye_job_id_dentro_del_contexto():
    with job_context("job-123"):
        salida = json.loads(JsonFormatter().format(_registro("hola")))

    assert salida["job_id"] == "job-123"
    assert salida["msg"] == "hola"
    assert salida["level"] == "INFO"


def test_el_contexto_se_libera_al_salir():
    with job_context("job-123"):
        pass

    salida = json.loads(JsonFormatter().format(_registro("fuera")))

    assert "job_id" not in salida


def test_incluye_trace_id_si_existe():
    with job_context("job-123", trace_id="trace-9"):
        salida = json.loads(JsonFormatter().format(_registro("hola")))

    assert salida["trace_id"] == "trace-9"
