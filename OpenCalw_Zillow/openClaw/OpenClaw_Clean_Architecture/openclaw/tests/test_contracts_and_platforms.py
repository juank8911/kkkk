"""Pruebas de contratos Pydantic (S1-02) y catálogo de 9 plataformas (S1-07)."""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI, status
from pydantic import ValidationError

from app.models import BarridoRequest, ErrorDetail, ErrorResponse, JobResponse, JobStatus
from app.services.platforms import (
    PLATFORM_CATALOG,
    VALID_PLATFORM_KEYS,
    InvalidPlatformError,
    InvalidQueryError,
    build_platform_url,
    validate_platforms,
)


def _build_contract_app() -> FastAPI:
    """App FastAPI mínima para validar publicación OpenAPI y HTTP 422 de los contratos S1-02/S1-07."""
    app = FastAPI(title="OpenClaw Contracts Verification")

    @app.post(
        "/v1/openclaw/barridos",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=JobResponse,
        responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    )
    async def crear_barrido(body: BarridoRequest) -> JobResponse:
        return JobResponse(
            job_id=uuid4(),
            status=JobStatus.QUEUED,
            query=body.query,
            plataformas=body.plataformas,
            progress=dict.fromkeys(body.plataformas, "queued"),
        )

    return app


def test_catalog_contains_exactly_nine_adr01_platforms() -> None:
    expected = {
        "zillow",
        "realtor",
        "redfin",
        "trulia",
        "loopnet",
        "craigslist",
        "century21",
        "coldwell",
        "remax",
    }
    assert len(PLATFORM_CATALOG) == 9
    assert set(PLATFORM_CATALOG.keys()) == expected
    assert set(VALID_PLATFORM_KEYS) == expected


@pytest.mark.parametrize("platform_key", VALID_PLATFORM_KEYS)
def test_build_platform_url_for_all_nine_platforms(platform_key: str) -> None:
    url = build_platform_url(platform_key, "Miami FL")
    spec = PLATFORM_CATALOG[platform_key]
    assert url.startswith("https://")
    assert spec.domain in url


def test_build_platform_url_prevents_ssrf_and_path_injection() -> None:
    # Intentar inyectar otro host o traversal en el query debe quedar saneado dentro del dominio
    url_zillow = build_platform_url("zillow", "../../etc/passwd@evil.com?x=1#frag")
    assert url_zillow.startswith("https://www.zillow.com/")
    assert "@" not in url_zillow
    assert ".." not in url_zillow

    url_cl = build_platform_url("craigslist", "miami.evil.com/admin")
    assert url_cl.endswith(".craigslist.org/search/rea")

    with pytest.raises(InvalidQueryError):
        build_platform_url("zillow", "   ///   ")


def test_validate_platforms_normalizes_and_deduplicates() -> None:
    result = validate_platforms([" Zillow ", "REALTOR", "zillow", "Redfin"])
    assert result == ["zillow", "realtor", "redfin"]


@pytest.mark.parametrize("bad_key", ["idealista", "https://www.zillow.com", "http://169.254.169.254"])
def test_validate_platforms_rejects_unknown_or_raw_url_with_valid_list(bad_key: str) -> None:
    with pytest.raises(InvalidPlatformError) as exc_info:
        validate_platforms(["zillow", bad_key])

    message = str(exc_info.value)
    assert bad_key in message
    for valid_key in VALID_PLATFORM_KEYS:
        assert valid_key in message


async def test_unknown_platform_returns_422_with_list_of_valid_platforms() -> None:
    """Criterio S1-07: Clave desconocida devuelve 422 con lista de válidas."""
    contract_app = _build_contract_app()
    transport = httpx.ASGITransport(app=contract_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/openclaw/barridos",
            json={"query": "Cleveland OH", "plataformas": ["zillow", "plataforma_falsa"]},
        )

    assert response.status_code == 422
    detail_text = str(response.json())
    assert "plataforma_falsa" in detail_text
    for valid_key in VALID_PLATFORM_KEYS:
        assert valid_key in detail_text


async def test_valid_barrido_request_and_openapi_contracts() -> None:
    """Criterio S1-02: BarridoRequest, JobResponse y ErrorResponse publicados en OpenAPI."""
    contract_app = _build_contract_app()
    transport = httpx.ASGITransport(app=contract_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        openapi_resp = await client.get("/openapi.json")
        assert openapi_resp.status_code == 200
        schemas = openapi_resp.json()["components"]["schemas"]
        assert "BarridoRequest" in schemas
        assert "JobResponse" in schemas
        assert "ErrorResponse" in schemas

        post_resp = await client.post(
            "/v1/openclaw/barridos",
            json={"query": "  Cleveland, OH  ", "plataformas": ["Zillow", "Realtor"]},
        )
        assert post_resp.status_code == 202
        job = JobResponse.model_validate(post_resp.json())
        assert job.status == JobStatus.QUEUED
        assert job.query == "Cleveland, OH"
        assert job.plataformas == ["zillow", "realtor"]


def test_barrido_request_forbids_extra_fields_and_invalid_query() -> None:
    with pytest.raises(ValidationError):
        BarridoRequest(query="Miami FL", plataformas=["zillow"], campo_extra="no_permitido")  # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        BarridoRequest(query="a", plataformas=["zillow"])

    with pytest.raises(ValidationError):
        BarridoRequest(query="Miami\nFL", plataformas=["zillow"])


def test_error_response_model_serialization() -> None:
    err = ErrorResponse(
        code="invalid_platform",
        message="Plataforma desconocida",
        details=[
            ErrorDetail(
                field="plataformas",
                reason="Clave no perteneciente al catálogo",
                valid_values=list(VALID_PLATFORM_KEYS),
            )
        ],
        trace_id="trace-123",
    )
    dumped = err.model_dump(mode="json")
    assert dumped["code"] == "invalid_platform"
    assert dumped["trace_id"] == "trace-123"
    assert len(dumped["details"][0]["valid_values"]) == 9
