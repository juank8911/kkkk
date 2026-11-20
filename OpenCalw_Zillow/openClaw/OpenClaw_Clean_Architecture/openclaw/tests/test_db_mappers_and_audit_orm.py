"""Pruebas de integración ORM entre dominio Dev 2 y esquema PostgreSQL Dev 1 (S2-02 / S2-07)."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import httpx
import pytest

from app.db.models import Decision, Evento, Propiedad
from app.integrations.uao_client import UaoClient
from app.models.buy import BuyMode, BuyOrderRequest, BuyOrderStatus
from app.models.property import DatosPropiedadNormalizados
from app.services.buy_service import BuyDisabledError, BuySimulationService
from app.services.db_mappers import (
    SqlAlchemyBuyAuditStore,
    derive_decision_veredicto,
    map_buy_audit_event_to_orm,
    map_normalized_property_to_orm,
    map_pipeline_state_to_decision_orm,
)


class _FakeScalarsResult:
    def __init__(self, rows: list[Evento]) -> None:
        self._rows = rows

    def scalars(self) -> _FakeScalarsResult:
        return self

    def all(self) -> list[Evento]:
        return list(self._rows)


class _FakeAsyncSession:
    """Doble ligero de AsyncSession para verificar persistencia ORM sin PostgreSQL externo."""

    def __init__(self) -> None:
        self.added: list[Any] = []
        self.flushed = 0

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    async def flush(self) -> None:
        self.flushed += 1

    async def execute(self, stmt: Any) -> _FakeScalarsResult:
        eventos = [obj for obj in self.added if isinstance(obj, Evento)]
        return _FakeScalarsResult(eventos)


def test_map_normalized_property_to_orm_matches_s2_02_schema() -> None:
    job_id = uuid4()
    prop = DatosPropiedadNormalizados(
        id_plataforma_origen="https://www.zillow.com/homedetails/1420-maple-ave/1001_zpid/",
        ubicacion="1420 Maple Ave, Cleveland, OH 44102",
        precio_usd=115_000.0,
        ingreso_anual_estimado=17_400.0,
        imagenes=["https://photos.zillowstatic.com/1001.jpg"],
        condado="Cuyahoga",
        latitud=41.4822,
        longitud=-81.7341,
        descripcion_cruda="Solid brick colonial in Cuyahoga County.",
    )

    orm_prop = map_normalized_property_to_orm(job_id=job_id, plataforma="Zillow", propiedad=prop)
    assert isinstance(orm_prop, Propiedad)
    assert orm_prop.job_id == job_id
    assert orm_prop.plataforma == "zillow"
    assert orm_prop.precio_usd == 115_000.0
    assert orm_prop.huella_dedupe is not None
    assert len(orm_prop.huella_dedupe) == 24


@pytest.mark.parametrize(
    ("state", "expected_veredicto"),
    [
        (
            {
                "estado": "ALERTA_ENVIADA",
                "requiere_revision_humana": False,
                "vision_disponible": True,
            },
            "comprar",
        ),
        (
            {
                "estado": "ALERTA_ENVIADA",
                "requiere_revision_humana": True,
                "vision_disponible": True,
            },
            "revisar",
        ),
        ({"estado": "VISION_NO_DISPONIBLE", "vision_disponible": False}, "revisar"),
        ({"estado": "RECHAZADO", "motivo_financiero": "cap_rate bajo"}, "descartar"),
        ({"estado": "DESCARTADO_PRECIO"}, "descartar"),
        ({"estado": "ALERTA_ERROR"}, "error"),
    ],
)
def test_map_pipeline_state_to_decision_orm_respects_check_constraint(
    state: dict[str, Any],
    expected_veredicto: str,
) -> None:
    prop_id = uuid4()
    full_state = {
        "cap_rate_ajustado": 10.42,
        "capex": 12_000.0,
        "vision_motivo": "Estructura sólida.",
        **state,
    }
    assert derive_decision_veredicto(full_state) == expected_veredicto

    orm_decision = map_pipeline_state_to_decision_orm(propiedad_id=prop_id, state=full_state)
    assert isinstance(orm_decision, Decision)
    assert orm_decision.propiedad_id == prop_id
    assert orm_decision.veredicto == expected_veredicto
    assert orm_decision.cap_rate == pytest.approx(10.42)
    assert orm_decision.capex_estimado == pytest.approx(12_000.0)


async def test_sqlalchemy_buy_audit_store_persists_into_evento_table() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"reference_id": "uao-orm-tx-1"})

    uao = UaoClient(
        base_url="http://uao.local",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    fake_session = _FakeAsyncSession()
    audit_store = SqlAlchemyBuyAuditStore(fake_session)  # type: ignore[arg-type]

    # 1. Rechazo en disabled audita con nivel warning
    disabled_svc = BuySimulationService(
        uao_client=uao, mode=BuyMode.DISABLED, audit_store=audit_store
    )
    order = BuyOrderRequest(
        property_id="prop-orm-1",
        asset="USDT",
        amount=80_000.0,
        destination="sandbox_escrow_vault",
        idempotency_key="orm-idem-0001",
        requested_by="chat-1001",
    )
    with pytest.raises(BuyDisabledError):
        await disabled_svc.execute_buy(order)

    # 2. Ejecución + replay en sandbox auditan ambos eventos en tabla `eventos`
    sandbox_svc = BuySimulationService(
        uao_client=uao,
        mode=BuyMode.SANDBOX,
        require_confirmation=False,
        audit_store=audit_store,
    )
    await sandbox_svc.execute_buy(order)
    await sandbox_svc.execute_buy(order)

    assert fake_session.flushed == 3
    assert len(fake_session.added) == 3
    assert all(isinstance(row, Evento) for row in fake_session.added)
    assert fake_session.added[0].nivel == "warning"
    assert fake_session.added[1].nivel == "info"
    assert fake_session.added[1].tipo == "buy.execute_buy_sandbox"

    stored = await audit_store.list_events(idempotency_key="orm-idem-0001")
    assert [ev.status for ev in stored] == [
        BuyOrderStatus.REJECTED_DISABLED,
        BuyOrderStatus.SIMULATED_EXECUTED,
        BuyOrderStatus.DUPLICATE_REPLAYED,
    ]
    row_direct = map_buy_audit_event_to_orm(stored[1])
    assert row_direct.payload["upstream_reference"] == "uao-orm-tx-1"
