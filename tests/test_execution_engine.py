"""Tests for ExecutionEngine — intent -> risk -> broker pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from coin_trader.core.models import (
    ExchangeName,
    Order,
    OrderIntent,
    OrderSide,
    OrderType,
    RiskDecision,
)
from coin_trader.execution.engine import ExecutionEngine
from coin_trader.execution.idempotency import IdempotencyManager


@dataclass
class MockRiskResult:
    decision: RiskDecision
    reason: str = ""


def make_intent(
    *,
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.LIMIT,
) -> OrderIntent:
    return OrderIntent(
        signal_id=uuid4(),
        exchange=ExchangeName.UPBIT,
        symbol="BTC/KRW",
        side=side,
        order_type=order_type,
        quote_quantity=Decimal("100000"),
        price=Decimal("50000000"),
        reason="test",
        timestamp=datetime.now(UTC),
    )


def make_order(*, created_at: datetime | None = None) -> Order:
    now = datetime.now(UTC)
    ts = created_at if created_at is not None else now
    return Order(
        client_order_id="test_001",
        intent_id=uuid4(),
        exchange=ExchangeName.UPBIT,
        symbol="BTC/KRW",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.001"),
        price=Decimal("50000000"),
        created_at=ts,
        updated_at=max(ts, now),
    )


def make_engine(
    *,
    risk_decision: RiskDecision = RiskDecision.APPROVED,
    risk_reason: str = "",
    broker_order: Order | None = None,
    broker_place_raises: Exception | None = None,
    kill_switch_active: bool = False,
    open_orders: list[Order] | None = None,
    broker_cancel_raises: dict[str, Exception] | None = None,
) -> tuple[ExecutionEngine, AsyncMock, AsyncMock, IdempotencyManager, AsyncMock]:
    broker = AsyncMock()

    if broker_place_raises is not None:
        broker.place.side_effect = broker_place_raises
    else:
        broker.place.return_value = broker_order or make_order()

    broker.fetch_open_orders.return_value = open_orders or []

    if broker_cancel_raises:
        def _cancel_side_effect(order_id: str) -> None:
            if order_id in broker_cancel_raises:
                raise broker_cancel_raises[order_id]
        broker.cancel.side_effect = _cancel_side_effect

    risk_manager = AsyncMock()
    risk_manager.validate.return_value = MockRiskResult(
        decision=risk_decision, reason=risk_reason
    )

    idempotency = IdempotencyManager()

    state_store = AsyncMock()

    kill_switch = Mock()
    kill_switch.is_active = Mock(return_value=kill_switch_active)

    engine = ExecutionEngine(broker, risk_manager, idempotency, state_store, kill_switch)
    return engine, broker, risk_manager, idempotency, state_store


DEFAULT_STATE: dict[str, object] = {
    "total_balance": Decimal("1000000"),
    "position_count": 0,
    "market_price": Decimal("50000000"),
}


# ===========================================================================
# execute() tests
# ===========================================================================

async def test_execute_kill_switch_blocks() -> None:
    engine, broker, _, _, state_store = make_engine(kill_switch_active=True)
    intent = make_intent()
    result = await engine.execute(intent, DEFAULT_STATE)
    assert result is None
    broker.place.assert_not_called()
    state_store.save_decision.assert_not_called()


async def test_execute_idempotency_duplicate_blocks() -> None:
    engine, broker, _, idempotency, state_store = make_engine()
    intent = make_intent()
    # Pre-mark this intent as already processed
    key = idempotency.generate_key(intent.intent_id)
    idempotency.mark_processed(key)

    result = await engine.execute(intent, DEFAULT_STATE)
    assert result is None
    broker.place.assert_not_called()
    state_store.save_decision.assert_not_called()


async def test_execute_risk_rejection_returns_none() -> None:
    engine, broker, risk_manager, _, state_store = make_engine(
        risk_decision=RiskDecision.REJECTED, risk_reason="position too large"
    )
    intent = make_intent()
    result = await engine.execute(intent, DEFAULT_STATE)
    assert result is None
    broker.place.assert_not_called()
    state_store.save_decision.assert_called_once()


async def test_execute_successful_order() -> None:
    placed_order = make_order()
    engine, broker, _, idempotency, state_store = make_engine(
        risk_decision=RiskDecision.APPROVED, broker_order=placed_order
    )
    intent = make_intent()
    result = await engine.execute(intent, DEFAULT_STATE)

    assert result is placed_order
    state_store.save_order.assert_called_once_with(placed_order)
    # Verify idempotency key was marked after success
    key = idempotency.generate_key(intent.intent_id)
    assert key in idempotency._processed


async def test_execute_broker_exception_propagates() -> None:
    engine, broker, _, _, _ = make_engine(
        risk_decision=RiskDecision.APPROVED,
        broker_place_raises=RuntimeError("exchange down"),
    )
    intent = make_intent()
    with pytest.raises(RuntimeError, match="exchange down"):
        await engine.execute(intent, DEFAULT_STATE)


# ===========================================================================
# cancel_stale_orders() tests
# ===========================================================================

async def test_cancel_stale_no_open_orders() -> None:
    engine, broker, _, _, _ = make_engine(open_orders=[])
    result = await engine.cancel_stale_orders(timeout_sec=60)
    assert result == []
    broker.cancel.assert_not_called()


async def test_cancel_stale_young_orders_skipped() -> None:
    recent_order = make_order(created_at=datetime.now(UTC) - timedelta(seconds=10))
    engine, broker, _, _, _ = make_engine(open_orders=[recent_order])
    result = await engine.cancel_stale_orders(timeout_sec=60)
    assert result == []
    broker.cancel.assert_not_called()


async def test_cancel_stale_old_orders_cancelled() -> None:
    old_order = make_order(created_at=datetime.now(UTC) - timedelta(seconds=120))
    engine, broker, _, _, _ = make_engine(open_orders=[old_order])
    result = await engine.cancel_stale_orders(timeout_sec=60)
    assert result == [old_order]
    broker.cancel.assert_called_once()


async def test_cancel_stale_exception_logged_continues() -> None:
    old_order1 = make_order(created_at=datetime.now(UTC) - timedelta(seconds=120))
    old_order1 = old_order1.model_copy(update={"client_order_id": "order_a", "order_id": "order_a"})
    old_order2 = make_order(created_at=datetime.now(UTC) - timedelta(seconds=120))
    old_order2 = old_order2.model_copy(update={"client_order_id": "order_b", "order_id": "order_b"})

    engine, broker, _, _, _ = make_engine(
        open_orders=[old_order1, old_order2],
        broker_cancel_raises={"order_a": RuntimeError("cancel failed")},
    )
    # order_a raises, order_b should still be cancelled
    result = await engine.cancel_stale_orders(timeout_sec=60)
    # Only order_b succeeds
    assert old_order2 in result
    assert old_order1 not in result


# ===========================================================================
# cancel_all_open_orders() tests
# ===========================================================================

async def test_cancel_all_empty_returns_zero() -> None:
    engine, broker, _, _, _ = make_engine(open_orders=[])
    result = await engine.cancel_all_open_orders()
    assert result == 0
    broker.cancel.assert_not_called()


async def test_cancel_all_multiple_orders() -> None:
    orders = [
        make_order().model_copy(update={"client_order_id": f"o{i}", "order_id": f"o{i}"})
        for i in range(3)
    ]
    engine, broker, _, _, _ = make_engine(open_orders=orders)
    result = await engine.cancel_all_open_orders()
    assert result == 3
    assert broker.cancel.call_count == 3


async def test_cancel_all_partial_failure() -> None:
    order_a = make_order().model_copy(update={"client_order_id": "oa", "order_id": "oa"})
    order_b = make_order().model_copy(update={"client_order_id": "ob", "order_id": "ob"})
    order_c = make_order().model_copy(update={"client_order_id": "oc", "order_id": "oc"})

    engine, broker, _, _, _ = make_engine(
        open_orders=[order_a, order_b, order_c],
        broker_cancel_raises={"ob": RuntimeError("network error")},
    )
    result = await engine.cancel_all_open_orders()
    # order_a and order_c succeed, order_b fails
    assert result == 2
    assert broker.cancel.call_count == 3
