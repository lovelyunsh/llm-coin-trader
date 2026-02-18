from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

from coin_trader.core.models import ExchangeName, Order, OrderSide, OrderStatus, OrderType
from coin_trader.execution.idempotency import IdempotencyManager
from coin_trader.state.store import StateStore


def test_generate_key_deterministic() -> None:
    mgr = IdempotencyManager()
    intent_id = UUID("00000000-0000-0000-0000-000000000001")

    assert mgr.generate_key(intent_id) == mgr.generate_key(intent_id)
    assert mgr.generate_key(intent_id) == "intent_00000000-0000-0000-0000-000000000001"


def test_is_duplicate_false_for_new_true_for_processed() -> None:
    mgr = IdempotencyManager()
    client_order_id = "client_001"

    assert mgr.is_duplicate(client_order_id) is False
    mgr.mark_processed(client_order_id)
    assert mgr.is_duplicate(client_order_id) is True


def test_mark_processed_makes_subsequent_is_duplicate_true() -> None:
    mgr = IdempotencyManager()
    client_order_id = "client_002"

    mgr.mark_processed(client_order_id)
    assert mgr.is_duplicate(client_order_id) is True


def test_db_backed_duplicate_detection(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    client_order_id = "client_db_001"

    with StateStore(db_path=db_path) as store:
        order = Order(
            client_order_id=client_order_id,
            intent_id=uuid4(),
            exchange=ExchangeName.UPBIT,
            symbol="BTC/KRW",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000000"),
            status=OrderStatus.PENDING,
        )
        store.save_order(order)

        mgr = IdempotencyManager()
        mgr.__dict__["store"] = store
        assert mgr.is_duplicate(client_order_id) is True
