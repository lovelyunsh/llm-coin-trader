from __future__ import annotations

from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from coin_trader.core.models import ExchangeName, Order, OrderSide, OrderStatus, OrderType
from coin_trader.execution.idempotency import IdempotencyManager


def test_generate_key_deterministic() -> None:
    mgr = IdempotencyManager()
    intent_id = UUID("00000000-0000-0000-0000-000000000001")

    assert mgr.generate_key(intent_id) == mgr.generate_key(intent_id)
    assert mgr.generate_key(intent_id) == "intent_00000000-0000-0000-0000-000000000001"


async def test_is_duplicate_false_for_new_true_for_processed() -> None:
    mgr = IdempotencyManager()
    client_order_id = "client_001"

    assert await mgr.is_duplicate(client_order_id) is False
    mgr.mark_processed(client_order_id)
    assert await mgr.is_duplicate(client_order_id) is True


async def test_mark_processed_makes_subsequent_is_duplicate_true() -> None:
    mgr = IdempotencyManager()
    client_order_id = "client_002"

    mgr.mark_processed(client_order_id)
    assert await mgr.is_duplicate(client_order_id) is True


async def test_db_backed_duplicate_detection(tmp_path: object) -> None:
    """Test duplicate detection backed by PostgreSQL.

    Requires a running PostgreSQL instance. Set DATABASE_URL env var or
    skip with: pytest -k 'not db_backed'
    """
    import os

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not set — skipping PostgreSQL-backed test")

    from coin_trader.state.store import StateStore

    store = await StateStore.create(database_url, min_size=1, max_size=2)
    client_order_id = f"client_db_{uuid4().hex[:8]}"

    try:
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
        await store.save_order(order)

        mgr = IdempotencyManager(state_store=store)
        assert await mgr.is_duplicate(client_order_id) is True
    finally:
        await store.close()
