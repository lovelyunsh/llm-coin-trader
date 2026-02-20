from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from coin_trader.broker.paper_futures import PaperFuturesBroker
from coin_trader.core.models import (
    ExchangeName,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FEE_RATE = Decimal("0.0004")
_SLIPPAGE_RATE = Decimal("0.001")


class _MockTickerAdapter:
    def __init__(self, price: Decimal = Decimal("50000")) -> None:
        self._price = price

    async def get_ticker(self, symbol: str) -> Mapping[str, object]:
        return {"price": self._price, "last": self._price}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_intent(
    side: OrderSide,
    quantity: Decimal,
    *,
    order_type: OrderType = OrderType.MARKET,
    symbol: str = "BTC/USDT",
    position_side: PositionSide | None = None,
    reduce_only: bool = False,
    price: Decimal | None = None,
) -> OrderIntent:
    return OrderIntent(
        signal_id=uuid4(),
        exchange=ExchangeName.BINANCE,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        reason="test",
        timestamp=_now(),
        position_side=position_side,
        reduce_only=reduce_only,
        price=price,
    )


def _broker(
    price: Decimal = Decimal("50000"),
    initial_balance: Decimal = Decimal("10000"),
    leverage: int = 1,
) -> PaperFuturesBroker:
    return PaperFuturesBroker(
        exchange=ExchangeName.BINANCE,
        exchange_adapter=_MockTickerAdapter(price),
        initial_balance_usdt=initial_balance,
        default_leverage=leverage,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_initial_balance() -> None:
    broker = _broker()
    snap = await broker.fetch_balances()
    assert snap.balances["USDT"] == Decimal("10000")
    assert snap.total_value_quote == Decimal("10000")


async def test_long_market_order() -> None:
    broker = _broker(price=Decimal("50000"))
    intent = _make_intent(OrderSide.BUY, Decimal("0.1"))
    order = await broker.place(intent, client_order_id="cid-long-1")

    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == Decimal("0.1")


async def test_short_market_order() -> None:
    broker = _broker(price=Decimal("50000"))
    intent = _make_intent(
        OrderSide.SELL,
        Decimal("0.05"),
        position_side=PositionSide.SHORT,
    )
    order = await broker.place(intent, client_order_id="cid-short-1")

    assert order.status == OrderStatus.FILLED

    positions = await broker.fetch_positions()
    assert len(positions) == 1
    assert positions[0].side == PositionSide.SHORT
    assert positions[0].symbol == "BTC/USDT"


async def test_close_long_position() -> None:
    price = Decimal("50000")
    broker = _broker(price=price, initial_balance=Decimal("10000"))

    # Open long
    open_intent = _make_intent(OrderSide.BUY, Decimal("0.1"))
    await broker.place(open_intent, client_order_id="cid-open")

    balance_after_open = (await broker.fetch_balances()).balances["USDT"]

    # Close long (reduce_only SELL)
    close_intent = _make_intent(
        OrderSide.SELL,
        Decimal("0.1"),
        reduce_only=True,
    )
    await broker.place(close_intent, client_order_id="cid-close")

    balance_after_close = (await broker.fetch_balances()).balances["USDT"]

    # After close, balance should be different from mid-trade (fees apply twice)
    # Both are fees, so final balance should be less than initial 10000
    assert balance_after_close > balance_after_open
    # Rough sanity: fees = 2 * qty * price * fee_rate ~ 2 * 0.1 * 50050 * 0.0004 ≈ 4
    assert balance_after_close < Decimal("10000")

    positions = await broker.fetch_positions()
    assert positions == []


async def test_close_short_position() -> None:
    open_price = Decimal("50000")
    close_price = Decimal("45000")

    # Use an adapter whose price we'll control externally by building the broker
    # with different adapter for open vs close via a mutable adapter.
    class _MutableAdapter:
        price: Decimal = open_price

        async def get_ticker(self, symbol: str) -> Mapping[str, object]:
            return {"price": self.price, "last": self.price}

    adapter = _MutableAdapter()
    broker = PaperFuturesBroker(
        exchange=ExchangeName.BINANCE,
        exchange_adapter=adapter,
        initial_balance_usdt=Decimal("10000"),
        default_leverage=1,
    )

    # Open short at 50000
    open_intent = _make_intent(
        OrderSide.SELL,
        Decimal("0.1"),
        position_side=PositionSide.SHORT,
    )
    await broker.place(open_intent, client_order_id="cid-short-open")

    balance_after_open = (await broker.fetch_balances()).balances["USDT"]

    # Price drops to 45000 — profit for short
    adapter.price = close_price

    # Close short with reduce_only BUY
    close_intent = _make_intent(
        OrderSide.BUY,
        Decimal("0.1"),
        reduce_only=True,
    )
    await broker.place(close_intent, client_order_id="cid-short-close")

    balance_after_close = (await broker.fetch_balances()).balances["USDT"]

    # Short profit: (50000 - 45000) * 0.1 = 500 (minus fees/slippage)
    # balance_after_close should be greater than balance_after_open (realized profit)
    assert balance_after_close > balance_after_open

    positions = await broker.fetch_positions()
    assert positions == []


async def test_leverage_margin() -> None:
    leverage = 10
    price = Decimal("50000")
    broker = _broker(price=price, initial_balance=Decimal("10000"), leverage=leverage)

    qty = Decimal("0.1")
    intent = _make_intent(OrderSide.BUY, qty)
    order = await broker.place(intent, client_order_id="cid-lev")
    assert order.status == OrderStatus.FILLED

    balance = (await broker.fetch_balances()).balances["USDT"]

    # Full notional would be qty * fill_price ~ 0.1 * 50050 = 5005
    # With leverage=10, margin required ~ 500.5; fee ~ 0.1*50050*0.0004 ~ 2.002
    # So balance deducted ~ 502.5, leaving ~ 9497.5
    full_notional = qty * (price * (Decimal("1") + _SLIPPAGE_RATE))
    required_margin = full_notional / Decimal(str(leverage))
    assert balance > Decimal("10000") - full_notional  # margin << full notional
    assert balance > Decimal("9000")  # sanity: still most balance left
    _ = required_margin  # used in comment above for clarity


async def test_insufficient_margin_rejected() -> None:
    # Balance of 100, trying to open a position needing ~5005 margin (leverage=1)
    broker = _broker(price=Decimal("50000"), initial_balance=Decimal("100"), leverage=1)
    intent = _make_intent(OrderSide.BUY, Decimal("0.1"))
    order = await broker.place(intent, client_order_id="cid-reject")

    assert order.status == OrderStatus.REJECTED
    assert "insufficient_margin" in str(order.metadata)


async def test_fetch_positions_empty() -> None:
    broker = _broker()
    positions = await broker.fetch_positions()
    assert positions == []
