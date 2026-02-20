from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from coin_trader.core.models import (
    BalanceSnapshot,
    ExchangeName,
    OrderIntent,
    OrderSide,
    OrderType,
    Position,
    PositionSide,
    SignalType,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Enum value tests
# ---------------------------------------------------------------------------


def test_position_side_enum_values() -> None:
    assert PositionSide.LONG == "long"
    assert PositionSide.SHORT == "short"
    assert PositionSide.BOTH == "both"


def test_signal_type_short_and_cover_exist() -> None:
    assert SignalType.SHORT == "short"
    assert SignalType.COVER == "cover"


def test_order_type_futures_variants_exist() -> None:
    assert OrderType.STOP_MARKET == "stop_market"
    assert OrderType.TAKE_PROFIT_MARKET == "take_profit_market"


# ---------------------------------------------------------------------------
# Position – short side
# ---------------------------------------------------------------------------


def test_position_short_profitable_when_price_drops() -> None:
    pos = Position(
        exchange=ExchangeName.UPBIT,
        symbol="BTC/KRW",
        quantity=Decimal("1"),
        average_entry_price=Decimal("50000"),
        current_price=Decimal("45000"),
        timestamp=_now(),
        side=PositionSide.SHORT,
    )
    assert pos.unrealized_pnl == Decimal("5000")
    assert pos.unrealized_pnl_pct is not None
    assert pos.unrealized_pnl_pct > Decimal("0")


def test_position_short_loss_when_price_rises() -> None:
    pos = Position(
        exchange=ExchangeName.UPBIT,
        symbol="BTC/KRW",
        quantity=Decimal("1"),
        average_entry_price=Decimal("50000"),
        current_price=Decimal("55000"),
        timestamp=_now(),
        side=PositionSide.SHORT,
    )
    assert pos.unrealized_pnl == Decimal("-5000")


def test_position_default_side_is_long() -> None:
    pos = Position(
        exchange=ExchangeName.UPBIT,
        symbol="BTC/KRW",
        quantity=Decimal("1"),
        average_entry_price=Decimal("50000"),
        timestamp=_now(),
    )
    assert pos.side == PositionSide.LONG


def test_position_long_unrealized_pnl() -> None:
    pos = Position(
        exchange=ExchangeName.UPBIT,
        symbol="ETH/KRW",
        quantity=Decimal("2"),
        average_entry_price=Decimal("100"),
        current_price=Decimal("110"),
        timestamp=_now(),
    )
    assert pos.unrealized_pnl == Decimal("20")


# ---------------------------------------------------------------------------
# OrderIntent – futures fields
# ---------------------------------------------------------------------------


def test_order_intent_futures_field_defaults() -> None:
    intent = OrderIntent(
        signal_id=uuid4(),
        exchange=ExchangeName.UPBIT,
        symbol="BTC/KRW",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.001"),
        reason="test",
        timestamp=_now(),
    )
    assert intent.reduce_only is False
    assert intent.position_side is None
    assert intent.stop_price is None


def test_order_intent_with_futures_fields() -> None:
    intent = OrderIntent(
        signal_id=uuid4(),
        exchange=ExchangeName.BINANCE,
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_MARKET,
        quantity=Decimal("0.01"),
        reason="stop loss",
        timestamp=_now(),
        reduce_only=True,
        position_side=PositionSide.SHORT,
        stop_price=Decimal("48000"),
    )
    assert intent.reduce_only is True
    assert intent.position_side == PositionSide.SHORT
    assert intent.stop_price == Decimal("48000")


# ---------------------------------------------------------------------------
# BalanceSnapshot – futures fields
# ---------------------------------------------------------------------------


def test_balance_snapshot_total_value_quote_and_currency() -> None:
    snap = BalanceSnapshot(
        exchange=ExchangeName.BINANCE,
        timestamp=_now(),
        balances={"USDT": Decimal("10000")},
        total_value_quote=Decimal("10000"),
        quote_currency="USDT",
    )
    assert snap.total_value_quote == Decimal("10000")
    assert snap.quote_currency == "USDT"
