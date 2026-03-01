"""Unit tests for async action helpers in coin_trader/main.py.

Covers _log_and_notify_order, _execute_buy, _execute_sell,
_execute_short, and _execute_cover with ~20 async test cases.
"""

from __future__ import annotations

import importlib
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from coin_trader.config.settings import Settings
from coin_trader.core.models import (
    ExchangeName,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    Signal,
    SignalType,
)
from coin_trader.llm.advisory import LLMAdvice

# ---------------------------------------------------------------------------
# Module-level imports via importlib (mirrors existing test_tick_loop.py pattern)
# ---------------------------------------------------------------------------
_main_mod = importlib.import_module("coin_trader.main")
_log_and_notify_order = _main_mod._log_and_notify_order
_execute_buy = _main_mod._execute_buy
_execute_sell = _main_mod._execute_sell
_execute_short = _main_mod._execute_short
_execute_cover = _main_mod._execute_cover


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings() -> Settings:
    return Settings.load_safe()


def _make_order(
    *,
    side: OrderSide = OrderSide.BUY,
    symbol: str = "BTC/KRW",
    quantity: Decimal = Decimal("0.001"),
    price: Decimal = Decimal("50000000"),
) -> Order:
    return Order(
        client_order_id="test-cid",
        intent_id=uuid4(),
        exchange=ExchangeName.UPBIT,
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        price=price,
        status=OrderStatus.SUBMITTED,
    )


def _make_signal(
    symbol: str = "BTC/KRW",
    signal_type: str = "buy",
    confidence: str = "0.8",
) -> Signal:
    return Signal(
        strategy_name="test",
        symbol=symbol,
        signal_type=SignalType(signal_type),
        timestamp=datetime.now(UTC),
        confidence=Decimal(confidence),
        metadata={"reason": "test"},
    )


def _make_advice(
    action: str = "BUY_CONSIDER",
    confidence: str = "0.85",
    buy_pct: Decimal | None = Decimal("20"),
    sell_pct: Decimal | None = None,
    short_pct: Decimal | None = None,
    order_type: str = "limit",
) -> LLMAdvice:
    return LLMAdvice(
        action=action,
        confidence=Decimal(confidence),
        reasoning="test reasoning",
        risk_notes="",
        buy_pct=buy_pct,
        sell_pct=sell_pct,
        short_pct=short_pct,
        order_type=order_type,
    )


def _make_short_position(
    symbol: str = "BTC/USDT",
    quantity: Decimal = Decimal("0.01"),
    price: Decimal = Decimal("50000"),
) -> Position:
    return Position(
        exchange=ExchangeName.BINANCE,
        symbol=symbol,
        quantity=quantity,
        average_entry_price=price,
        current_price=price,
        timestamp=datetime.now(UTC),
        side=PositionSide.SHORT,
    )


def _make_long_position(
    symbol: str = "BTC/KRW",
    quantity: Decimal = Decimal("0.01"),
    price: Decimal = Decimal("50000000"),
) -> Position:
    return Position(
        exchange=ExchangeName.UPBIT,
        symbol=symbol,
        quantity=quantity,
        average_entry_price=price,
        current_price=price,
        timestamp=datetime.now(UTC),
        side=PositionSide.LONG,
    )


def _make_engine(order: Order | None = None) -> Any:
    engine = Mock()
    engine.execute = AsyncMock(return_value=order)
    return engine


def _make_notifier() -> Any:
    notifier = Mock()
    notifier.send_alert = AsyncMock(return_value=None)
    return notifier


def _make_state() -> dict[str, Any]:
    return {}


_NOW = datetime.now(UTC)
_SYMBOL = "BTC/KRW"
_TRADE_PRICE = Decimal("50000000")
_TOTAL_BALANCE = Decimal("1000000")


# ---------------------------------------------------------------------------
# _log_and_notify_order tests (4 tests)
# ---------------------------------------------------------------------------


async def test_log_and_notify_order_logs_event() -> None:
    """_log_and_notify_order always calls log_event with 'order' type."""
    order = _make_order()
    settings = _settings()
    events: list[tuple[str, dict[str, object]]] = []

    def _log_event(event_type: str, data: dict[str, object]) -> None:
        events.append((event_type, data))

    with patch("coin_trader.main.log_event", new=_log_event):
        await _log_and_notify_order(
            order,
            settings=settings,
            notifier=None,
            sizing_pct=Decimal("20"),
            alert_title="Test Buy",
            alert_label="BUY",
            severity="medium",
        )

    assert len(events) == 1
    assert events[0][0] == "order"
    assert events[0][1]["symbol"] == "BTC/KRW"


async def test_log_and_notify_order_notifies_in_live_mode() -> None:
    """Notifier.send_alert is called when is_live_mode() returns True."""
    order = _make_order()
    settings = _settings()
    notifier = _make_notifier()

    # Patch is_live_mode at the class level to avoid Pydantic frozen-model restrictions
    with patch("coin_trader.main.log_event"):
        with patch.object(type(settings), "is_live_mode", return_value=True):
            await _log_and_notify_order(
                order,
                settings=settings,
                notifier=notifier,
                sizing_pct=Decimal("20"),
                alert_title="Live Buy",
                alert_label="BUY",
                severity="medium",
            )

    notifier.send_alert.assert_awaited_once()
    call_args = notifier.send_alert.call_args
    assert call_args[0][0] == "Live Buy"


async def test_log_and_notify_order_skips_notification_when_no_notifier() -> None:
    """No error when notifier is None."""
    order = _make_order()
    settings = _settings()

    with patch("coin_trader.main.log_event"):
        # Should not raise
        await _log_and_notify_order(
            order,
            settings=settings,
            notifier=None,
            sizing_pct=Decimal("20"),
            alert_title="Test",
            alert_label="BUY",
            severity="medium",
        )


async def test_log_and_notify_order_skips_notification_when_not_live() -> None:
    """Notifier.send_alert is NOT called in paper (non-live) mode."""
    order = _make_order()
    settings = _settings()
    notifier = _make_notifier()

    # Patch is_live_mode at class level to force paper (non-live) mode
    with patch("coin_trader.main.log_event"):
        with patch.object(type(settings), "is_live_mode", return_value=False):
            await _log_and_notify_order(
                order,
                settings=settings,
                notifier=notifier,
                sizing_pct=Decimal("20"),
                alert_title="Paper Buy",
                alert_label="BUY",
                severity="medium",
            )

    notifier.send_alert.assert_not_awaited()


# ---------------------------------------------------------------------------
# _execute_buy tests (5 tests)
# ---------------------------------------------------------------------------


async def test_execute_buy_cooldown_blocks_execution() -> None:
    """Buy is skipped when cooldown has not elapsed for the symbol."""
    engine = _make_engine()
    settings = _settings()

    _main_mod._last_buy_ts[_SYMBOL] = time.time()  # cooldown active
    try:
        with patch("coin_trader.main.log_event"):
            await _execute_buy(
                symbol=_SYMBOL,
                signals=[_make_signal()],
                advice=_make_advice(),
                settings=settings,
                engine=engine,
                state=_make_state(),
                notifier=None,
                trade_price=_TRADE_PRICE,
                total_balance=_TOTAL_BALANCE,
                now=_NOW,
                surge_context=None,
            )
        engine.execute.assert_not_awaited()
    finally:
        _main_mod._last_buy_ts.pop(_SYMBOL, None)


async def test_execute_buy_surge_gating_skips_when_cache_full_low_confidence() -> None:
    """Surge buy skipped when universe is full and confidence < 0.8."""
    engine = _make_engine()
    settings = _settings()
    surge_sym = "XRP/KRW"

    # Fill the cache to max_symbols
    max_sym = int(settings.dynamic_symbol_max_symbols)
    orig_cache = list(_main_mod._dynamic_symbols_cache)
    orig_last_buy = dict(_main_mod._last_buy_ts)
    try:
        _main_mod._dynamic_symbols_cache.clear()
        for i in range(max_sym):
            _main_mod._dynamic_symbols_cache.append(f"FAKE{i}/KRW")

        with patch("coin_trader.main.log_event"):
            await _execute_buy(
                symbol=surge_sym,
                signals=[_make_signal(symbol=surge_sym)],
                advice=_make_advice(confidence="0.7"),  # < 0.8
                settings=settings,
                engine=engine,
                state=_make_state(),
                notifier=None,
                trade_price=_TRADE_PRICE,
                total_balance=_TOTAL_BALANCE,
                now=_NOW,
                surge_context={"is_surge": True},
            )
        engine.execute.assert_not_awaited()
    finally:
        _main_mod._dynamic_symbols_cache.clear()
        _main_mod._dynamic_symbols_cache.extend(orig_cache)
        _main_mod._last_buy_ts.clear()
        _main_mod._last_buy_ts.update(orig_last_buy)


async def test_execute_buy_skips_when_order_value_below_minimum() -> None:
    """Buy is skipped when computed order value is below the minimum threshold."""
    engine = _make_engine()
    settings = _settings()

    orig_last_buy = dict(_main_mod._last_buy_ts)
    try:
        _main_mod._last_buy_ts.pop(_SYMBOL, None)
        with patch("coin_trader.main.log_event"):
            await _execute_buy(
                symbol=_SYMBOL,
                signals=[_make_signal()],
                advice=_make_advice(buy_pct=Decimal("1")),
                settings=settings,
                engine=engine,
                state=_make_state(),
                notifier=None,
                trade_price=_TRADE_PRICE,
                total_balance=Decimal("100"),  # 1% of 100 = 1 KRW < 5000 minimum
                now=_NOW,
                surge_context=None,
            )
        engine.execute.assert_not_awaited()
    finally:
        _main_mod._last_buy_ts.clear()
        _main_mod._last_buy_ts.update(orig_last_buy)


async def test_execute_buy_successful_updates_last_buy_ts() -> None:
    """Successful buy calls engine.execute and sets _last_buy_ts."""
    order = _make_order()
    engine = _make_engine(order)
    settings = _settings()

    orig_last_buy = dict(_main_mod._last_buy_ts)
    try:
        _main_mod._last_buy_ts.pop(_SYMBOL, None)
        with patch("coin_trader.main.log_event"):
            await _execute_buy(
                symbol=_SYMBOL,
                signals=[_make_signal()],
                advice=_make_advice(buy_pct=Decimal("20")),
                settings=settings,
                engine=engine,
                state=_make_state(),
                notifier=None,
                trade_price=_TRADE_PRICE,
                total_balance=_TOTAL_BALANCE,
                now=_NOW,
                surge_context=None,
            )
        engine.execute.assert_awaited_once()
        assert _SYMBOL in _main_mod._last_buy_ts
    finally:
        _main_mod._last_buy_ts.clear()
        _main_mod._last_buy_ts.update(orig_last_buy)


async def test_execute_buy_surge_symbol_added_to_cache_on_success() -> None:
    """When surge_context is set and buy succeeds, symbol is added to _dynamic_symbols_cache."""
    order = _make_order(symbol="XRP/KRW")
    engine = _make_engine(order)
    settings = _settings()
    surge_sym = "XRP/KRW"

    orig_cache = list(_main_mod._dynamic_symbols_cache)
    orig_last_buy = dict(_main_mod._last_buy_ts)
    orig_surge_cooldowns = dict(_main_mod._surge_cooldowns)
    try:
        _main_mod._dynamic_symbols_cache.clear()
        _main_mod._last_buy_ts.pop(surge_sym, None)

        with patch("coin_trader.main.log_event"):
            await _execute_buy(
                symbol=surge_sym,
                signals=[_make_signal(symbol=surge_sym)],
                advice=_make_advice(buy_pct=Decimal("10"), confidence="0.85"),
                settings=settings,
                engine=engine,
                state=_make_state(),
                notifier=None,
                trade_price=_TRADE_PRICE,
                total_balance=_TOTAL_BALANCE,
                now=_NOW,
                surge_context={"is_surge": True},
            )

        assert surge_sym in _main_mod._dynamic_symbols_cache
    finally:
        _main_mod._dynamic_symbols_cache.clear()
        _main_mod._dynamic_symbols_cache.extend(orig_cache)
        _main_mod._last_buy_ts.clear()
        _main_mod._last_buy_ts.update(orig_last_buy)
        _main_mod._surge_cooldowns.clear()
        _main_mod._surge_cooldowns.update(orig_surge_cooldowns)


# ---------------------------------------------------------------------------
# _execute_sell tests (4 tests)
# ---------------------------------------------------------------------------


async def test_execute_sell_quick_flip_blocked() -> None:
    """Sell is skipped when buy happened within the cooldown window."""
    engine = _make_engine()
    settings = _settings()
    position = _make_long_position()

    _main_mod._last_buy_ts[_SYMBOL] = time.time()  # fresh buy
    try:
        with patch("coin_trader.main.log_event"):
            await _execute_sell(
                symbol=_SYMBOL,
                signals=[_make_signal(signal_type="sell")],
                advice=_make_advice(action="SELL_CONSIDER"),
                settings=settings,
                engine=engine,
                state=_make_state(),
                notifier=None,
                positions=[position],
                trade_price=_TRADE_PRICE,
                now=_NOW,
            )
        engine.execute.assert_not_awaited()
    finally:
        _main_mod._last_buy_ts.pop(_SYMBOL, None)


async def test_execute_sell_partial_sell_respects_sell_pct() -> None:
    """Partial sell: engine.execute is called with sell_pct=50 of position."""
    order = _make_order(side=OrderSide.SELL, quantity=Decimal("0.005"))
    engine = _make_engine(order)
    settings = _settings()
    position = _make_long_position(quantity=Decimal("0.01"))

    orig_last_buy = dict(_main_mod._last_buy_ts)
    try:
        # Set last buy far in the past so cooldown is elapsed
        _main_mod._last_buy_ts[_SYMBOL] = time.time() - _main_mod._BUY_COOLDOWN_SECONDS - 1

        with patch("coin_trader.main.log_event"):
            await _execute_sell(
                symbol=_SYMBOL,
                signals=[_make_signal(signal_type="sell")],
                advice=_make_advice(
                    action="SELL_CONSIDER",
                    sell_pct=Decimal("50"),
                    buy_pct=None,
                ),
                settings=settings,
                engine=engine,
                state=_make_state(),
                notifier=None,
                positions=[position],
                trade_price=_TRADE_PRICE,
                now=_NOW,
            )

        engine.execute.assert_awaited_once()
        intent = engine.execute.call_args[0][0]
        # 50% of 0.01 = 0.005; remaining value = 0.005 * 50M = 250000 > min 5000 so partial
        assert intent.quantity == Decimal("0.005")
    finally:
        _main_mod._last_buy_ts.clear()
        _main_mod._last_buy_ts.update(orig_last_buy)


async def test_execute_sell_full_sell_when_remainder_below_minimum() -> None:
    """When remainder after partial sell is below min order value, sell full qty."""
    order = _make_order(side=OrderSide.SELL, quantity=Decimal("0.01"))
    engine = _make_engine(order)
    settings = _settings()

    # Small position: 50% of 0.0002 @ 50M = 5000 KRW remainder, just at edge
    # Use a price where remainder would be just below minimum after partial sell
    position = _make_long_position(quantity=Decimal("0.0001"), price=Decimal("50000000"))

    orig_last_buy = dict(_main_mod._last_buy_ts)
    try:
        _main_mod._last_buy_ts[_SYMBOL] = time.time() - _main_mod._BUY_COOLDOWN_SECONDS - 1

        with patch("coin_trader.main.log_event"):
            await _execute_sell(
                symbol=_SYMBOL,
                signals=[_make_signal(signal_type="sell")],
                advice=_make_advice(
                    action="SELL_CONSIDER",
                    sell_pct=Decimal("50"),
                    buy_pct=None,
                ),
                settings=settings,
                engine=engine,
                state=_make_state(),
                notifier=None,
                positions=[position],
                trade_price=Decimal("50000000"),
                now=_NOW,
            )

        engine.execute.assert_awaited_once()
        intent = engine.execute.call_args[0][0]
        # 50% of 0.0001 = 0.00005; remainder = 0.00005 * 50000000 = 2500 < 5000 min
        # So full qty (0.0001) should be used
        assert intent.quantity == Decimal("0.0001")
    finally:
        _main_mod._last_buy_ts.clear()
        _main_mod._last_buy_ts.update(orig_last_buy)


async def test_execute_sell_uses_sell_pct_from_advice() -> None:
    """sell_pct field on advice is used to compute sell quantity."""
    order = _make_order(side=OrderSide.SELL)
    engine = _make_engine(order)
    settings = _settings()
    position = _make_long_position(quantity=Decimal("1.0"))

    orig_last_buy = dict(_main_mod._last_buy_ts)
    try:
        _main_mod._last_buy_ts[_SYMBOL] = time.time() - _main_mod._BUY_COOLDOWN_SECONDS - 1

        with patch("coin_trader.main.log_event"):
            await _execute_sell(
                symbol=_SYMBOL,
                signals=[_make_signal(signal_type="sell")],
                advice=_make_advice(
                    action="SELL_CONSIDER",
                    sell_pct=Decimal("75"),
                    buy_pct=None,
                ),
                settings=settings,
                engine=engine,
                state=_make_state(),
                notifier=None,
                positions=[position],
                trade_price=_TRADE_PRICE,
                now=_NOW,
            )

        engine.execute.assert_awaited_once()
        intent = engine.execute.call_args[0][0]
        # 75% of 1.0 = 0.75; remaining = 0.25 * 50M = 12500000 >> 5000 min
        assert intent.quantity == Decimal("0.75")
    finally:
        _main_mod._last_buy_ts.clear()
        _main_mod._last_buy_ts.update(orig_last_buy)


# ---------------------------------------------------------------------------
# _execute_short tests (3 tests)
# ---------------------------------------------------------------------------


async def test_execute_short_basic_calls_engine() -> None:
    """Basic short: engine.execute is called with SELL side and SHORT position_side."""
    order = _make_order(side=OrderSide.SELL, symbol="BTC/USDT")
    engine = _make_engine(order)
    settings = _settings()

    with patch("coin_trader.main.log_event"):
        await _execute_short(
            symbol="BTC/USDT",
            signals=[_make_signal(symbol="BTC/USDT", signal_type="short")],
            advice=_make_advice(action="SHORT_CONSIDER", short_pct=Decimal("10")),
            settings=settings,
            engine=engine,
            state=_make_state(),
            notifier=None,
            trade_price=Decimal("50000"),
            total_balance=Decimal("10000"),
            now=_NOW,
        )

    engine.execute.assert_awaited_once()
    intent = engine.execute.call_args[0][0]
    assert intent.side.value == "sell"
    assert intent.position_side == PositionSide.SHORT


async def test_execute_short_pct_capped_by_max_position_size() -> None:
    """short_pct from advice is capped at settings.risk.max_position_size_pct."""
    order = _make_order(side=OrderSide.SELL, symbol="BTC/USDT")
    engine = _make_engine(order)
    settings = _settings()
    max_pct = settings.risk.max_position_size_pct  # 30

    with patch("coin_trader.main.log_event"):
        await _execute_short(
            symbol="BTC/USDT",
            signals=[_make_signal(symbol="BTC/USDT", signal_type="short")],
            advice=_make_advice(action="SHORT_CONSIDER", short_pct=Decimal("99")),
            settings=settings,
            engine=engine,
            state=_make_state(),
            notifier=None,
            trade_price=Decimal("50000"),
            total_balance=Decimal("10000"),
            now=_NOW,
        )

    engine.execute.assert_awaited_once()
    intent = engine.execute.call_args[0][0]
    # order_value = 10000 * max_pct / 100
    expected_value = Decimal("10000") * max_pct / Decimal("100")
    assert intent.quote_quantity == expected_value


async def test_execute_short_zero_balance_skips() -> None:
    """Short is skipped when total_balance is 0 (zero order value)."""
    engine = _make_engine()
    settings = _settings()

    with patch("coin_trader.main.log_event"):
        await _execute_short(
            symbol="BTC/USDT",
            signals=[_make_signal(symbol="BTC/USDT", signal_type="short")],
            advice=_make_advice(action="SHORT_CONSIDER", short_pct=Decimal("10")),
            settings=settings,
            engine=engine,
            state=_make_state(),
            notifier=None,
            trade_price=Decimal("50000"),
            total_balance=Decimal("0"),
            now=_NOW,
        )

    engine.execute.assert_not_awaited()


# ---------------------------------------------------------------------------
# _execute_cover tests (4 tests)
# ---------------------------------------------------------------------------


async def test_execute_cover_covers_short_position() -> None:
    """Cover: engine.execute called with BUY side and reduce_only=True for SHORT position."""
    short_pos = _make_short_position()
    order = _make_order(side=OrderSide.BUY, symbol="BTC/USDT")
    engine = _make_engine(order)
    settings = _settings()

    orig_low = dict(_main_mod._low_watermarks)
    try:
        with patch("coin_trader.main.log_event"):
            await _execute_cover(
                symbol="BTC/USDT",
                signals=[_make_signal(symbol="BTC/USDT", signal_type="cover")],
                advice=_make_advice(action="COVER_CONSIDER", sell_pct=Decimal("100")),
                settings=settings,
                engine=engine,
                state=_make_state(),
                notifier=None,
                positions=[short_pos],
                trade_price=Decimal("50000"),
                now=_NOW,
            )

        engine.execute.assert_awaited_once()
        intent = engine.execute.call_args[0][0]
        assert intent.side == OrderSide.BUY
        assert intent.reduce_only is True
        assert intent.position_side == PositionSide.SHORT
    finally:
        _main_mod._low_watermarks.clear()
        _main_mod._low_watermarks.update(orig_low)


async def test_execute_cover_partial_cover_uses_sell_pct() -> None:
    """Partial cover: cover_qty = quantity * sell_pct / 100."""
    short_pos = _make_short_position(quantity=Decimal("1.0"))
    order = _make_order(side=OrderSide.BUY, symbol="BTC/USDT")
    engine = _make_engine(order)
    settings = _settings()

    orig_low = dict(_main_mod._low_watermarks)
    try:
        with patch("coin_trader.main.log_event"):
            await _execute_cover(
                symbol="BTC/USDT",
                signals=[_make_signal(symbol="BTC/USDT", signal_type="cover")],
                advice=_make_advice(
                    action="COVER_CONSIDER",
                    sell_pct=Decimal("50"),
                    buy_pct=None,
                ),
                settings=settings,
                engine=engine,
                state=_make_state(),
                notifier=None,
                positions=[short_pos],
                trade_price=Decimal("50000"),
                now=_NOW,
            )

        engine.execute.assert_awaited_once()
        intent = engine.execute.call_args[0][0]
        assert intent.quantity == Decimal("0.5")
    finally:
        _main_mod._low_watermarks.clear()
        _main_mod._low_watermarks.update(orig_low)


async def test_execute_cover_pops_low_watermark_on_success() -> None:
    """After a successful cover, _low_watermarks entry for the symbol is removed."""
    short_pos = _make_short_position()
    order = _make_order(side=OrderSide.BUY, symbol="BTC/USDT")
    engine = _make_engine(order)
    settings = _settings()

    orig_low = dict(_main_mod._low_watermarks)
    try:
        _main_mod._low_watermarks["BTC/USDT"] = Decimal("48000")

        with patch("coin_trader.main.log_event"):
            await _execute_cover(
                symbol="BTC/USDT",
                signals=[_make_signal(symbol="BTC/USDT", signal_type="cover")],
                advice=_make_advice(action="COVER_CONSIDER", sell_pct=Decimal("100")),
                settings=settings,
                engine=engine,
                state=_make_state(),
                notifier=None,
                positions=[short_pos],
                trade_price=Decimal("50000"),
                now=_NOW,
            )

        assert "BTC/USDT" not in _main_mod._low_watermarks
    finally:
        _main_mod._low_watermarks.clear()
        _main_mod._low_watermarks.update(orig_low)


async def test_execute_cover_skips_non_short_positions() -> None:
    """Cover is skipped when the position side is LONG (not SHORT)."""
    long_pos = _make_long_position()
    engine = _make_engine()
    settings = _settings()

    with patch("coin_trader.main.log_event"):
        await _execute_cover(
            symbol=_SYMBOL,
            signals=[_make_signal(signal_type="cover")],
            advice=_make_advice(action="COVER_CONSIDER", sell_pct=Decimal("100")),
            settings=settings,
            engine=engine,
            state=_make_state(),
            notifier=None,
            positions=[long_pos],
            trade_price=_TRADE_PRICE,
            now=_NOW,
        )

    engine.execute.assert_not_awaited()
