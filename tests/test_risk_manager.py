"""Tests for RiskManager safety kernel."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from coin_trader.core.models import (
    ExchangeName,
    OrderIntent,
    OrderSide,
    OrderType,
    PositionSide,
    RiskDecision,
)
from coin_trader.risk.limits import RiskLimits
from coin_trader.risk.manager import RiskManager


def make_intent(
    *,
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.LIMIT,
    quote_quantity: Decimal | None = Decimal("100000"),
    price: Decimal | None = Decimal("50000000"),
    symbol: str = "BTC/KRW",
    reduce_only: bool = False,
    position_side: PositionSide | None = None,
    quantity: Decimal | None = None,
) -> OrderIntent:
    return OrderIntent(
        signal_id=uuid4(),
        exchange=ExchangeName.UPBIT,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quote_quantity=quote_quantity,
        quantity=quantity,
        price=price,
        reason="test",
        timestamp=datetime.now(UTC),
        reduce_only=reduce_only,
        position_side=position_side,
    )


def make_state(**kwargs: object) -> dict[str, object]:
    base: dict[str, object] = {
        "total_balance": Decimal("1000000"),
        "position_count": 0,
        "market_price": Decimal("50000000"),
    }
    base.update(kwargs)
    return base


def make_manager(**limit_kwargs: object) -> RiskManager:
    limits = RiskLimits(**limit_kwargs)  # type: ignore[arg-type]
    return RiskManager(limits=limits, state_store=None)


# ---------------------------------------------------------------------------
# Test 1: Market order policy
# ---------------------------------------------------------------------------
async def test_market_order_policy_rejected() -> None:
    manager = make_manager(default_order_type="limit")
    intent = make_intent(order_type=OrderType.MARKET)
    result = await manager.validate(intent, make_state())
    assert result.decision == RiskDecision.REJECTED
    assert "Market orders not allowed" in result.reason


# ---------------------------------------------------------------------------
# Test 2: Position sizing — too large
# ---------------------------------------------------------------------------
async def test_position_sizing_rejected() -> None:
    manager = make_manager(max_position_size_pct=Decimal("30"))
    # 400000 / 1000000 = 40% > 30%
    intent = make_intent(quote_quantity=Decimal("400000"))
    result = await manager.validate(intent, make_state())
    assert result.decision == RiskDecision.REJECTED
    assert "exceeds limit" in result.reason


# ---------------------------------------------------------------------------
# Test 3: Position sizing — within limit
# ---------------------------------------------------------------------------
async def test_position_sizing_approved() -> None:
    manager = make_manager(max_position_size_pct=Decimal("30"))
    # 200000 / 1000000 = 20% < 30%
    intent = make_intent(quote_quantity=Decimal("200000"))
    result = await manager.validate(intent, make_state())
    assert result.decision == RiskDecision.APPROVED


# ---------------------------------------------------------------------------
# Test 4: Single-coin exposure
# ---------------------------------------------------------------------------
async def test_single_coin_exposure_rejected() -> None:
    # current_symbol_value=200000, order=100000 → 300000/1000000=30% > 25%
    manager = make_manager(max_single_coin_exposure_pct=Decimal("25"))
    intent = make_intent(quote_quantity=Decimal("100000"))
    state = make_state(current_symbol_value=Decimal("200000"))
    result = await manager.validate(intent, state)
    assert result.decision == RiskDecision.REJECTED
    assert "Single-coin exposure" in result.reason


# ---------------------------------------------------------------------------
# Test 5: Daily drawdown
# ---------------------------------------------------------------------------
async def test_daily_drawdown_rejected() -> None:
    # today_pnl=-60000, total=1000000 → 6% >= 5%
    manager = make_manager(daily_max_drawdown_pct=Decimal("5"))
    intent = make_intent(quote_quantity=Decimal("100000"))
    state = make_state(today_pnl=Decimal("-60000"))
    result = await manager.validate(intent, state)
    assert result.decision == RiskDecision.REJECTED
    assert "Daily drawdown" in result.reason


# ---------------------------------------------------------------------------
# Test 6: Rate limit
# ---------------------------------------------------------------------------
async def test_rate_limit_rejected() -> None:
    manager = make_manager(max_orders_per_second=1)
    now = datetime.now(UTC)
    # Fill with a recent timestamp so rate limit triggers
    manager._order_timestamps = [now]
    intent = make_intent(quote_quantity=Decimal("100000"))
    result = await manager.validate(intent, make_state())
    assert result.decision == RiskDecision.REJECTED
    assert "Rate limit exceeded" in result.reason


# ---------------------------------------------------------------------------
# Test 7: Daily order count
# ---------------------------------------------------------------------------
async def test_daily_order_count_rejected() -> None:
    manager = make_manager(max_orders_per_day=10)
    manager._daily_order_count = 10  # already at the max
    intent = make_intent(quote_quantity=Decimal("100000"))
    result = await manager.validate(intent, make_state())
    assert result.decision == RiskDecision.REJECTED
    assert "Daily order limit exceeded" in result.reason


# ---------------------------------------------------------------------------
# Test 8: Futures disabled
# ---------------------------------------------------------------------------
async def test_futures_disabled_rejected() -> None:
    manager = make_manager(futures_enabled=False)
    intent = make_intent(symbol="BTC/USDT-perp", quote_quantity=Decimal("100000"))
    result = await manager.validate(intent, make_state())
    assert result.decision == RiskDecision.REJECTED
    assert "Futures/derivatives trading is disabled" in result.reason


# ---------------------------------------------------------------------------
# Test 9: Slippage check
# ---------------------------------------------------------------------------
async def test_slippage_rejected() -> None:
    # market_price=50000000, order price=49000000 → 200bps > 100bps limit
    manager = make_manager(max_slippage_bps=100)
    intent = make_intent(
        order_type=OrderType.LIMIT,
        price=Decimal("49000000"),
        quote_quantity=Decimal("100000"),
    )
    state = make_state(market_price=Decimal("50000000"))
    result = await manager.validate(intent, state)
    assert result.decision == RiskDecision.REJECTED
    assert "Slippage" in result.reason


# ---------------------------------------------------------------------------
# Test 10: Futures max notional per position
# ---------------------------------------------------------------------------
async def test_futures_max_notional_rejected() -> None:
    # order_value=100000, leverage=10 → effective=1000000 > max_notional_per_position=500000
    manager = make_manager(
        futures_enabled=True,
        max_leverage=10,
        max_notional_per_position=Decimal("500000"),
        max_total_notional=Decimal("9999999"),
        max_slippage_bps=99999,
    )
    intent = make_intent(quote_quantity=Decimal("100000"))
    # Patch leverage attribute on the intent (code uses getattr)
    object.__setattr__(
        intent, "__dict__", dict(intent.__dict__, leverage=10),
    )

    # Use a fresh intent via a workaround: build a state that triggers the check
    # effective = 100000 * 10 = 1000000 > 500000
    state = make_state(total_notional_exposure=Decimal("0"), available_margin=Decimal("999999999"))
    result = await manager.validate(intent, state)
    assert result.decision == RiskDecision.REJECTED
    assert "Notional" in result.reason


# ---------------------------------------------------------------------------
# Test 11: Futures total notional
# ---------------------------------------------------------------------------
async def test_futures_total_notional_rejected() -> None:
    # With leverage=1, order_value=100000, total already=150000 → 250000 > max_total=200000
    manager = make_manager(
        futures_enabled=True,
        max_leverage=1,
        max_notional_per_position=Decimal("9999999"),
        max_total_notional=Decimal("200000"),
        max_slippage_bps=99999,
    )
    intent = make_intent(quote_quantity=Decimal("100000"))
    state = make_state(
        total_notional_exposure=Decimal("150000"), available_margin=Decimal("999999999"),
    )
    result = await manager.validate(intent, state)
    assert result.decision == RiskDecision.REJECTED
    assert "Total notional" in result.reason


# ---------------------------------------------------------------------------
# Test 12: Futures margin insufficient
# ---------------------------------------------------------------------------
async def test_futures_margin_insufficient_rejected() -> None:
    # required_margin = order_value = 100000, available = 50000
    manager = make_manager(
        futures_enabled=True,
        max_leverage=1,
        max_notional_per_position=Decimal("9999999"),
        max_total_notional=Decimal("9999999"),
        max_slippage_bps=99999,
    )
    intent = make_intent(quote_quantity=Decimal("100000"))
    state = make_state(
        total_notional_exposure=Decimal("0"),
        available_margin=Decimal("50000"),
    )
    result = await manager.validate(intent, state)
    assert result.decision == RiskDecision.REJECTED
    assert "Insufficient margin" in result.reason


# ---------------------------------------------------------------------------
# Test 13: Futures liquidation proximity
# ---------------------------------------------------------------------------
async def test_futures_liquidation_proximity_rejected() -> None:
    # market=50000, liquidation=48000 → dist=4% < threshold=20%
    manager = make_manager(
        futures_enabled=True,
        max_leverage=1,
        max_notional_per_position=Decimal("9999999"),
        max_total_notional=Decimal("9999999"),
        liquidation_warning_threshold_pct=Decimal("20"),
        max_slippage_bps=99999,
    )
    intent = make_intent(
        quote_quantity=Decimal("100000"),
        price=Decimal("50000"),
    )
    state = make_state(
        market_price=Decimal("50000"),
        liquidation_price=Decimal("48000"),
        total_notional_exposure=Decimal("0"),
        available_margin=Decimal("999999999"),
    )
    result = await manager.validate(intent, state)
    assert result.decision == RiskDecision.REJECTED
    assert "liquidation" in result.reason.lower()


# ---------------------------------------------------------------------------
# Test 14: Futures funding rate
# ---------------------------------------------------------------------------
async def test_futures_funding_rate_rejected() -> None:
    # Long BUY + positive funding_bps > max → rejected
    manager = make_manager(
        futures_enabled=True,
        max_leverage=1,
        max_notional_per_position=Decimal("9999999"),
        max_total_notional=Decimal("9999999"),
        max_funding_rate_bps=50,
        liquidation_warning_threshold_pct=Decimal("1"),  # won't trigger (large gap)
        max_slippage_bps=99999,
    )
    intent = make_intent(
        side=OrderSide.BUY,
        quote_quantity=Decimal("100000"),
        price=Decimal("50000000"),
    )
    state = make_state(
        market_price=Decimal("50000000"),
        liquidation_price=Decimal("0"),  # 0 means no liq check
        total_notional_exposure=Decimal("0"),
        available_margin=Decimal("999999999"),
        funding_rate_bps=100,  # > 50
    )
    result = await manager.validate(intent, state)
    assert result.decision == RiskDecision.REJECTED
    assert "funding rate" in result.reason.lower()


# ---------------------------------------------------------------------------
# Test 15: SELL reduce_only skips opening checks (MARKET order allowed)
# ---------------------------------------------------------------------------
async def test_sell_reduce_only_market_approved() -> None:
    manager = make_manager(default_order_type="limit")
    intent = make_intent(
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quote_quantity=None,
        quantity=Decimal("0.01"),
        price=None,
        reduce_only=True,
    )
    result = await manager.validate(intent, make_state())
    assert result.decision == RiskDecision.APPROVED


# ---------------------------------------------------------------------------
# Test 16: check_circuit_breaker (sync)
# ---------------------------------------------------------------------------
def test_check_circuit_breaker() -> None:
    manager = make_manager(circuit_breaker_threshold_pct=Decimal("15"))
    # Below threshold
    assert manager.check_circuit_breaker(Decimal("10")) is False
    # At threshold
    assert manager.check_circuit_breaker(Decimal("15")) is True
    # Above threshold
    assert manager.check_circuit_breaker(Decimal("20")) is True
    # Negative (absolute value)
    assert manager.check_circuit_breaker(Decimal("-16")) is True


# ---------------------------------------------------------------------------
# Test 17: _reset_daily_if_needed resets counters when day changes
# ---------------------------------------------------------------------------
async def test_reset_daily_if_needed() -> None:
    manager = make_manager()
    # Manually set _day_start to yesterday
    manager._day_start = datetime.now(UTC).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - timedelta(days=1)
    manager._daily_order_count = 99
    manager._daily_pnl = Decimal("-50000")

    # Call validate to trigger reset
    intent = make_intent(quote_quantity=Decimal("100000"))
    await manager.validate(intent, make_state())

    # Counters should have been reset (and then incremented once by this BUY)
    assert manager._daily_order_count == 1
    assert manager._daily_pnl == Decimal("0")


# ---------------------------------------------------------------------------
# Test 18: Happy path — all limits satisfied → APPROVED
# ---------------------------------------------------------------------------
async def test_approve_happy_path() -> None:
    manager = make_manager()
    intent = make_intent(
        quote_quantity=Decimal("200000"),
        price=Decimal("50000000"),
    )
    state = make_state(
        total_balance=Decimal("1000000"),
        market_price=Decimal("50000000"),
        today_pnl=Decimal("0"),
        current_symbol_value=Decimal("0"),
        position_count=0,
    )
    result = await manager.validate(intent, state)
    assert result.decision == RiskDecision.APPROVED
