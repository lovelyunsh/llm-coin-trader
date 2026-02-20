from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from coin_trader.config.settings import RiskLimits, Settings
from coin_trader.core.models import (
    ExchangeName,
    MarketData,
    OrderIntent,
    OrderSide,
    OrderType,
    PositionSide,
    RiskDecision,
    SignalType,
)
from coin_trader.llm.advisory import LLMAdvice
from coin_trader.risk.manager import RiskManager
from coin_trader.strategy.conservative import ConservativeStrategy


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_intent(
    *,
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.LIMIT,
    quantity: Decimal = Decimal("0.001"),
    price: Decimal | None = Decimal("50000"),
    reduce_only: bool = False,
    position_side: PositionSide | None = None,
    symbol: str = "BTC/USDT",
) -> OrderIntent:
    return OrderIntent(
        signal_id=uuid4(),
        exchange=ExchangeName.BINANCE,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        reason="test",
        timestamp=_now(),
        reduce_only=reduce_only,
        position_side=position_side,
    )


# ---------------------------------------------------------------------------
# Settings tests
# ---------------------------------------------------------------------------


def test_settings_apply_exchange_defaults_binance() -> None:
    settings = Settings.model_construct(
        exchange=ExchangeName.BINANCE,
        quote_currency="KRW",
        btc_reference_symbol="BTC/KRW",
        trading_symbols=["BTC/KRW", "ETH/KRW"],
        dynamic_symbol_min_turnover_24h=1_000_000_000,
        risk=RiskLimits(),
    )
    settings.apply_exchange_defaults()
    assert settings.quote_currency == "USDT"
    assert settings.btc_reference_symbol == "BTC/USDT"
    assert settings.trading_symbols == ["BTC/USDT", "ETH/USDT"]
    assert settings.dynamic_symbol_min_turnover_24h == 10_000_000


def test_settings_apply_exchange_defaults_upbit_noop() -> None:
    settings = Settings.model_construct(
        exchange=ExchangeName.UPBIT,
        quote_currency="KRW",
        btc_reference_symbol="BTC/KRW",
        risk=RiskLimits(),
    )
    settings.apply_exchange_defaults()
    assert settings.quote_currency == "KRW"
    assert settings.btc_reference_symbol == "BTC/KRW"


def test_risk_limits_futures_fields() -> None:
    limits = RiskLimits()
    assert limits.max_notional_per_position == Decimal("50000")
    assert limits.max_total_notional == Decimal("200000")
    assert limits.liquidation_warning_threshold_pct == Decimal("20")
    assert limits.max_funding_rate_bps == 50


def test_settings_load_safe_fallback() -> None:
    settings = Settings.load_safe()
    assert settings is not None


# ---------------------------------------------------------------------------
# LLM tests
# ---------------------------------------------------------------------------


def test_llm_advice_short_consider_valid() -> None:
    data: dict[str, object] = {
        "action": "SHORT_CONSIDER",
        "confidence": "0.8",
        "reasoning": "Strong downtrend",
        "risk_notes": "Funding rate risk",
        "short_pct": "5",
    }
    advice = LLMAdvice.from_mapping(data)
    assert advice is not None
    assert advice.action == "SHORT_CONSIDER"
    assert advice.short_pct == Decimal("5")


def test_llm_advice_cover_consider_valid() -> None:
    data: dict[str, object] = {
        "action": "COVER_CONSIDER",
        "confidence": "0.7",
        "reasoning": "Reversal signal",
        "risk_notes": "Bounce risk",
    }
    advice = LLMAdvice.from_mapping(data)
    assert advice is not None
    assert advice.action == "COVER_CONSIDER"


def test_llm_advice_short_pct_clamped() -> None:
    data: dict[str, object] = {
        "action": "SHORT_CONSIDER",
        "confidence": "0.8",
        "reasoning": "test",
        "risk_notes": "",
        "short_pct": "150",
    }
    advice = LLMAdvice.from_mapping(data)
    assert advice is not None
    assert advice.short_pct == Decimal("100")


# ---------------------------------------------------------------------------
# RiskManager tests
# ---------------------------------------------------------------------------


async def test_risk_is_opening_buy() -> None:
    limits = RiskLimits()
    manager = RiskManager(limits=limits, state_store=object())
    intent = _make_intent(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=Decimal("50000"))
    state: dict[str, object] = {
        "total_balance": Decimal("1000000"),
        "today_pnl": Decimal("0"),
        "position_count": 0,
    }
    record = await manager.validate(intent, state)
    assert record.decision == RiskDecision.APPROVED


async def test_risk_is_opening_short_entry() -> None:
    limits = RiskLimits(futures_enabled=True)
    manager = RiskManager(limits=limits, state_store=object())
    intent = _make_intent(
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=Decimal("50000"),
        position_side=PositionSide.SHORT,
        reduce_only=False,
    )
    state: dict[str, object] = {
        "total_balance": Decimal("1000000"),
        "today_pnl": Decimal("0"),
        "position_count": 0,
        "total_notional_exposure": Decimal("0"),
    }
    record = await manager.validate(intent, state)
    # Should not be rejected for wrong reasons; approve or reject on risk grounds only
    assert record.decision in (RiskDecision.APPROVED, RiskDecision.REJECTED)
    # Must not be rejected due to "Futures/derivatives trading is disabled"
    assert "disabled" not in record.reason


async def test_risk_reduce_only_not_opening() -> None:
    limits = RiskLimits()
    manager = RiskManager(limits=limits, state_store=object())
    # reduce_only=True means this is a cover/close, not an opening position
    intent = _make_intent(
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        price=None,
        reduce_only=True,
    )
    state: dict[str, object] = {
        "total_balance": Decimal("1000000"),
        "today_pnl": Decimal("0"),
        "position_count": 1,
    }
    record = await manager.validate(intent, state)
    # reduce_only orders skip position-sizing checks and market-order policy
    assert record.decision == RiskDecision.APPROVED


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------


async def test_strategy_futures_disabled_no_short_signals() -> None:
    strategy = ConservativeStrategy(futures_enabled=False)
    md = MarketData(
        symbol="BTC/USDT",
        exchange=ExchangeName.BINANCE,
        timestamp=_now(),
        open=Decimal("50000"),
        high=Decimal("51000"),
        low=Decimal("49000"),
        close=Decimal("50000"),
        volume=Decimal("1000"),
    )
    signals = await strategy.on_tick(md)
    signal_types = {s.signal_type for s in signals}
    assert SignalType.SHORT not in signal_types
    assert SignalType.COVER not in signal_types


async def test_strategy_futures_enabled_flag() -> None:
    strategy = ConservativeStrategy(futures_enabled=True)
    assert strategy._futures_enabled is True
