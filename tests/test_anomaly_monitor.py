"""Tests for AnomalyMonitor and helpers in coin_trader/safety/monitor.py"""

from datetime import UTC, datetime
from decimal import Decimal

from coin_trader.core.models import ExchangeName, MarketData, SafetyEventType
from coin_trader.safety.monitor import AnomalyMonitor, _cfg_decimal, _cfg_int

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _market_data(close: Decimal, symbol: str = "BTC/KRW") -> MarketData:
    return MarketData(
        exchange=ExchangeName.UPBIT,
        symbol=symbol,
        timestamp=datetime.now(UTC),
        open=Decimal("100"),
        high=Decimal("100"),
        low=Decimal("100"),
        close=close,
        volume=Decimal("1"),
    )


def _market_data_with_spread(bid: Decimal, ask: Decimal, symbol: str = "BTC/KRW") -> MarketData:
    return MarketData(
        exchange=ExchangeName.UPBIT,
        symbol=symbol,
        timestamp=datetime.now(UTC),
        open=Decimal("100"),
        high=Decimal("100"),
        low=Decimal("100"),
        close=ask,
        volume=Decimal("1"),
        bid=bid,
        ask=ask,
    )


# ---------------------------------------------------------------------------
# _cfg_int
# ---------------------------------------------------------------------------


def test_cfg_int_missing_key_returns_default() -> None:
    assert _cfg_int({}, "missing", 7) == 7


def test_cfg_int_int_value_returned_as_is() -> None:
    assert _cfg_int({"k": 42}, "k", 0) == 42


def test_cfg_int_float_truncated() -> None:
    assert _cfg_int({"k": 3.9}, "k", 0) == 3


def test_cfg_int_str_numeric() -> None:
    assert _cfg_int({"k": "42"}, "k", 0) == 42


def test_cfg_int_bool_true_is_one() -> None:
    assert _cfg_int({"k": True}, "k", 0) == 1


def test_cfg_int_invalid_str_returns_default() -> None:
    assert _cfg_int({"k": "abc"}, "k", 99) == 99


def test_cfg_int_none_value_returns_default() -> None:
    assert _cfg_int({"k": None}, "k", 5) == 5


# ---------------------------------------------------------------------------
# _cfg_decimal
# ---------------------------------------------------------------------------


def test_cfg_decimal_missing_key_returns_default() -> None:
    default = Decimal("9.9")
    assert _cfg_decimal({}, "missing", default) == default


def test_cfg_decimal_decimal_value_returned_as_is() -> None:
    val = Decimal("1.23")
    assert _cfg_decimal({"k": val}, "k", Decimal("0")) == val


def test_cfg_decimal_int_converted() -> None:
    assert _cfg_decimal({"k": 5}, "k", Decimal("0")) == Decimal("5")


def test_cfg_decimal_float_converted_via_str() -> None:
    result = _cfg_decimal({"k": 1.5}, "k", Decimal("0"))
    assert result == Decimal("1.5")


def test_cfg_decimal_str_numeric() -> None:
    assert _cfg_decimal({"k": "1.5"}, "k", Decimal("0")) == Decimal("1.5")


def test_cfg_decimal_invalid_str_returns_default() -> None:
    default = Decimal("0")
    assert _cfg_decimal({"k": "not_a_number"}, "k", default) == default


def test_cfg_decimal_none_value_returns_default() -> None:
    default = Decimal("3.14")
    assert _cfg_decimal({"k": None}, "k", default) == default


# ---------------------------------------------------------------------------
# AnomalyMonitor.record_api_success
# ---------------------------------------------------------------------------


def test_record_api_success_resets_failure_counter() -> None:
    monitor = AnomalyMonitor({"api_failure_threshold": 2})
    # First failure — below threshold, returns None
    result = monitor.record_api_failure()
    assert result is None
    # Success resets counter
    monitor.record_api_success()
    # Another failure — still below threshold after reset
    result = monitor.record_api_failure()
    assert result is None


# ---------------------------------------------------------------------------
# AnomalyMonitor.record_api_failure
# ---------------------------------------------------------------------------


def test_record_api_failure_below_threshold_returns_none() -> None:
    monitor = AnomalyMonitor({"api_failure_threshold": 5})
    for _ in range(4):
        result = monitor.record_api_failure()
        assert result is None


def test_record_api_failure_at_threshold_returns_event() -> None:
    monitor = AnomalyMonitor({"api_failure_threshold": 3})
    monitor.record_api_failure()
    monitor.record_api_failure()
    event = monitor.record_api_failure()
    assert event is not None
    assert event.event_type == SafetyEventType.API_FAILURE


# ---------------------------------------------------------------------------
# AnomalyMonitor.check_balance_mismatch
# ---------------------------------------------------------------------------


def test_check_balance_mismatch_both_zero_returns_none() -> None:
    monitor = AnomalyMonitor({})
    assert monitor.check_balance_mismatch(Decimal("0"), Decimal("0")) is None


def test_check_balance_mismatch_within_threshold_returns_none() -> None:
    # Default threshold is 1.0%
    monitor = AnomalyMonitor({})
    # 0.5% difference — below threshold
    assert monitor.check_balance_mismatch(Decimal("1000"), Decimal("1005")) is None


def test_check_balance_mismatch_exceeds_threshold_returns_event() -> None:
    monitor = AnomalyMonitor({"balance_mismatch_threshold_pct": "1.0"})
    # 10% difference — above threshold
    event = monitor.check_balance_mismatch(Decimal("1000"), Decimal("1100"))
    assert event is not None
    assert event.event_type == SafetyEventType.BALANCE_MISMATCH


# ---------------------------------------------------------------------------
# AnomalyMonitor.check_price_anomaly
# ---------------------------------------------------------------------------


def test_check_price_anomaly_first_call_returns_none() -> None:
    monitor = AnomalyMonitor({})
    md = _market_data(Decimal("100"))
    assert monitor.check_price_anomaly(md) is None


def test_check_price_anomaly_small_change_returns_none() -> None:
    monitor = AnomalyMonitor({"price_change_threshold_pct": "10.0"})
    monitor.check_price_anomaly(_market_data(Decimal("100")))
    # 1% change — well below threshold
    result = monitor.check_price_anomaly(_market_data(Decimal("101")))
    assert result is None


def test_check_price_anomaly_large_change_returns_event() -> None:
    monitor = AnomalyMonitor({"price_change_threshold_pct": "10.0"})
    monitor.check_price_anomaly(_market_data(Decimal("100")))
    # 50% jump — way above 10% threshold
    event = monitor.check_price_anomaly(_market_data(Decimal("150")))
    assert event is not None
    assert event.event_type == SafetyEventType.PRICE_ANOMALY


# ---------------------------------------------------------------------------
# AnomalyMonitor.check_spread_anomaly
# ---------------------------------------------------------------------------


def test_check_spread_anomaly_no_bid_ask_returns_none() -> None:
    monitor = AnomalyMonitor({})
    md = _market_data(Decimal("100"))
    assert monitor.check_spread_anomaly(md) is None


def test_check_spread_anomaly_tight_spread_returns_none() -> None:
    monitor = AnomalyMonitor({"spread_threshold_pct": "5.0"})
    # 1% spread — below threshold
    md = _market_data_with_spread(Decimal("100"), Decimal("101"))
    assert monitor.check_spread_anomaly(md) is None


def test_check_spread_anomaly_wide_spread_returns_event() -> None:
    monitor = AnomalyMonitor({"spread_threshold_pct": "5.0"})
    # 20% spread — way above threshold
    md = _market_data_with_spread(Decimal("100"), Decimal("120"))
    event = monitor.check_spread_anomaly(md)
    assert event is not None
    assert event.event_type == SafetyEventType.PRICE_ANOMALY
