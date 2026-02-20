from __future__ import annotations

from coin_trader.exchange.upbit import (
    _normalize_candle,
    _normalize_orderbook,
    _normalize_ticker,
    _to_upbit_interval,
)


# ---------------------------------------------------------------------------
# _to_upbit_interval
# ---------------------------------------------------------------------------


def test_to_upbit_interval_1h() -> None:
    assert _to_upbit_interval("1h") == "minutes/60"


def test_to_upbit_interval_4h() -> None:
    assert _to_upbit_interval("4h") == "minutes/240"


def test_to_upbit_interval_1d() -> None:
    assert _to_upbit_interval("1d") == "days"


def test_to_upbit_interval_1w() -> None:
    assert _to_upbit_interval("1w") == "weeks"


def test_to_upbit_interval_1M() -> None:
    assert _to_upbit_interval("1M") == "months"


def test_to_upbit_interval_passthrough_raw_upbit_format() -> None:
    # Values already in Upbit format are returned unchanged.
    assert _to_upbit_interval("minutes/60") == "minutes/60"


# ---------------------------------------------------------------------------
# _normalize_ticker
# ---------------------------------------------------------------------------


def test_normalize_ticker_canonical_keys_present() -> None:
    raw = {
        "trade_price": 95000000,
        "opening_price": 94000000,
        "high_price": 96000000,
        "low_price": 93000000,
        "acc_trade_volume_24h": 1234.5,
        "highest_bid": 94999000,
        "lowest_ask": 95001000,
        "prev_closing_price": 94500000,
        "acc_trade_price_24h": 117000000000,
        "signed_change_rate": 0.005,
    }
    result = _normalize_ticker(raw)

    expected_keys = {
        "price",
        "open",
        "high",
        "low",
        "volume_24h",
        "bid",
        "ask",
        "prev_close",
        "turnover_24h",
        "change_rate",
    }
    assert expected_keys <= result.keys()


def test_normalize_ticker_price_value() -> None:
    raw = {
        "trade_price": 95000000,
        "opening_price": 94000000,
        "high_price": 96000000,
        "low_price": 93000000,
        "acc_trade_volume_24h": 1234.5,
        "highest_bid": 94999000,
        "lowest_ask": 95001000,
        "prev_closing_price": 94500000,
        "acc_trade_price_24h": 117000000000,
        "signed_change_rate": 0.005,
    }
    result = _normalize_ticker(raw)

    assert result["price"] == 95000000
    assert result["open"] == 94000000


def test_normalize_ticker_extra_key_not_in_output() -> None:
    # _normalize_ticker returns only canonical fields; raw-only keys are not forwarded.
    raw = {
        "trade_price": 95000000,
        "opening_price": 94000000,
        "high_price": 96000000,
        "low_price": 93000000,
        "acc_trade_volume_24h": 1234.5,
        "highest_bid": 94999000,
        "lowest_ask": 95001000,
        "prev_closing_price": 94500000,
        "acc_trade_price_24h": 117000000000,
        "signed_change_rate": 0.005,
        "market": "KRW-BTC",
    }
    result = _normalize_ticker(raw)

    # Canonical keys are always present.
    assert result["price"] == 95000000
    # Raw Upbit key "market" is not forwarded by the current implementation.
    assert "market" not in result


# ---------------------------------------------------------------------------
# _normalize_candle
# ---------------------------------------------------------------------------


def test_normalize_candle_fields() -> None:
    raw = {
        "opening_price": 94000000,
        "high_price": 96000000,
        "low_price": 93000000,
        "trade_price": 95000000,
        "candle_acc_trade_volume": 500.5,
    }
    result = _normalize_candle(raw)

    assert result["open"] == 94000000
    assert result["high"] == 96000000
    assert result["low"] == 93000000
    assert result["close"] == 95000000
    assert result["volume"] == 500.5


# ---------------------------------------------------------------------------
# _normalize_orderbook
# ---------------------------------------------------------------------------


def test_normalize_orderbook_bids_and_asks() -> None:
    raw = {
        "orderbook_units": [
            {"bid_price": 95000000, "bid_size": 1.5, "ask_price": 95100000, "ask_size": 2.0},
            {"bid_price": 94900000, "bid_size": 0.8, "ask_price": 95200000, "ask_size": 1.2},
        ]
    }
    result = _normalize_orderbook(raw)

    assert result["bids"] == [[95000000, 1.5], [94900000, 0.8]]
    assert result["asks"] == [[95100000, 2.0], [95200000, 1.2]]


def test_normalize_orderbook_keys_present() -> None:
    raw = {
        "orderbook_units": [
            {"bid_price": 95000000, "bid_size": 1.5, "ask_price": 95100000, "ask_size": 2.0},
        ]
    }
    result = _normalize_orderbook(raw)

    assert "bids" in result
    assert "asks" in result


def test_normalize_orderbook_empty_units() -> None:
    raw: dict[str, object] = {"orderbook_units": []}
    result = _normalize_orderbook(raw)

    assert result["bids"] == []
    assert result["asks"] == []
