"""Unit tests for pure helper functions in coin_trader/main.py."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Module import
# ---------------------------------------------------------------------------
_main_mod = importlib.import_module("coin_trader.main")

_safe_decimal = _main_mod._safe_decimal
_chunk_symbols = _main_mod._chunk_symbols
_compute_universe_candidate_k = _main_mod._compute_universe_candidate_k
_build_order_reason = _main_mod._build_order_reason
_compute_portfolio_exposure = _main_mod._compute_portfolio_exposure
_compute_recent_structure = _main_mod._compute_recent_structure
_compute_volume_context = _main_mod._compute_volume_context
_compute_candidate_score = _main_mod._compute_candidate_score
_build_surge_context = _main_mod._build_surge_context
_check_trailing_stop = _main_mod._check_trailing_stop


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

@dataclass
class _Signal:
    metadata: dict[str, Any]


@dataclass
class _Advice:
    reasoning: str


@dataclass
class _Position:
    symbol: str
    quantity: Decimal
    current_price: Decimal | None
    average_entry_price: Decimal = Decimal("0")


@dataclass
class _RiskLimits:
    trailing_stop_pct: Decimal


# ---------------------------------------------------------------------------
# _safe_decimal
# ---------------------------------------------------------------------------

class TestSafeDecimal:
    def test_none_returns_default(self):
        assert _safe_decimal(None) == Decimal("0")

    def test_none_returns_custom_default(self):
        assert _safe_decimal(None, Decimal("99")) == Decimal("99")

    def test_valid_string(self):
        assert _safe_decimal("123.45") == Decimal("123.45")

    def test_invalid_string_returns_default(self):
        assert _safe_decimal("abc") == Decimal("0")

    def test_decimal_passthrough(self):
        d = Decimal("7.77")
        assert _safe_decimal(d) == d

    def test_float_value(self):
        result = _safe_decimal(1.5)
        assert result == Decimal("1.5")

    def test_int_value(self):
        assert _safe_decimal(42) == Decimal("42")

    def test_invalid_with_custom_default(self):
        assert _safe_decimal("bad", Decimal("-1")) == Decimal("-1")


# ---------------------------------------------------------------------------
# _chunk_symbols
# ---------------------------------------------------------------------------

class TestChunkSymbols:
    def test_empty_list(self):
        assert _chunk_symbols([], 3) == []

    def test_exact_multiple(self):
        symbols = ["A", "B", "C", "D", "E", "F"]
        result = _chunk_symbols(symbols, 3)
        assert result == [["A", "B", "C"], ["D", "E", "F"]]

    def test_remainder(self):
        symbols = ["A", "B", "C", "D", "E"]
        result = _chunk_symbols(symbols, 2)
        assert result == [["A", "B"], ["C", "D"], ["E"]]

    def test_size_zero_returns_whole_list_wrapped(self):
        symbols = ["A", "B", "C"]
        result = _chunk_symbols(symbols, 0)
        assert result == [["A", "B", "C"]]

    def test_size_negative_returns_whole_list_wrapped(self):
        symbols = ["A", "B"]
        result = _chunk_symbols(symbols, -5)
        assert result == [["A", "B"]]

    def test_size_one(self):
        symbols = ["A", "B", "C"]
        result = _chunk_symbols(symbols, 1)
        assert result == [["A"], ["B"], ["C"]]


# ---------------------------------------------------------------------------
# _compute_universe_candidate_k
# ---------------------------------------------------------------------------

class TestComputeUniverseCandidateK:
    def test_top_n_1_gives_12(self):
        assert _compute_universe_candidate_k(1) == 12

    def test_top_n_4_gives_12(self):
        # 4*3=12, so clamped to min 12
        assert _compute_universe_candidate_k(4) == 12

    def test_top_n_5_gives_15(self):
        # 5*3=15
        assert _compute_universe_candidate_k(5) == 15

    def test_top_n_7_gives_20(self):
        # 7*3=21 → clamped to max 20
        assert _compute_universe_candidate_k(7) == 20

    def test_top_n_20_gives_20(self):
        # 20*3=60 → clamped to max 20
        assert _compute_universe_candidate_k(20) == 20


# ---------------------------------------------------------------------------
# _build_order_reason
# ---------------------------------------------------------------------------

class TestBuildOrderReason:
    def test_signals_only(self):
        signals = [_Signal(metadata={"reason": "EMA cross"})]
        result = _build_order_reason(signals, None, "buy")
        assert result == "Strategy: EMA cross"

    def test_advice_only(self):
        advice = _Advice(reasoning="Strong momentum detected")
        result = _build_order_reason([], advice, "buy")
        assert result == "LLM: Strong momentum detected"

    def test_both_signals_and_advice(self):
        signals = [_Signal(metadata={"reason": "RSI oversold"})]
        advice = _Advice(reasoning="Good entry point")
        result = _build_order_reason(signals, advice, "buy")
        assert result == "Strategy: RSI oversold | LLM: Good entry point"

    def test_empty_signals_no_advice(self):
        result = _build_order_reason([], None, "buy")
        assert result == ""

    def test_signal_missing_reason_key_uses_default_verb(self):
        signals = [_Signal(metadata={})]
        result = _build_order_reason(signals, None, "sell")
        assert result == "Strategy: sell"

    def test_advice_reasoning_truncated_at_100_chars(self):
        long_reasoning = "x" * 150
        advice = _Advice(reasoning=long_reasoning)
        result = _build_order_reason([], advice, "buy")
        assert result == f"LLM: {'x' * 100}"


# ---------------------------------------------------------------------------
# _compute_portfolio_exposure
# ---------------------------------------------------------------------------

class TestComputePortfolioExposure:
    def test_no_positions(self):
        result = _compute_portfolio_exposure([], Decimal("1000000"))
        assert result == {"alt_total_pct": "0.00"}

    def test_single_btc_position(self):
        pos = _Position(
            symbol="BTC/KRW", quantity=Decimal("0.5"), current_price=Decimal("100000000"),
        )
        result = _compute_portfolio_exposure([pos], Decimal("100000000"))
        assert "BTC_pct" in result
        assert float(result["BTC_pct"]) == pytest.approx(50.0)
        assert result["alt_total_pct"] == "0.00"

    def test_multiple_positions_btc_and_alts(self):
        btc = _Position(symbol="BTC/KRW", quantity=Decimal("1"), current_price=Decimal("50000000"))
        eth = _Position(symbol="ETH/KRW", quantity=Decimal("10"), current_price=Decimal("3000000"))
        total = Decimal("80000000")
        result = _compute_portfolio_exposure([btc, eth], total)
        assert "BTC_pct" in result
        assert "ETH_pct" in result
        alt_total = float(result["alt_total_pct"])
        eth_pct = float(result["ETH_pct"])
        assert alt_total == pytest.approx(eth_pct)

    def test_zero_balance_returns_empty(self):
        pos = _Position(symbol="BTC/KRW", quantity=Decimal("1"), current_price=Decimal("50000000"))
        result = _compute_portfolio_exposure([pos], Decimal("0"))
        assert result == {}

    def test_position_with_none_price_skipped(self):
        pos = _Position(symbol="ETH/KRW", quantity=Decimal("5"), current_price=None)
        result = _compute_portfolio_exposure([pos], Decimal("1000000"))
        assert "ETH_pct" not in result
        assert "alt_total_pct" in result

    def test_position_with_zero_quantity_skipped(self):
        pos = _Position(symbol="XRP/KRW", quantity=Decimal("0"), current_price=Decimal("1000"))
        result = _compute_portfolio_exposure([pos], Decimal("1000000"))
        assert "XRP_pct" not in result


# ---------------------------------------------------------------------------
# _compute_recent_structure
# ---------------------------------------------------------------------------

class TestComputeRecentStructure:
    def _make_candles(self, highs: list[float], lows: list[float]) -> list[dict]:
        return [{"high": h, "low": lv} for h, lv in zip(highs, lows, strict=False)]

    def test_insufficient_candles_returns_range_defaults(self):
        candles = self._make_candles([100, 110], [90, 95])
        result = _compute_recent_structure(candles, lookback=20)
        assert result["higher_high"] is False
        assert result["higher_low"] is False
        assert result["lower_high"] is False
        assert result["lower_low"] is False
        assert result["range_market"] is True

    def test_uptrend_hh_and_hl(self):
        # 20 candles: second half has higher highs and higher lows
        highs = [100.0] * 10 + [110.0] * 10
        lows = [90.0] * 10 + [95.0] * 10
        candles = self._make_candles(highs, lows)
        result = _compute_recent_structure(candles, lookback=20)
        assert result["higher_high"] is True
        assert result["higher_low"] is True
        assert result["lower_high"] is False
        assert result["lower_low"] is False
        assert result["range_market"] is False

    def test_downtrend_lh_and_ll(self):
        highs = [110.0] * 10 + [100.0] * 10
        lows = [95.0] * 10 + [85.0] * 10
        candles = self._make_candles(highs, lows)
        result = _compute_recent_structure(candles, lookback=20)
        assert result["lower_high"] is True
        assert result["lower_low"] is True
        assert result["higher_high"] is False
        # is_range = not hh and not ll; ll=True here, so range_market is False
        assert result["range_market"] is False

    def test_range_market(self):
        # Highs and lows identical across halves
        highs = [100.0] * 20
        lows = [90.0] * 20
        candles = self._make_candles(highs, lows)
        result = _compute_recent_structure(candles, lookback=20)
        assert result["range_market"] is True
        assert result["higher_high"] is False
        assert result["lower_low"] is False


# ---------------------------------------------------------------------------
# _compute_volume_context
# ---------------------------------------------------------------------------

class TestComputeVolumeContext:
    def _make_candles(self, volumes: list[float]) -> list[dict]:
        return [{"volume": v} for v in volumes]

    def test_insufficient_candles_returns_defaults(self):
        # window=10 requires at least 11 candles
        candles = self._make_candles([100.0] * 5)
        result = _compute_volume_context(candles, window=10)
        assert result["volume_vs_avg_ratio"] == 1.0
        assert result["volume_trend"] == "flat"

    def test_increasing_volume(self):
        # First 5 volumes low, last 5 volumes high (window=10, need 11 candles total)
        volumes = [1.0] + [100.0] * 5 + [1000.0] * 6
        candles = self._make_candles(volumes)
        result = _compute_volume_context(candles, window=10)
        assert result["volume_trend"] == "increasing"

    def test_decreasing_volume(self):
        volumes = [1.0] + [1000.0] * 5 + [100.0] * 6
        candles = self._make_candles(volumes)
        result = _compute_volume_context(candles, window=10)
        assert result["volume_trend"] == "decreasing"

    def test_flat_volume(self):
        volumes = [1.0] + [500.0] * 10
        candles = self._make_candles(volumes)
        result = _compute_volume_context(candles, window=10)
        assert result["volume_trend"] == "flat"

    def test_edge_window_2(self):
        # window=2 requires at least 3 candles
        volumes = [100.0, 100.0, 200.0]
        candles = self._make_candles(volumes)
        result = _compute_volume_context(candles, window=2)
        # ratio = 200 / avg([100, 200]) = 200/150 ≈ 1.33
        assert result["volume_vs_avg_ratio"] == pytest.approx(1.33, abs=0.01)


# ---------------------------------------------------------------------------
# _compute_candidate_score
# ---------------------------------------------------------------------------

class TestComputeCandidateScore:
    def _score(
        self,
        spread_bps: float = 5,
        change_24h_pct: float = 3.0,
        intraday_range_pct: float = 5.0,
        distance_from_high_pct: float = 5.0,
        turnover: float = 1.0,
        max_turnover: float = 1.0,
    ) -> Decimal:
        metrics: dict[str, object] = {
            "spread_bps": spread_bps,
            "change_24h_pct": change_24h_pct,
            "intraday_range_pct": intraday_range_pct,
            "distance_from_high_pct": distance_from_high_pct,
        }
        return _compute_candidate_score(
            metrics, Decimal(str(turnover)), Decimal(str(max_turnover))
        )

    def test_ideal_metrics_score_near_one(self):
        # All metrics in ideal ranges, max liquidity
        score = self._score(
            spread_bps=0,
            change_24h_pct=3.0,
            intraday_range_pct=5.0,
            distance_from_high_pct=5.0,
            turnover=1.0,
            max_turnover=1.0,
        )
        assert score > Decimal("0.9")

    def test_zero_turnover_reduces_liquidity(self):
        score = self._score(turnover=0.0, max_turnover=1.0)
        score_full = self._score(turnover=1.0, max_turnover=1.0)
        assert score < score_full

    def test_extreme_spread_reduces_score(self):
        score_tight = self._score(spread_bps=0)
        score_wide = self._score(spread_bps=200)
        assert score_wide < score_tight

    def test_extreme_change_reduces_score(self):
        score_moderate = self._score(change_24h_pct=3.0)
        score_extreme = self._score(change_24h_pct=50.0)
        assert score_extreme < score_moderate

    def test_zero_max_turnover_gives_zero_liquidity(self):
        score = self._score(turnover=0.0, max_turnover=0.0)
        # liquidity=0, but other components still contribute
        assert score >= Decimal("0")

    def test_result_is_decimal(self):
        score = self._score()
        assert isinstance(score, Decimal)


# ---------------------------------------------------------------------------
# _build_surge_context
# ---------------------------------------------------------------------------

class TestBuildSurgeContext:
    def test_no_history_returns_default(self):
        with patch.object(_main_mod, "_surge_turnover_history", {}):
            result = _build_surge_context("BTC/KRW")
        assert result["is_surge"] is True
        assert result["surge_volume_ratio"] == 0.0

    def test_short_history_single_entry_returns_default(self):
        with patch.object(_main_mod, "_surge_turnover_history", {"ETH/KRW": [500]}):
            result = _build_surge_context("ETH/KRW")
        assert result["is_surge"] is True
        assert result["surge_volume_ratio"] == 0.0

    def test_normal_surge_with_ratio(self):
        # baseline=[100, 100], latest=300 → ratio = 300/100 = 3.0
        history = [100, 100, 300]
        with patch.object(_main_mod, "_surge_turnover_history", {"XRP/KRW": history}):
            result = _build_surge_context("XRP/KRW")
        assert result["is_surge"] is True
        assert result["surge_volume_ratio"] == pytest.approx(3.0)
        assert "surge_delta_krw" in result
        assert "surge_baseline_avg_krw" in result

    def test_ratio_equals_one_for_flat_volume(self):
        history = [200, 200, 200]
        with patch.object(_main_mod, "_surge_turnover_history", {"SOL/KRW": history}):
            result = _build_surge_context("SOL/KRW")
        assert result["surge_volume_ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _check_trailing_stop
# ---------------------------------------------------------------------------

class TestCheckTrailingStop:
    def _pos(self, entry: str) -> _Position:
        return _Position(
            symbol="BTC/KRW",
            quantity=Decimal("1"),
            current_price=Decimal(entry),
            average_entry_price=Decimal(entry),
        )

    def _limits(self, pct: str = "5") -> _RiskLimits:
        return _RiskLimits(trailing_stop_pct=Decimal(pct))

    # --- Long tests ---

    def test_long_no_trigger_below_entry(self):
        """Price below entry, no high watermark set yet — no trigger."""
        with patch.object(_main_mod, "_high_watermarks", {}), \
             patch.object(_main_mod, "_low_watermarks", {}):
            pos = self._pos("100")
            # Price at entry — hw=entry, drop=0% < 5%
            result = _check_trailing_stop("BTC/KRW", Decimal("100"), pos, self._limits("5"), False)
        assert result is None

    def test_long_trigger_drop_from_high(self):
        """High watermark set above entry; price drops enough to trigger."""
        hw = {"BTC/KRW": Decimal("110")}
        lw: dict = {}
        with patch.object(_main_mod, "_high_watermarks", hw), \
             patch.object(_main_mod, "_low_watermarks", lw):
            pos = self._pos("100")
            # drop = (110 - 104) / 110 * 100 ≈ 5.45% ≥ 5%
            result = _check_trailing_stop("BTC/KRW", Decimal("104"), pos, self._limits("5"), False)
        assert result is not None
        assert "trailing_stop" in result

    def test_long_no_trigger_small_drop(self):
        """Drop is below trailing_stop_pct — no trigger."""
        hw = {"BTC/KRW": Decimal("110")}
        lw: dict = {}
        with patch.object(_main_mod, "_high_watermarks", hw), \
             patch.object(_main_mod, "_low_watermarks", lw):
            pos = self._pos("100")
            # drop = (110 - 109) / 110 * 100 ≈ 0.9% < 5%
            result = _check_trailing_stop("BTC/KRW", Decimal("109"), pos, self._limits("5"), False)
        assert result is None

    def test_long_updates_high_watermark(self):
        """New price above existing watermark updates the watermark."""
        hw: dict = {"BTC/KRW": Decimal("100")}
        lw: dict = {}
        with patch.object(_main_mod, "_high_watermarks", hw), \
             patch.object(_main_mod, "_low_watermarks", lw):
            pos = self._pos("100")
            # New price 120 > hw 100 → hw updated to 120, drop = 0% < 5%
            result = _check_trailing_stop("BTC/KRW", Decimal("120"), pos, self._limits("5"), False)
            assert hw["BTC/KRW"] == Decimal("120")
        assert result is None

    # --- Short tests ---

    def test_short_no_trigger_above_entry(self):
        """Price at entry for short — no bounce yet."""
        lw: dict = {}
        hw: dict = {}
        with patch.object(_main_mod, "_high_watermarks", hw), \
             patch.object(_main_mod, "_low_watermarks", lw):
            pos = self._pos("100")
            result = _check_trailing_stop("BTC/KRW", Decimal("100"), pos, self._limits("5"), True)
        assert result is None

    def test_short_trigger_bounce_from_low(self):
        """Low watermark set below entry; price bounces enough to trigger."""
        lw = {"BTC/KRW": Decimal("90")}
        hw: dict = {}
        with patch.object(_main_mod, "_high_watermarks", hw), \
             patch.object(_main_mod, "_low_watermarks", lw):
            pos = self._pos("100")
            # bounce = (96 - 90) / 90 * 100 ≈ 6.67% ≥ 5%
            result = _check_trailing_stop("BTC/KRW", Decimal("96"), pos, self._limits("5"), True)
        assert result is not None
        assert "trailing_stop_short" in result

    def test_short_no_trigger_small_bounce(self):
        """Bounce below threshold — no trigger."""
        lw = {"BTC/KRW": Decimal("90")}
        hw: dict = {}
        with patch.object(_main_mod, "_high_watermarks", hw), \
             patch.object(_main_mod, "_low_watermarks", lw):
            pos = self._pos("100")
            # bounce = (91 - 90) / 90 * 100 ≈ 1.1% < 5%
            result = _check_trailing_stop("BTC/KRW", Decimal("91"), pos, self._limits("5"), True)
        assert result is None
