"""Conservative trend-following strategy with volatility filter.

Safety-first approach:
- Low frequency (1h candles)
- EMA crossover for trend confirmation
- ATR volatility filter (don't trade in high volatility)
- RSI overbought/oversold filter
- Strict confidence thresholds
- Default action is HOLD (do nothing)
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from decimal import Decimal
from collections.abc import Mapping
from typing import ClassVar

from coin_trader.core.models import MarketData, Signal, SignalType


@dataclasses.dataclass(frozen=True, slots=True)
class TechnicalIndicators:
    """Computed technical indicators for external consumption (e.g. LLM)."""

    fast_ema: float
    slow_ema: float
    ema200: float
    rsi: float
    atr: float
    vol_ratio: float
    avg_volume: float
    current_volume: float
    current_price: float
    trend_strength: float
    adx: float
    plus_di: float
    minus_di: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    uptrend: bool
    price_above_slow: bool
    low_volatility: bool
    rsi_ok_buy: bool
    volume_ok: bool
    downtrend: bool
    rsi_overbought: bool
    _BOOL_FLAGS: ClassVar[frozenset[str]] = frozenset(
        {
            "uptrend",
            "price_above_slow",
            "low_volatility",
            "rsi_ok_buy",
            "volume_ok",
            "downtrend",
            "rsi_overbought",
        }
    )

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    def to_llm_dict(self) -> dict[str, object]:
        """Return indicators for LLM consumption (raw values only, no booleans)."""
        return {k: v for k, v in dataclasses.asdict(self).items() if k not in self._BOOL_FLAGS}


class ConservativeStrategy:
    """Conservative trend+volatility strategy.

    Entry conditions (ALL must be true for BUY):
    1. Fast EMA > Slow EMA (uptrend)
    2. Price above Slow EMA (trend confirmation)
    3. ATR-based volatility below threshold (calm market)
    4. RSI between 30-65 (not overbought)
    5. Volume above average (liquidity check)

    Exit conditions (ANY triggers SELL):
    1. Fast EMA < Slow EMA (trend reversal)
    2. RSI > 75 (overbought)
    3. Stop-loss hit (handled by risk manager)
    """

    def __init__(
        self,
        fast_ema_period: int = 12,
        slow_ema_period: int = 26,
        atr_period: int = 14,
        rsi_period: int = 14,
        volatility_threshold: float = 0.03,  # 3% ATR/price ratio
        min_confidence: float = 0.6,
    ) -> None:
        self.fast_period: int = fast_ema_period
        self.slow_period: int = slow_ema_period
        self.atr_period: int = atr_period
        self.rsi_period: int = rsi_period
        self.vol_threshold: float = volatility_threshold
        self.min_confidence: float = min_confidence
        self._candle_history: dict[str, list[Mapping[str, object]]] = {}

    def update_candles(self, symbol: str, candles: list[Mapping[str, object]]) -> None:
        """Update candle history for a symbol from exchange API response."""
        self._candle_history[symbol] = candles

    def compute_indicators(self, symbol: str) -> TechnicalIndicators | None:
        candles = self._candle_history.get(symbol, [])
        if len(candles) < self.slow_period + 10:
            return None

        closes = [float(str(c.get("trade_price", c.get("close", 0)) or 0)) for c in candles]
        highs = [float(str(c.get("high_price", c.get("high", 0)) or 0)) for c in candles]
        lows = [float(str(c.get("low_price", c.get("low", 0)) or 0)) for c in candles]
        volumes = [
            float(str(c.get("candle_acc_trade_volume", c.get("volume", 0)) or 0)) for c in candles
        ]

        if closes[-1] <= 0:
            return None

        fast_ema = self._ema(closes, self.fast_period)
        slow_ema = self._ema(closes, self.slow_period)
        ema200 = self._ema(closes, 200)
        atr = self._atr(highs, lows, closes, self.atr_period)
        rsi = self._rsi(closes, self.rsi_period)
        adx_val, plus_di_val, minus_di_val = self._adx(highs, lows, closes, self.atr_period)
        macd_line, signal_line, histogram = self._macd(closes)
        bb_upper, bb_middle, bb_lower, bb_width = self._bollinger_bands(closes)

        window = volumes[-20:] if len(volumes) >= 20 else volumes
        avg_volume = (sum(window) / len(window)) if window else 0.0

        current_price = closes[-1]
        current_fast = fast_ema[-1]
        current_slow = slow_ema[-1]
        current_ema200 = ema200[-1] if ema200 else 0.0
        current_atr = atr[-1]
        current_rsi = rsi[-1]
        current_vol = volumes[-1]

        vol_ratio = current_atr / current_price if current_price > 0 else 1.0
        trend_strength = (
            (current_fast - current_slow) / current_slow * 100 if current_slow > 0 else 0.0
        )

        return TechnicalIndicators(
            fast_ema=round(current_fast, 2),
            slow_ema=round(current_slow, 2),
            ema200=round(current_ema200, 2),
            rsi=round(current_rsi, 2),
            atr=round(current_atr, 2),
            vol_ratio=round(vol_ratio, 4),
            avg_volume=round(avg_volume, 2),
            current_volume=round(current_vol, 2),
            current_price=round(current_price, 2),
            trend_strength=round(trend_strength, 4),
            adx=round(adx_val, 2),
            plus_di=round(plus_di_val, 2),
            minus_di=round(minus_di_val, 2),
            macd=round(macd_line, 4),
            macd_signal=round(signal_line, 4),
            macd_histogram=round(histogram, 4),
            bb_upper=round(bb_upper, 2),
            bb_middle=round(bb_middle, 2),
            bb_lower=round(bb_lower, 2),
            bb_width=round(bb_width, 4),
            uptrend=current_fast > current_slow,
            price_above_slow=current_price > current_slow,
            low_volatility=vol_ratio < self.vol_threshold,
            rsi_ok_buy=30 < current_rsi < 65,
            volume_ok=current_vol > avg_volume * 0.5,
            downtrend=current_fast < current_slow,
            rsi_overbought=current_rsi > 75,
        )

    async def on_tick(self, md: MarketData) -> list[Signal]:
        indicators = self.compute_indicators(md.symbol)
        if indicators is None:
            return [self._hold_signal(md, "Insufficient candle data")]

        if (
            indicators.uptrend
            and indicators.price_above_slow
            and indicators.low_volatility
            and indicators.rsi_ok_buy
            and indicators.volume_ok
        ):
            confidence = min(0.9, self.min_confidence + indicators.trend_strength / 100)
            if confidence >= self.min_confidence:
                return [
                    Signal(
                        strategy_name="conservative_trend",
                        symbol=md.symbol,
                        signal_type=SignalType.BUY,
                        timestamp=datetime.now(timezone.utc),
                        confidence=Decimal(str(round(confidence, 4))),
                        metadata={
                            "fast_ema": indicators.fast_ema,
                            "slow_ema": indicators.slow_ema,
                            "rsi": indicators.rsi,
                            "atr": indicators.atr,
                            "vol_ratio": indicators.vol_ratio,
                            "reason": "Uptrend confirmed with low volatility",
                        },
                    )
                ]

        if indicators.downtrend or indicators.rsi_overbought:
            reason = "Trend reversal" if indicators.downtrend else "RSI overbought"
            return [
                Signal(
                    strategy_name="conservative_trend",
                    symbol=md.symbol,
                    signal_type=SignalType.SELL,
                    timestamp=datetime.now(timezone.utc),
                    confidence=Decimal("0.7"),
                    metadata={
                        "fast_ema": indicators.fast_ema,
                        "slow_ema": indicators.slow_ema,
                        "rsi": indicators.rsi,
                        "reason": reason,
                    },
                )
            ]

        return [self._hold_signal(md, "No clear signal")]

    def _hold_signal(self, md: MarketData, reason: str) -> Signal:
        return Signal(
            strategy_name="conservative_trend",
            symbol=md.symbol,
            signal_type=SignalType.HOLD,
            timestamp=datetime.now(timezone.utc),
            confidence=Decimal("0.5"),
            metadata={"reason": reason},
        )

    @staticmethod
    def _ema(data: list[float], period: int) -> list[float]:
        """Exponential Moving Average."""
        if not data:
            return []
        alpha = 2.0 / (period + 1)
        ema: list[float] = [data[0]]
        for i in range(1, len(data)):
            ema.append(alpha * data[i] + (1 - alpha) * ema[i - 1])
        return ema

    @staticmethod
    def _atr(
        highs: list[float], lows: list[float], closes: list[float], period: int
    ) -> list[float]:
        """Average True Range."""
        if not highs or not lows or not closes:
            return []
        n = min(len(highs), len(lows), len(closes))
        highs = highs[:n]
        lows = lows[:n]
        closes = closes[:n]

        tr: list[float] = [highs[0] - lows[0]]
        for i in range(1, n):
            tr.append(
                max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
            )

        if n < period or period <= 0:
            avg = (sum(tr) / len(tr)) if tr else 0.0
            return [avg for _ in range(n)]

        first_atr = sum(tr[:period]) / period
        atr: list[float] = [first_atr for _ in range(period)]
        for i in range(period, n):
            atr.append((atr[i - 1] * (period - 1) + tr[i]) / period)
        return atr

    @staticmethod
    def _rsi(closes: list[float], period: int) -> list[float]:
        """Relative Strength Index."""
        n = len(closes)
        if n == 0:
            return []
        if n < period + 1 or period <= 0:
            return [50.0 for _ in range(n)]

        deltas = [closes[i] - closes[i - 1] for i in range(1, n)]
        gains = [d if d > 0 else 0.0 for d in deltas]
        losses = [-d if d < 0 else 0.0 for d in deltas]

        rsi: list[float] = [50.0 for _ in range(n)]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    @staticmethod
    def _adx(
        highs: list[float], lows: list[float], closes: list[float], period: int
    ) -> tuple[float, float, float]:
        """Average Directional Index. Returns (ADX, +DI, -DI)."""
        n = min(len(highs), len(lows), len(closes))
        if n < period + 1 or period <= 0:
            return (0.0, 0.0, 0.0)

        plus_dm: list[float] = []
        minus_dm: list[float] = []
        tr: list[float] = []

        for i in range(1, n):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            plus_dm.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
            minus_dm.append(down_move if (down_move > up_move and down_move > 0) else 0.0)
            tr.append(
                max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
            )

        if len(tr) < period:
            return (0.0, 0.0, 0.0)

        alpha = 1.0 / period
        smoothed_tr = sum(tr[:period])
        smoothed_plus = sum(plus_dm[:period])
        smoothed_minus = sum(minus_dm[:period])

        dx_values: list[float] = []

        for i in range(period, len(tr)):
            smoothed_tr = smoothed_tr - smoothed_tr * alpha + tr[i]
            smoothed_plus = smoothed_plus - smoothed_plus * alpha + plus_dm[i]
            smoothed_minus = smoothed_minus - smoothed_minus * alpha + minus_dm[i]

            pdi = (smoothed_plus / smoothed_tr * 100) if smoothed_tr > 0 else 0.0
            mdi = (smoothed_minus / smoothed_tr * 100) if smoothed_tr > 0 else 0.0
            di_sum = pdi + mdi
            dx = (abs(pdi - mdi) / di_sum * 100) if di_sum > 0 else 0.0
            dx_values.append(dx)

        if not dx_values:
            return (0.0, 0.0, 0.0)

        if len(dx_values) < period:
            adx = sum(dx_values) / len(dx_values)
        else:
            adx = sum(dx_values[:period]) / period
            for i in range(period, len(dx_values)):
                adx = (adx * (period - 1) + dx_values[i]) / period

        final_pdi = (smoothed_plus / smoothed_tr * 100) if smoothed_tr > 0 else 0.0
        final_mdi = (smoothed_minus / smoothed_tr * 100) if smoothed_tr > 0 else 0.0

        return (adx, final_pdi, final_mdi)

    @staticmethod
    def _macd(
        closes: list[float],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> tuple[float, float, float]:
        """MACD. Returns (macd_line, signal_line, histogram)."""
        if len(closes) < slow_period + signal_period:
            return (0.0, 0.0, 0.0)

        fast_ema = ConservativeStrategy._ema(closes, fast_period)
        slow_ema = ConservativeStrategy._ema(closes, slow_period)

        macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
        signal_ema = ConservativeStrategy._ema(macd_line, signal_period)

        current_macd = macd_line[-1]
        current_signal = signal_ema[-1]
        current_histogram = current_macd - current_signal

        return (current_macd, current_signal, current_histogram)

    @staticmethod
    def _bollinger_bands(
        closes: list[float], period: int = 20, num_std: float = 2.0
    ) -> tuple[float, float, float, float]:
        """Bollinger Bands. Returns (upper, middle, lower, width)."""
        if len(closes) < period:
            p = closes[-1] if closes else 0.0
            return (p, p, p, 0.0)

        window = closes[-period:]
        middle = sum(window) / period
        variance = sum((x - middle) ** 2 for x in window) / period
        std = variance**0.5

        upper = middle + num_std * std
        lower = middle - num_std * std
        width = ((upper - lower) / middle) if middle > 0 else 0.0

        return (upper, middle, lower, width)
