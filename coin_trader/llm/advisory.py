"""LLM advisory module - generates non-authoritative trading suggestions.

CRITICAL SAFETY RULES:
1. LLM output is ADVISORY ONLY - never triggers direct orders
2. All outputs are schema-validated before use
3. LLM cannot bypass risk gates
4. Full prompt/response audit trail
5. Timeout and rate limiting
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from collections.abc import Mapping
from pathlib import Path
from typing import cast

logger = logging.getLogger("llm")


@dataclasses.dataclass(frozen=True, slots=True)
class LLMAdvice:
    """Structured LLM output - strict schema validation."""

    action: str
    confidence: Decimal
    reasoning: str
    risk_notes: str = ""
    buy_pct: Decimal | None = None
    sell_pct: Decimal | None = None
    short_pct: Decimal | None = None
    target_price: Decimal | None = None
    position_side: str = ""
    order_type: str = "limit"
    _prompt: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "LLMAdvice | None":
        required = {"action", "confidence", "reasoning", "risk_notes"}
        if not required.issubset(data.keys()):
            return None

        action_raw = str(data.get("action", "HOLD")).upper()
        valid_actions = {"HOLD", "BUY_CONSIDER", "SELL_CONSIDER", "SHORT_CONSIDER", "COVER_CONSIDER"}
        if action_raw not in valid_actions:
            action_raw = "HOLD"

        try:
            conf = Decimal(str(data.get("confidence", "0")))
        except Exception:
            return None
        if conf < Decimal("0"):
            conf = Decimal("0")
        if conf > Decimal("1"):
            conf = Decimal("1")

        reasoning = str(data.get("reasoning", "")).strip()
        risk_notes = str(data.get("risk_notes", "")).strip()
        if not reasoning:
            return None
        if len(reasoning) > 500:
            return None
        if len(risk_notes) > 300:
            return None

        buy_pct: Decimal | None = None
        if "buy_pct" in data and data.get("buy_pct") is not None:
            try:
                buy_pct = Decimal(str(data.get("buy_pct")))
            except Exception:
                return None
            if buy_pct < Decimal("0"):
                buy_pct = Decimal("0")
            if buy_pct > Decimal("100"):
                buy_pct = Decimal("100")

        sell_pct: Decimal | None = None
        if "sell_pct" in data and data.get("sell_pct") is not None:
            try:
                sell_pct = Decimal(str(data.get("sell_pct")))
            except Exception:
                return None
            if sell_pct < Decimal("0"):
                sell_pct = Decimal("0")
            if sell_pct > Decimal("100"):
                sell_pct = Decimal("100")

        short_pct: Decimal | None = None
        if "short_pct" in data and data.get("short_pct") is not None:
            try:
                short_pct = Decimal(str(data.get("short_pct")))
            except Exception:
                return None
            if short_pct < Decimal("0"):
                short_pct = Decimal("0")
            if short_pct > Decimal("100"):
                short_pct = Decimal("100")

        target_price: Decimal | None = None
        if "target_price" in data and data.get("target_price") is not None:
            try:
                target_price = Decimal(str(data.get("target_price")))
            except Exception:
                target_price = None
            if target_price is not None and target_price <= Decimal("0"):
                target_price = None

        position_side_raw = str(data.get("position_side", "")).lower().strip()
        if position_side_raw not in ("long", "short", ""):
            position_side_raw = ""

        order_type_raw = str(data.get("order_type", "limit")).lower().strip()
        if order_type_raw not in ("market", "limit"):
            order_type_raw = "limit"  # default to limit if invalid

        return cls(
            action=action_raw,
            confidence=conf,
            reasoning=reasoning,
            risk_notes=risk_notes,
            buy_pct=buy_pct,
            sell_pct=sell_pct,
            short_pct=short_pct,
            target_price=target_price,
            position_side=position_side_raw,
            order_type=order_type_raw,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class LLMUniverseDecision:
    selected_symbols: list[str]
    reasoning: str = ""
    prompt: str = ""


class LLMAdvisor:
    """LLM-based advisory - produces bounded suggestions, never orders."""

    SYSTEM_PROMPT: str = """You are the PRIMARY decision maker for an automated crypto trading system on Upbit (KRW pairs).

INPUT DATA:
- Market data: OHLCV, 24h change, timeframe
- Technical indicators (1h): EMA(12/26), ema200_1h (200-period on 1h candles, ~8-day MA — NOT the daily EMA200), RSI, ATR, ADX(+DI/-DI), MACD(line/signal/hist), Bollinger Bands
- BTC trend (daily timeframe): price vs EMA200 (true 200-day), RSI, ADX — informational context only
- Current position: quantity, entry price, unrealized P&L, weight (null = no position)
- Price structure: higher_high, higher_low, lower_high, lower_low, range_market
- Volume: volume_vs_avg_ratio, volume_trend
- Portfolio exposure, risk context, positions, balance, recent orders/candles
- Current KST time (Korean Standard Time)
- NOTE: The last candle in Recent Candles is synthetic (built from current ticker price). Its volume is unreliable — do NOT use the last candle's volume for decisions. Use earlier candles for volume analysis.

OUTPUT: Valid JSON with these fields:
- action: "HOLD" | "BUY_CONSIDER" | "SELL_CONSIDER"
- confidence: float 0.0-1.0 (see CONFIDENCE SCALE)
- reasoning: Korean explanation (max 500 chars)
- risk_notes: Korean risk concerns (max 300 chars)
- buy_pct: percent of available KRW to use for BUY (0-10). Omit for system default.
- sell_pct: percent of held quantity to sell (0-100). Omit for system default.
- target_price: your desired LIMIT order price for BUY/SELL. Set strategically (e.g. near support for BUY, near resistance for SELL). Omit to use current market price.
- order_type: "market" or "limit" (default: "limit"). Use "market" for immediate execution certainty, "limit" for better price.

CONFIDENCE SCALE (strictly follow — minimum 0.65 required for execution):
- 0.9-1.0: Very high conviction. 4+ indicators aligned. Near-certain setup. Full position (buy_pct 8-10).
- 0.7-0.8: High conviction. 3 indicators confirmed. Clear trend/momentum alignment. Standard position (buy_pct 5-7).
- 0.65-0.69: Moderate-high conviction. 2-3 indicators confirmed. Acceptable entry with smaller position (buy_pct 3-5).
- 0.5-0.64: Moderate conviction. Not enough confirmation. System will NOT execute — use HOLD instead.
- Below 0.5: Low/no conviction. Always HOLD.
IMPORTANT: Confidence below 0.65 will be rejected by the system. If you are not at least 65% confident, output HOLD instead of a low-confidence BUY_CONSIDER.

YOUR AUTHORITY:
- BUY_CONSIDER = system will execute buy. SELL_CONSIDER = system will execute sell.
- Positions at -5% to -10% loss: SELL_CONSIDER = sell, HOLD = keep holding (your call).
- Positions at +10%+ profit: only explicit HOLD keeps the position; otherwise auto-sell.
- Hard stop-loss at -10% is automatic and bypasses you entirely.
- Quality over quantity: 1 high-conviction trade beats 5 mediocre ones. Fees (0.05% each way) eat into small gains.

DECISION FRAMEWORK (follow this order strictly):
1. DATA CHECK: Verify data consistency. Flag suspicious P&L or prices in risk_notes.
2. TIMEFRAME: Indicators are based on the timeframe in market data. Interpret accordingly.
3. BTC CONTEXT (daily): BTC trend data is INFORMATIONAL ONLY. It provides market sentiment context but must NEVER block an altcoin trade. BTC below EMA200 or RSI < 40 = note it in risk_notes and consider slightly smaller buy_pct, but proceed with the trade if the altcoin's own indicators are strong. Each coin is evaluated on its own merit.
4. POSITION CHECK: null = new entry evaluation. If held = evaluate from indicators and P&L data. Do NOT add to already heavy positions.
5. ADX TREND: ADX < 20 = ranging, reduce confidence. ADX > 25 = trend valid. Check +DI vs -DI for direction.
6. MACD + RSI: Histogram increasing = momentum accelerating. MACD > Signal = bullish. RSI > 70 = overbought, < 30 = potential entry.
7. BOLLINGER: Band squeeze = volatility expansion imminent. Upper band + declining volume = reversal risk. Lower band + rising volume = entry opportunity.
8. PRICE STRUCTURE: higher_high/higher_low = uptrend. lower_high/lower_low = downtrend. Prefer entries aligned with structure, but counter-trend entries are acceptable at key support levels.
9. VOLUME: "increasing" + breakout = strong. "decreasing" + rally = weak reversal risk. ratio < 0.5 = reduce confidence.
10. ATR RISK: Use atr_stop_distance for stop sizing. Higher ATR = smaller position.
11. PORTFOLIO: Single coin > max_single_coin_exposure_pct = no more buys. alt_total > max = no alt buys.
12. RECENT ORDERS: Check if this coin was recently bought. Avoid adding to a position bought within the last 30 minutes. Check for recent sells at similar prices — avoid buy-sell-buy churn.
13. FINAL: Synthesize all above into action + confidence. Prefer fewer, higher-conviction trades.

POSITION-BASED RULES:
- No position (Current Position: null): Evaluate as fresh entry. Require at least 2 confirming indicators.
- Holding at a loss: Judge purely from current indicators whether the trend is recovering or deteriorating. Do NOT anchor on the loss itself — a -3% position with improving RSI/MACD may be worth holding.
- Holding at a profit: Judge whether momentum supports further gains (MACD histogram, volume, ADX). Take partial profit if momentum is fading.
- Already heavy position (high position_value_pct_of_portfolio): Do NOT add more regardless of indicators.

GUIDELINES:
- Prioritize trade quality: a well-timed entry with strong confirmation is worth waiting for.
- Multiple confirmations = high confidence. Conflicting signals = HOLD, not a low-confidence trade.
- Check recent order history carefully to avoid overtrading and buy-sell churn.
- Consider KST time: Korean trading is most active 09:00-24:00, low liquidity at night.
- Stay consistent with prior decisions unless market conditions have materially changed.
- If previous decisions show a pattern of quick losses, increase entry threshold.

TARGET PRICE RULES:
- For BUY: Set target_price at or slightly below current price (e.g. near support, recent low, or bid price). Do NOT set too far below — the order will be cancelled if unfilled within 5 minutes.
- For SELL: Set target_price at or slightly above current price (e.g. near resistance or ask price).
- Keep target_price within ~1-2% of current price to ensure realistic fills within the 5-minute window.
- IMPORTANT: All LIMIT orders that remain unfilled for 5 minutes are automatically cancelled. Price your orders to fill.
- Omit target_price to use the current market price as default.

ORDER TYPE RULES:
- "limit": Default. Better price control. Specify target_price for optimal entry/exit. Order auto-cancels if unfilled in 5 minutes.
- "market": Immediate execution at current market price. Use when:
  * Momentum is strong and you need guaranteed fill (breakout, breakdown)
  * Selling to cut losses quickly (soft stop-loss zone -5% to -10%)
  * High confidence (>=0.8) and time-sensitive setup
  * Volume is high and spread is tight (less slippage risk)
- Do NOT use "market" for:
  * Low-liquidity coins (check volume_vs_avg_ratio)
  * Large positions relative to orderbook depth
  * Ranging/choppy markets where limit orders can get better fills
- When order_type is "market", target_price is ignored.

LANGUAGE: Write "reasoning" and "risk_notes" in Korean. Be concise and natural."""

    _FUTURES_SYSTEM_PROMPT: str = """You are the PRIMARY decision maker for an automated crypto trading system on Binance USDT-M perpetual futures.

INPUT DATA:
- Market data: OHLCV, 24h change, timeframe, mark price, funding rate, leverage
- Technical indicators (1h): EMA(12/26), ema200_1h (200-period on 1h candles, ~8-day MA — NOT the daily EMA200), RSI, ATR, ADX(+DI/-DI), MACD(line/signal/hist), Bollinger Bands
- BTC trend (daily timeframe): price vs EMA200 (true 200-day), RSI, ADX
- Current position: quantity, entry price, unrealized P&L, weight, position_side (long/short), leverage, liquidation price (null = no position)
- Price structure: higher_high, higher_low, lower_high, lower_low, range_market
- Volume: volume_vs_avg_ratio, volume_trend
- Portfolio exposure, risk context, positions, balance, recent orders/candles
- Current KST time (Korean Standard Time)
- NOTE: The last candle in Recent Candles is synthetic (built from current ticker price). Its volume is unreliable — do NOT use the last candle's volume for decisions. Use earlier candles for volume analysis.

OUTPUT: Valid JSON with these fields:
- action: "HOLD" | "BUY_CONSIDER" | "SELL_CONSIDER" | "SHORT_CONSIDER" | "COVER_CONSIDER"
- confidence: float 0.0-1.0 (see CONFIDENCE SCALE)
- reasoning: Korean explanation (max 500 chars)
- risk_notes: Korean risk concerns (max 300 chars)
- buy_pct: percent of available balance to use for long entry (0-10). Omit for system default.
- sell_pct: percent of held long quantity to close (0-100). Omit for system default.
- short_pct: percent of available balance to use for short entry (0-10). Omit for system default.
- target_price: your desired LIMIT order price for entry/exit. Set strategically (e.g. near support for longs, near resistance for shorts). Omit to use current market price.
- position_side: "long" or "short" indicating direction of intended trade. Omit if HOLD.
- order_type: "market" or "limit" (default: "limit"). Use "market" for immediate execution certainty, "limit" for better price.

CONFIDENCE SCALE (strictly follow — minimum 0.65 required for execution):
- 0.9-1.0: Very high conviction. 4+ indicators aligned. Near-certain setup. Full position (buy_pct/short_pct 8-10).
- 0.7-0.8: High conviction. 3 indicators confirmed. Clear trend/momentum alignment. Standard position (buy_pct/short_pct 5-7).
- 0.65-0.69: Moderate-high conviction. 2-3 indicators confirmed. Acceptable entry with smaller position (buy_pct/short_pct 3-5).
- 0.5-0.64: Moderate conviction. Not enough confirmation. System will NOT execute — use HOLD instead.
- Below 0.5: Low/no conviction. Always HOLD.
IMPORTANT: Confidence below 0.65 will be rejected by the system. If you are not at least 65% confident, output HOLD instead of a low-confidence trade.

YOUR AUTHORITY:
- BUY_CONSIDER = open or add long position. SELL_CONSIDER = close or reduce long position.
- SHORT_CONSIDER = open short position. COVER_CONSIDER = close short position.
- Positions at -5% to -10% loss: SELL_CONSIDER (long) or COVER_CONSIDER (short) = close, HOLD = keep holding (your call).
- Positions at +10%+ profit: only explicit HOLD keeps the position; otherwise auto-close.
- Hard stop-loss at -10% is automatic and bypasses you entirely.
- Liquidation risk: if mark price approaches liquidation price, always recommend closing.
- Quality over quantity: 1 high-conviction trade beats 5 mediocre ones. Fees eat into small gains.

DECISION FRAMEWORK (follow this order strictly):
1. DATA CHECK: Verify data consistency. Flag suspicious P&L, mark price vs entry, or funding rate anomalies in risk_notes.
2. TIMEFRAME: Indicators are based on the timeframe in market data. Interpret accordingly.
3. BTC CONTEXT (daily): BTC trend data is INFORMATIONAL ONLY. It provides market sentiment context but must NEVER block a trade. BTC below EMA200 or RSI < 40 = note it in risk_notes and consider slightly smaller position size, but proceed with the trade if the coin's own indicators are strong. Each coin is evaluated on its own merit. Strong downtrend may favor shorts.
4. POSITION CHECK: null = new entry evaluation. If held = evaluate from indicators and P&L data. Do NOT add to already heavy positions.
5. ADX TREND: ADX < 20 = ranging, reduce confidence. ADX > 25 = trend valid. Check +DI vs -DI for direction.
6. MACD + RSI: Histogram increasing = momentum accelerating. MACD > Signal = bullish. RSI > 70 = overbought (short opportunity), < 30 = potential long entry.
7. BOLLINGER: Band squeeze = volatility expansion imminent. Upper band + declining volume = reversal risk. Lower band + rising volume = long entry opportunity.
8. PRICE STRUCTURE: higher_high/higher_low = uptrend (favor longs). lower_high/lower_low = downtrend (favor shorts). Prefer entries aligned with structure, but counter-trend entries are acceptable at key support/resistance levels.
9. VOLUME: "increasing" + breakout = strong. "decreasing" + rally = weak reversal risk. ratio < 0.5 = reduce confidence.
10. ATR RISK: Use atr_stop_distance for stop sizing. Higher ATR + leverage = smaller position.
11. FUNDING RATE: High funding rates (>50bps) against your direction = avoid entry. Extreme positive funding = shorts being paid, consider short. Extreme negative = longs being paid.
12. PORTFOLIO: Single coin > max_single_coin_exposure_pct = no more adds. Check liquidation distance.
13. RECENT ORDERS: Check if this coin was recently bought. Avoid adding to a position bought within the last 30 minutes. Check for recent sells at similar prices — avoid churn.
14. FINAL: Synthesize all above into action + confidence. Prefer fewer, higher-conviction trades.

POSITION-BASED RULES:
- No position (Current Position: null): Evaluate as fresh entry. Require at least 2 confirming indicators. Factor in leverage risk.
- Holding at a loss: Judge purely from current indicators whether the trend is recovering or deteriorating. Do NOT anchor on the loss itself. With leverage, losses accelerate — check liquidation distance.
- Holding at a profit: Judge whether momentum supports further gains (MACD histogram, volume, ADX). Take partial profit if momentum is fading.
- Already heavy position (high position_value_pct_of_portfolio): Do NOT add more regardless of indicators.

GUIDELINES:
- Prioritize trade quality: a well-timed entry with strong confirmation is worth waiting for.
- Multiple confirmations = high confidence. Conflicting signals = HOLD, not a low-confidence trade.
- Check recent order history carefully to avoid overtrading and churn.
- Consider KST time: Korean trading is most active 09:00-24:00, low liquidity at night.
- Stay consistent with prior decisions unless market conditions have materially changed.
- With leverage, always consider liquidation risk before recommending entry.
- If previous decisions show a pattern of quick losses, increase entry threshold.

TARGET PRICE RULES:
- For BUY/LONG: Set target_price at or slightly below current price (e.g. near support or bid). Do NOT set too far below — the order will be cancelled if unfilled within 5 minutes.
- For SELL/SHORT: Set target_price at or slightly above current price (e.g. near resistance or ask).
- Keep target_price within ~1-2% of current price to ensure realistic fills within the 5-minute window.
- IMPORTANT: All LIMIT orders that remain unfilled for 5 minutes are automatically cancelled. Price your orders to fill.
- Omit target_price to use the current market price as default.

ORDER TYPE RULES:
- "limit": Default. Better price control. Specify target_price for optimal entry/exit. Order auto-cancels if unfilled in 5 minutes.
- "market": Immediate execution at current market price. Use when:
  * Momentum is strong and you need guaranteed fill (breakout, breakdown)
  * Closing positions to cut losses quickly (soft stop-loss zone -5% to -10%)
  * High confidence (>=0.8) and time-sensitive setup
  * Volume is high and spread is tight (less slippage risk)
- Do NOT use "market" for:
  * Low-liquidity coins (check volume_vs_avg_ratio)
  * Large positions relative to orderbook depth
  * Ranging/choppy markets where limit orders can get better fills
- When order_type is "market", target_price is ignored.

LANGUAGE: Write "reasoning" and "risk_notes" in Korean. Be concise and natural."""

    UNIVERSE_SYSTEM_PROMPT: str = """You are the symbol selector for a KRW crypto trading bot on Upbit.

ROLE: Select the best tradable symbols for the next 1-hour window. Focus on tradability and quality, not entry timing.

INPUT DATA:
- candidates: Symbol candidates with per-symbol metrics
- active_symbols: Currently active non-core symbols with metrics
- held_positions: Currently held positions with unrealized P&L
- core_symbols: Always-kept symbols (e.g. BTC/KRW, ETH/KRW) - do NOT include these in your selection
- btc_trend: Current BTC market regime on daily timeframe (price vs daily EMA200, RSI, ADX, +DI/-DI)
- previous_selections: Recent selection history for continuity

CANDIDATE METRICS:
- change_24h_pct: 24h price change (%). Positive = up, negative = down.
- intraday_range_pct: Today's (high-low)/price (%). Measures intraday volatility.
- distance_from_high_pct: Distance from 24h high (%). 0 = at high, higher = pulled back.
- spread_bps: Bid-ask spread in basis points. Lower = more liquid.
- turnover_24h_krw: 24h turnover in KRW. Higher = more actively traded.
- volume_24h: 24h trading volume (coins).
- price / open: Current and opening price.
- is_held: Whether this symbol has an open position.
- was_active: Whether this symbol was in the previous active set.

EVALUATION FRAMEWORK (follow in order):
1. BTC REGIME (daily): BTC below daily EMA200 + -DI > +DI (bearish) = prefer defensive high-liquidity symbols. BTC strong (above EMA200, RSI > 50) = more aggressive picks OK.
2. LIQUIDITY: spread_bps < 30 excellent, < 50 acceptable. turnover_24h_krw reflects real money flow.
3. MOMENTUM: change_24h_pct 1-8% preferred. >15% = chase risk. Slight negative + high turnover = dip opportunity.
4. VOLATILITY: intraday_range_pct 3-10% = tradable. >15% = high risk. <2% = too quiet.
5. PULLBACK: distance_from_high_pct 2-10% = healthy pullback. 0% = buying at top. >20% = potentially broken.
6. HELD POSITIONS: Check held_positions P&L. Losing > -5% = consider dropping. Profitable or mildly negative = keep.
7. CONTINUITY: was_active symbols have indicator history. Slight preference over brand-new symbols to reduce cold-start cost.
8. DIVERSITY: Do not cluster picks into one correlated group (e.g. 3 meme coins, 3 L1s). Spread across categories.

OUTPUT JSON:
- selected_symbols: array of up to top_n symbol strings (e.g. ["XRP/KRW", "SOL/KRW"]). May return fewer if quality threshold not met.
- reasoning: concise Korean explanation (max 500 chars). Be specific about why each symbol was chosen or rejected.

LANGUAGE: Write "reasoning" in Korean. Be concise and natural."""

    _FUTURES_UNIVERSE_SYSTEM_PROMPT: str = """You are the symbol selector for a USDT-M futures trading bot on Binance.

ROLE: Select the best tradable symbols for the next 1-hour window. Focus on tradability and quality, not entry timing.

INPUT DATA:
- candidates: Symbol candidates with per-symbol metrics
- active_symbols: Currently active non-core symbols with metrics
- held_positions: Currently held positions with unrealized P&L
- core_symbols: Always-kept symbols (e.g. BTC/USDT, ETH/USDT) - do NOT include these in your selection
- btc_trend: Current BTC market regime on daily timeframe (price vs daily EMA200, RSI, ADX, +DI/-DI)
- previous_selections: Recent selection history for continuity

CANDIDATE METRICS:
- change_24h_pct: 24h price change (%). Positive = up, negative = down.
- intraday_range_pct: Today's (high-low)/price (%). Measures intraday volatility.
- distance_from_high_pct: Distance from 24h high (%). 0 = at high, higher = pulled back.
- spread_bps: Bid-ask spread in basis points. Lower = more liquid.
- turnover_24h: 24h turnover in USDT. Higher = more actively traded.
- volume_24h: 24h trading volume (coins).
- price / open: Current and opening price.
- is_held: Whether this symbol has an open position.
- was_active: Whether this symbol was in the previous active set.

EVALUATION FRAMEWORK (follow in order):
1. BTC REGIME (daily): BTC below daily EMA200 + -DI > +DI (bearish) = prefer defensive high-liquidity symbols. BTC strong (above EMA200, RSI > 50) = more aggressive picks OK.
2. LIQUIDITY: spread_bps < 30 excellent, < 50 acceptable. turnover_24h reflects real money flow.
3. MOMENTUM: change_24h_pct 1-8% preferred. >15% = chase risk. Slight negative + high turnover = dip opportunity.
4. VOLATILITY: intraday_range_pct 3-10% = tradable. >15% = high risk. <2% = too quiet.
5. PULLBACK: distance_from_high_pct 2-10% = healthy pullback. 0% = buying at top. >20% = potentially broken.
6. HELD POSITIONS: Check held_positions P&L. Losing > -5% = consider dropping. Profitable or mildly negative = keep.
7. CONTINUITY: was_active symbols have indicator history. Slight preference over brand-new symbols to reduce cold-start cost.
8. DIVERSITY: Do not cluster picks into one correlated group (e.g. 3 meme coins, 3 L1s). Spread across categories.

OUTPUT JSON:
- selected_symbols: array of up to top_n symbol strings (e.g. ["XRP/USDT", "SOL/USDT"]). May return fewer if quality threshold not met.
- reasoning: concise Korean explanation (max 500 chars). Be specific about why each symbol was chosen or rejected.

LANGUAGE: Write "reasoning" in Korean. Be concise and natural."""

    def __init__(
        self,
        api_key: str = "",
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        timeout: float = 30.0,
        min_interval_seconds: float = 2.0,
        *,
        auth_mode: str = "api_key",
        oauth_auth_file: Path | None = None,
        oauth_open_browser: bool = True,
        oauth_force_login: bool = False,
        futures_enabled: bool = False,
        leverage: int = 1,
        exchange_context: str = "upbit",
    ) -> None:
        self._api_key: str = api_key
        self._provider: str = provider
        self._model: str = model
        self._timeout: float = timeout
        self._min_interval: float = min_interval_seconds
        self._auth_mode: str = auth_mode
        self._oauth_auth_file: Path | None = oauth_auth_file
        self._oauth_open_browser: bool = oauth_open_browser
        self._oauth_force_login: bool = oauth_force_login
        self._futures_enabled: bool = futures_enabled
        self._leverage: int = leverage
        self._exchange_context: str = exchange_context

        self._rate_lock: asyncio.Lock = asyncio.Lock()
        self._last_call_ts: float = 0.0

        if provider == "openai":
            self._base_url: str = "https://api.openai.com/v1"
        elif provider == "anthropic":
            self._base_url = "https://api.anthropic.com/v1"
        else:
            self._base_url = "https://api.openai.com/v1"

    @property
    def system_prompt(self) -> str:
        if self._exchange_context == "binance":
            return self._FUTURES_SYSTEM_PROMPT
        return self.SYSTEM_PROMPT

    @property
    def universe_system_prompt(self) -> str:
        if self._exchange_context == "binance":
            return self._FUTURES_UNIVERSE_SYSTEM_PROMPT
        return self.UNIVERSE_SYSTEM_PROMPT

    @classmethod
    def create_oauth(
        cls,
        auth_file: Path,
        model: str = "gpt-5.2-codex",
        open_browser: bool = True,
        force_login: bool = False,
        timeout: float = 60.0,
        min_interval_seconds: float = 10.0,
    ) -> "LLMAdvisor":
        return cls(
            api_key="",
            provider="openai",
            model=model,
            timeout=timeout,
            min_interval_seconds=min_interval_seconds,
            auth_mode="oauth",
            oauth_auth_file=auth_file,
            oauth_open_browser=open_browser,
            oauth_force_login=force_login,
        )

    async def get_advice(
        self,
        symbol: str,
        market_summary: dict[str, object],
        strategy_signals: list[dict[str, object]] | None = None,
        *,
        technical_indicators: dict[str, object] | None = None,
        positions: list[dict[str, object]] | None = None,
        balance_info: dict[str, object] | None = None,
        recent_orders: list[dict[str, object]] | None = None,
        recent_candles: list[dict[str, object]] | None = None,
        previous_decisions: list[dict[str, str]] | None = None,
        btc_trend: dict[str, object] | None = None,
        portfolio_exposure: dict[str, object] | None = None,
        risk_context: dict[str, object] | None = None,
        current_symbol_position: dict[str, object] | None = None,
        recent_structure: dict[str, object] | None = None,
        volume_context: dict[str, object] | None = None,
    ) -> LLMAdvice | None:
        if self._auth_mode == "api_key" and not self._api_key:
            return None

        user_prompt = self._build_prompt(
            symbol,
            market_summary,
            strategy_signals,
            technical_indicators=technical_indicators,
            positions=positions,
            balance_info=balance_info,
            recent_orders=recent_orders,
            recent_candles=recent_candles,
            previous_decisions=previous_decisions,
            btc_trend=btc_trend,
            portfolio_exposure=portfolio_exposure,
            risk_context=risk_context,
            current_symbol_position=current_symbol_position,
            recent_structure=recent_structure,
            volume_context=volume_context,
        )

        try:
            async with self._rate_lock:
                now = time.time()
                delay = self._min_interval - (now - self._last_call_ts)
                if delay > 0:
                    await asyncio.sleep(delay)

                if self._auth_mode == "oauth":
                    response = await self._call_oauth_codex(user_prompt)
                elif self._provider == "anthropic":
                    response = await self._call_anthropic(user_prompt)
                else:
                    response = await self._call_openai(user_prompt)
                self._last_call_ts = time.time()

            if response is None:
                return None

            advice = self._parse_response(response)

            logger.info(
                "llm_advice symbol=%s action=%s confidence=%s mode=%s",
                symbol,
                advice.action if advice else "PARSE_FAILED",
                str(advice.confidence) if advice else "0",
                self._auth_mode,
            )

            if advice is not None:
                object.__setattr__(advice, "_prompt", user_prompt)
            return advice
        except Exception as e:
            logger.error("llm_error error=%s symbol=%s mode=%s", str(e), symbol, self._auth_mode)
            return None

    async def get_symbol_universe(
        self,
        *,
        top_n: int,
        candidates: list[dict[str, object]],
        active_symbols: list[dict[str, object]] | None = None,
        held_symbols: list[str] | None = None,
        core_symbols: list[str] | None = None,
        btc_trend: dict[str, object] | None = None,
        held_positions: list[dict[str, object]] | None = None,
        previous_selections: list[dict[str, object]] | None = None,
    ) -> LLMUniverseDecision | None:
        if self._auth_mode == "api_key" and not self._api_key:
            return None
        if top_n <= 0 or not candidates:
            return LLMUniverseDecision(selected_symbols=[])

        user_prompt = self._build_universe_prompt(
            top_n=top_n,
            candidates=candidates,
            active_symbols=active_symbols or [],
            held_symbols=held_symbols or [],
            core_symbols=core_symbols or [],
            btc_trend=btc_trend,
            held_positions=held_positions or [],
            previous_selections=previous_selections or [],
        )

        try:
            async with self._rate_lock:
                now = time.time()
                delay = self._min_interval - (now - self._last_call_ts)
                if delay > 0:
                    await asyncio.sleep(delay)

                if self._auth_mode == "oauth":
                    response = await self._call_oauth_universe(user_prompt)
                elif self._provider == "anthropic":
                    response = await self._call_anthropic_universe(user_prompt)
                else:
                    response = await self._call_openai_universe(user_prompt)
                self._last_call_ts = time.time()

            if response is None:
                return None

            parsed = self._parse_universe_response(response)
            if parsed is None or not parsed.selected_symbols:
                return None

            candidate_symbols = {
                str(item.get("symbol", "")).upper()
                for item in candidates
                if isinstance(item.get("symbol"), str)
            }
            filtered = [s for s in parsed.selected_symbols if s in candidate_symbols]
            if not filtered:
                return None

            return LLMUniverseDecision(
                selected_symbols=filtered[:top_n],
                reasoning=parsed.reasoning,
                prompt=user_prompt,
            )
        except Exception as e:
            logger.error("llm_universe_error error=%s", str(e))
            return None

    async def _call_openai_universe(self, user_prompt: str) -> str | None:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self.universe_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 600,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        data = await self._post_json(f"{self._base_url}/chat/completions", body, headers)
        if data is None:
            return None
        choices_obj = data.get("choices")
        if not isinstance(choices_obj, list) or not choices_obj:
            return None
        choice0 = choices_obj[0]
        if not isinstance(choice0, dict):
            return None
        msg_obj = choice0.get("message")
        if not isinstance(msg_obj, dict):
            return None
        content = msg_obj.get("content")
        return str(content) if content is not None else None

    async def _call_anthropic_universe(self, user_prompt: str) -> str | None:
        headers = {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": self._model,
            "system": self.universe_system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.2,
            "max_tokens": 600,
        }
        data = await self._post_json(f"{self._base_url}/messages", body, headers)
        if data is None:
            return None
        content_obj = data.get("content")
        if not isinstance(content_obj, list) or not content_obj:
            return None
        item0 = content_obj[0]
        if not isinstance(item0, dict):
            return None
        text = item0.get("text")
        return str(text) if text is not None else None

    async def _call_oauth_universe(self, user_prompt: str) -> str | None:
        try:
            from coin_trader.llm.oauth_openai import get_reusable_auth, normalize_model
            from coin_trader.llm.codex_client import query_codex
        except ImportError as e:
            logger.error("oauth_import_error error=%s", str(e))
            return None

        auth_file = self._oauth_auth_file or Path("data/.auth/openai-oauth.json")
        try:
            auth = await asyncio.to_thread(
                get_reusable_auth,
                auth_file=auth_file,
                force_login=self._oauth_force_login,
                open_browser=self._oauth_open_browser,
            )
        except Exception as e:
            logger.error("oauth_auth_error error=%s", str(e))
            return None

        if not auth.access or not auth.account_id:
            logger.error("oauth_missing_credentials")
            return None

        try:
            model = normalize_model(self._model)
        except ValueError as e:
            logger.error("oauth_model_error error=%s", str(e))
            return None

        return await query_codex(
            access_token=auth.access,
            account_id=auth.account_id,
            model=model,
            system_prompt=self.universe_system_prompt,
            user_message=user_prompt,
            timeout=self._timeout,
        )

    async def _call_oauth_codex(self, user_prompt: str) -> str | None:
        try:
            from coin_trader.llm.oauth_openai import get_reusable_auth, normalize_model
            from coin_trader.llm.codex_client import query_codex
        except ImportError as e:
            logger.error("oauth_import_error error=%s", str(e))
            return None

        auth_file = self._oauth_auth_file or Path("data/.auth/openai-oauth.json")
        try:
            auth = await asyncio.to_thread(
                get_reusable_auth,
                auth_file=auth_file,
                force_login=self._oauth_force_login,
                open_browser=self._oauth_open_browser,
            )
        except Exception as e:
            logger.error("oauth_auth_error error=%s", str(e))
            return None

        if not auth.access or not auth.account_id:
            logger.error("oauth_missing_credentials")
            return None

        try:
            model = normalize_model(self._model)
        except ValueError as e:
            logger.error("oauth_model_error error=%s", str(e))
            return None

        return await query_codex(
            access_token=auth.access,
            account_id=auth.account_id,
            model=model,
            system_prompt=self.system_prompt,
            user_message=user_prompt,
            timeout=self._timeout,
        )

    async def _call_openai(self, user_prompt: str) -> str | None:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 500,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        data = await self._post_json(f"{self._base_url}/chat/completions", body, headers)
        if data is None:
            return None
        choices_obj = data.get("choices")
        if not isinstance(choices_obj, list) or not choices_obj:
            return None
        choice0 = choices_obj[0]
        if not isinstance(choice0, dict):
            return None
        msg_obj = choice0.get("message")
        if not isinstance(msg_obj, dict):
            return None
        content = msg_obj.get("content")
        return str(content) if content is not None else None

    async def _call_anthropic(self, user_prompt: str) -> str | None:
        headers = {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": self._model,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.3,
            "max_tokens": 500,
        }
        data = await self._post_json(f"{self._base_url}/messages", body, headers)
        if data is None:
            return None
        content_obj = data.get("content")
        if not isinstance(content_obj, list) or not content_obj:
            return None
        item0 = content_obj[0]
        if not isinstance(item0, dict):
            return None
        text = item0.get("text")
        return str(text) if text is not None else None

    @staticmethod
    def _compact(obj: object) -> str:
        """Serialize to compact JSON (no indent) to save tokens."""
        return json.dumps(obj, separators=(",", ":"), default=str, ensure_ascii=False)

    def _build_prompt(
        self,
        symbol: str,
        market_summary: dict[str, object],
        strategy_signals: list[dict[str, object]] | None,
        *,
        technical_indicators: dict[str, object] | None = None,
        positions: list[dict[str, object]] | None = None,
        balance_info: dict[str, object] | None = None,
        recent_orders: list[dict[str, object]] | None = None,
        recent_candles: list[dict[str, object]] | None = None,
        previous_decisions: list[dict[str, str]] | None = None,
        btc_trend: dict[str, object] | None = None,
        portfolio_exposure: dict[str, object] | None = None,
        risk_context: dict[str, object] | None = None,
        current_symbol_position: dict[str, object] | None = None,
        recent_structure: dict[str, object] | None = None,
        volume_context: dict[str, object] | None = None,
    ) -> str:
        _c = self._compact
        kst = datetime.now(timezone(timedelta(hours=9)))
        parts = [
            f"Analyze {symbol} | KST: {kst.strftime('%Y-%m-%d %H:%M')} ({kst.strftime('%A')})",
            f"\nMarket Data:\n{_c(market_summary)}",
        ]
        if current_symbol_position is not None:
            parts.append(f"\nCurrent Position:\n{_c(current_symbol_position)}")
        else:
            parts.append("\nCurrent Position: null (no position held)")
        if technical_indicators:
            parts.append(f"\nTechnical Indicators:\n{_c(technical_indicators)}")
        if recent_structure:
            parts.append(f"\nPrice Structure:\n{_c(recent_structure)}")
        if volume_context:
            parts.append(f"\nVolume:\n{_c(volume_context)}")
        if btc_trend:
            parts.append(f"\nBTC Trend:\n{_c(btc_trend)}")
        if portfolio_exposure:
            parts.append(f"\nPortfolio Exposure:\n{_c(portfolio_exposure)}")
        if risk_context:
            parts.append(f"\nRisk Context:\n{_c(risk_context)}")
        if strategy_signals:
            parts.append(f"\nStrategy Signals:\n{_c(strategy_signals)}")
        if positions:
            parts.append(f"\nAll Positions:\n{_c(positions)}")
        if balance_info:
            parts.append(f"\nBalance:\n{_c(balance_info)}")
        if recent_orders:
            parts.append(f"\nRecent Orders:\n{_c(recent_orders)}")
        if recent_candles:
            parts.append(f"\nRecent Candles:\n{_c(recent_candles)}")
        if previous_decisions:
            parts.append(
                f"\nPrevious Decisions (oldest to newest):\n{_c(previous_decisions)}"
            )
        parts.append("\nProvide your analysis as JSON.")
        return "\n".join(parts)

    def _parse_response(self, text: str) -> LLMAdvice | None:
        """Parse LLM response with strict validation. Returns None on failure."""
        try:
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data_obj: object = json.loads(text)
            if not isinstance(data_obj, dict):
                return None
            return LLMAdvice.from_mapping(cast(dict[str, object], data_obj))
        except Exception:
            return None

    def _build_universe_prompt(
        self,
        *,
        top_n: int,
        candidates: list[dict[str, object]],
        active_symbols: list[dict[str, object]],
        held_symbols: list[str],
        core_symbols: list[str],
        btc_trend: dict[str, object] | None = None,
        held_positions: list[dict[str, object]] | None = None,
        previous_selections: list[dict[str, object]] | None = None,
    ) -> str:
        _c = self._compact
        kst = datetime.now(timezone(timedelta(hours=9)))
        parts: list[str] = [
            f"=== Symbol Selection (up to {top_n}) | KST: {kst.strftime('%Y-%m-%d %H:%M')} ==="
        ]

        if btc_trend:
            parts.append(f"\nBTC Regime:\n{_c(btc_trend)}")

        parts.append(f"\nCore (excluded from selection):\n{_c(core_symbols)}")

        if held_positions:
            parts.append(f"\nHeld Positions:\n{_c(held_positions)}")
        elif held_symbols:
            parts.append(f"\nHeld: {_c(held_symbols)}")

        if active_symbols:
            parts.append(f"\nActive Non-Core:\n{_c(active_symbols)}")

        if previous_selections:
            parts.append(f"\nPrevious Selections:\n{_c(previous_selections)}")

        parts.append(f"\nCandidates ({len(candidates)}):\n{_c(candidates)}")

        parts.append(
            f"\nSelect up to {top_n} symbols from candidates. Return fewer if quality threshold not met. JSON only."
        )
        return "\n".join(parts)

    def _parse_universe_response(self, text: str) -> LLMUniverseDecision | None:
        try:
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data_obj: object = json.loads(text)
            if not isinstance(data_obj, dict):
                return None
            arr = data_obj.get("selected_symbols")
            if not isinstance(arr, list):
                return None

            out: list[str] = []
            for item in arr:
                if not isinstance(item, str):
                    continue
                symbol = item.upper().strip()
                if not symbol or "/" not in symbol:
                    continue
                if symbol not in out:
                    out.append(symbol)
            reasoning_raw = data_obj.get("reasoning")
            reasoning = str(reasoning_raw).strip() if reasoning_raw is not None else ""
            if len(reasoning) > 500:
                reasoning = reasoning[:500]
            return LLMUniverseDecision(selected_symbols=out, reasoning=reasoning)
        except Exception:
            return None

    async def close(self) -> None:
        return

    async def _post_json(
        self,
        url: str,
        body: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> dict[str, object] | None:
        payload = json.dumps(body).encode("utf-8")
        raw = await asyncio.to_thread(self._post_sync, url, payload, dict(headers))
        try:
            parsed_obj: object = json.loads(raw)
        except Exception:
            return None
        if not isinstance(parsed_obj, dict):
            return None
        return cast(dict[str, object], parsed_obj)

    def _post_sync(self, url: str, payload: bytes, headers: dict[str, str]) -> str:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            raw = resp.read()
            if not isinstance(raw, (bytes, bytearray)):
                return ""
            return bytes(raw).decode("utf-8", errors="replace")
