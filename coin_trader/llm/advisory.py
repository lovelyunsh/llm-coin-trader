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
    _prompt: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "LLMAdvice | None":
        required = {"action", "confidence", "reasoning", "risk_notes"}
        if not required.issubset(data.keys()):
            return None

        action_raw = str(data.get("action", "HOLD")).upper()
        if action_raw not in {"HOLD", "BUY_CONSIDER", "SELL_CONSIDER"}:
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

        return cls(
            action=action_raw,
            confidence=conf,
            reasoning=reasoning,
            risk_notes=risk_notes,
            buy_pct=buy_pct,
            sell_pct=sell_pct,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class LLMUniverseDecision:
    selected_symbols: list[str]
    reasoning: str = ""
    prompt: str = ""


class LLMAdvisor:
    """LLM-based advisory - produces bounded suggestions, never orders."""

    SYSTEM_PROMPT: str = """You are the PRIMARY decision maker for an automated crypto trading system on Upbit (KRW pairs).

You will receive:
- Market data (OHLCV, 24h change, timeframe)
- Technical indicators (1h): EMA(12/26/200), RSI, ATR, ADX(+DI/-DI), MACD(line/signal/histogram), Bollinger Bands(upper/middle/lower/width)
- BTC trend filter (BTC price vs EMA200, RSI, ADX)
- Current symbol position (if held): quantity, entry price, unrealized P&L, portfolio weight
- Decision context: new_entry / add_position / reduce_position / risk_management
- Recent price structure: higher_high, higher_low, lower_high, lower_low, range_market
- Volume context: volume_vs_avg_ratio, volume_trend (increasing/decreasing/flat)
- Portfolio exposure (per-coin %, alt total %)
- Risk context (ATR stop distance, exposure limits)
- All positions, account balance, recent orders, recent candles

Your output MUST be valid JSON with these fields:
- action: one of \"HOLD\", \"BUY_CONSIDER\", \"SELL_CONSIDER\"
- confidence: float between 0.0 and 1.0
- reasoning: brief explanation (max 500 chars)
- risk_notes: any risk concerns (max 300 chars)
- buy_pct: optional float percent for BUY size (0-10). If omitted, system default is used.
- sell_pct: optional float percent for SELL size (0-100). If omitted, system default is used.

YOUR AUTHORITY:
- Your decision IS the trading decision. BUY_CONSIDER = system will buy. SELL_CONSIDER = system will sell.
- For positions at -5% to -10% loss: SELL_CONSIDER = sell, HOLD = keep holding.
- For positions at +10%+ profit: Only explicit HOLD keeps the position; otherwise it sells.
- Hard stop-loss at -10% is automatic and bypasses you entirely.

DECISION FRAMEWORK (follow this order strictly):
1. DATA CHECK: Verify data consistency. If positions show suspicious P&L or prices seem wrong, note in risk_notes.
2. TIMEFRAME: All indicators are based on the timeframe specified in market data. Interpret accordingly.
3. BTC FILTER: Check btc_trend. If BTC is below EMA200 with ADX rising and -DI > +DI (downtrend strengthening), block new alt buys. If BTC RSI < 40 and falling, increase caution.
4. POSITION CHECK: Check current_symbol_position. If null → new entry evaluation. If held → consider decision_context (add_position / reduce_position / risk_management). Avoid repeated buys into already heavy positions.
5. ADX TREND FILTER: If ADX < 20, market is ranging — avoid trend-following entries. If ADX > 25, trend strategies are valid. Check +DI vs -DI for direction.
6. MACD + RSI MOMENTUM: Histogram increasing = momentum accelerating. MACD crossing above signal = bullish. RSI >70 = overbought caution, <30 = potential entry. Combine both for momentum direction.
7. BOLLINGER BANDS: Band squeeze (low bb_width) = volatility expansion imminent. Price near upper band + declining volume = potential reversal. Price bouncing off lower band + rising volume = potential entry.
8. RECENT STRUCTURE: Check higher_high/higher_low (uptrend structure), lower_high/lower_low (downtrend structure), range_market. Align entries with structure direction.
9. VOLUME TREND: volume_trend "increasing" with breakout = strong signal. "decreasing" during rally = weak, potential reversal. volume_vs_avg_ratio < 0.5 = low conviction move.
10. ATR RISK: Use atr_stop_distance as reference for stop-loss sizing. Higher ATR = wider stops needed = smaller position size.
11. PORTFOLIO CHECK: Respect exposure limits. If single coin > max_single_coin_exposure_pct, do NOT buy more. If alt_total > max_alt_total_exposure_pct, do NOT buy alts.
12. FINAL DECISION: Synthesize all above into BUY_CONSIDER / SELL_CONSIDER / HOLD with confidence level.

GUIDELINES:
- Be decisive. Multiple confirming signals = high confidence. Conflicting signals = HOLD.
- Factor in recent order history to avoid overtrading.
- High confidence (>0.8) only with strong multi-indicator confirmation.
- If action is BUY_CONSIDER, include buy_pct (0-10) based on conviction and volatility.
- If action is SELL_CONSIDER, include sell_pct (0-100) based on risk urgency.
- For risk_management context: cut early if trend (ADX + MACD) is against you.
- For reduce_position context: consider if momentum (MACD histogram) supports further gains.
- For new_entry: require at least 3 confirming factors (trend + momentum + structure).
- For add_position: be stricter than new_entry — only add if conviction is higher than initial entry.

LANGUAGE: Write "reasoning" and "risk_notes" in Korean (한국어). Be concise and natural."""

    UNIVERSE_SYSTEM_PROMPT: str = """You select tradable symbols for a KRW crypto bot.

You receive a candidate list with per-symbol metrics. Your task is portfolio construction, not trade entry timing.

Selection rules:
- Return exactly top_n symbols unless candidates are fewer.
- Prefer liquid symbols with healthy trend/momentum and acceptable spread.
- Avoid overheated one-candle spikes and thin/erratic symbols.
- Keep diversity; do not cluster all picks into one very correlated micro-theme.
- Existing held symbols should be respected unless quality is clearly weak.

Output JSON fields:
- selected_symbols: array of symbol strings (e.g. ["XRP/KRW", "SOL/KRW"])
- reasoning: concise Korean explanation (max 300 chars)
"""

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

        self._rate_lock: asyncio.Lock = asyncio.Lock()
        self._last_call_ts: float = 0.0

        if provider == "openai":
            self._base_url: str = "https://api.openai.com/v1"
        elif provider == "anthropic":
            self._base_url = "https://api.anthropic.com/v1"
        else:
            self._base_url = "https://api.openai.com/v1"

    @classmethod
    def create_oauth(
        cls,
        auth_file: Path,
        model: str = "gpt-5.2-codex",
        open_browser: bool = True,
        force_login: bool = False,
        timeout: float = 60.0,
        min_interval_seconds: float = 5.0,
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
        decision_context: str | None = None,
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
            decision_context=decision_context,
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
                {"role": "system", "content": self.UNIVERSE_SYSTEM_PROMPT},
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
            "system": self.UNIVERSE_SYSTEM_PROMPT,
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
            system_prompt=self.UNIVERSE_SYSTEM_PROMPT,
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
            system_prompt=self.SYSTEM_PROMPT,
            user_message=user_prompt,
            timeout=self._timeout,
        )

    async def _call_openai(self, user_prompt: str) -> str | None:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
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
            "system": self.SYSTEM_PROMPT,
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
        decision_context: str | None = None,
    ) -> str:
        parts = [
            f"Analyze {symbol} for trading decision.\n\nMarket Data:\n{json.dumps(market_summary, indent=2, default=str)}"
        ]
        if decision_context:
            parts.append(f"\nDecision Context: {decision_context}")
        if current_symbol_position is not None:
            parts.append(
                f"\nCurrent Symbol Position:\n{json.dumps(current_symbol_position, indent=2, default=str)}"
            )
        else:
            parts.append("\nCurrent Symbol Position: null (no position held)")
        if technical_indicators:
            parts.append(
                f"\nTechnical Indicators:\n{json.dumps(technical_indicators, indent=2, default=str)}"
            )
        if recent_structure:
            parts.append(
                f"\nRecent Price Structure:\n{json.dumps(recent_structure, indent=2, default=str)}"
            )
        if volume_context:
            parts.append(f"\nVolume Context:\n{json.dumps(volume_context, indent=2, default=str)}")
        if btc_trend:
            parts.append(f"\nBTC Trend Filter:\n{json.dumps(btc_trend, indent=2, default=str)}")
        if portfolio_exposure:
            parts.append(
                f"\nPortfolio Exposure:\n{json.dumps(portfolio_exposure, indent=2, default=str)}"
            )
        if risk_context:
            parts.append(f"\nRisk Context:\n{json.dumps(risk_context, indent=2, default=str)}")
        if strategy_signals:
            parts.append(
                f"\nStrategy Signals:\n{json.dumps(strategy_signals, indent=2, default=str)}"
            )
        if positions:
            parts.append(f"\nCurrent Positions:\n{json.dumps(positions, indent=2, default=str)}")
        if balance_info:
            parts.append(f"\nAccount Balance:\n{json.dumps(balance_info, indent=2, default=str)}")
        if recent_orders:
            parts.append(f"\nRecent Orders:\n{json.dumps(recent_orders, indent=2, default=str)}")
        if recent_candles:
            parts.append(
                f"\nRecent Candles (OHLCV):\n{json.dumps(recent_candles, indent=2, default=str)}"
            )
        if previous_decisions:
            parts.append(
                f"\nYour Previous Decisions (oldest→newest):\n{json.dumps(previous_decisions, indent=2, default=str, ensure_ascii=False)}"
                "\nBe consistent with your prior reasoning unless market conditions have materially changed."
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
    ) -> str:
        parts = [
            "Select tradable symbols for the next refresh window.",
            f"top_n={top_n}",
            f"held_symbols={json.dumps(held_symbols, ensure_ascii=False)}",
            f"core_symbols={json.dumps(core_symbols, ensure_ascii=False)}",
            f"active_symbols={json.dumps(active_symbols, ensure_ascii=False)}",
            f"candidates={json.dumps(candidates, ensure_ascii=False)}",
            "Return JSON only.",
        ]
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
            if len(reasoning) > 300:
                reasoning = reasoning[:300]
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
