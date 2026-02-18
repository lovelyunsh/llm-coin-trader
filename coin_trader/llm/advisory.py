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

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "LLMAdvice | None":
        required = {"action", "confidence", "reasoning", "risk_notes"}
        if set(data.keys()) != required:
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

        return cls(action=action_raw, confidence=conf, reasoning=reasoning, risk_notes=risk_notes)


class LLMAdvisor:
    """LLM-based advisory - produces bounded suggestions, never orders."""

    SYSTEM_PROMPT: str = """You are the PRIMARY decision maker for an automated crypto trading system on Upbit (KRW pairs).
Technical strategy signals (EMA crossover, RSI, ATR) are provided as REFERENCE DATA — you make the final call.

You will receive:
- Current market data (OHLCV)
- Technical indicators (EMA crossover, RSI, ATR, volatility ratio)
- Current positions and unrealized P&L
- Account balance information
- Recent order history
- Recent candlestick data

Your output MUST be valid JSON with exactly these fields:
- action: one of \"HOLD\", \"BUY_CONSIDER\", \"SELL_CONSIDER\"
- confidence: float between 0.0 and 1.0
- reasoning: brief explanation (max 500 chars)
- risk_notes: any risk concerns (max 300 chars)

YOUR AUTHORITY:
- Your decision IS the trading decision. BUY_CONSIDER = system will buy. SELL_CONSIDER = system will sell.
- Strategy signals are just indicators for your reference, not co-decision-makers.
- For positions at -5% to -10% loss: system asks you whether to cut losses. SELL_CONSIDER = sell, HOLD = keep holding.
- For positions at +10%+ profit: system asks you whether to take profit. Only explicit HOLD keeps the position; otherwise it sells.
- Hard stop-loss at -10% is automatic and bypasses you entirely.

GUIDELINES:
- You have full trading authority. Be decisive, not always conservative.
- Consider position sizing: if already heavily invested, be cautious about additional buys.
- Factor in recent order history to avoid overtrading.
- Use technical indicators as one of many inputs, not as rules.
- High confidence (>0.8) should be used when you have strong conviction.
- When genuinely uncertain, recommend HOLD with low confidence.
- For loss positions (-5% to -10%): consider if the loss is likely to deepen or recover. Cut early if trend is against you.
- For profit positions (+10%+): consider if momentum supports further gains. Don't be greedy — taking profit is often correct.

LANGUAGE: Write "reasoning" and "risk_notes" fields in Korean (한국어). Be concise and natural."""

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

            return advice
        except Exception as e:
            logger.error("llm_error error=%s symbol=%s mode=%s", str(e), symbol, self._auth_mode)
            return None

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
    ) -> str:
        parts = [
            f"Analyze {symbol} for trading decision.\n\nMarket Data:\n{json.dumps(market_summary, indent=2, default=str)}"
        ]
        if technical_indicators:
            parts.append(
                f"\nTechnical Indicators:\n{json.dumps(technical_indicators, indent=2, default=str)}"
            )
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
