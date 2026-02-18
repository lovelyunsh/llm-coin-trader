from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

from coin_trader.llm.advisory import LLMAdvice, LLMAdvisor


class _TestAdvisor(LLMAdvisor):
    def parse(self, text: str) -> LLMAdvice | None:
        return self._parse_response(text)


def test_llm_advice_from_mapping_valid() -> None:
    data: dict[str, object] = {
        "action": "BUY_CONSIDER",
        "confidence": "0.7",
        "reasoning": "Trend looks constructive; wait for confirmation.",
        "risk_notes": "Low liquidity can spike spreads.",
    }
    advice = LLMAdvice.from_mapping(data)
    assert advice is not None
    assert advice.action == "BUY_CONSIDER"
    assert advice.confidence == Decimal("0.7")
    assert advice.reasoning == "Trend looks constructive; wait for confirmation."
    assert advice.risk_notes == "Low liquidity can spike spreads."


def test_llm_advice_from_mapping_invalid_action_defaults_to_hold() -> None:
    data: dict[str, object] = {
        "action": "BUY",
        "confidence": 1,
        "reasoning": "Not a valid action; should clamp to HOLD.",
        "risk_notes": "",
    }
    advice = LLMAdvice.from_mapping(data)
    assert advice is not None
    assert advice.action == "HOLD"


def test_llm_advice_from_mapping_missing_required_fields_returns_none() -> None:
    data_missing: dict[str, object] = {
        "action": "HOLD",
        "confidence": 0.2,
        "reasoning": "Missing risk_notes key.",
    }
    assert LLMAdvice.from_mapping(data_missing) is None


def test_llm_advice_from_mapping_overlong_reasoning_returns_none() -> None:
    data: dict[str, object] = {
        "action": "HOLD",
        "confidence": 0.1,
        "reasoning": "a" * 501,
        "risk_notes": "",
    }
    assert LLMAdvice.from_mapping(data) is None


def test_llm_advisor_parse_response_valid_json() -> None:
    advisor = _TestAdvisor(api_key="test")
    text = '{"action":"HOLD","confidence":0.3,"reasoning":"ok","risk_notes":""}'
    advice = advisor.parse(text)
    assert advice is not None
    assert advice.action == "HOLD"
    assert advice.confidence == Decimal("0.3")


def test_llm_advisor_parse_response_json_code_fence() -> None:
    advisor = _TestAdvisor(api_key="test")
    text = """```json
{"action":"SELL_CONSIDER","confidence":0.9,"reasoning":"r","risk_notes":"n"}
```"""
    advice = advisor.parse(text)
    assert advice is not None
    assert advice.action == "SELL_CONSIDER"
    assert advice.confidence == Decimal("0.9")


def test_llm_advisor_parse_response_invalid_json_returns_none() -> None:
    advisor = _TestAdvisor(api_key="test")
    assert advisor.parse("{not-json") is None


async def test_llm_advisor_create_oauth_sets_auth_mode(tmp_path: Path) -> None:
    auth_file = tmp_path / "openai-oauth.json"
    advisor = LLMAdvisor.create_oauth(auth_file=auth_file)

    call_mock = AsyncMock(
        return_value='{"action":"HOLD","confidence":0.1,"reasoning":"ok","risk_notes":""}'
    )
    setattr(advisor, "_call_oauth_codex", call_mock)

    advice = await advisor.get_advice("BTC/KRW", {"price": "1"}, [])
    assert advice is not None
    call_mock.assert_awaited_once()
