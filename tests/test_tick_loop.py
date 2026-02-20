from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import NotRequired, TypedDict, cast
from unittest.mock import AsyncMock, Mock, patch

from coin_trader.config.settings import Settings
from coin_trader.core.models import BalanceSnapshot, ExchangeName, Position
from coin_trader.llm.advisory import LLMAdvice
from coin_trader.state.store import StateStore


@dataclass(slots=True)
class _DummyKillSwitch:
    active: bool = False

    def is_active(self) -> bool:
        return self.active


@dataclass(slots=True)
class _ExchangeAdapter:
    get_ticker: AsyncMock
    get_candles: AsyncMock


@dataclass(slots=True)
class _Strategy:
    update_candles: Mock
    on_tick: AsyncMock
    compute_indicators: Mock = dataclasses.field(default_factory=lambda: Mock(return_value=None))
    _candle_history: dict[str, list[object]] = dataclasses.field(default_factory=dict)


@dataclass(slots=True)
class _Broker:
    fetch_balances: AsyncMock
    fetch_positions: AsyncMock


@dataclass(slots=True)
class _Engine:
    execute: AsyncMock


@dataclass(slots=True)
class _AnomalyMonitor:
    record_api_success: Mock
    record_api_failure: Mock
    check_price_anomaly: Mock


@dataclass(slots=True)
class _RiskManager:
    check_circuit_breaker: Mock


@dataclass(slots=True)
class _LLMAdvisorStub:
    get_advice: AsyncMock


class _Components(TypedDict):
    engine: _Engine
    strategy: _Strategy
    broker: _Broker
    exchange_adapter: _ExchangeAdapter
    anomaly_monitor: _AnomalyMonitor
    risk_manager: _RiskManager
    kill_switch: _DummyKillSwitch
    store: StateStore
    settings: Settings
    notifier: object | None
    llm_advisor: NotRequired[_LLMAdvisorStub | None]


_main_mod = importlib.import_module("coin_trader.main")
_RunTick = Callable[[_Components, str], Awaitable[None]]
_run_tick = cast(_RunTick, getattr(_main_mod, "_run_tick"))
_resolve_action = getattr(_main_mod, "_resolve_action")


def _make_components(tmp_path: Path) -> _Components:
    settings = Settings.load_safe()
    store = StateStore(db_path=tmp_path / "state.sqlite3")

    exchange_adapter = _ExchangeAdapter(
        get_ticker=AsyncMock(
            return_value={
                "price": 50000000,
                "open": 49500000,
                "high": 51000000,
                "low": 49000000,
                "volume_24h": 123.45,
                "bid": 49990000,
                "ask": 50010000,
                "prev_close": 50000000,
                "turnover_24h": 0,
                "change_rate": 0,
            }
        ),
        get_candles=AsyncMock(return_value=[]),
    )

    strategy = _Strategy(
        update_candles=Mock(return_value=None),
        on_tick=AsyncMock(return_value=[]),
    )

    broker = _Broker(
        fetch_balances=AsyncMock(
            return_value=BalanceSnapshot(
                exchange=ExchangeName.UPBIT,
                timestamp=datetime.now(timezone.utc),
                balances={"KRW": Decimal("1000000")},
                total_value_krw=Decimal("1000000"),
            )
        ),
        fetch_positions=AsyncMock(return_value=[]),
    )

    engine = _Engine(execute=AsyncMock(return_value=None))

    anomaly_monitor = _AnomalyMonitor(
        record_api_success=Mock(return_value=None),
        record_api_failure=Mock(return_value=None),
        check_price_anomaly=Mock(return_value=None),
    )
    risk_manager = _RiskManager(check_circuit_breaker=Mock(return_value=False))
    kill_switch = _DummyKillSwitch(active=False)

    return {
        "engine": engine,
        "strategy": strategy,
        "broker": broker,
        "exchange_adapter": exchange_adapter,
        "anomaly_monitor": anomaly_monitor,
        "risk_manager": risk_manager,
        "kill_switch": kill_switch,
        "store": store,
        "settings": settings,
        "notifier": None,
    }


async def test_run_tick_with_llm_advisor_calls_get_advice_and_logs_no_order(tmp_path: Path) -> None:
    components = _make_components(tmp_path)
    components["settings"].llm_trading_enabled = False

    # Clear global caches so LLM skip logic doesn't trigger
    _main_mod._last_llm_prices.pop("BTC/KRW", None)
    _main_mod._last_llm_times.pop("BTC/KRW", None)

    advice_calls: list[tuple[str, dict[str, object]]] = []

    async def _get_advice(
        symbol: str,
        market_summary: dict[str, object],
        strategy_signals: list[dict[str, object]] | None = None,
        **kwargs: object,
    ) -> LLMAdvice | None:
        advice_calls.append((symbol, market_summary))
        return LLMAdvice(
            action="BUY_CONSIDER",
            confidence=Decimal("0.9"),
            reasoning="Looks interesting, but strategy has no signals.",
            risk_notes="",
        )

    llm_advisor = _LLMAdvisorStub(get_advice=AsyncMock(side_effect=_get_advice))
    components["llm_advisor"] = llm_advisor

    execute_calls: list[object] = []

    async def _execute(intent: object, state: object) -> None:
        execute_calls.append((intent, state))
        return None

    components["engine"] = _Engine(execute=AsyncMock(side_effect=_execute))

    events: list[tuple[str, dict[str, object]]] = []

    def _log_event(event_type: str, data: dict[str, object]) -> None:
        events.append((event_type, data))

    try:
        with patch("coin_trader.main.log_event", new=_log_event):
            await _run_tick(components, "BTC/KRW")

        assert len(advice_calls) == 1
        assert execute_calls == []

        # Actual code logs event_type="decision" with llm_action field
        decision_events = [
            e for e in events
            if e[0] == "decision" and e[1].get("llm_action") is not None
        ]
        assert len(decision_events) == 1
        _, payload = decision_events[0]
        assert payload.get("symbol") == "BTC/KRW"
        assert payload.get("llm_action") == "BUY_CONSIDER"
        assert payload.get("final_action") == "HOLD"
    finally:
        _main_mod._last_llm_prices.pop("BTC/KRW", None)
        _main_mod._last_llm_times.pop("BTC/KRW", None)
        components["store"].close()


async def test_run_tick_without_llm_advisor_no_error(tmp_path: Path) -> None:
    components = _make_components(tmp_path)

    execute_calls: list[object] = []

    async def _execute(intent: object, state: object) -> None:
        execute_calls.append((intent, state))
        return None

    components["engine"] = _Engine(execute=AsyncMock(side_effect=_execute))

    events: list[tuple[str, dict[str, object]]] = []

    def _log_event(event_type: str, data: dict[str, object]) -> None:
        events.append((event_type, data))

    try:
        with patch("coin_trader.main.log_event", new=_log_event):
            await _run_tick(components, "BTC/KRW")

        assert execute_calls == []
        # No LLM advisor → no decision event with llm_action
        decision_events = [
            e for e in events
            if e[0] == "decision" and "final_action" in e[1]
        ]
        # Without LLM advisor, _skip_llm stays False so decision is still logged
        if decision_events:
            assert decision_events[0][1].get("final_action") == "HOLD"
    finally:
        components["store"].close()


async def test_stop_loss_triggers_sell_when_position_loses(tmp_path: Path) -> None:
    components = _make_components(tmp_path)

    entry_price = Decimal("50000000")
    current_price = Decimal("44000000")  # -12%, triggers hard stop-loss at -10%

    components["exchange_adapter"] = _ExchangeAdapter(
        get_ticker=AsyncMock(
            return_value={
                "price": int(current_price),
                "open": 50000000,
                "high": 50500000,
                "low": 43500000,
                "volume_24h": 100.0,
                "bid": 43900000,
                "ask": 44100000,
                "prev_close": 50000000,
                "turnover_24h": 0,
                "change_rate": 0,
            }
        ),
        get_candles=AsyncMock(return_value=[]),
    )

    losing_position = Position(
        exchange=ExchangeName.UPBIT,
        symbol="BTC/KRW",
        quantity=Decimal("0.01"),
        average_entry_price=entry_price,
        current_price=current_price,
        timestamp=datetime.now(timezone.utc),
    )
    components["broker"].fetch_positions = AsyncMock(return_value=[losing_position])

    execute_calls: list[object] = []

    async def _execute(intent: object, state: object) -> None:
        execute_calls.append(intent)
        return None

    components["engine"] = _Engine(execute=AsyncMock(side_effect=_execute))

    events: list[tuple[str, dict[str, object]]] = []

    def _log_event(event_type: str, data: dict[str, object]) -> None:
        events.append((event_type, data))

    try:
        with patch("coin_trader.main.log_event", new=_log_event):
            await _run_tick(components, "BTC/KRW")

        protection_events = [
            e for e in events
            if e[0] == "decision" and e[1].get("trigger") == "position_protection"
        ]
        assert len(protection_events) == 1
        assert "stop_loss" in str(protection_events[0][1].get("reason", ""))

        assert len(execute_calls) == 1
    finally:
        components["store"].close()


async def test_take_profit_triggers_sell_when_position_gains(tmp_path: Path) -> None:
    components = _make_components(tmp_path)

    entry_price = Decimal("50000000")
    current_price = Decimal("56000000")

    components["exchange_adapter"] = _ExchangeAdapter(
        get_ticker=AsyncMock(
            return_value={
                "price": int(current_price),
                "open": 50000000,
                "high": 56500000,
                "low": 50000000,
                "volume_24h": 200.0,
                "bid": 55900000,
                "ask": 56100000,
                "prev_close": 50000000,
                "turnover_24h": 0,
                "change_rate": 0,
            }
        ),
        get_candles=AsyncMock(return_value=[]),
    )

    winning_position = Position(
        exchange=ExchangeName.UPBIT,
        symbol="BTC/KRW",
        quantity=Decimal("0.01"),
        average_entry_price=entry_price,
        current_price=current_price,
        timestamp=datetime.now(timezone.utc),
    )
    components["broker"].fetch_positions = AsyncMock(return_value=[winning_position])

    execute_calls: list[object] = []

    async def _execute(intent: object, state: object) -> None:
        execute_calls.append(intent)
        return None

    components["engine"] = _Engine(execute=AsyncMock(side_effect=_execute))

    events: list[tuple[str, dict[str, object]]] = []

    def _log_event(event_type: str, data: dict[str, object]) -> None:
        events.append((event_type, data))

    try:
        with patch("coin_trader.main.log_event", new=_log_event):
            await _run_tick(components, "BTC/KRW")

        protection_events = [
            e for e in events
            if e[0] == "decision" and e[1].get("trigger") == "position_protection"
        ]
        assert len(protection_events) == 1
        assert "take_profit" in str(protection_events[0][1].get("reason", ""))

        assert len(execute_calls) == 1
    finally:
        components["store"].close()


async def test_no_protection_trigger_within_safe_range(tmp_path: Path) -> None:
    components = _make_components(tmp_path)

    entry_price = Decimal("50000000")
    current_price = Decimal("51000000")

    components["exchange_adapter"] = _ExchangeAdapter(
        get_ticker=AsyncMock(
            return_value={
                "price": int(current_price),
                "open": 50000000,
                "high": 51500000,
                "low": 49500000,
                "volume_24h": 150.0,
                "bid": 50900000,
                "ask": 51100000,
                "prev_close": 50000000,
                "turnover_24h": 0,
                "change_rate": 0,
            }
        ),
        get_candles=AsyncMock(return_value=[]),
    )

    safe_position = Position(
        exchange=ExchangeName.UPBIT,
        symbol="BTC/KRW",
        quantity=Decimal("0.01"),
        average_entry_price=entry_price,
        current_price=current_price,
        timestamp=datetime.now(timezone.utc),
    )
    components["broker"].fetch_positions = AsyncMock(return_value=[safe_position])

    execute_calls: list[object] = []

    async def _execute(intent: object, state: object) -> None:
        execute_calls.append(intent)
        return None

    components["engine"] = _Engine(execute=AsyncMock(side_effect=_execute))

    events: list[tuple[str, dict[str, object]]] = []

    def _log_event(event_type: str, data: dict[str, object]) -> None:
        events.append((event_type, data))

    try:
        with patch("coin_trader.main.log_event", new=_log_event):
            await _run_tick(components, "BTC/KRW")

        protection_events = [
            e for e in events
            if e[0] == "decision" and e[1].get("trigger") == "position_protection"
        ]
        assert len(protection_events) == 0

        assert len(execute_calls) == 0
    finally:
        components["store"].close()


def _make_signal(signal_type: str, confidence: str = "0.7") -> object:
    from coin_trader.core.models import Signal, SignalType
    from datetime import datetime, timezone

    return Signal(
        strategy_name="test",
        symbol="BTC/KRW",
        signal_type=SignalType(signal_type),
        timestamp=datetime.now(timezone.utc),
        confidence=Decimal(confidence),
        metadata={},
    )


def _make_settings(
    *,
    llm_trading_enabled: bool = True,
    llm_min_confidence: float = 0.7,
    llm_solo_min_confidence: float = 0.8,
) -> Settings:
    settings = Settings.load_safe()
    settings.llm_trading_enabled = llm_trading_enabled
    settings.llm_min_confidence = llm_min_confidence
    settings.llm_solo_min_confidence = llm_solo_min_confidence
    return settings


def test_resolve_action_strategy_buy_llm_buy_consider() -> None:
    signals = [_make_signal("buy")]
    advice = LLMAdvice(
        action="BUY_CONSIDER", confidence=Decimal("0.8"), reasoning="r", risk_notes=""
    )
    assert _resolve_action(signals, advice, _make_settings()) == "BUY"


def test_resolve_action_strategy_buy_llm_hold_vetoes() -> None:
    signals = [_make_signal("buy")]
    advice = LLMAdvice(action="HOLD", confidence=Decimal("0.8"), reasoning="r", risk_notes="")
    assert _resolve_action(signals, advice, _make_settings()) == "HOLD"


def test_resolve_action_strategy_buy_llm_sell_conflict() -> None:
    """LLM-primary model: LLM SELL_CONSIDER overrides strategy BUY."""
    signals = [_make_signal("buy")]
    advice = LLMAdvice(
        action="SELL_CONSIDER", confidence=Decimal("0.9"), reasoning="r", risk_notes=""
    )
    assert _resolve_action(signals, advice, _make_settings()) == "SELL"


def test_resolve_action_strategy_sell_llm_sell_consider() -> None:
    signals = [_make_signal("sell")]
    advice = LLMAdvice(
        action="SELL_CONSIDER", confidence=Decimal("0.8"), reasoning="r", risk_notes=""
    )
    assert _resolve_action(signals, advice, _make_settings()) == "SELL"


def test_resolve_action_strategy_sell_llm_hold_vetoes() -> None:
    signals = [_make_signal("sell")]
    advice = LLMAdvice(action="HOLD", confidence=Decimal("0.8"), reasoning="r", risk_notes="")
    assert _resolve_action(signals, advice, _make_settings()) == "HOLD"


def test_resolve_action_strategy_hold_llm_buy_high_confidence() -> None:
    signals = [_make_signal("hold")]
    advice = LLMAdvice(
        action="BUY_CONSIDER", confidence=Decimal("0.85"), reasoning="r", risk_notes=""
    )
    assert _resolve_action(signals, advice, _make_settings()) == "BUY"


def test_resolve_action_strategy_hold_llm_buy_low_confidence_no_trade() -> None:
    """LLM confidence below min threshold → treated as HOLD."""
    signals = [_make_signal("hold")]
    advice = LLMAdvice(
        action="BUY_CONSIDER", confidence=Decimal("0.65"), reasoning="r", risk_notes=""
    )
    assert _resolve_action(signals, advice, _make_settings()) == "HOLD"


def test_resolve_action_strategy_hold_llm_sell_high_confidence() -> None:
    signals = [_make_signal("hold")]
    advice = LLMAdvice(
        action="SELL_CONSIDER", confidence=Decimal("0.9"), reasoning="r", risk_notes=""
    )
    assert _resolve_action(signals, advice, _make_settings()) == "SELL"


def test_resolve_action_llm_disabled_follows_strategy() -> None:
    signals = [_make_signal("buy")]
    advice = LLMAdvice(action="HOLD", confidence=Decimal("0.9"), reasoning="r", risk_notes="")
    assert _resolve_action(signals, advice, _make_settings(llm_trading_enabled=False)) == "BUY"


def test_resolve_action_no_advice_holds() -> None:
    """LLM-primary model: no LLM advice → HOLD (not strategy fallback)."""
    signals = [_make_signal("sell")]
    assert _resolve_action(signals, None, _make_settings()) == "HOLD"


def test_resolve_action_llm_below_min_confidence_treated_as_hold() -> None:
    signals = [_make_signal("buy")]
    advice = LLMAdvice(
        action="BUY_CONSIDER", confidence=Decimal("0.5"), reasoning="r", risk_notes=""
    )
    assert _resolve_action(signals, advice, _make_settings(llm_min_confidence=0.7)) == "HOLD"
