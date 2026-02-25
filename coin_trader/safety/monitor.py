"""Anomaly detection - monitors for dangerous conditions"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal
from typing import Protocol, cast

from coin_trader.core.models import MarketData, SafetyEvent, SafetyEventType


class _SafetyEventCtor(Protocol):
    def __call__(
        self,
        *,
        event_type: SafetyEventType,
        timestamp: datetime,
        description: str,
        severity: str,
        triggered_by: str,
        metadata: dict[str, object] | None = None,
    ) -> SafetyEvent: ...


_safety_event = cast(_SafetyEventCtor, cast(object, SafetyEvent))


ZERO = Decimal("0")
HUNDRED = Decimal("100")


def _cfg_int(config: Mapping[str, object], key: str, default: int) -> int:
    value = config.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except Exception:
            return default
    return default


def _cfg_decimal(config: Mapping[str, object], key: str, default: Decimal) -> Decimal:
    value = config.get(key)
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, str):
        try:
            return Decimal(value)
        except Exception:
            return default
    return default


class AnomalyMonitor:
    api_failure_threshold: int
    balance_mismatch_threshold_pct: Decimal
    price_change_threshold_pct: Decimal
    spread_threshold_pct: Decimal
    _consecutive_api_failures: int
    _prev_prices: dict[str, Decimal]

    def __init__(self, config: Mapping[str, object]) -> None:
        self.api_failure_threshold = _cfg_int(config, "api_failure_threshold", 5)
        self.balance_mismatch_threshold_pct = _cfg_decimal(
            config, "balance_mismatch_threshold_pct", Decimal("1.0")
        )
        self.price_change_threshold_pct = _cfg_decimal(
            config, "price_change_threshold_pct", Decimal("10.0")
        )
        self.spread_threshold_pct = _cfg_decimal(
            config, "spread_threshold_pct", Decimal("5.0")
        )
        self._consecutive_api_failures = 0
        self._prev_prices = {}

    def record_api_success(self) -> None:
        self._consecutive_api_failures = 0

    def record_api_failure(self) -> SafetyEvent | None:
        self._consecutive_api_failures += 1
        if self._consecutive_api_failures >= self.api_failure_threshold:
            return _safety_event(
                event_type=SafetyEventType.API_FAILURE,
                timestamp=datetime.now(timezone.utc),
                description=f"API consecutive failures: {self._consecutive_api_failures}",
                severity="critical",
                triggered_by="AnomalyMonitor",
                metadata={"consecutive_failures": self._consecutive_api_failures},
            )
        return None

    def check_balance_mismatch(
        self, expected: Decimal, actual: Decimal
    ) -> SafetyEvent | None:
        if expected == ZERO and actual == ZERO:
            return None
        max_val = max(abs(expected), abs(actual))
        if max_val == ZERO:
            return None
        diff_pct = abs(expected - actual) / max_val * HUNDRED
        if diff_pct >= self.balance_mismatch_threshold_pct:
            return _safety_event(
                event_type=SafetyEventType.BALANCE_MISMATCH,
                timestamp=datetime.now(timezone.utc),
                description=f"Balance mismatch: {diff_pct:.2f}%",
                severity="critical",
                triggered_by="AnomalyMonitor",
                metadata={
                    "expected": str(expected),
                    "actual": str(actual),
                    "diff_pct": str(diff_pct),
                },
            )
        return None

    def check_price_anomaly(self, md: MarketData) -> SafetyEvent | None:
        prev = self._prev_prices.get(md.symbol)
        self._prev_prices[md.symbol] = md.close
        if prev is None or prev == ZERO:
            return None
        change_pct = abs(md.close - prev) / prev * HUNDRED
        if change_pct >= self.price_change_threshold_pct:
            return _safety_event(
                event_type=SafetyEventType.PRICE_ANOMALY,
                timestamp=datetime.now(timezone.utc),
                description=f"Price anomaly on {md.symbol}: {change_pct:.2f}% change",
                severity="high",
                triggered_by="AnomalyMonitor",
                metadata={
                    "symbol": md.symbol,
                    "prev": str(prev),
                    "current": str(md.close),
                    "change_pct": str(change_pct),
                },
            )
        return None

    def check_spread_anomaly(self, md: MarketData) -> SafetyEvent | None:
        if md.bid is None or md.ask is None or md.bid == ZERO:
            return None
        spread_pct = (md.ask - md.bid) / md.bid * HUNDRED
        if spread_pct >= self.spread_threshold_pct:
            return _safety_event(
                event_type=SafetyEventType.PRICE_ANOMALY,
                timestamp=datetime.now(timezone.utc),
                description=f"Wide spread on {md.symbol}: {spread_pct:.2f}%",
                severity="medium",
                triggered_by="AnomalyMonitor",
                metadata={
                    "symbol": md.symbol,
                    "bid": str(md.bid),
                    "ask": str(md.ask),
                    "spread_pct": str(spread_pct),
                },
            )
        return None
