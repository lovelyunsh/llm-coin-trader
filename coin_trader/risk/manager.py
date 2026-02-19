"""Risk Manager - the non-bypassable safety kernel"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Protocol, cast
from uuid import UUID

from coin_trader.core.models import (
    OrderIntent,
    OrderSide,
    OrderType,
    RiskDecision,
    RiskDecisionRecord,
)
from coin_trader.risk.limits import RiskLimits


class _RiskDecisionRecordCtor(Protocol):
    def __call__(
        self,
        *,
        intent_id: UUID,
        decision: RiskDecision,
        reason: str,
        timestamp: datetime,
        metadata: dict[str, object] | None = None,
    ) -> RiskDecisionRecord:
        ...


_risk_record = cast(_RiskDecisionRecordCtor, cast(object, RiskDecisionRecord))


ZERO = Decimal("0")


def _as_decimal(value: object, default: Decimal = ZERO) -> Decimal:
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


def _as_int(value: object, default: int = 0) -> int:
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


class RiskManager:
    def __init__(self, limits: RiskLimits, state_store: object) -> None:
        self.limits: RiskLimits = limits
        self.store: object = state_store
        self._order_timestamps: list[datetime] = []
        self._daily_pnl: Decimal = ZERO
        self._daily_order_count: int = 0
        self._day_start: datetime = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    def _reset_daily_if_needed(self) -> None:
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if today_start > self._day_start:
            self._day_start = today_start
            self._daily_pnl = ZERO
            self._daily_order_count = 0

    async def validate(self, intent: OrderIntent, state: dict[str, object]) -> RiskDecisionRecord:
        now = datetime.now(timezone.utc)
        self._reset_daily_if_needed()
        is_buy = intent.side == OrderSide.BUY

        # 1. Market order policy: must be LIMIT unless explicit exception
        if (
            is_buy
            and intent.order_type == OrderType.MARKET
            and self.limits.default_order_type == "limit"
        ):
            return self._reject(
                intent, now, "Market orders not allowed by default policy. Use limit orders."
            )

        total_balance = _as_decimal(state.get("total_balance"), ZERO)
        order_value = intent.quote_quantity or (intent.quantity or ZERO) * (intent.price or ZERO)
        current_positions = _as_int(state.get("position_count"), 0)

        # 2. Position sizing and exposure checks (buy only)
        if is_buy and total_balance > 0 and order_value > 0:
            position_pct = (order_value / total_balance) * Decimal("100")
            if position_pct > self.limits.max_position_size_pct:
                return self._reject(
                    intent,
                    now,
                    f"Position size {position_pct:.1f}% exceeds limit {self.limits.max_position_size_pct}%",
                )

            current_symbol_value = _as_decimal(state.get("current_symbol_value"), ZERO)
            projected_symbol_pct = ((current_symbol_value + order_value) / total_balance) * Decimal(
                "100"
            )
            if projected_symbol_pct > self.limits.max_single_coin_exposure_pct:
                return self._reject(
                    intent,
                    now,
                    f"Single-coin exposure {projected_symbol_pct:.1f}% exceeds limit {self.limits.max_single_coin_exposure_pct}%",
                )

        # 3. Daily drawdown check (buy only)
        today_pnl = _as_decimal(state.get("today_pnl"), ZERO)
        if is_buy and total_balance > 0:
            drawdown_pct = abs(min(today_pnl, ZERO)) / total_balance * Decimal("100")
            if drawdown_pct >= self.limits.daily_max_drawdown_pct:
                return self._reject(
                    intent,
                    now,
                    f"Daily drawdown {drawdown_pct:.1f}% exceeds limit {self.limits.daily_max_drawdown_pct}%",
                )

        # 4. Rate limit check (buy only)
        cutoff = now - timedelta(seconds=1)
        self._order_timestamps = [t for t in self._order_timestamps if t > cutoff]
        if is_buy and len(self._order_timestamps) >= self.limits.max_orders_per_second:
            return self._reject(
                intent, now, f"Rate limit exceeded: {len(self._order_timestamps)}/s"
            )

        # 5. Daily order count (buy only)
        if is_buy:
            self._daily_order_count += 1
        if is_buy and self._daily_order_count > self.limits.max_orders_per_day:
            return self._reject(
                intent,
                now,
                f"Daily order limit exceeded: {self._daily_order_count}/{self.limits.max_orders_per_day}",
            )

        # 6. Futures check
        if not self.limits.futures_enabled:
            symbol_lower = intent.symbol.lower()
            if any(kw in symbol_lower for kw in ("perp", "future", "swap")):
                return self._reject(intent, now, "Futures/derivatives trading is disabled")

        # 7. Slippage check for limit orders
        if is_buy and intent.order_type == OrderType.LIMIT and intent.price:
            market_price = _as_decimal(state.get("market_price"), ZERO)
            if market_price > 0:
                slippage_bps = abs(intent.price - market_price) / market_price * Decimal("10000")
                if slippage_bps > Decimal(self.limits.max_slippage_bps):
                    return self._reject(
                        intent,
                        now,
                        f"Slippage {slippage_bps:.0f}bps exceeds limit {self.limits.max_slippage_bps}bps",
                    )

        # All checks passed
        if is_buy:
            self._order_timestamps.append(now)
        return _risk_record(
            intent_id=intent.intent_id,
            decision=RiskDecision.APPROVED,
            reason="All risk checks passed",
            timestamp=now,
            metadata={"order_value": str(order_value), "position_count": current_positions},
        )

    def _reject(self, intent: OrderIntent, ts: datetime, reason: str) -> RiskDecisionRecord:
        return _risk_record(
            intent_id=intent.intent_id,
            decision=RiskDecision.REJECTED,
            reason=reason,
            timestamp=ts,
        )

    def update_daily_pnl(self, pnl: Decimal) -> None:
        self._daily_pnl = pnl

    def check_circuit_breaker(self, price_change_pct: Decimal) -> bool:
        """Returns True if circuit breaker should trigger"""
        return abs(price_change_pct) >= self.limits.circuit_breaker_threshold_pct
