"""Domain enums and models for the trading system.

This module is the single source of truth for domain types used across the
system. All monetary values are represented with Decimal.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, TypeVar
from uuid import UUID, uuid4

T = TypeVar("T")
F = TypeVar("F")

if TYPE_CHECKING:
    # Minimal stubs so the module type-checks without importing pydantic.
    class BaseModel:  # noqa: D101
        def __init__(self, **data: object) -> None:
            ...

        @classmethod
        def model_validate(cls: type["BaseModel"], _data: object) -> "BaseModel":
            ...

    def Field(
        _default: object = ..., *, _expected_type: type[T] | None = None, **_kwargs: object
    ) -> T:  # noqa: D103
        raise NotImplementedError

    def computed_field(func: T) -> T:  # noqa: D103
        return func

    def model_validator(*_args: object, **_kwargs: object) -> Callable[[F], F]:  # noqa: D103
        def _decorator(func: F) -> F:
            return func

        return _decorator

else:
    from pydantic import BaseModel, Field, computed_field, model_validator


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class ExchangeName(str, Enum):
    UPBIT = "upbit"
    BINANCE = "binance"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


class SignalType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class RiskDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class SafetyEventType(str, Enum):
    SOFT_STOP = "soft_stop"
    HARD_STOP = "hard_stop"
    CIRCUIT_BREAKER = "circuit_breaker"
    BALANCE_MISMATCH = "balance_mismatch"
    API_FAILURE = "api_failure"
    PRICE_ANOMALY = "price_anomaly"


class MarketData(BaseModel):
    exchange: ExchangeName = Field(...)
    symbol: str = Field(...)
    timestamp: datetime = Field(...)

    open: Decimal = Field(...)
    high: Decimal = Field(...)
    low: Decimal = Field(...)
    close: Decimal = Field(...)
    volume: Decimal = Field(...)

    bid: Decimal | None = None
    ask: Decimal | None = None
    spread: Decimal | None = None

    @model_validator(mode="after")
    def _auto_calc_spread(self) -> "MarketData":
        if self.bid is not None and self.ask is not None:
            if self.ask < self.bid:
                raise ValueError("ask must be >= bid")
            if self.spread is None:
                self.spread = self.ask - self.bid
        return self


class Signal(BaseModel):
    signal_id: UUID = Field(default_factory=uuid4)
    strategy_name: str = Field(...)
    symbol: str = Field(...)
    signal_type: SignalType = Field(...)
    timestamp: datetime = Field(...)
    confidence: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    metadata: dict[str, object] = Field(default_factory=dict)


class OrderIntent(BaseModel):
    intent_id: UUID = Field(default_factory=uuid4)
    signal_id: UUID = Field(...)
    exchange: ExchangeName = Field(...)
    symbol: str = Field(...)
    side: OrderSide = Field(...)
    order_type: OrderType = Field(...)
    quantity: Decimal | None = None
    quote_quantity: Decimal | None = None
    price: Decimal | None = None
    reason: str = Field(...)
    timestamp: datetime = Field(...)

    @model_validator(mode="after")
    def _validate_size(self) -> "OrderIntent":
        if self.quantity is None and self.quote_quantity is None:
            raise ValueError("quantity or quote_quantity must be provided")
        return self


class Order(BaseModel):
    order_id: str | None = None
    client_order_id: str = Field(...)
    intent_id: UUID = Field(...)

    exchange: ExchangeName = Field(...)
    symbol: str = Field(...)
    side: OrderSide = Field(...)
    order_type: OrderType = Field(...)

    quantity: Decimal = Field(...)
    price: Decimal | None = None

    status: OrderStatus = Field(default=OrderStatus.PENDING)
    filled_quantity: Decimal = Field(default=Decimal("0"))
    average_fill_price: Decimal | None = None

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    metadata: dict[str, object] = Field(default_factory=dict)

    @computed_field
    @property
    def remaining_quantity(self) -> Decimal:
        return self.quantity - self.filled_quantity

    @model_validator(mode="after")
    def _validate_invariants(self) -> "Order":
        if self.quantity <= Decimal("0"):
            raise ValueError("quantity must be > 0")

        if self.filled_quantity < Decimal("0"):
            raise ValueError("filled_quantity must be >= 0")
        if self.filled_quantity > self.quantity:
            raise ValueError("filled_quantity must be <= quantity")

        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("price is required for LIMIT orders")
        if self.created_at > self.updated_at:
            raise ValueError("updated_at must be >= created_at")

        return self


class Fill(BaseModel):
    fill_id: str = Field(...)
    order_id: str = Field(...)
    client_order_id: str = Field(...)
    exchange: ExchangeName = Field(...)
    symbol: str = Field(...)
    side: OrderSide = Field(...)
    quantity: Decimal = Field(...)
    price: Decimal = Field(...)
    fee: Decimal = Field(...)
    fee_currency: str = Field(...)
    timestamp: datetime = Field(...)
    metadata: dict[str, object] = Field(default_factory=dict)


class Position(BaseModel):
    exchange: ExchangeName = Field(...)
    symbol: str = Field(...)
    quantity: Decimal = Field(...)
    average_entry_price: Decimal = Field(...)
    current_price: Decimal | None = None
    timestamp: datetime = Field(...)

    @computed_field
    @property
    def unrealized_pnl(self) -> Decimal | None:
        if self.current_price is None:
            return None
        return (self.current_price - self.average_entry_price) * self.quantity

    @computed_field
    @property
    def unrealized_pnl_pct(self) -> Decimal | None:
        pnl = self.unrealized_pnl
        if pnl is None:
            return None

        notional = abs(self.quantity) * self.average_entry_price
        if notional == Decimal("0"):
            return None
        return (pnl / notional) * Decimal("100")


class BalanceSnapshot(BaseModel):
    snapshot_id: UUID = Field(default_factory=uuid4)
    exchange: ExchangeName = Field(...)
    timestamp: datetime = Field(...)
    balances: dict[str, Decimal] = Field(...)
    total_value_krw: Decimal | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class RiskDecisionRecord(BaseModel):
    decision_id: UUID = Field(default_factory=uuid4)
    intent_id: UUID = Field(...)
    decision: RiskDecision = Field(...)
    reason: str = Field(...)
    timestamp: datetime = Field(...)
    metadata: dict[str, object] = Field(default_factory=dict)


class SafetyEvent(BaseModel):
    event_id: UUID = Field(default_factory=uuid4)
    event_type: SafetyEventType = Field(...)
    timestamp: datetime = Field(...)
    description: str = Field(...)
    severity: str = Field(...)
    triggered_by: str = Field(...)
    metadata: dict[str, object] = Field(default_factory=dict)


class DecisionLog(BaseModel):
    log_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(...)
    component: str = Field(...)
    action: str = Field(...)
    input_data: dict[str, object] = Field(...)
    output_data: dict[str, object] = Field(...)
    metadata: dict[str, object] = Field(default_factory=dict)
