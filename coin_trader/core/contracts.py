"""Abstract contracts for core trading components.

All interfaces in this module depend on domain models defined in
`coin_trader.core.models`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from .models import (
    BalanceSnapshot,
    Fill,
    MarketData,
    Order,
    OrderIntent,
    Position,
    RiskDecisionRecord,
    SafetyEvent,
    Signal,
)  # noqa: F401 — PositionSide used in type hints below


class IStrategy(ABC):
    @abstractmethod
    async def on_tick(self, md: MarketData) -> list[Signal]:
        raise NotImplementedError


class ISignalToIntent(ABC):
    @abstractmethod
    def to_intents(self, signals: list[Signal], md: MarketData) -> list[OrderIntent]:
        raise NotImplementedError


class IRiskManager(ABC):
    @abstractmethod
    async def validate(
        self, intent: OrderIntent, state: dict[str, object]
    ) -> RiskDecisionRecord:
        raise NotImplementedError


class IBroker(ABC):
    @abstractmethod
    async def place(self, intent: OrderIntent, client_order_id: str) -> Order:
        raise NotImplementedError

    @abstractmethod
    async def cancel(self, order_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def fetch_open_orders(self) -> list[Order]:
        raise NotImplementedError

    @abstractmethod
    async def fetch_balances(self) -> BalanceSnapshot:
        raise NotImplementedError

    @abstractmethod
    async def fetch_positions(self) -> list[Position]:
        raise NotImplementedError

    @abstractmethod
    async def fetch_fills(self, order_id: str) -> list[Fill]:
        raise NotImplementedError


class IExchangeAdapter(ABC):
    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal | None,
    ) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_open_orders(
        self, symbol: str | None = None
    ) -> list[dict[str, object]]:
        raise NotImplementedError

    @abstractmethod
    async def get_balances(self) -> dict[str, Decimal]:
        raise NotImplementedError

    @abstractmethod
    async def get_ticker(self, symbol: str) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    async def get_tickers(self, symbols: list[str]) -> dict[str, dict[str, object]]:
        raise NotImplementedError

    @abstractmethod
    async def get_orderbook(self, symbol: str) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    async def get_candles(
        self, symbol: str, interval: str, count: int
    ) -> list[dict[str, object]]:
        raise NotImplementedError

    @abstractmethod
    def normalize_symbol(self, symbol: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def denormalize_symbol(self, symbol: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def get_tradeable_markets(self) -> list[str]:
        raise NotImplementedError

    # Futures-only methods (default NotImplementedError for spot adapters)
    async def set_leverage(self, symbol: str, leverage: int) -> None:
        raise NotImplementedError("set_leverage not supported by this adapter")

    async def set_margin_mode(self, symbol: str, mode: str) -> None:
        raise NotImplementedError("set_margin_mode not supported by this adapter")

    async def get_funding_rate(self, symbol: str) -> dict[str, object]:
        raise NotImplementedError("get_funding_rate not supported by this adapter")

    async def get_positions(self) -> list[dict[str, object]]:
        raise NotImplementedError("get_positions not supported by this adapter")

    async def get_mark_price(self, symbol: str) -> Decimal:
        raise NotImplementedError("get_mark_price not supported by this adapter")


class IStateStore(ABC):
    @abstractmethod
    def save_order(self, order: Order) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_fill(self, fill: Fill) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_decision(self, decision: RiskDecisionRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_safety_event(self, event: SafetyEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_open_orders(self) -> list[Order]:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> list[Position]:
        raise NotImplementedError


class INotifier(ABC):
    @abstractmethod
    async def send_alert(self, title: str, message: str, severity: str) -> None:
        raise NotImplementedError
