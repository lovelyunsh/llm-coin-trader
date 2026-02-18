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
)


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
    async def validate(self, intent: OrderIntent, state: dict[str, object]) -> RiskDecisionRecord:
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
    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, object]]:
        raise NotImplementedError

    @abstractmethod
    async def get_balances(self) -> dict[str, Decimal]:
        raise NotImplementedError

    @abstractmethod
    async def get_ticker(self, symbol: str) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    async def get_orderbook(self, symbol: str) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    async def get_candles(self, symbol: str, interval: str, count: int) -> list[dict[str, object]]:
        raise NotImplementedError

    @abstractmethod
    def normalize_symbol(self, symbol: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def denormalize_symbol(self, symbol: str) -> str:
        raise NotImplementedError


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
