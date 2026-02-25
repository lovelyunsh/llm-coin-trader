"""Paper trading broker - simulates order execution.

In-memory simulation with:
- Upbit-like fees (0.05%)
- Market slippage (0.1%); limit orders have zero slippage
- Limit order fill only when current price crosses the limit
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal
from typing import Protocol
from uuid import uuid4

from coin_trader.core.models import (
    BalanceSnapshot,
    ExchangeName,
    Fill,
    Order,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


class _TickerAdapter(Protocol):
    async def get_ticker(self, symbol: str) -> Mapping[str, object]: ...


DEFAULT_INITIAL_BALANCE = Decimal("1000000")


class PaperBroker:
    def __init__(
        self,
        exchange: ExchangeName,
        exchange_adapter: _TickerAdapter,
        initial_balance_krw: Decimal = DEFAULT_INITIAL_BALANCE,
        quote_currency: str = "KRW",
    ) -> None:
        self.exchange: ExchangeName = exchange
        self.adapter: _TickerAdapter = exchange_adapter
        self._quote_currency: str = quote_currency

        self.balances: dict[str, Decimal] = {quote_currency: initial_balance_krw}
        self.open_orders: dict[str, Order] = {}  # client_order_id -> Order

        self._orders_by_id: dict[str, Order] = {}
        self._orders_by_client_id: dict[str, Order] = {}

        self.all_fills: dict[str, list[Fill]] = {}  # client_order_id -> [Fill]
        self._fills_by_order_id: dict[str, list[Fill]] = {}

        self._position_cost_krw: dict[str, Decimal] = (
            {}
        )  # base currency -> KRW cost basis

        self.fee_rate: Decimal = Decimal("0.0005")  # 0.05%
        self.slippage_rate: Decimal = Decimal("0.001")  # 0.1% for market orders

    async def place(self, intent: OrderIntent, client_order_id: str) -> Order:
        now = datetime.now(timezone.utc)

        current_price = await self._get_price(intent.symbol)

        limit_price_invalid = intent.order_type == OrderType.LIMIT and (
            intent.price is None or intent.price <= 0
        )
        effective_price = intent.price
        if limit_price_invalid:
            effective_price = current_price if current_price > 0 else Decimal("1")

        quantity = intent.quantity
        if quantity is None and intent.quote_quantity is not None:
            ref_price = (
                effective_price if effective_price is not None else current_price
            )
            if ref_price <= 0:
                raise ValueError("Invalid reference price for sizing")
            quantity = intent.quote_quantity / ref_price

        if quantity is None or quantity <= 0:
            raise ValueError("Order quantity must be > 0")

        order = Order(
            order_id=f"paper_{uuid4().hex[:12]}",
            client_order_id=client_order_id,
            intent_id=intent.intent_id,
            exchange=self.exchange,
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            quantity=quantity,
            price=effective_price,
            status=OrderStatus.REJECTED if limit_price_invalid else OrderStatus.PENDING,
            filled_quantity=Decimal("0"),
            created_at=now,
            updated_at=now,
            metadata={"error": "invalid_limit_price"} if limit_price_invalid else {},
        )
        if order.order_id is not None:
            self._orders_by_id[str(order.order_id)] = order
        self._orders_by_client_id[client_order_id] = order

        if limit_price_invalid:
            return order

        if intent.order_type == OrderType.MARKET:
            if intent.side == OrderSide.BUY:
                fill_price = current_price * (Decimal("1") + self.slippage_rate)
            else:
                fill_price = current_price * (Decimal("1") - self.slippage_rate)
            await self._execute_fill(order, quantity, fill_price, now)
            return order

        if intent.order_type == OrderType.LIMIT:
            if intent.price is None or intent.price <= 0:
                order.status = OrderStatus.REJECTED
                order.updated_at = now
                return order

            can_fill = False
            if intent.side == OrderSide.BUY:
                can_fill = current_price <= intent.price
            elif intent.side == OrderSide.SELL:
                can_fill = current_price >= intent.price

            if can_fill:
                await self._execute_fill(order, quantity, intent.price, now)
            else:
                order.status = OrderStatus.SUBMITTED
                order.updated_at = now
                self.open_orders[client_order_id] = order

            return order

        order.status = OrderStatus.REJECTED
        order.updated_at = now
        return order

    async def _execute_fill(
        self, order: Order, qty: Decimal, price: Decimal, ts: datetime
    ) -> None:
        base, quote = order.symbol.split("/")
        fee = qty * price * self.fee_rate

        if order.side == OrderSide.BUY:
            cost = qty * price + fee
            available = self.balances.get(quote, Decimal("0"))
            if available < cost:
                order.status = OrderStatus.REJECTED
                order.updated_at = ts
                return

            self.balances[quote] = available - cost
            self.balances[base] = self.balances.get(base, Decimal("0")) + qty

            if quote == self._quote_currency:
                self._position_cost_krw[base] = (
                    self._position_cost_krw.get(base, Decimal("0")) + cost
                )

        else:
            available = self.balances.get(base, Decimal("0"))
            if available < qty:
                order.status = OrderStatus.REJECTED
                order.updated_at = ts
                return

            self.balances[base] = available - qty
            proceeds = qty * price - fee
            self.balances[quote] = self.balances.get(quote, Decimal("0")) + proceeds

            if quote == self._quote_currency:
                held_qty = available
                held_cost = self._position_cost_krw.get(base, Decimal("0"))
                if held_qty > 0 and held_cost > 0:
                    reduction = held_cost * (qty / held_qty)
                    new_cost = held_cost - reduction
                    self._position_cost_krw[base] = (
                        new_cost if new_cost > 0 else Decimal("0")
                    )
                if self.balances.get(base, Decimal("0")) <= 0:
                    _ = self._position_cost_krw.pop(base, None)

        order.status = OrderStatus.FILLED
        order.filled_quantity = qty
        order.average_fill_price = price
        order.updated_at = ts

        fill = Fill(
            fill_id=f"fill_{uuid4().hex[:12]}",
            order_id=str(order.order_id or ""),
            client_order_id=order.client_order_id,
            exchange=self.exchange,
            symbol=order.symbol,
            side=order.side,
            quantity=qty,
            price=price,
            fee=fee,
            fee_currency=quote,
            timestamp=ts,
        )
        self.all_fills.setdefault(order.client_order_id, []).append(fill)
        if order.order_id is not None:
            self._fills_by_order_id.setdefault(str(order.order_id), []).append(fill)

        _ = self.open_orders.pop(order.client_order_id, None)

    async def _reconcile_open_orders(self) -> None:
        if not self.open_orders:
            return

        now = datetime.now(timezone.utc)
        for client_order_id, order in list(self.open_orders.items()):
            if order.status not in {OrderStatus.SUBMITTED, OrderStatus.PENDING}:
                _ = self.open_orders.pop(client_order_id, None)
                continue

            if order.price is None or order.price <= 0:
                continue

            try:
                current_price = await self._get_price(order.symbol)
            except Exception:
                continue

            can_fill = False
            if order.side == OrderSide.BUY:
                can_fill = current_price <= order.price
            elif order.side == OrderSide.SELL:
                can_fill = current_price >= order.price

            if can_fill:
                await self._execute_fill(
                    order, order.remaining_quantity, order.price, now
                )

    async def _get_price(self, symbol: str) -> Decimal:
        ticker = await self.adapter.get_ticker(symbol)
        for key in ("price", "last", "last_price"):
            if key in ticker and ticker.get(key) is not None:
                return Decimal(str(ticker.get(key)))
        return Decimal("0")

    async def cancel(self, order_id: str) -> None:
        await self._reconcile_open_orders()
        for cid, order in list(self.open_orders.items()):
            if order.order_id == order_id or order.client_order_id == order_id:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now(timezone.utc)
                del self.open_orders[cid]
                return
        raise ValueError(f"Order not found: {order_id}")

    async def fetch_open_orders(self) -> list[Order]:
        await self._reconcile_open_orders()
        return list(self.open_orders.values())

    async def fetch_balances(self) -> BalanceSnapshot:
        await self._reconcile_open_orders()

        now = datetime.now(timezone.utc)
        balances = {k: v for k, v in self.balances.items() if v > 0}
        qc = self._quote_currency

        total_value_quote = balances.get(qc, Decimal("0"))
        for currency, qty in balances.items():
            if currency == qc or qty <= 0:
                continue
            symbol = f"{currency}/{qc}"
            try:
                price = await self._get_price(symbol)
                if price > 0:
                    total_value_quote += qty * price
            except Exception:
                continue

        return BalanceSnapshot(
            exchange=self.exchange,
            timestamp=now,
            balances=balances,
            total_value_krw=total_value_quote if qc == "KRW" else None,
            total_value_quote=total_value_quote,
            quote_currency=qc,
        )

    async def fetch_positions(self) -> list[Position]:
        await self._reconcile_open_orders()

        positions: list[Position] = []
        now = datetime.now(timezone.utc)
        qc = self._quote_currency
        for currency, qty in self.balances.items():
            if currency == qc or qty <= 0:
                continue

            symbol = f"{currency}/{qc}"
            try:
                price = await self._get_price(symbol)
            except Exception:
                continue

            cost = self._position_cost_krw.get(currency, Decimal("0"))
            avg_entry = (cost / qty) if qty > 0 else Decimal("0")

            positions.append(
                Position(
                    exchange=self.exchange,
                    symbol=symbol,
                    quantity=qty,
                    average_entry_price=avg_entry,
                    current_price=price,
                    timestamp=now,
                )
            )

        return positions

    async def fetch_fills(self, order_id: str) -> list[Fill]:
        await self._reconcile_open_orders()

        if order_id in self._fills_by_order_id:
            return list(self._fills_by_order_id[order_id])
        if order_id in self.all_fills:
            return list(self.all_fills[order_id])
        for client_id, fills in self.all_fills.items():
            if not fills:
                continue
            first = fills[0]
            if (
                first.order_id == order_id
                or first.client_order_id == order_id
                or client_id == order_id
            ):
                return list(fills)
        return []
