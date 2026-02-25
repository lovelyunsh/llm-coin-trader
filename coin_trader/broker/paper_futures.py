"""Paper futures broker - simulates Binance USDT-M futures order execution.

In-memory simulation with:
- Futures fee: 0.04% taker
- Market slippage: 0.1%
- Isolated margin mode by default
- LONG/SHORT position tracking with liquidation price calculation
- Maintenance margin rate: 0.4%
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
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
    PositionSide,
)

_logger = logging.getLogger("broker.paper_futures")

_FEE_RATE = Decimal("0.0004")  # 0.04% taker
_SLIPPAGE_RATE = Decimal("0.001")  # 0.1% market slippage
_MAINTENANCE_MARGIN_RATE = Decimal("0.004")  # 0.4%


class _TickerAdapter(Protocol):
    async def get_ticker(self, symbol: str) -> Mapping[str, object]: ...


@dataclass
class _FuturesPosition:
    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    leverage: int
    margin: Decimal
    liquidation_price: Decimal
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))

    def update_pnl(self, current_price: Decimal) -> None:
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity


def _calc_liquidation_price(
    entry_price: Decimal,
    leverage: int,
    side: PositionSide,
) -> Decimal:
    lev = Decimal(str(leverage))
    if side == PositionSide.LONG:
        return entry_price * (
            Decimal("1") - Decimal("1") / lev + _MAINTENANCE_MARGIN_RATE
        )
    else:
        return entry_price * (
            Decimal("1") + Decimal("1") / lev - _MAINTENANCE_MARGIN_RATE
        )


class PaperFuturesBroker:
    """Simulated Binance USDT-M futures broker for paper trading."""

    def __init__(
        self,
        exchange: ExchangeName,
        exchange_adapter: _TickerAdapter,
        initial_balance_usdt: Decimal = Decimal("10000"),
        default_leverage: int = 1,
        margin_type: str = "isolated",
    ) -> None:
        self.exchange = exchange
        self.adapter = exchange_adapter
        self._balance_usdt: Decimal = initial_balance_usdt
        self._default_leverage: int = default_leverage
        self._margin_type: str = margin_type

        # symbol -> _FuturesPosition (one per symbol per side in hedge mode)
        self._positions: dict[str, _FuturesPosition] = {}
        self._open_orders: dict[str, Order] = {}  # client_order_id -> Order

        self._orders_by_id: dict[str, Order] = {}
        self._orders_by_client_id: dict[str, Order] = {}

        self._fills_by_order_id: dict[str, list[Fill]] = {}
        self._fills_by_client_id: dict[str, list[Fill]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_price(self, symbol: str) -> Decimal:
        ticker = await self.adapter.get_ticker(symbol)
        for key in ("price", "last", "last_price", "markPrice"):
            val = ticker.get(key)
            if val is not None:
                return Decimal(str(val))
        return Decimal("0")

    def _position_key(self, symbol: str, side: PositionSide) -> str:
        return f"{symbol}:{side.value}"

    def _open_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: Decimal,
        price: Decimal,
        leverage: int,
    ) -> None:
        key = self._position_key(symbol, side)
        existing = self._positions.get(key)

        margin = (quantity * price) / Decimal(str(leverage))
        liq_price = _calc_liquidation_price(price, leverage, side)

        if existing is None:
            self._positions[key] = _FuturesPosition(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                leverage=leverage,
                margin=margin,
                liquidation_price=liq_price,
            )
        else:
            # Average into existing position
            total_qty = existing.quantity + quantity
            avg_entry = (
                existing.entry_price * existing.quantity + price * quantity
            ) / total_qty
            new_margin = existing.margin + margin
            new_liq = _calc_liquidation_price(avg_entry, leverage, side)
            self._positions[key] = _FuturesPosition(
                symbol=symbol,
                side=side,
                quantity=total_qty,
                entry_price=avg_entry,
                leverage=leverage,
                margin=new_margin,
                liquidation_price=new_liq,
            )

    def _close_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: Decimal,
        price: Decimal,
    ) -> Decimal:
        """Close (part of) a position; return realized PnL."""
        key = self._position_key(symbol, side)
        pos = self._positions.get(key)
        if pos is None:
            _logger.warning("close_position: no position found key=%s", key)
            return Decimal("0")

        close_qty = min(quantity, pos.quantity)
        if side == PositionSide.LONG:
            realized_pnl = (price - pos.entry_price) * close_qty
        else:
            realized_pnl = (pos.entry_price - price) * close_qty

        margin_released = pos.margin * (close_qty / pos.quantity)

        remaining_qty = pos.quantity - close_qty
        if remaining_qty <= Decimal("0"):
            del self._positions[key]
        else:
            pos.quantity = remaining_qty
            pos.margin = pos.margin - margin_released

        return realized_pnl

    def _resolve_intent_side(self, intent: OrderIntent) -> tuple[PositionSide, bool]:
        """Return (position_side, is_close).

        Logic:
        - reduce_only=True → closing the opposite side position
        - position_side=SHORT → short (opening/adding)
        - position_side=LONG or None with side=BUY → long (opening/adding)
        - side=SELL without reduce_only → open short
        """
        if intent.reduce_only:
            # Closing: the position being reduced is opposite to order side
            if intent.side == OrderSide.BUY:
                return PositionSide.SHORT, True
            else:
                return PositionSide.LONG, True

        if intent.position_side == PositionSide.SHORT:
            return PositionSide.SHORT, False

        if intent.side == OrderSide.BUY:
            return PositionSide.LONG, False

        # SELL without reduce_only → open/add short
        return PositionSide.SHORT, False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def place(self, intent: OrderIntent, client_order_id: str) -> Order:
        now = datetime.now(timezone.utc)
        current_price = await self._get_price(intent.symbol)

        quantity = intent.quantity
        if quantity is None and intent.quote_quantity is not None:
            ref_price = (
                intent.price if intent.price and intent.price > 0 else current_price
            )
            if ref_price <= 0:
                raise ValueError("Invalid reference price for sizing")
            quantity = intent.quote_quantity / ref_price

        if quantity is None or quantity <= 0:
            raise ValueError("Order quantity must be > 0")

        # Determine fill price with slippage for market orders
        if intent.order_type == OrderType.MARKET:
            if intent.side == OrderSide.BUY:
                fill_price = current_price * (Decimal("1") + _SLIPPAGE_RATE)
            else:
                fill_price = current_price * (Decimal("1") - _SLIPPAGE_RATE)
        elif intent.order_type == OrderType.LIMIT:
            if intent.price is None or intent.price <= 0:
                order = self._make_order(
                    client_order_id,
                    intent,
                    quantity,
                    intent.price,
                    now,
                    status=OrderStatus.REJECTED,
                    metadata={"error": "invalid_limit_price"},
                )
                return order
            fill_price = intent.price
        elif intent.order_type == OrderType.STOP_MARKET:
            fill_price = (
                intent.stop_price
                if intent.stop_price and intent.stop_price > 0
                else current_price
            )
        else:
            fill_price = current_price

        if fill_price <= 0:
            order = self._make_order(
                client_order_id,
                intent,
                quantity,
                intent.price,
                now,
                status=OrderStatus.REJECTED,
                metadata={"error": "zero_fill_price"},
            )
            return order

        pos_side, is_close = self._resolve_intent_side(intent)
        leverage = self._default_leverage

        fee = quantity * fill_price * _FEE_RATE

        if is_close:
            realized_pnl = self._close_position(
                intent.symbol, pos_side, quantity, fill_price
            )
            proceeds = quantity * fill_price - fee + realized_pnl
            self._balance_usdt += proceeds
            _logger.debug(
                "close_position symbol=%s side=%s qty=%s pnl=%s",
                intent.symbol,
                pos_side.value,
                quantity,
                realized_pnl,
            )
        else:
            required_margin = (quantity * fill_price) / Decimal(str(leverage))
            total_cost = required_margin + fee
            if self._balance_usdt < total_cost:
                order = self._make_order(
                    client_order_id,
                    intent,
                    quantity,
                    fill_price,
                    now,
                    status=OrderStatus.REJECTED,
                    metadata={
                        "error": "insufficient_margin",
                        "required": str(total_cost),
                    },
                )
                return order

            self._balance_usdt -= total_cost
            self._open_position(intent.symbol, pos_side, quantity, fill_price, leverage)
            _logger.debug(
                "open_position symbol=%s side=%s qty=%s price=%s margin=%s",
                intent.symbol,
                pos_side.value,
                quantity,
                fill_price,
                required_margin,
            )

        order = self._make_order(
            client_order_id,
            intent,
            quantity,
            fill_price,
            now,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            avg_fill_price=fill_price,
        )

        fill = Fill(
            fill_id=f"fill_{uuid4().hex[:12]}",
            order_id=str(order.order_id or ""),
            client_order_id=client_order_id,
            exchange=self.exchange,
            symbol=intent.symbol,
            side=intent.side,
            quantity=quantity,
            price=fill_price,
            fee=fee,
            fee_currency="USDT",
            timestamp=now,
            metadata={"position_side": pos_side.value, "is_close": str(is_close)},
        )
        self._fills_by_client_id.setdefault(client_order_id, []).append(fill)
        if order.order_id is not None:
            self._fills_by_order_id.setdefault(str(order.order_id), []).append(fill)

        return order

    def _make_order(
        self,
        client_order_id: str,
        intent: OrderIntent,
        quantity: Decimal,
        price: Decimal | None,
        now: datetime,
        *,
        status: OrderStatus = OrderStatus.PENDING,
        filled_quantity: Decimal = Decimal("0"),
        avg_fill_price: Decimal | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Order:
        order = Order(
            order_id=f"pfut_{uuid4().hex[:12]}",
            client_order_id=client_order_id,
            intent_id=intent.intent_id,
            exchange=self.exchange,
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            quantity=quantity,
            price=price,
            status=status,
            filled_quantity=filled_quantity,
            average_fill_price=avg_fill_price,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        self._orders_by_client_id[client_order_id] = order
        if order.order_id is not None:
            self._orders_by_id[str(order.order_id)] = order
        return order

    async def cancel(self, order_id: str) -> None:
        for cid, order in list(self._open_orders.items()):
            if order.order_id == order_id or order.client_order_id == order_id:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now(timezone.utc)
                del self._open_orders[cid]
                return
        raise ValueError(f"Order not found: {order_id}")

    async def fetch_open_orders(self) -> list[Order]:
        return list(self._open_orders.values())

    async def fetch_balances(self) -> BalanceSnapshot:
        now = datetime.now(timezone.utc)
        total = self._balance_usdt
        # Add unrealized PnL from open positions to total equity
        for pos in self._positions.values():
            total += pos.unrealized_pnl

        return BalanceSnapshot(
            exchange=self.exchange,
            timestamp=now,
            balances={"USDT": self._balance_usdt},
            total_value_quote=total,
            quote_currency="USDT",
        )

    async def fetch_positions(self) -> list[Position]:
        now = datetime.now(timezone.utc)
        result: list[Position] = []
        for pos in self._positions.values():
            try:
                current_price = await self._get_price(pos.symbol)
                if current_price > 0:
                    pos.update_pnl(current_price)
            except Exception:
                current_price = None

            result.append(
                Position(
                    exchange=self.exchange,
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    average_entry_price=pos.entry_price,
                    current_price=(
                        current_price if current_price and current_price > 0 else None
                    ),
                    timestamp=now,
                    side=pos.side,
                    leverage=pos.leverage,
                    liquidation_price=pos.liquidation_price,
                    margin=pos.margin,
                    margin_type=self._margin_type,
                )
            )
        return result

    async def fetch_fills(self, order_id: str) -> list[Fill]:
        if order_id in self._fills_by_order_id:
            return list(self._fills_by_order_id[order_id])
        if order_id in self._fills_by_client_id:
            return list(self._fills_by_client_id[order_id])
        return []
