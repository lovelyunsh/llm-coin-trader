"""Live futures broker - wraps BinanceFuturesAdapter for real Binance USDT-M futures execution."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal
from typing import Protocol
from uuid import UUID, uuid4

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

_logger = logging.getLogger("broker.live_futures")

_CACHE_TTL = 5.0


class BinanceFuturesAdapter(Protocol):
    """Protocol for Binance USDT-M futures adapter methods used by this broker."""

    async def place_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal | None,
        reduce_only: bool,
        position_side: str | None,
        stop_price: Decimal | None,
        time_in_force: str | None,
    ) -> Mapping[str, object]: ...

    async def cancel_order(self, order_id: str, symbol: str) -> object: ...

    async def get_open_orders(self, symbol: str | None) -> object: ...

    async def get_order(self, order_id: str, symbol: str) -> Mapping[str, object]: ...

    async def get_balances(self) -> object: ...

    async def get_positions(self) -> object: ...

    def normalize_symbol(self, symbol: str) -> str: ...


# Binance futures order status mapping
_BINANCE_STATUS_MAP: dict[str, OrderStatus] = {
    "new": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIAL,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
    "expired": OrderStatus.CANCELLED,
    "pending_cancel": OrderStatus.SUBMITTED,
}

# Binance futures order type mapping
_ORDER_TYPE_MAP: dict[OrderType, str] = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.STOP_MARKET: "STOP_MARKET",
    OrderType.TAKE_PROFIT_MARKET: "TAKE_PROFIT_MARKET",
}

# Binance position side mapping
_POSITION_SIDE_MAP: dict[str, PositionSide] = {
    "long": PositionSide.LONG,
    "short": PositionSide.SHORT,
    "both": PositionSide.BOTH,
}


class LiveFuturesBroker:
    """Real Binance USDT-M futures broker."""

    def __init__(
        self,
        exchange: ExchangeName,
        exchange_adapter: BinanceFuturesAdapter,
        trading_symbols: list[str] | None = None,
        quote_currency: str = "USDT",
    ) -> None:
        self.exchange = exchange
        self.adapter = exchange_adapter
        self._quote_currency = quote_currency
        self._trading_symbols: set[str] = set(trading_symbols or [])

        self._snapshot_cache: tuple[float, BalanceSnapshot | None, list[Position]] = (
            0.0,
            None,
            [],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        self._snapshot_cache = (0.0, None, [])

    def _map_order_type(self, order_type: OrderType) -> str:
        return _ORDER_TYPE_MAP.get(order_type, "MARKET")

    def _map_position_side(self, ps: PositionSide | None) -> str | None:
        if ps is None:
            return None
        return ps.value.upper()

    def _parse_order_status(self, raw_status: str) -> OrderStatus:
        return _BINANCE_STATUS_MAP.get(raw_status.lower(), OrderStatus.SUBMITTED)

    def _parse_order_from_raw(
        self,
        raw: Mapping[str, object],
        intent_id: object = None,
        now: datetime | None = None,
    ) -> Order | None:
        if now is None:
            now = datetime.now(timezone.utc)

        symbol_raw = str(raw.get("symbol", ""))
        symbol = self.adapter.normalize_symbol(symbol_raw)

        side_raw = str(raw.get("side", "")).lower()
        side = OrderSide.BUY if side_raw == "buy" else OrderSide.SELL

        ord_type_raw = str(raw.get("type", "MARKET")).upper()
        type_reverse: dict[str, OrderType] = {v: k for k, v in _ORDER_TYPE_MAP.items()}
        order_type = type_reverse.get(ord_type_raw, OrderType.MARKET)

        qty_raw = raw.get("origQty", raw.get("qty", "0"))
        try:
            quantity = Decimal(str(qty_raw))
        except Exception:
            return None
        if quantity <= 0:
            return None

        price_raw = raw.get("price")
        price: Decimal | None = None
        if price_raw is not None:
            try:
                p = Decimal(str(price_raw))
                price = p if p > 0 else None
            except Exception:
                pass

        executed_raw = raw.get("executedQty", "0")
        try:
            executed = Decimal(str(executed_raw))
        except Exception:
            executed = Decimal("0")
        if executed < 0:
            executed = Decimal("0")

        avg_price_raw = raw.get("avgPrice")
        avg_price: Decimal | None = None
        if avg_price_raw is not None:
            try:
                ap = Decimal(str(avg_price_raw))
                avg_price = ap if ap > 0 else None
            except Exception:
                pass

        status = self._parse_order_status(str(raw.get("status", "new")))

        order_id = raw.get("orderId", raw.get("id"))
        client_order_id_raw = raw.get("clientOrderId", raw.get("newClientOrderId"))

        time_raw = raw.get("time", raw.get("updateTime"))
        created_at = now
        if time_raw is not None:
            try:
                ts_ms = int(str(time_raw))
                created_at = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            except Exception:
                pass

        resolved_intent_id: object
        if intent_id is None:
            resolved_intent_id = uuid4()
        elif isinstance(intent_id, UUID):
            resolved_intent_id = intent_id
        else:
            try:
                resolved_intent_id = UUID(str(intent_id))
            except Exception:
                resolved_intent_id = uuid4()

        return Order(
            order_id=str(order_id) if order_id is not None else None,
            client_order_id=(
                str(client_order_id_raw) if client_order_id_raw else str(uuid4())
            ),
            intent_id=resolved_intent_id,  # type: ignore[arg-type]
            exchange=self.exchange,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=status,
            filled_quantity=executed,
            average_fill_price=avg_price,
            created_at=created_at,
            updated_at=now,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def place(self, intent: OrderIntent, client_order_id: str) -> Order:
        now = datetime.now(timezone.utc)

        quantity = intent.quantity
        if quantity is None and intent.quote_quantity is not None:
            # Futures: size in base asset only; use price for conversion
            ref_price = intent.price
            if ref_price is None or ref_price <= 0:
                raise ValueError(
                    "price must be provided to convert quote_quantity for futures"
                )
            quantity = intent.quote_quantity / ref_price

        if quantity is None or quantity <= 0:
            raise ValueError("Order quantity must be > 0")

        binance_order_type = self._map_order_type(intent.order_type)
        binance_side = intent.side.value.upper()
        binance_pos_side = self._map_position_side(intent.position_side)

        # LIMIT orders require GTC time-in-force on Binance futures
        time_in_force: str | None = None
        if intent.order_type == OrderType.LIMIT:
            time_in_force = "GTC"

        try:
            result = await self.adapter.place_order(
                symbol=intent.symbol,
                side=binance_side,
                order_type=binance_order_type,
                quantity=quantity,
                price=intent.price if intent.order_type == OrderType.LIMIT else None,
                reduce_only=intent.reduce_only,
                position_side=binance_pos_side,
                stop_price=intent.stop_price,
                time_in_force=time_in_force,
            )

            order_id = result.get("orderId", result.get("id"))
            status_str = str(result.get("status", "new")).lower()
            status = self._parse_order_status(status_str)

            executed_raw = result.get("executedQty", "0")
            executed = Decimal(str(executed_raw)) if executed_raw else Decimal("0")

            avg_price_raw = result.get("avgPrice")
            avg_price: Decimal | None = None
            if avg_price_raw is not None:
                try:
                    ap = Decimal(str(avg_price_raw))
                    avg_price = ap if ap > 0 else None
                except Exception:
                    pass

            self.invalidate_cache()

            return Order(
                order_id=str(order_id) if order_id is not None else None,
                client_order_id=client_order_id,
                intent_id=intent.intent_id,
                exchange=self.exchange,
                symbol=intent.symbol,
                side=intent.side,
                order_type=intent.order_type,
                quantity=quantity,
                price=intent.price,
                status=status,
                filled_quantity=executed,
                average_fill_price=avg_price,
                created_at=now,
                updated_at=now,
                metadata={
                    "reduce_only": str(intent.reduce_only),
                    "position_side": binance_pos_side or "",
                },
            )

        except Exception as e:
            _logger.error(
                "futures_place_failed symbol=%s side=%s error=%s",
                intent.symbol,
                intent.side.value,
                str(e),
            )
            return Order(
                order_id=None,
                client_order_id=client_order_id,
                intent_id=intent.intent_id,
                exchange=self.exchange,
                symbol=intent.symbol,
                side=intent.side,
                order_type=intent.order_type,
                quantity=quantity,
                price=intent.price,
                status=OrderStatus.FAILED,
                filled_quantity=Decimal("0"),
                created_at=now,
                updated_at=now,
                metadata={"error": str(e)},
            )

    async def cancel(self, order_id: str) -> None:
        # symbol is required by Binance; best-effort with empty string if unknown
        await self.adapter.cancel_order(order_id, "")

    async def fetch_open_orders(self) -> list[Order]:
        now = datetime.now(timezone.utc)
        raw_obj = await self.adapter.get_open_orders(None)
        if not isinstance(raw_obj, (list, tuple)):
            return []

        orders: list[Order] = []
        for raw in raw_obj:
            if not isinstance(raw, dict):
                continue
            order = self._parse_order_from_raw(raw, now=now)
            if order is not None:
                orders.append(order)
        return orders

    async def _build_snapshot(self) -> tuple[BalanceSnapshot, list[Position]]:
        now_mono = time.monotonic()
        cached_ts, cached_snap, cached_pos = self._snapshot_cache
        if now_mono - cached_ts < _CACHE_TTL and cached_snap is not None:
            return cached_snap, cached_pos

        now = datetime.now(timezone.utc)
        balances: dict[str, Decimal] = {}
        total_usdt = Decimal("0")

        try:
            raw_balances = await self.adapter.get_balances()
            if isinstance(raw_balances, (list, tuple)):
                for item in raw_balances:
                    if not isinstance(item, dict):
                        continue
                    asset = str(item.get("asset", ""))
                    wb_raw = item.get("walletBalance", item.get("balance", "0"))
                    try:
                        wb = Decimal(str(wb_raw))
                    except Exception:
                        wb = Decimal("0")
                    if asset and wb != 0:
                        balances[asset] = wb
                    if asset.upper() == "USDT":
                        total_usdt = wb
            elif isinstance(raw_balances, dict):
                for asset, val in raw_balances.items():
                    try:
                        amount = Decimal(str(val))
                    except Exception:
                        amount = Decimal("0")
                    if amount != 0:
                        balances[str(asset)] = amount
                total_usdt = balances.get("USDT", Decimal("0"))
        except Exception as e:
            _logger.warning("futures_get_balances_failed error=%s", str(e))

        positions: list[Position] = []
        try:
            raw_positions = await self.adapter.get_positions()
            if isinstance(raw_positions, (list, tuple)):
                for raw in raw_positions:
                    if not isinstance(raw, dict):
                        continue

                    symbol_raw = str(raw.get("symbol", ""))
                    symbol = self.adapter.normalize_symbol(symbol_raw)

                    pos_amt_raw = raw.get("positionAmt", raw.get("quantity", "0"))
                    try:
                        pos_amt = Decimal(str(pos_amt_raw))
                    except Exception:
                        continue

                    if pos_amt == 0:
                        continue

                    entry_raw = raw.get("entryPrice", raw.get("entry_price", "0"))
                    try:
                        entry_price = Decimal(str(entry_raw))
                    except Exception:
                        entry_price = Decimal("0")

                    mark_raw = raw.get("markPrice", raw.get("currentPrice", "0"))
                    current_price: Decimal | None = None
                    try:
                        mp = Decimal(str(mark_raw))
                        current_price = mp if mp > 0 else None
                    except Exception:
                        pass

                    liq_raw = raw.get("liquidationPrice", "0")
                    liq_price: Decimal | None = None
                    try:
                        lp = Decimal(str(liq_raw))
                        liq_price = lp if lp > 0 else None
                    except Exception:
                        pass

                    leverage_raw = raw.get("leverage", "1")
                    try:
                        leverage = int(str(leverage_raw))
                    except Exception:
                        leverage = 1

                    margin_raw = raw.get("isolatedMargin", raw.get("margin", "0"))
                    margin: Decimal | None = None
                    try:
                        m = Decimal(str(margin_raw))
                        margin = m if m > 0 else None
                    except Exception:
                        pass

                    margin_type_raw = str(raw.get("marginType", "isolated")).lower()
                    margin_type = (
                        "isolated" if margin_type_raw == "isolated" else "cross"
                    )

                    pos_side_raw = str(raw.get("positionSide", "both")).lower()
                    # Infer side from sign of positionAmt if positionSide is "both"
                    if pos_side_raw == "both":
                        side = PositionSide.LONG if pos_amt > 0 else PositionSide.SHORT
                    else:
                        side = _POSITION_SIDE_MAP.get(pos_side_raw, PositionSide.LONG)

                    quantity = abs(pos_amt)

                    positions.append(
                        Position(
                            exchange=self.exchange,
                            symbol=symbol,
                            quantity=quantity,
                            average_entry_price=entry_price,
                            current_price=current_price,
                            timestamp=now,
                            side=side,
                            leverage=leverage,
                            liquidation_price=liq_price,
                            margin=margin,
                            margin_type=margin_type,
                        )
                    )
        except Exception as e:
            _logger.warning("futures_get_positions_failed error=%s", str(e))

        snapshot = BalanceSnapshot(
            exchange=self.exchange,
            timestamp=now,
            balances=balances,
            total_value_quote=total_usdt,
            quote_currency=self._quote_currency,
        )

        self._snapshot_cache = (now_mono, snapshot, positions)
        return snapshot, positions

    async def fetch_balances(self) -> BalanceSnapshot:
        snapshot, _ = await self._build_snapshot()
        return snapshot

    async def fetch_positions(self) -> list[Position]:
        _, positions = await self._build_snapshot()
        return positions

    async def fetch_fills(self, order_id: str) -> list[Fill]:
        # Binance fills (trades) require a separate /fapi/v1/userTrades call.
        # Return empty list; can be enhanced by adding get_trades() to adapter.
        return []

    async def sync_order_statuses(self, orders: list[Order]) -> list[Order]:
        updated: list[Order] = []
        for order in orders:
            if order.order_id is None:
                continue
            try:
                raw = await self.adapter.get_order(order.order_id, order.symbol)
                if not raw:
                    continue

                new_status = self._parse_order_status(
                    str(raw.get("status", "")).lower()
                )
                if new_status == order.status:
                    continue

                executed_raw = raw.get("executedQty", "0")
                executed = Decimal(str(executed_raw)) if executed_raw else Decimal("0")

                avg_price_raw = raw.get("avgPrice")
                avg_price: Decimal | None = None
                if avg_price_raw is not None:
                    try:
                        ap = Decimal(str(avg_price_raw))
                        avg_price = ap if ap > 0 else None
                    except Exception:
                        pass

                now = datetime.now(timezone.utc)
                updated_order = order.model_copy(
                    update={
                        "status": new_status,
                        "filled_quantity": executed,
                        "average_fill_price": avg_price,
                        "updated_at": now,
                    }
                )
                updated.append(updated_order)
            except Exception as e:
                _logger.debug(
                    "sync_order_status_failed order_id=%s error=%s",
                    order.order_id,
                    str(e),
                )
        return updated
