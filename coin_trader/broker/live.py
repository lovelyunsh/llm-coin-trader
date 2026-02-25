"""Live trading broker - wraps exchange adapter for real execution."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal
from typing import Protocol
from uuid import uuid4

_logger = logging.getLogger("broker.live")

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

_CACHE_TTL = 5.0
_NET_DEPOSITS_CACHE_TTL = 300.0


class _ExchangeAdapter(Protocol):
    async def place_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal | None,
    ) -> Mapping[str, object]: ...

    async def cancel_order(self, order_id: str, symbol: str) -> object: ...

    async def get_open_orders(self) -> object: ...

    async def get_order(self, order_id: str) -> Mapping[str, object]: ...

    async def get_balances(self) -> object: ...

    async def get_accounts_raw(self) -> list[Mapping[str, object]]: ...

    async def get_ticker(self, symbol: str) -> Mapping[str, object]: ...

    async def get_tickers(
        self, symbols: list[str]
    ) -> Mapping[str, Mapping[str, object]]: ...

    async def get_deposits(self, limit: int = ...) -> list[Mapping[str, object]]: ...

    async def get_withdraws(self, limit: int = ...) -> list[Mapping[str, object]]: ...

    def normalize_symbol(self, symbol: str) -> str: ...


class LiveBroker:
    exchange: ExchangeName
    adapter: _ExchangeAdapter

    def __init__(
        self,
        exchange: ExchangeName,
        exchange_adapter: _ExchangeAdapter,
        trading_symbols: list[str] | None = None,
        quote_currency: str = "KRW",
    ) -> None:
        self.exchange = exchange
        self.adapter = exchange_adapter
        self._quote_currency: str = quote_currency
        self._trading_currencies: set[str] = set()
        if trading_symbols:
            for s in trading_symbols:
                base = s.split("/")[0] if "/" in s else s
                self._trading_currencies.add(base)
        self._accounts_cache: list[Mapping[str, object]] = []
        self._accounts_cache_ts: float = 0.0
        self._ticker_cache: dict[str, tuple[float, Mapping[str, object]]] = {}
        self._snapshot_cache: tuple[float, BalanceSnapshot | None, list[Position]] = (
            0.0,
            None,
            [],
        )
        self._net_deposits_cache: dict[str, tuple[float, Decimal]] = {}

    async def place(self, intent: OrderIntent, client_order_id: str) -> Order:
        now = datetime.now(timezone.utc)

        quantity = intent.quantity
        if quantity is None and intent.quote_quantity is not None:
            ref_price = intent.price
            if ref_price is None:
                ticker = await self.adapter.get_ticker(intent.symbol)
                ref_price = Decimal(str(ticker.get("price", 0)))
            if ref_price <= 0:
                raise ValueError("Invalid reference price for sizing")
            quantity = intent.quote_quantity / ref_price

        if quantity is None or quantity <= 0:
            raise ValueError("Order quantity must be > 0")

        try:
            result = await self.adapter.place_order(
                symbol=intent.symbol,
                side=intent.side.value,
                order_type=intent.order_type.value,
                quantity=quantity,
                price=intent.price,
            )

            order_id = result.get("uuid", result.get("orderId", result.get("id", "")))
            status_str = str(
                result.get("state", result.get("status", "submitted"))
            ).lower()
            status_map = {
                "wait": OrderStatus.SUBMITTED,
                "watch": OrderStatus.SUBMITTED,
                "submitted": OrderStatus.SUBMITTED,
                "open": OrderStatus.SUBMITTED,
                "done": OrderStatus.FILLED,
                "filled": OrderStatus.FILLED,
                "cancel": OrderStatus.CANCELLED,
                "cancelled": OrderStatus.CANCELLED,
                "canceled": OrderStatus.CANCELLED,
                "rejected": OrderStatus.REJECTED,
                "failed": OrderStatus.FAILED,
            }
            status = status_map.get(status_str, OrderStatus.SUBMITTED)

            executed = Decimal(
                str(result.get("executed_volume", result.get("filled", "0")))
            )
            remaining_obj = result.get("remaining_volume", result.get("remaining"))
            remaining = (
                Decimal(str(remaining_obj))
                if remaining_obj is not None
                else (quantity - executed)
            )
            if remaining < 0:
                remaining = Decimal("0")

            self.invalidate_cache()
            return Order(
                order_id=str(order_id) if order_id else None,
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
                average_fill_price=None,
                created_at=now,
                updated_at=now,
            )
        except Exception as e:
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
        _ = await self.adapter.cancel_order(order_id, "")

    async def fetch_open_orders(self) -> list[Order]:
        raw_orders_obj = await self.adapter.get_open_orders()
        if not isinstance(raw_orders_obj, (list, tuple)):
            return []
        orders: list[Order] = []
        now = datetime.now(timezone.utc)

        for raw in raw_orders_obj:
            if not isinstance(raw, dict):
                continue
            market = str(raw.get("market", raw.get("symbol", "")))
            symbol = self.adapter.normalize_symbol(market)

            side_raw = str(raw.get("side", "")).lower()
            if side_raw in {"bid", "buy"}:
                side = OrderSide.BUY
            else:
                side = OrderSide.SELL

            ord_type_raw = str(raw.get("ord_type", raw.get("type", ""))).lower()
            if ord_type_raw == "limit":
                order_type = OrderType.LIMIT
            else:
                order_type = OrderType.MARKET

            quantity = Decimal(str(raw.get("volume", raw.get("origQty", "0"))))
            if quantity <= 0:
                continue

            price_obj = raw.get("price")
            price = Decimal(str(price_obj)) if price_obj is not None else None
            if order_type == OrderType.LIMIT and price is None:
                continue

            executed = Decimal(
                str(raw.get("executed_volume", raw.get("executedQty", "0")))
            )
            if executed < 0:
                executed = Decimal("0")

            created_at_raw = raw.get("created_at")
            if created_at_raw:
                try:
                    created_at = datetime.fromisoformat(
                        str(created_at_raw).replace("+09:00", "+09:00")
                    )
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    created_at = now
            else:
                created_at = now

            orders.append(
                Order(
                    order_id=str(raw.get("uuid", raw.get("orderId", raw.get("id", ""))))
                    or None,
                    client_order_id=str(
                        raw.get("uuid", raw.get("clientOrderId", raw.get("id", "")))
                    )
                    or str(raw.get("uuid", ""))
                    or str(uuid4()),
                    intent_id=uuid4(),
                    exchange=self.exchange,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    status=OrderStatus.SUBMITTED,
                    filled_quantity=executed,
                    average_fill_price=None,
                    created_at=created_at,
                    updated_at=now,
                )
            )

        return orders

    async def _get_accounts_cached(self) -> list[Mapping[str, object]]:
        now = time.monotonic()
        if now - self._accounts_cache_ts < _CACHE_TTL and self._accounts_cache:
            return self._accounts_cache
        self._accounts_cache = await self.adapter.get_accounts_raw()
        self._accounts_cache_ts = now
        return self._accounts_cache

    async def _get_ticker_cached(self, symbol: str) -> Mapping[str, object]:
        now = time.monotonic()
        cached = self._ticker_cache.get(symbol)
        if cached and now - cached[0] < _CACHE_TTL:
            return cached[1]
        ticker = await self.adapter.get_ticker(symbol)
        self._ticker_cache[symbol] = (now, ticker)
        return ticker

    def invalidate_cache(self) -> None:
        self._accounts_cache_ts = 0.0
        self._ticker_cache.clear()
        self._snapshot_cache = (0.0, None, [])

    async def get_net_deposits(self, since: str = "2026-01-01") -> Decimal:
        now_mono = time.monotonic()
        cached = self._net_deposits_cache.get(since)
        if cached and now_mono - cached[0] < _NET_DEPOSITS_CACHE_TTL:
            return cached[1]

        since_dt = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)

        deposits = await self.adapter.get_deposits()
        withdraws = await self.adapter.get_withdraws()

        net = Decimal("0")

        for dep in deposits:
            if str(dep.get("currency", "")).upper() != "KRW":
                continue
            raw_ts = dep.get("created_at")
            if not isinstance(raw_ts, str):
                continue
            try:
                ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            if ts < since_dt:
                continue
            try:
                net += Decimal(str(dep.get("amount", "0")))
            except Exception:
                continue

        for wd in withdraws:
            if str(wd.get("currency", "")).upper() != "KRW":
                continue
            raw_ts = wd.get("created_at")
            if not isinstance(raw_ts, str):
                continue
            try:
                ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            if ts < since_dt:
                continue
            try:
                net -= Decimal(str(wd.get("amount", "0")))
            except Exception:
                continue

        self._net_deposits_cache[since] = (now_mono, net)
        return net

    async def _build_snapshot(self) -> tuple[BalanceSnapshot, list[Position]]:
        now_mono = time.monotonic()
        cached_ts, cached_snap, cached_pos = self._snapshot_cache
        if now_mono - cached_ts < _CACHE_TTL and cached_snap is not None:
            return cached_snap, cached_pos

        accounts = await self._get_accounts_cached()
        balances: dict[str, Decimal] = {}
        positions: list[Position] = []
        now = datetime.now(timezone.utc)

        for acct in accounts:
            currency = acct.get("currency")
            if not isinstance(currency, str):
                continue
            balance = Decimal(str(acct.get("balance", "0")))
            locked = Decimal(str(acct.get("locked", "0")))
            total = balance + locked
            if total > 0:
                balances[currency] = total

        qc = self._quote_currency
        total_quote = balances.get(qc, Decimal("0"))
        coin_currencies = [c for c in balances if c != qc and balances[c] > 0]

        known = [f"{c}/{qc}" for c in coin_currencies if c in self._trading_currencies]
        dust = [
            f"{c}/{qc}" for c in coin_currencies if c not in self._trading_currencies
        ]

        tickers: dict[str, Mapping[str, object]] = {}
        if known:
            try:
                tickers = dict(await self.adapter.get_tickers(known))
                for sym, t in tickers.items():
                    self._ticker_cache[sym] = (now_mono, t)
            except Exception as e:
                _logger.warning("batch_ticker_failed error=%s", str(e))

        for sym in dust:
            try:
                t = await self.adapter.get_ticker(sym)
                if t:
                    tickers[sym] = t
                    self._ticker_cache[sym] = (now_mono, t)
            except Exception:
                pass

        for currency in coin_currencies:
            qty = balances[currency]
            symbol = f"{currency}/{qc}"

            ticker = tickers.get(symbol)
            if ticker is None:
                cached = self._ticker_cache.get(symbol)
                ticker = cached[1] if cached else None

            price = Decimal(str(ticker.get("price", 0))) if ticker else Decimal("0")
            total_quote += qty * price

            avg_buy_price = Decimal("0")
            for acct in accounts:
                if acct.get("currency") == currency:
                    avg_buy_price = Decimal(str(acct.get("avg_buy_price", "0")))
                    break

            if price > 0:
                positions.append(
                    Position(
                        exchange=self.exchange,
                        symbol=symbol,
                        quantity=qty,
                        average_entry_price=avg_buy_price,
                        current_price=price,
                        timestamp=now,
                    )
                )
            elif qty > 0:
                _logger.debug(
                    "no_price_for_coin currency=%s qty=%s (dust?)", currency, str(qty)
                )

        snapshot = BalanceSnapshot(
            exchange=self.exchange,
            timestamp=now,
            balances=balances,
            total_value_krw=total_quote if qc == "KRW" else None,
            total_value_quote=total_quote,
            quote_currency=qc,
        )
        self._snapshot_cache = (now_mono, snapshot, positions)
        return snapshot, positions

    async def fetch_balances(self) -> BalanceSnapshot:
        snapshot, _ = await self._build_snapshot()
        return snapshot

    async def fetch_positions(self) -> list[Position]:
        _, positions = await self._build_snapshot()
        return positions

    async def sync_order_statuses(self, orders: list[Order]) -> list[Order]:
        updated: list[Order] = []
        for order in orders:
            if order.order_id is None:
                continue
            try:
                raw = await self.adapter.get_order(order.order_id)
                if not raw:
                    continue

                status_str = str(raw.get("state", "")).lower()
                status_map = {
                    "wait": OrderStatus.SUBMITTED,
                    "watch": OrderStatus.SUBMITTED,
                    "done": OrderStatus.FILLED,
                    "cancel": OrderStatus.CANCELLED,
                }
                new_status = status_map.get(status_str)
                if new_status is None or new_status == order.status:
                    continue

                executed = Decimal(str(raw.get("executed_volume", "0")))
                avg_price_raw = raw.get("avg_price", raw.get("price"))
                avg_price = Decimal(str(avg_price_raw)) if avg_price_raw else None

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

    async def fetch_fills(self, _order_id: str) -> list[Fill]:
        return []
