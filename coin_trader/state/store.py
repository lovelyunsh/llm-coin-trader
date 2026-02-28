from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib import import_module
from typing import TYPE_CHECKING, Any

import asyncpg

if TYPE_CHECKING:
    from coin_trader.core.models import (
        BalanceSnapshot,
        Fill,
        Order,
        OrderIntent,
        Position,
        RiskDecisionRecord,
        SafetyEvent,
    )


_MODELS_MODULE: Any | None = None


def _models() -> Any:
    global _MODELS_MODULE
    if _MODELS_MODULE is None:
        _MODELS_MODULE = import_module("coin_trader.core.models")
    return _MODELS_MODULE


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, str):
        return value
    return str(value)


def _get_attr(model: Any, name: str) -> Any:
    if hasattr(model, name):
        return getattr(model, name)
    model_dump = getattr(model, "model_dump", None)
    if callable(model_dump):
        try:
            data = model_dump()
            if isinstance(data, dict) and name in data:
                return data[name]
        except Exception:
            return None
    if isinstance(model, dict):
        return model.get(name)
    return None


def _to_jsonb(model: Any) -> str:
    """Serialize a pydantic model to a JSON string for JSONB columns."""
    dump_json = getattr(model, "model_dump_json", None)
    if callable(dump_json):
        return dump_json()
    return json.dumps(model, default=str)


def _parse_model(model_cls: Any, data: Any) -> Any:
    """Parse a model from JSONB data (dict from asyncpg) or JSON string."""
    if data is None:
        return None
    if isinstance(data, dict):
        return model_cls.model_validate(data)
    if isinstance(data, str):
        return model_cls.model_validate_json(data)
    return model_cls.model_validate(data)


class StateStore:
    """PostgreSQL-backed state store with an append-only audit log."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self._closed = False

    @classmethod
    async def create(
        cls,
        database_url: str,
        *,
        min_size: int = 2,
        max_size: int = 10,
    ) -> "StateStore":
        pool = await asyncpg.create_pool(
            database_url,
            min_size=min_size,
            max_size=max_size,
        )
        store = cls(pool)
        await store._init_schema()
        return store

    async def _init_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id SERIAL PRIMARY KEY,
                    event_type TEXT,
                    timestamp TEXT,
                    data JSONB
                )
                """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS order_intents (
                    intent_id TEXT PRIMARY KEY,
                    signal_id TEXT,
                    exchange TEXT,
                    symbol TEXT,
                    side TEXT,
                    order_type TEXT,
                    quantity TEXT,
                    quote_quantity TEXT,
                    price TEXT,
                    reason TEXT,
                    timestamp TEXT,
                    data JSONB
                )
                """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    client_order_id TEXT PRIMARY KEY,
                    order_id TEXT,
                    intent_id TEXT,
                    exchange TEXT,
                    symbol TEXT,
                    side TEXT,
                    order_type TEXT,
                    quantity TEXT,
                    price TEXT,
                    status TEXT,
                    filled_quantity TEXT,
                    remaining_quantity TEXT,
                    average_fill_price TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    data JSONB
                )
                """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS fills (
                    fill_id TEXT PRIMARY KEY,
                    order_id TEXT,
                    client_order_id TEXT,
                    exchange TEXT,
                    symbol TEXT,
                    side TEXT,
                    quantity TEXT,
                    price TEXT,
                    fee TEXT,
                    fee_currency TEXT,
                    timestamp TEXT,
                    data JSONB
                )
                """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS positions_snapshot (
                    id SERIAL PRIMARY KEY,
                    exchange TEXT,
                    symbol TEXT,
                    quantity TEXT,
                    avg_entry_price TEXT,
                    current_price TEXT,
                    unrealized_pnl TEXT,
                    timestamp TEXT,
                    data JSONB
                )
                """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS balances_snapshot (
                    snapshot_id TEXT PRIMARY KEY,
                    exchange TEXT,
                    timestamp TEXT,
                    total_value_krw TEXT,
                    data JSONB
                )
                """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions_log (
                    decision_id TEXT PRIMARY KEY,
                    intent_id TEXT,
                    decision TEXT,
                    reason TEXT,
                    timestamp TEXT,
                    data JSONB
                )
                """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS safety_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT,
                    timestamp TEXT,
                    description TEXT,
                    severity TEXT,
                    triggered_by TEXT,
                    data JSONB
                )
                """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS news (
                    id SERIAL PRIMARY KEY,
                    message_id TEXT UNIQUE,
                    content TEXT NOT NULL,
                    published_at TEXT,
                    fetched_at TEXT NOT NULL,
                    source TEXT DEFAULT 'coinness_telegram'
                )
                """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS news_summaries (
                    id SERIAL PRIMARY KEY,
                    summary TEXT NOT NULL,
                    news_count INTEGER,
                    created_at TEXT NOT NULL
                )
                """)

            # Indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_news_published_at ON news(published_at DESC)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_news_summaries_created ON news_summaries(created_at DESC)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_orders_exchange_symbol ON orders(exchange, symbol)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_positions_snapshot_timestamp_desc ON positions_snapshot(timestamp DESC)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_balances_snapshot_timestamp_desc ON balances_snapshot(timestamp DESC)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_safety_events_timestamp_desc ON safety_events(timestamp DESC)"
            )

    async def _append_event(
        self, conn: asyncpg.Connection, event_type: str, timestamp: str, data_json: str
    ) -> None:
        await conn.execute(
            "INSERT INTO events (event_type, timestamp, data) VALUES ($1, $2, $3::jsonb)",
            event_type, timestamp, data_json,
        )

    async def save_intent(self, intent: OrderIntent) -> None:
        data_json = _to_jsonb(intent)
        ts = _to_text(_get_attr(intent, "timestamp")) or _now_iso()
        intent_id = _to_text(_get_attr(intent, "intent_id"))
        if not intent_id:
            raise ValueError("OrderIntent.intent_id is required")

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._append_event(conn, "order_intent", ts, data_json)
                await conn.execute(
                    """
                    INSERT INTO order_intents (
                        intent_id, signal_id, exchange, symbol, side, order_type,
                        quantity, quote_quantity, price, reason, timestamp, data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb)
                    ON CONFLICT DO NOTHING
                    """,
                    intent_id,
                    _to_text(_get_attr(intent, "signal_id")),
                    _to_text(_get_attr(intent, "exchange")),
                    _to_text(_get_attr(intent, "symbol")),
                    _to_text(_get_attr(intent, "side")),
                    _to_text(_get_attr(intent, "order_type")),
                    _to_text(_get_attr(intent, "quantity")),
                    _to_text(_get_attr(intent, "quote_quantity")),
                    _to_text(_get_attr(intent, "price")),
                    _to_text(_get_attr(intent, "reason")),
                    ts,
                    data_json,
                )

    async def save_order(self, order: Order) -> None:
        data_json = _to_jsonb(order)
        updated_at = _to_text(_get_attr(order, "updated_at"))
        ts = updated_at or _to_text(_get_attr(order, "created_at")) or _now_iso()
        client_order_id = _to_text(_get_attr(order, "client_order_id"))
        if not client_order_id:
            raise ValueError("Order.client_order_id is required")

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._append_event(conn, "order", ts, data_json)
                await conn.execute(
                    """
                    INSERT INTO orders (
                        client_order_id, order_id, intent_id, exchange, symbol, side, order_type,
                        quantity, price, status, filled_quantity, remaining_quantity, average_fill_price,
                        created_at, updated_at, data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16::jsonb)
                    ON CONFLICT(client_order_id) DO UPDATE SET
                        order_id=EXCLUDED.order_id,
                        intent_id=EXCLUDED.intent_id,
                        exchange=EXCLUDED.exchange,
                        symbol=EXCLUDED.symbol,
                        side=EXCLUDED.side,
                        order_type=EXCLUDED.order_type,
                        quantity=EXCLUDED.quantity,
                        price=EXCLUDED.price,
                        status=EXCLUDED.status,
                        filled_quantity=EXCLUDED.filled_quantity,
                        remaining_quantity=EXCLUDED.remaining_quantity,
                        average_fill_price=EXCLUDED.average_fill_price,
                        created_at=EXCLUDED.created_at,
                        updated_at=EXCLUDED.updated_at,
                        data=EXCLUDED.data
                    """,
                    client_order_id,
                    _to_text(_get_attr(order, "order_id")),
                    _to_text(_get_attr(order, "intent_id")),
                    _to_text(_get_attr(order, "exchange")),
                    _to_text(_get_attr(order, "symbol")),
                    _to_text(_get_attr(order, "side")),
                    _to_text(_get_attr(order, "order_type")),
                    _to_text(_get_attr(order, "quantity")),
                    _to_text(_get_attr(order, "price")),
                    _to_text(_get_attr(order, "status")),
                    _to_text(_get_attr(order, "filled_quantity")),
                    _to_text(_get_attr(order, "remaining_quantity")),
                    _to_text(_get_attr(order, "average_fill_price")),
                    _to_text(_get_attr(order, "created_at")),
                    updated_at,
                    data_json,
                )

    async def save_fill(self, fill: Fill) -> None:
        data_json = _to_jsonb(fill)
        ts = _to_text(_get_attr(fill, "timestamp")) or _now_iso()
        fill_id = _to_text(_get_attr(fill, "fill_id"))
        if not fill_id:
            raise ValueError("Fill.fill_id is required")

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._append_event(conn, "fill", ts, data_json)
                await conn.execute(
                    """
                    INSERT INTO fills (
                        fill_id, order_id, client_order_id, exchange, symbol, side,
                        quantity, price, fee, fee_currency, timestamp, data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb)
                    ON CONFLICT DO NOTHING
                    """,
                    fill_id,
                    _to_text(_get_attr(fill, "order_id")),
                    _to_text(_get_attr(fill, "client_order_id")),
                    _to_text(_get_attr(fill, "exchange")),
                    _to_text(_get_attr(fill, "symbol")),
                    _to_text(_get_attr(fill, "side")),
                    _to_text(_get_attr(fill, "quantity")),
                    _to_text(_get_attr(fill, "price")),
                    _to_text(_get_attr(fill, "fee")),
                    _to_text(_get_attr(fill, "fee_currency")),
                    ts,
                    data_json,
                )

    async def save_decision(self, decision: RiskDecisionRecord) -> None:
        data_json = _to_jsonb(decision)
        ts = _to_text(_get_attr(decision, "timestamp")) or _now_iso()
        decision_id = _to_text(_get_attr(decision, "decision_id"))
        if not decision_id:
            raise ValueError("RiskDecisionRecord.decision_id is required")

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._append_event(conn, "decision", ts, data_json)
                await conn.execute(
                    """
                    INSERT INTO decisions_log (
                        decision_id, intent_id, decision, reason, timestamp, data
                    ) VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                    ON CONFLICT DO NOTHING
                    """,
                    decision_id,
                    _to_text(_get_attr(decision, "intent_id")),
                    _to_text(_get_attr(decision, "decision")),
                    _to_text(_get_attr(decision, "reason")),
                    ts,
                    data_json,
                )

    async def save_safety_event(self, event: SafetyEvent) -> None:
        data_json = _to_jsonb(event)
        ts = _to_text(_get_attr(event, "timestamp")) or _now_iso()
        event_id = _to_text(_get_attr(event, "event_id"))
        if not event_id:
            raise ValueError("SafetyEvent.event_id is required")

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._append_event(conn, "safety", ts, data_json)
                await conn.execute(
                    """
                    INSERT INTO safety_events (
                        event_id, event_type, timestamp, description, severity, triggered_by, data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                    ON CONFLICT DO NOTHING
                    """,
                    event_id,
                    _to_text(_get_attr(event, "event_type")),
                    ts,
                    _to_text(_get_attr(event, "description")),
                    _to_text(_get_attr(event, "severity")),
                    _to_text(_get_attr(event, "triggered_by")),
                    data_json,
                )

    async def save_balance_snapshot(self, snapshot: BalanceSnapshot) -> None:
        data_json = _to_jsonb(snapshot)
        ts = _to_text(_get_attr(snapshot, "timestamp")) or _now_iso()
        snapshot_id = _to_text(_get_attr(snapshot, "snapshot_id"))
        if not snapshot_id:
            raise ValueError("BalanceSnapshot.snapshot_id is required")

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._append_event(conn, "balance", ts, data_json)
                await conn.execute(
                    """
                    INSERT INTO balances_snapshot (
                        snapshot_id, exchange, timestamp, total_value_krw, data
                    ) VALUES ($1, $2, $3, $4, $5::jsonb)
                    ON CONFLICT DO NOTHING
                    """,
                    snapshot_id,
                    _to_text(_get_attr(snapshot, "exchange")),
                    ts,
                    _to_text(_get_attr(snapshot, "total_value_krw")),
                    data_json,
                )

    async def save_position_snapshot(self, position: Position) -> None:
        data_json = _to_jsonb(position)
        ts = _to_text(_get_attr(position, "timestamp")) or _now_iso()

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._append_event(conn, "position", ts, data_json)
                await conn.execute(
                    """
                    INSERT INTO positions_snapshot (
                        exchange, symbol, quantity, avg_entry_price, current_price, unrealized_pnl, timestamp, data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                    """,
                    _to_text(_get_attr(position, "exchange")),
                    _to_text(_get_attr(position, "symbol")),
                    _to_text(_get_attr(position, "quantity")),
                    _to_text(_get_attr(position, "avg_entry_price")),
                    _to_text(_get_attr(position, "current_price")),
                    _to_text(_get_attr(position, "unrealized_pnl")),
                    ts,
                    data_json,
                )

    async def get_open_orders(self) -> list[Order]:
        OrderModel = _models().Order
        terminal = (
            "filled", "canceled", "cancelled", "rejected",
            "expired", "done", "closed", "failed",
        )
        placeholders = ", ".join(f"${i + 1}" for i in range(len(terminal)))
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT data FROM orders WHERE status IS NULL OR lower(status) NOT IN ({placeholders}) ORDER BY updated_at DESC",
                *terminal,
            )

        orders: list[Order] = []
        for row in rows:
            data = row["data"]
            if data is None:
                continue
            orders.append(_parse_model(OrderModel, data))
        return orders

    async def get_positions(self) -> list[Position]:
        PositionModel = _models().Position
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT p.data
                FROM positions_snapshot p
                JOIN (
                    SELECT exchange, symbol, MAX(timestamp) AS ts
                    FROM positions_snapshot
                    GROUP BY exchange, symbol
                ) latest
                ON p.exchange = latest.exchange AND p.symbol = latest.symbol AND p.timestamp = latest.ts
                ORDER BY p.exchange, p.symbol
                """)

        positions: list[Position] = []
        for row in rows:
            data = row["data"]
            if data is None:
                continue
            positions.append(_parse_model(PositionModel, data))
        return positions

    async def get_latest_balance(self, exchange: str) -> BalanceSnapshot | None:
        BalanceSnapshotModel = _models().BalanceSnapshot
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM balances_snapshot WHERE exchange = $1 ORDER BY timestamp DESC LIMIT 1",
                exchange,
            )
        if row is None or row["data"] is None:
            return None
        return _parse_model(BalanceSnapshotModel, row["data"])

    async def get_all_orders(self, limit: int = 100) -> list[Order]:
        """Get recent orders, newest first."""
        OrderModel = _models().Order
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT data FROM orders ORDER BY updated_at DESC LIMIT $1",
                limit,
            )
        orders: list[Order] = []
        for row in rows:
            data = row["data"]
            if data is None:
                continue
            orders.append(_parse_model(OrderModel, data))
        return orders

    async def get_recent_decisions(self, limit: int = 50) -> list[RiskDecisionRecord]:
        """Get recent risk decisions, newest first."""
        DecisionModel = _models().RiskDecisionRecord
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT data FROM decisions_log ORDER BY timestamp DESC LIMIT $1",
                limit,
            )
        decisions: list[RiskDecisionRecord] = []
        for row in rows:
            data = row["data"]
            if data is None:
                continue
            decisions.append(_parse_model(DecisionModel, data))
        return decisions

    async def get_safety_events(self, limit: int = 50) -> list[SafetyEvent]:
        """Get recent safety events, newest first."""
        SafetyEventModel = _models().SafetyEvent
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT data FROM safety_events ORDER BY timestamp DESC LIMIT $1",
                limit,
            )
        events: list[SafetyEvent] = []
        for row in rows:
            data = row["data"]
            if data is None:
                continue
            events.append(_parse_model(SafetyEventModel, data))
        return events

    async def get_recent_balances(self, limit: int = 20) -> list[BalanceSnapshot]:
        """Get recent balance snapshots, newest first."""
        BalanceSnapshotModel = _models().BalanceSnapshot
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT data FROM balances_snapshot ORDER BY timestamp DESC LIMIT $1",
                limit,
            )
        snapshots: list[BalanceSnapshot] = []
        for row in rows:
            data = row["data"]
            if data is None:
                continue
            snapshots.append(_parse_model(BalanceSnapshotModel, data))
        return snapshots

    async def save_news(self, items: list[Any]) -> int:
        """Save news items. Returns count of newly inserted rows."""
        now = _now_iso()
        inserted = 0
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for item in items:
                    msg_id = _to_text(_get_attr(item, "message_id"))
                    content = _to_text(_get_attr(item, "content"))
                    published_at = _to_text(_get_attr(item, "published_at"))
                    if not msg_id or not content:
                        continue
                    result = await conn.execute(
                        "INSERT INTO news (message_id, content, published_at, fetched_at) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING",
                        msg_id, content, published_at, now,
                    )
                    if result.endswith("1"):
                        inserted += 1
        return inserted

    async def get_recent_news(self, limit: int = 50) -> list[dict[str, str]]:
        """Get recent news items, newest first."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT message_id, content, published_at FROM news ORDER BY published_at DESC LIMIT $1",
                limit,
            )
        return [
            {
                "message_id": row["message_id"],
                "content": row["content"],
                "published_at": row["published_at"] or "",
            }
            for row in rows
        ]

    async def save_news_summary(self, summary: str, news_count: int) -> None:
        """Save a news summary."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO news_summaries (summary, news_count, created_at) VALUES ($1, $2, $3)",
                summary, news_count, _now_iso(),
            )

    async def get_latest_news_summary(self) -> str | None:
        """Get the most recent news summary."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT summary FROM news_summaries ORDER BY created_at DESC LIMIT 1"
            )
        if row is None:
            return None
        return str(row["summary"])

    async def order_exists(self, client_order_id: str) -> bool:
        """Check if an order with the given client_order_id exists."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM orders WHERE client_order_id = $1 LIMIT 1",
                client_order_id,
            )
        return row is not None

    async def close(self) -> None:
        if self._closed:
            return
        try:
            await self._pool.close()
        finally:
            self._closed = True

    async def __aenter__(self) -> "StateStore":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()
