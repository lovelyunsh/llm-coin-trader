from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


class StateStore:
    """SQLite-backed state store with an append-only audit log."""

    def __init__(self, db_path: Path | str = "state.sqlite3") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._closed = False

        self._configure_pragmas()
        self._init_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        """Public read-only accessor for the underlying connection."""
        return self._conn

    def _configure_pragmas(self) -> None:
        # WAL + NORMAL is a good default for durability/performance.
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                timestamp TEXT,
                data JSON
            )
            """
        )

        self._conn.execute(
            """
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
                data JSON
            )
            """
        )

        self._conn.execute(
            """
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
                data JSON
            )
            """
        )

        self._conn.execute(
            """
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
                data JSON
            )
            """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positions_snapshot (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exchange TEXT,
                symbol TEXT,
                quantity TEXT,
                avg_entry_price TEXT,
                current_price TEXT,
                unrealized_pnl TEXT,
                timestamp TEXT,
                data JSON
            )
            """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS balances_snapshot (
                snapshot_id TEXT PRIMARY KEY,
                exchange TEXT,
                timestamp TEXT,
                total_value_krw TEXT,
                data JSON
            )
            """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions_log (
                decision_id TEXT PRIMARY KEY,
                intent_id TEXT,
                decision TEXT,
                reason TEXT,
                timestamp TEXT,
                data JSON
            )
            """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS safety_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT,
                timestamp TEXT,
                description TEXT,
                severity TEXT,
                triggered_by TEXT,
                data JSON
            )
            """
        )

        # Indexes
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_orders_exchange_symbol ON orders(exchange, symbol)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_positions_snapshot_timestamp_desc ON positions_snapshot(timestamp DESC)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_balances_snapshot_timestamp_desc ON balances_snapshot(timestamp DESC)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_safety_events_timestamp_desc ON safety_events(timestamp DESC)"
        )

        self._conn.commit()

    def _append_event(self, event_type: str, timestamp: str, data_json: str) -> None:
        self._conn.execute(
            "INSERT INTO events (event_type, timestamp, data) VALUES (?, ?, ?)",
            (event_type, timestamp, data_json),
        )

    def save_intent(self, intent: OrderIntent) -> None:
        data_json = intent.model_dump_json()
        ts = _to_text(_get_attr(intent, "timestamp")) or _now_iso()
        intent_id = _to_text(_get_attr(intent, "intent_id"))
        if not intent_id:
            raise ValueError("OrderIntent.intent_id is required")

        with self._conn:
            self._append_event("order_intent", ts, data_json)
            self._conn.execute(
                """
                INSERT OR IGNORE INTO order_intents (
                    intent_id, signal_id, exchange, symbol, side, order_type,
                    quantity, quote_quantity, price, reason, timestamp, data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """.strip(),
                (
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
                ),
            )

    def save_order(self, order: Order) -> None:
        data_json = order.model_dump_json()
        updated_at = _to_text(_get_attr(order, "updated_at"))
        ts = updated_at or _to_text(_get_attr(order, "created_at")) or _now_iso()
        client_order_id = _to_text(_get_attr(order, "client_order_id"))
        if not client_order_id:
            raise ValueError("Order.client_order_id is required")

        with self._conn:
            self._append_event("order", ts, data_json)
            self._conn.execute(
                """
                INSERT INTO orders (
                    client_order_id, order_id, intent_id, exchange, symbol, side, order_type,
                    quantity, price, status, filled_quantity, remaining_quantity, average_fill_price,
                    created_at, updated_at, data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_order_id) DO UPDATE SET
                    order_id=excluded.order_id,
                    intent_id=excluded.intent_id,
                    exchange=excluded.exchange,
                    symbol=excluded.symbol,
                    side=excluded.side,
                    order_type=excluded.order_type,
                    quantity=excluded.quantity,
                    price=excluded.price,
                    status=excluded.status,
                    filled_quantity=excluded.filled_quantity,
                    remaining_quantity=excluded.remaining_quantity,
                    average_fill_price=excluded.average_fill_price,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    data=excluded.data
                """.strip(),
                (
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
                ),
            )

    def save_fill(self, fill: Fill) -> None:
        data_json = fill.model_dump_json()
        ts = _to_text(_get_attr(fill, "timestamp")) or _now_iso()
        fill_id = _to_text(_get_attr(fill, "fill_id"))
        if not fill_id:
            raise ValueError("Fill.fill_id is required")

        with self._conn:
            self._append_event("fill", ts, data_json)
            self._conn.execute(
                """
                INSERT OR IGNORE INTO fills (
                    fill_id, order_id, client_order_id, exchange, symbol, side,
                    quantity, price, fee, fee_currency, timestamp, data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """.strip(),
                (
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
                ),
            )

    def save_decision(self, decision: RiskDecisionRecord) -> None:
        data_json = decision.model_dump_json()
        ts = _to_text(_get_attr(decision, "timestamp")) or _now_iso()
        decision_id = _to_text(_get_attr(decision, "decision_id"))
        if not decision_id:
            raise ValueError("RiskDecisionRecord.decision_id is required")

        with self._conn:
            self._append_event("decision", ts, data_json)
            self._conn.execute(
                """
                INSERT OR IGNORE INTO decisions_log (
                    decision_id, intent_id, decision, reason, timestamp, data
                ) VALUES (?, ?, ?, ?, ?, ?)
                """.strip(),
                (
                    decision_id,
                    _to_text(_get_attr(decision, "intent_id")),
                    _to_text(_get_attr(decision, "decision")),
                    _to_text(_get_attr(decision, "reason")),
                    ts,
                    data_json,
                ),
            )

    def save_safety_event(self, event: SafetyEvent) -> None:
        data_json = event.model_dump_json()
        ts = _to_text(_get_attr(event, "timestamp")) or _now_iso()
        event_id = _to_text(_get_attr(event, "event_id"))
        if not event_id:
            raise ValueError("SafetyEvent.event_id is required")

        with self._conn:
            self._append_event("safety", ts, data_json)
            self._conn.execute(
                """
                INSERT OR IGNORE INTO safety_events (
                    event_id, event_type, timestamp, description, severity, triggered_by, data
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """.strip(),
                (
                    event_id,
                    _to_text(_get_attr(event, "event_type")),
                    ts,
                    _to_text(_get_attr(event, "description")),
                    _to_text(_get_attr(event, "severity")),
                    _to_text(_get_attr(event, "triggered_by")),
                    data_json,
                ),
            )

    def save_balance_snapshot(self, snapshot: BalanceSnapshot) -> None:
        data_json = snapshot.model_dump_json()
        ts = _to_text(_get_attr(snapshot, "timestamp")) or _now_iso()
        snapshot_id = _to_text(_get_attr(snapshot, "snapshot_id"))
        if not snapshot_id:
            raise ValueError("BalanceSnapshot.snapshot_id is required")

        with self._conn:
            self._append_event("balance", ts, data_json)
            self._conn.execute(
                """
                INSERT OR IGNORE INTO balances_snapshot (
                    snapshot_id, exchange, timestamp, total_value_krw, data
                ) VALUES (?, ?, ?, ?, ?)
                """.strip(),
                (
                    snapshot_id,
                    _to_text(_get_attr(snapshot, "exchange")),
                    ts,
                    _to_text(_get_attr(snapshot, "total_value_krw")),
                    data_json,
                ),
            )

    def save_position_snapshot(self, position: Position) -> None:
        data_json = position.model_dump_json()
        ts = _to_text(_get_attr(position, "timestamp")) or _now_iso()

        with self._conn:
            self._append_event("position", ts, data_json)
            self._conn.execute(
                """
                INSERT INTO positions_snapshot (
                    exchange, symbol, quantity, avg_entry_price, current_price, unrealized_pnl, timestamp, data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """.strip(),
                (
                    _to_text(_get_attr(position, "exchange")),
                    _to_text(_get_attr(position, "symbol")),
                    _to_text(_get_attr(position, "quantity")),
                    _to_text(_get_attr(position, "avg_entry_price")),
                    _to_text(_get_attr(position, "current_price")),
                    _to_text(_get_attr(position, "unrealized_pnl")),
                    ts,
                    data_json,
                ),
            )

    def get_open_orders(self) -> list[Order]:
        OrderModel = _models().Order
        terminal = (
            "filled",
            "canceled",
            "cancelled",
            "rejected",
            "expired",
            "done",
            "closed",
            "failed",
        )
        placeholders = ",".join("?" for _ in terminal)
        rows = self._conn.execute(
            f"SELECT data FROM orders WHERE status IS NULL OR lower(status) NOT IN ({placeholders}) ORDER BY updated_at DESC",
            terminal,
        ).fetchall()

        orders: list[Order] = []
        for row in rows:
            data = row["data"]
            if data is None:
                continue
            orders.append(OrderModel.model_validate_json(data))
        return orders

    def get_positions(self) -> list[Position]:
        PositionModel = _models().Position
        rows = self._conn.execute(
            """
            SELECT p.data
            FROM positions_snapshot p
            JOIN (
                SELECT exchange, symbol, MAX(timestamp) AS ts
                FROM positions_snapshot
                GROUP BY exchange, symbol
            ) latest
            ON p.exchange = latest.exchange AND p.symbol = latest.symbol AND p.timestamp = latest.ts
            ORDER BY p.exchange, p.symbol
            """.strip()
        ).fetchall()

        positions: list[Position] = []
        for row in rows:
            data = row["data"]
            if data is None:
                continue
            positions.append(PositionModel.model_validate_json(data))
        return positions

    def get_latest_balance(self, exchange: str) -> BalanceSnapshot | None:
        BalanceSnapshotModel = _models().BalanceSnapshot
        row = self._conn.execute(
            "SELECT data FROM balances_snapshot WHERE exchange = ? ORDER BY timestamp DESC LIMIT 1",
            (exchange,),
        ).fetchone()
        if row is None or row["data"] is None:
            return None
        return BalanceSnapshotModel.model_validate_json(row["data"])

    def get_all_orders(self, limit: int = 100) -> list[Order]:
        """Get recent orders, newest first."""
        OrderModel = _models().Order
        rows = self._conn.execute(
            "SELECT data FROM orders ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        orders: list[Order] = []
        for row in rows:
            data = row["data"]
            if data is None:
                continue
            orders.append(OrderModel.model_validate_json(data))
        return orders

    def get_recent_decisions(self, limit: int = 50) -> list[RiskDecisionRecord]:
        """Get recent risk decisions, newest first."""
        DecisionModel = _models().RiskDecisionRecord
        rows = self._conn.execute(
            "SELECT data FROM decisions_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        decisions: list[RiskDecisionRecord] = []
        for row in rows:
            data = row["data"]
            if data is None:
                continue
            decisions.append(DecisionModel.model_validate_json(data))
        return decisions

    def get_safety_events(self, limit: int = 50) -> list[SafetyEvent]:
        """Get recent safety events, newest first."""
        SafetyEventModel = _models().SafetyEvent
        rows = self._conn.execute(
            "SELECT data FROM safety_events ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        events: list[SafetyEvent] = []
        for row in rows:
            data = row["data"]
            if data is None:
                continue
            events.append(SafetyEventModel.model_validate_json(data))
        return events

    def get_recent_balances(self, limit: int = 20) -> list[BalanceSnapshot]:
        """Get recent balance snapshots, newest first."""
        BalanceSnapshotModel = _models().BalanceSnapshot
        rows = self._conn.execute(
            "SELECT data FROM balances_snapshot ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        snapshots: list[BalanceSnapshot] = []
        for row in rows:
            data = row["data"]
            if data is None:
                continue
            snapshots.append(BalanceSnapshotModel.model_validate_json(data))
        return snapshots

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._conn.close()
        finally:
            self._closed = True

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
