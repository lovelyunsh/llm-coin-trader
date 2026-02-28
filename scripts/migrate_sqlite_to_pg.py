#!/usr/bin/env python3
"""Migrate data from SQLite (data/coin_trader.db) to PostgreSQL.

Usage:
    python scripts/migrate_sqlite_to_pg.py [--sqlite PATH] [--pg-url URL]

Defaults:
    --sqlite  data/coin_trader.db
    --pg-url  postgresql://trader:trader@localhost:5432/coin_trader
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import json
from pathlib import Path

import asyncpg


# Tables and their columns (order matters for foreign-key-free bulk insert)
TABLES: list[tuple[str, list[str]]] = [
    ("events", ["id", "event_type", "timestamp", "data"]),
    ("order_intents", [
        "intent_id", "signal_id", "exchange", "symbol", "side", "order_type",
        "quantity", "quote_quantity", "price", "reason", "timestamp", "data",
    ]),
    ("orders", [
        "client_order_id", "order_id", "intent_id", "exchange", "symbol", "side",
        "order_type", "quantity", "price", "status", "filled_quantity",
        "remaining_quantity", "average_fill_price", "created_at", "updated_at", "data",
    ]),
    ("fills", [
        "fill_id", "order_id", "client_order_id", "exchange", "symbol", "side",
        "quantity", "price", "fee", "fee_currency", "timestamp", "data",
    ]),
    ("positions_snapshot", [
        "id", "exchange", "symbol", "quantity", "avg_entry_price",
        "current_price", "unrealized_pnl", "timestamp", "data",
    ]),
    ("balances_snapshot", [
        "snapshot_id", "exchange", "timestamp", "total_value_krw", "data",
    ]),
    ("decisions_log", [
        "decision_id", "intent_id", "decision", "reason", "timestamp", "data",
    ]),
    ("safety_events", [
        "event_id", "event_type", "timestamp", "description", "severity",
        "triggered_by", "data",
    ]),
    ("news", [
        "id", "message_id", "content", "published_at", "fetched_at", "source",
    ]),
    ("news_summaries", ["id", "summary", "news_count", "created_at"]),
]

# Columns that store JSON and need conversion to proper JSONB
JSON_COLUMNS = {"data"}

# Tables with SERIAL primary keys that need sequence resets
SERIAL_TABLES = ["events", "positions_snapshot", "news", "news_summaries"]


def _convert_row(columns: list[str], row: sqlite3.Row) -> list[object]:
    """Convert a SQLite row to a list of values suitable for asyncpg."""
    values: list[object] = []
    for col in columns:
        val = row[col]
        # JSONB columns: keep as string — asyncpg handles via ::jsonb cast
        values.append(val)
    return values


def _build_insert_sql(table_name: str, columns: list[str]) -> str:
    """Build INSERT SQL with ::jsonb casts for JSON columns."""
    col_list = ", ".join(columns)
    placeholders: list[str] = []
    for i, col in enumerate(columns):
        p = f"${i + 1}"
        if col in JSON_COLUMNS:
            p += "::jsonb"
        placeholders.append(p)
    placeholder_str = ", ".join(placeholders)
    return f"INSERT INTO {table_name} ({col_list}) VALUES ({placeholder_str}) ON CONFLICT DO NOTHING"


async def migrate(sqlite_path: str, pg_url: str) -> None:
    # 1. Open SQLite
    sqlite_db = Path(sqlite_path)
    if not sqlite_db.exists():
        print(f"SQLite database not found: {sqlite_db}")
        return

    sq_conn = sqlite3.connect(str(sqlite_db))
    sq_conn.row_factory = sqlite3.Row

    # 2. Connect to PostgreSQL and ensure schema exists
    from coin_trader.state.store import StateStore

    store = await StateStore.create(pg_url)

    pool = store._pool

    # 3. Migrate each table
    for table_name, columns in TABLES:
        # Check if table exists in SQLite
        check = sq_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        if check is None:
            print(f"  SKIP {table_name} (not in SQLite)")
            continue

        col_list = ", ".join(columns)
        rows = sq_conn.execute(f"SELECT {col_list} FROM {table_name}").fetchall()  # noqa: S608
        sqlite_count = len(rows)

        if sqlite_count == 0:
            print(f"  SKIP {table_name} (0 rows)")
            continue

        # Build INSERT with ON CONFLICT DO NOTHING to allow re-runs
        insert_sql = _build_insert_sql(table_name, columns)

        async with pool.acquire() as conn:
            batch: list[list[object]] = []
            for row in rows:
                batch.append(_convert_row(columns, row))

            # Use executemany for bulk insert
            await conn.executemany(insert_sql, batch)

            # Verify row count
            pg_count = await conn.fetchval(f"SELECT count(*) FROM {table_name}")  # noqa: S608
            print(f"  {table_name}: SQLite={sqlite_count} -> PG={pg_count}")
            assert pg_count >= sqlite_count, (
                f"Row count mismatch for {table_name}: expected >= {sqlite_count}, got {pg_count}"
            )

    # 4. Reset SERIAL sequences
    async with pool.acquire() as conn:
        for table_name in SERIAL_TABLES:
            seq_name = await conn.fetchval(
                "SELECT pg_get_serial_sequence($1, 'id')", table_name
            )
            if seq_name:
                max_id = await conn.fetchval(
                    f"SELECT COALESCE(MAX(id), 0) FROM {table_name}"  # noqa: S608
                )
                if max_id and max_id > 0:
                    await conn.execute(
                        f"SELECT setval($1, $2)", seq_name, max_id  # noqa: S608
                    )
                    print(f"  sequence {seq_name} reset to {max_id}")

    await store.close()
    sq_conn.close()
    print("\nMigration complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate SQLite to PostgreSQL")
    parser.add_argument(
        "--sqlite",
        default="data/coin_trader.db",
        help="Path to SQLite database (default: data/coin_trader.db)",
    )
    parser.add_argument(
        "--pg-url",
        default="postgresql://trader:trader@localhost:5432/coin_trader",
        help="PostgreSQL connection URL",
    )
    args = parser.parse_args()

    print(f"Migrating from {args.sqlite} to {args.pg_url}")
    asyncio.run(migrate(args.sqlite, args.pg_url))


if __name__ == "__main__":
    main()
