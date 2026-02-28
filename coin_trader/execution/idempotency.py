"""Idempotency manager - prevents duplicate orders."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID


class _Store(Protocol):
    async def order_exists(self, client_order_id: str) -> bool: ...


class IdempotencyManager:
    store: _Store | None

    def __init__(self, state_store: _Store | None = None) -> None:
        self._processed: set[str] = set()
        self.store = state_store

    def generate_key(self, intent_id: UUID) -> str:
        return f"intent_{intent_id}"

    async def is_duplicate(self, client_order_id: str) -> bool:
        if client_order_id in self._processed:
            return True

        if self.store is not None:
            try:
                if await self.store.order_exists(client_order_id):
                    return True
            except Exception:
                pass

        return False

    def mark_processed(self, client_order_id: str) -> None:
        self._processed.add(client_order_id)
