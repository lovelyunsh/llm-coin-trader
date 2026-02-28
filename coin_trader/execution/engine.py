"""Execution engine - coordinates intent -> risk check -> order placement."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from coin_trader.core.models import Order, OrderIntent, RiskDecision
from coin_trader.execution.idempotency import IdempotencyManager
from coin_trader.logging.logger import get_logger

logger = get_logger("execution")


class _Broker(Protocol):
    async def place(self, intent: OrderIntent, client_order_id: str) -> Order: ...

    async def cancel(self, order_id: str) -> None: ...

    async def fetch_open_orders(self) -> list[Order]: ...


class _RiskManager(Protocol):
    async def validate(
        self, intent: OrderIntent, state: dict[str, object]
    ) -> object: ...


class _StateStore(Protocol):
    async def save_decision(self, decision: object) -> None: ...

    async def save_order(self, order: Order) -> None: ...


class _KillSwitch(Protocol):
    def is_active(self) -> bool: ...


class ExecutionEngine:
    broker: _Broker
    risk: _RiskManager
    idempotency: IdempotencyManager
    store: _StateStore
    kill_switch: _KillSwitch

    def __init__(
        self,
        broker: _Broker,
        risk_manager: _RiskManager,
        idempotency: IdempotencyManager,
        state_store: _StateStore,
        kill_switch: _KillSwitch,
    ) -> None:
        self.broker = broker
        self.risk = risk_manager
        self.idempotency = idempotency
        self.store = state_store
        self.kill_switch = kill_switch

    async def execute(
        self, intent: OrderIntent, state: dict[str, object]
    ) -> Order | None:
        """Execute an order intent through the full safety pipeline.

        Pipeline: kill_switch -> idempotency -> risk_check -> place_order -> persist
        """
        if self.kill_switch.is_active():
            logger.warning(
                "execution_blocked", symbol=intent.symbol, reason="kill_switch"
            )
            return None

        client_order_id = self.idempotency.generate_key(intent.intent_id)
        if await self.idempotency.is_duplicate(client_order_id):
            return None

        decision = await self.risk.validate(intent, state)
        await self.store.save_decision(decision)

        if getattr(decision, "decision", None) == RiskDecision.REJECTED:
            logger.info(
                "execution_rejected",
                symbol=intent.symbol,
                side=intent.side.value,
                reason=getattr(decision, "reason", "risk_rejected"),
            )
            return None

        try:
            order = await self.broker.place(intent, client_order_id)
            await self.store.save_order(order)
            self.idempotency.mark_processed(client_order_id)

            logger.info(
                "order_placed",
                symbol=order.symbol,
                side=order.side.value,
                qty=str(order.quantity),
                price=str(order.price),
            )
            return order
        except Exception as e:
            logger.error("order_failed", symbol=intent.symbol, error=str(e))
            raise

    async def cancel_stale_orders(self, timeout_sec: int) -> list[Order]:
        open_orders = await self.broker.fetch_open_orders()
        now = datetime.now(timezone.utc)
        cancelled: list[Order] = []
        for order in open_orders:
            age_sec = (now - order.created_at).total_seconds()
            if age_sec < timeout_sec:
                continue
            try:
                await self.broker.cancel(order.order_id or order.client_order_id)
                cancelled.append(order)
                logger.info(
                    "stale_order_cancelled",
                    symbol=order.symbol,
                    order_id=order.order_id,
                    age_sec=int(age_sec),
                )
            except Exception as e:
                logger.error("cancel_failed", order_id=order.order_id, error=str(e))
        return cancelled

    async def cancel_all_open_orders(self) -> int:
        open_orders = await self.broker.fetch_open_orders()
        cancelled = 0
        for order in open_orders:
            try:
                await self.broker.cancel(order.order_id or order.client_order_id)
                cancelled += 1
                logger.info(
                    "order_cancelled", symbol=order.symbol, order_id=order.order_id
                )
            except Exception as e:
                logger.error("cancel_failed", order_id=order.order_id, error=str(e))
        return cancelled
