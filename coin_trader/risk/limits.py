"""Risk limits - immutable constants for safety"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class RiskLimits:
    """Hardcoded risk parameters - IMMUTABLE at runtime"""

    max_position_size_pct: Decimal = Decimal("10")
    max_positions: int = 10
    daily_max_drawdown_pct: Decimal = Decimal("5")
    stop_loss_pct: Decimal = Decimal("3")
    take_profit_pct: Decimal = Decimal("10")
    trailing_stop_enabled: bool = True
    trailing_stop_pct: Decimal = Decimal("5")
    circuit_breaker_threshold_pct: Decimal = Decimal("15")
    circuit_breaker_window_min: int = 5
    max_orders_per_second: int = 1
    max_orders_per_day: int = 100
    default_order_type: str = "limit"
    futures_enabled: bool = False
    max_leverage: int = 1
    max_slippage_bps: int = 100
    max_single_coin_exposure_pct: Decimal = Decimal("25")
    max_alt_total_exposure_pct: Decimal = Decimal("70")
    atr_stop_multiplier: Decimal = Decimal("1.5")
    max_notional_per_position: Decimal = Decimal("50000")
    max_total_notional: Decimal = Decimal("200000")
    liquidation_warning_threshold_pct: Decimal = Decimal("20")
    max_funding_rate_bps: int = 50


DEFAULT_LIMITS = RiskLimits()
