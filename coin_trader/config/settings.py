from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Self, TypeVar

if TYPE_CHECKING:
    T = TypeVar("T")

    class SettingsConfigDict(dict[str, object]):
        pass

    class BaseSettings:  # noqa: D101
        def __init__(self, **_data: object) -> None:  # noqa: D107
            raise NotImplementedError

        @classmethod
        def model_construct(cls, **_values: object) -> Self:  # noqa: D102
            raise NotImplementedError

    def Field(
        _default: object = ..., *, _expected_type: type[T] | None = None, **_kwargs: object
    ) -> T:  # noqa: D103
        raise NotImplementedError

else:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict

from coin_trader.core.models import ExchangeName, TradingMode


@dataclass(frozen=True)
class RiskLimits:
    """Hardcoded risk limits - IMMUTABLE at runtime"""

    max_position_size_pct: Decimal = Decimal("10")
    max_positions: int = 5
    daily_max_drawdown_pct: Decimal = Decimal("5")
    soft_stop_loss_pct: Decimal = Decimal("5")
    stop_loss_pct: Decimal = Decimal("10")
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
    max_slippage_bps: int = 50


class Settings(BaseSettings):
    # Mode - FAIL CLOSED to paper
    trading_mode: TradingMode = Field(default=TradingMode.PAPER)
    live_mode_token_path: Path = Field(default=Path("RUN/live_mode_token.txt"))

    # Exchange
    exchange: ExchangeName = Field(default=ExchangeName.UPBIT)
    trading_symbols: list[str] = Field(default=["BTC/KRW", "ETH/KRW"])

    # Upbit
    upbit_key_file: Path = Field(default=Path("data/upbit_keys.enc"))
    upbit_master_key: str = Field(default="")

    # Binance
    binance_key_file: Path = Field(default=Path("data/binance_keys.enc"))
    binance_master_key: str = Field(default="")

    # LLM
    llm_enabled: bool = Field(default=False)
    llm_auth_mode: str = Field(default="api_key")  # "api_key" or "oauth"
    llm_provider: str = Field(default="openai")
    llm_api_key: str = Field(default="")
    llm_model: str = Field(default="gpt-4o-mini")

    # LLM OAuth (when llm_auth_mode=oauth)
    llm_oauth_auth_file: Path = Field(default=Path("data/.auth/openai-oauth.json"))
    llm_oauth_model: str = Field(default="gpt-5.2-codex")
    llm_oauth_open_browser: bool = Field(default=True)
    llm_oauth_force_login: bool = Field(default=False)

    # LLM trading integration
    llm_trading_enabled: bool = Field(default=False)
    llm_min_confidence: float = Field(default=0.7)
    llm_solo_min_confidence: float = Field(default=0.8)
    llm_recent_candles_count: int = Field(default=24)
    llm_recent_orders_count: int = Field(default=10)

    # Stale order auto-cancel
    stale_order_timeout_sec: int = Field(default=300)

    # Data
    db_path: Path = Field(default=Path("data/coin_trader.db"))
    market_data_interval_sec: int = Field(default=60)

    # Logging
    log_level: str = Field(default="INFO")
    log_dir: Path = Field(default=Path("logs"))

    # Kill switch
    kill_switch_file: Path = Field(default=Path("RUN/kill_switch"))

    # Web auth
    web_master_code: str = Field(default="")

    # Notifications
    slack_webhook_url: str = Field(default="")

    # Risk (immutable, loaded once)
    risk: RiskLimits = Field(default_factory=RiskLimits)

    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    def is_live_mode(self) -> bool:
        """Check if live trading is enabled with all safety gates"""
        if self.trading_mode != TradingMode.LIVE:
            return False

        # 2nd gate: token file must exist and be readable
        if not self.live_mode_token_path.exists():
            return False

        try:
            token_content = self.live_mode_token_path.read_text().strip()
            if not token_content:
                return False

            # Token must contain "ARMED" (simple but effective 2-step)
            return "ARMED" in token_content
        except Exception:
            return False  # fail closed

    @classmethod
    def load_safe(cls) -> "Settings":
        """Load settings with fail-closed behavior"""
        try:
            return cls()
        except Exception:
            # On ANY config failure, return safe defaults (paper mode)
            return cls.model_construct(
                trading_mode=TradingMode.PAPER,
                exchange=ExchangeName.UPBIT,
                trading_symbols=["BTC/KRW"],
                db_path=Path("data/coin_trader.db"),
                log_level="INFO",
                log_dir=Path("logs"),
                kill_switch_file=Path("RUN/kill_switch"),
                risk=RiskLimits(),
            )
