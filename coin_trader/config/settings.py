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
    max_positions: int = 10
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
    max_slippage_bps: int = 100
    max_single_coin_exposure_pct: Decimal = Decimal("25")
    max_alt_total_exposure_pct: Decimal = Decimal("70")
    atr_stop_multiplier: Decimal = Decimal("1.5")
    max_notional_per_position: Decimal = Decimal("50000")
    max_total_notional: Decimal = Decimal("200000")
    liquidation_warning_threshold_pct: Decimal = Decimal("20")
    max_funding_rate_bps: int = 50


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
    binance_testnet: bool = Field(default=True)
    binance_margin_type: str = Field(default="isolated")
    binance_default_leverage: int = Field(default=1)

    # Exchange-agnostic
    quote_currency: str = Field(default="KRW")
    btc_reference_symbol: str = Field(default="BTC/KRW")

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
    llm_min_confidence: float = Field(default=0.65)
    llm_recent_candles_count: int = Field(default=24)
    llm_recent_orders_count: int = Field(default=10)

    # Stale order auto-cancel
    stale_order_timeout_sec: int = Field(default=300)

    # Data
    db_path: Path = Field(default=Path("data/coin_trader.db"))
    market_data_interval_sec: int = Field(default=120)
    candle_refresh_interval_sec: int = Field(default=3600)

    dynamic_symbol_selection_enabled: bool = Field(default=True)
    dynamic_symbol_refresh_sec: int = Field(default=3600)
    dynamic_symbol_top_k: int = Field(default=5)
    dynamic_symbol_max_symbols: int = Field(default=10)
    dynamic_symbol_min_krw_24h: int = Field(default=1_000_000_000)
    dynamic_symbol_min_turnover_24h: int = Field(default=1_000_000_000)
    dynamic_symbol_batch_size: int = Field(default=80)
    dynamic_symbol_max_spread_bps: Decimal = Field(default=Decimal("80"))
    dynamic_symbol_max_abs_change_24h_pct: Decimal = Field(default=Decimal("20"))
    dynamic_symbol_max_intraday_range_pct: Decimal = Field(default=Decimal("30"))
    always_keep_symbols: str = Field(default="")

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

    def apply_exchange_defaults(self) -> None:
        """Auto-set exchange-specific defaults when exchange is BINANCE."""
        if self.exchange == ExchangeName.BINANCE:
            if self.quote_currency == "KRW":
                object.__setattr__(self, "quote_currency", "USDT")
            if self.btc_reference_symbol == "BTC/KRW":
                object.__setattr__(self, "btc_reference_symbol", "BTC/USDT")
            if self.trading_symbols == ["BTC/KRW", "ETH/KRW"]:
                object.__setattr__(self, "trading_symbols", ["BTC/USDT", "ETH/USDT"])
            if self.dynamic_symbol_min_turnover_24h == 1_000_000_000:
                object.__setattr__(self, "dynamic_symbol_min_turnover_24h", 10_000_000)

    def get_always_keep_symbols(self) -> list[str]:
        parts = [p.strip().upper() for p in self.always_keep_symbols.split(",") if p.strip()]
        out: list[str] = []
        for symbol in parts:
            if "/" not in symbol:
                continue
            if symbol not in out:
                out.append(symbol)
        return out

    @classmethod
    def load_safe(cls) -> "Settings":
        """Load settings with fail-closed behavior"""
        try:
            settings = cls()
            settings.apply_exchange_defaults()
            return settings
        except Exception:
            # On ANY config failure, return safe defaults (paper mode)
            return cls.model_construct(
                trading_mode=TradingMode.PAPER,
                exchange=ExchangeName.UPBIT,
                trading_symbols=["BTC/KRW"],
                quote_currency="KRW",
                btc_reference_symbol="BTC/KRW",
                db_path=Path("data/coin_trader.db"),
                log_level="INFO",
                log_dir=Path("logs"),
                kill_switch_file=Path("RUN/kill_switch"),
                risk=RiskLimits(),
            )
