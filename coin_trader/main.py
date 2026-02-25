"""Coin Trader - Safety-first automated crypto trading system.

CLI entrypoint and main trading loop.
Usage:
    trader run [--mode paper|live] [--once]
    trader kill [--close-positions]
    trader selftest
    trader encrypt-keys --exchange upbit --master-key <key>
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import click

from coin_trader.config.settings import RiskLimits, Settings
from coin_trader.core.models import (
    ExchangeName,
    MarketData,
    OrderIntent,
    OrderSide,
    OrderType,
    PositionSide,
    SignalType,
    TradingMode,
)
from coin_trader.execution.engine import ExecutionEngine
from coin_trader.execution.idempotency import IdempotencyManager
from coin_trader.logging.logger import (
    close_logging,
    get_logger,
    log_event,
    setup_logging,
)
from coin_trader.risk.manager import RiskManager
from coin_trader.safety.kill_switch import KillSwitch
from coin_trader.safety.monitor import AnomalyMonitor
from coin_trader.state.store import StateStore

logger = get_logger("main")

_high_watermarks: dict[str, Decimal] = {}
_low_watermarks: dict[str, Decimal] = {}  # for short trailing stops
_last_llm_prices: dict[str, Decimal] = {}
_last_llm_times: dict[str, float] = {}
# LLM skip: 가격 변동 < 1% AND 경과 < 30분이면 LLM 호출 생략 (API 비용 절감)
# 단, 소프트 손절/익절 구간이면 강제 호출 (main.py 내 force override 참고)
_LLM_SKIP_THRESHOLD_PCT = Decimal("1.0")
_LLM_MAX_SKIP_SECONDS = 1800.0
_recent_decisions: dict[str, list[dict[str, str]]] = {}
_MAX_DECISION_HISTORY = 5
# 심볼별 매수 쿨다운: 동일 코인 반복매수 방지 (RENDER 49분 3회, ETH 43분 5회 문제 해결)
_last_buy_ts: dict[str, float] = {}
_BUY_COOLDOWN_SECONDS = 1800.0
# 최소 주문금액: 수수료(0.05% x 2) 대비 의미없는 소액거래 차단
_MIN_ORDER_VALUE_KRW = Decimal("5000")

_btc_trend_cache: dict[str, object] = {}
_btc_trend_cache_ts: float = 0.0
_BTC_TREND_CACHE_TTL = 900.0  # 15 min — daily candles change slowly
_btc_reference_symbol = "BTC/KRW"  # set from settings at startup
_btc_daily_candles: list[dict[str, object]] = []
_btc_daily_candles_ts: float = 0.0
_BTC_DAILY_CANDLE_REFRESH = 3600.0  # refetch daily candles hourly
_btc_daily_strategy: Any = None
_symbol_candle_refresh_ts: dict[str, float] = {}
_base_candle_cache: dict[str, list[dict[str, object]]] = {}
_dynamic_symbols_cache: list[str] = []
_dynamic_symbols_cache_ts: float = 0.0
_previous_universe_selections: list[dict[str, object]] = []
_MAX_UNIVERSE_HISTORY = 3


def _build_system(
    settings: Settings, *, install_signal_handlers: bool = False
) -> dict[str, Any]:
    setup_logging(log_dir=settings.log_dir, log_level=settings.log_level)
    store = StateStore(db_path=settings.db_path)
    kill_switch = KillSwitch(
        settings.kill_switch_file, install_signal_handlers=install_signal_handlers
    )
    risk_manager = RiskManager(limits=settings.risk, state_store=store)
    anomaly_monitor = AnomalyMonitor(
        config={
            "api_failure_threshold": 5,
            "balance_mismatch_threshold_pct": 1.0,
            "price_change_threshold_pct": 10.0,
            "spread_threshold_pct": 5.0,
        }
    )

    idempotency = IdempotencyManager(state_store=store)
    exchange_adapter = None
    broker = None

    global _btc_reference_symbol
    _btc_reference_symbol = settings.btc_reference_symbol

    if settings.exchange == ExchangeName.UPBIT:
        if settings.is_live_mode():
            from coin_trader.security.key_manager import KeyManager

            access_key, secret_key = KeyManager.decrypt_keys(
                settings.upbit_key_file, settings.upbit_master_key
            )
            from coin_trader.exchange.upbit import UpbitAdapter

            exchange_adapter = UpbitAdapter(
                access_key=access_key, secret_key=secret_key
            )

            from coin_trader.broker.live import LiveBroker

            broker = LiveBroker(
                exchange=ExchangeName.UPBIT,
                exchange_adapter=exchange_adapter,
                trading_symbols=settings.trading_symbols,
            )
            logger.warning("live_mode_active", exchange="upbit")
        else:
            from coin_trader.exchange.upbit import UpbitAdapter

            exchange_adapter = UpbitAdapter(access_key="", secret_key="")

            from coin_trader.broker.paper import PaperBroker

            broker = PaperBroker(
                exchange=ExchangeName.UPBIT,
                exchange_adapter=exchange_adapter,
                initial_balance_krw=Decimal("1000000"),
            )
            logger.info("paper_mode_active", exchange="upbit")

    elif settings.exchange == ExchangeName.BINANCE:
        from coin_trader.exchange.binance_futures import BinanceFuturesAdapter

        if settings.is_live_mode():
            from coin_trader.security.key_manager import KeyManager

            api_key, api_secret = KeyManager.decrypt_keys(
                settings.binance_key_file, settings.binance_master_key
            )
            exchange_adapter = BinanceFuturesAdapter(
                api_key=api_key,
                api_secret=api_secret,
                testnet=settings.binance_testnet,
            )

            from coin_trader.broker.live_futures import LiveFuturesBroker

            broker = LiveFuturesBroker(
                exchange=ExchangeName.BINANCE,
                exchange_adapter=exchange_adapter,
                trading_symbols=settings.trading_symbols,
                quote_currency=settings.quote_currency,
                default_leverage=settings.binance_default_leverage,
                margin_type=settings.binance_margin_type,
            )
            logger.warning(
                "live_mode_active",
                exchange="binance",
                testnet=settings.binance_testnet,
            )
        else:
            exchange_adapter = BinanceFuturesAdapter(
                api_key="",
                api_secret="",
                testnet=True,
            )

            from coin_trader.broker.paper_futures import PaperFuturesBroker

            broker = PaperFuturesBroker(
                exchange=ExchangeName.BINANCE,
                exchange_adapter=exchange_adapter,
                initial_balance_usdt=Decimal("10000"),
                quote_currency=settings.quote_currency,
                default_leverage=settings.binance_default_leverage,
            )
            logger.info("paper_mode_active", exchange="binance", testnet=True)

    if broker is None:
        raise RuntimeError(f"Unsupported exchange: {settings.exchange}")

    engine = ExecutionEngine(
        broker=broker,
        risk_manager=risk_manager,
        idempotency=idempotency,
        state_store=store,
        kill_switch=kill_switch,
    )

    from coin_trader.strategy.conservative import ConservativeStrategy

    strategy = ConservativeStrategy(futures_enabled=settings.risk.futures_enabled)

    llm_advisor = None
    if settings.llm_enabled:
        from coin_trader.llm.advisory import LLMAdvisor

        _futures = settings.risk.futures_enabled
        _exchange_ctx = settings.exchange.value

        if settings.llm_auth_mode == "oauth":
            llm_advisor = LLMAdvisor.create_oauth(
                auth_file=settings.llm_oauth_auth_file,
                model=settings.llm_oauth_model,
                open_browser=settings.llm_oauth_open_browser,
                force_login=settings.llm_oauth_force_login,
            )
            llm_advisor._futures_enabled = _futures
            llm_advisor._exchange_context = _exchange_ctx
        elif settings.llm_api_key:
            llm_advisor = LLMAdvisor(
                api_key=settings.llm_api_key,
                provider=settings.llm_provider,
                model=settings.llm_model,
                futures_enabled=_futures,
                exchange_context=_exchange_ctx,
            )

    notifier = None
    if settings.slack_webhook_url:
        from coin_trader.notify.slack import SlackNotifier

        notifier = SlackNotifier(webhook_url=settings.slack_webhook_url)

    return {
        "settings": settings,
        "store": store,
        "kill_switch": kill_switch,
        "risk_manager": risk_manager,
        "anomaly_monitor": anomaly_monitor,
        "engine": engine,
        "broker": broker,
        "exchange_adapter": exchange_adapter,
        "strategy": strategy,
        "llm_advisor": llm_advisor,
        "notifier": notifier,
    }


def _resolve_action(
    signals: list[Any],
    advice: Any | None,
    settings: Settings,
) -> str:
    """LLM-primary decision model. Strategy signals are reference data only.

    LLM BUY_CONSIDER  (confidence >= min)  -> BUY
    LLM SELL_CONSIDER (confidence >= min)  -> SELL
    LLM HOLD                               -> HOLD
    No LLM / LLM disabled                  -> follow strategy as fallback
    """
    strategy_type = signals[0].signal_type if signals else SignalType.HOLD

    if not settings.llm_trading_enabled:
        return strategy_type.value.upper()

    if advice is None:
        # LLM skipped or failed: hold position.
        # Hard stop-loss / trailing stop are handled independently in position_protection (section 8).
        return "HOLD"

    llm_action = advice.action
    llm_conf = float(advice.confidence)

    if llm_conf < settings.llm_min_confidence:
        return "HOLD"

    action_map = {
        "BUY_CONSIDER": "BUY",
        "SELL_CONSIDER": "SELL",
        "SHORT_CONSIDER": "SHORT",
        "COVER_CONSIDER": "COVER",
    }
    mapped = action_map.get(llm_action)
    if mapped:
        if mapped in ("SHORT", "COVER") and not settings.risk.futures_enabled:
            return "HOLD"
        return mapped

    return "HOLD"


def _compute_portfolio_exposure(
    positions: list[Any],
    total_balance: Decimal,
) -> dict[str, object]:
    if total_balance <= 0:
        return {}
    exposure: dict[str, object] = {}
    alt_total = Decimal("0")
    for p in positions:
        if p.quantity <= 0 or p.current_price is None:
            continue
        value = p.quantity * p.current_price
        pct = value / total_balance * Decimal("100")
        coin = p.symbol.split("/")[0] if "/" in p.symbol else p.symbol
        exposure[f"{coin}_pct"] = str(round(pct, 2))
        if coin != "BTC":
            alt_total += pct
    exposure["alt_total_pct"] = str(round(alt_total, 2))
    return exposure


def _compute_recent_structure(
    candles: list[Any],
    lookback: int = 20,
) -> dict[str, object]:
    if len(candles) < lookback:
        return {
            "higher_high": False,
            "higher_low": False,
            "lower_high": False,
            "lower_low": False,
            "range_market": True,
        }

    recent = candles[-lookback:]
    highs = [float(str(c.get("high", 0) or 0)) for c in recent]
    lows = [float(str(c.get("low", 0) or 0)) for c in recent]

    mid = len(highs) // 2
    first_half_high = max(highs[:mid]) if highs[:mid] else 0
    second_half_high = max(highs[mid:]) if highs[mid:] else 0
    first_half_low = min(lows[:mid]) if lows[:mid] else 0
    second_half_low = min(lows[mid:]) if lows[mid:] else 0

    hh = second_half_high > first_half_high
    hl = second_half_low > first_half_low
    lh = second_half_high < first_half_high
    ll = second_half_low < first_half_low

    is_range = not hh and not ll

    return {
        "higher_high": hh,
        "higher_low": hl,
        "lower_high": lh,
        "lower_low": ll,
        "range_market": is_range,
    }


def _compute_volume_context(
    candles: list[Any],
    window: int = 10,
) -> dict[str, object]:
    if len(candles) < window + 1:
        return {"volume_vs_avg_ratio": 1.0, "volume_trend": "flat"}

    volumes = [float(str(c.get("volume", 0) or 0)) for c in candles]
    recent = volumes[-window:]
    avg = sum(recent) / len(recent) if recent else 1.0
    current = volumes[-1] if volumes else 0.0
    ratio = round(current / avg, 2) if avg > 0 else 0.0

    first_half = sum(recent[: window // 2]) / (window // 2) if window >= 2 else avg
    second_half = (
        sum(recent[window // 2 :]) / (window - window // 2) if window >= 2 else avg
    )
    threshold = 0.1
    if second_half > first_half * (1 + threshold):
        trend = "increasing"
    elif second_half < first_half * (1 - threshold):
        trend = "decreasing"
    else:
        trend = "flat"

    return {"volume_vs_avg_ratio": ratio, "volume_trend": trend}


async def _compute_btc_trend(
    exchange_adapter: Any,
    strategy: Any = None,
    current_symbol: str = "",
) -> dict[str, object] | None:
    """Compute BTC macro trend using daily candles (true EMA200 = 200-day).

    ``strategy`` and ``current_symbol`` are accepted for backward compatibility
    but no longer used internally.
    """
    _ = strategy, current_symbol  # keep signature; silence unused warnings
    global _btc_trend_cache, _btc_trend_cache_ts
    global _btc_daily_candles, _btc_daily_candles_ts, _btc_daily_strategy

    now = time.time()
    if now - _btc_trend_cache_ts < _BTC_TREND_CACHE_TTL and _btc_trend_cache:
        return _btc_trend_cache  # type: ignore[return-value]

    try:
        # Fetch daily candles (refresh hourly) — separate from hourly trading candles
        if (
            not _btc_daily_candles
            or now - _btc_daily_candles_ts >= _BTC_DAILY_CANDLE_REFRESH
        ):
            _btc_daily_candles = await exchange_adapter.get_candles(
                _btc_reference_symbol, interval="1d", count=200
            )
            _btc_daily_candles_ts = now

        if not _btc_daily_candles:
            return None

        # Use separate strategy instance to avoid polluting hourly candle history
        if _btc_daily_strategy is None:
            from coin_trader.strategy.conservative import ConservativeStrategy

            _btc_daily_strategy = ConservativeStrategy()

        _btc_daily_strategy.update_candles(_btc_reference_symbol, _btc_daily_candles)
        btc_indicators = _btc_daily_strategy.compute_indicators(_btc_reference_symbol)
        if btc_indicators is None:
            return None

        result: dict[str, object] = {
            "timeframe": "daily",
            "price_above_ema200": btc_indicators.current_price > btc_indicators.ema200,
            "rsi": btc_indicators.rsi,
            "adx": btc_indicators.adx,
            "plus_di": btc_indicators.plus_di,
            "minus_di": btc_indicators.minus_di,
            "ema200": btc_indicators.ema200,
            "current_price": btc_indicators.current_price,
            "trend_strength": btc_indicators.trend_strength,
        }
        _btc_trend_cache = result  # type: ignore[assignment]
        _btc_trend_cache_ts = now
        return result
    except Exception:
        return None


def _chunk_symbols(symbols: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        return [symbols]
    return [symbols[i : i + size] for i in range(0, len(symbols), size)]


async def _fetch_batch_tickers(
    exchange_adapter: Any, symbols: list[str]
) -> dict[str, dict[str, object]]:
    if not symbols or not hasattr(exchange_adapter, "get_tickers"):
        return {}
    try:
        raw = await exchange_adapter.get_tickers(symbols)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _should_refresh_candles(symbol: str, settings: Settings) -> bool:
    last = _symbol_candle_refresh_ts.get(symbol)
    if last is None:
        return True
    # Refresh when a new UTC hour starts (= new completed 1h candle available)
    now_dt = datetime.now(timezone.utc)
    last_dt = datetime.fromtimestamp(last, tz=timezone.utc)
    current_hour_start = now_dt.replace(minute=0, second=0, microsecond=0)
    return last_dt < current_hour_start


def _build_synthetic_candle(
    ticker: dict[str, object],
    last_candle: dict[str, object] | None,
) -> dict[str, object]:
    """Build a synthetic in-progress candle from live ticker data.

    Uses the last completed candle's close as open, current ticker price as close.
    Keeps indicators current without additional API calls.
    """
    trade_price = float(str(ticker.get("price", 0)))
    if last_candle is not None:
        raw = last_candle.get("close", trade_price)
        synthetic_open = float(str(raw or trade_price))
    else:
        synthetic_open = trade_price

    return {
        "open": synthetic_open,
        "high": max(synthetic_open, trade_price),
        "low": min(synthetic_open, trade_price),
        "close": trade_price,
        "volume": 0,
    }


def _compute_universe_candidate_k(final_top_n: int) -> int:
    return min(20, max(12, final_top_n * 3))


def _compute_candidate_score(
    metrics: dict[str, object],
    turnover: Decimal,
    max_turnover: Decimal,
) -> Decimal:
    """Composite score for candidate ranking. Higher = better tradability.

    Weights:
      - liquidity  25%: turnover relative to market leader
      - spread     20%: tighter spread = better execution
      - momentum   20%: moderate positive change (1-8%) preferred
      - volatility 15%: 3-10% intraday range = tradable
      - pullback   20%: 2-10% from 24h high = entry opportunity
    """
    spread = _safe_decimal(metrics.get("spread_bps", 0))
    change = _safe_decimal(metrics.get("change_24h_pct", 0))
    range_pct = _safe_decimal(metrics.get("intraday_range_pct", 0))
    dist_high = _safe_decimal(metrics.get("distance_from_high_pct", 0))

    # Liquidity: normalized against top coin
    liquidity = turnover / max_turnover if max_turnover > 0 else Decimal("0")

    # Spread: lower is better (0→1.0, 100bps→0.0)
    spread_score = max(Decimal("0"), Decimal("1") - spread / Decimal("100"))

    # Momentum: 1-8% ideal, <1% too quiet, >8% chasing risk
    abs_change = abs(change)
    if Decimal("1") <= abs_change <= Decimal("8"):
        momentum = Decimal("1")
    elif abs_change < Decimal("1"):
        momentum = abs_change
    else:
        momentum = max(
            Decimal("0"), Decimal("1") - (abs_change - Decimal("8")) / Decimal("20")
        )

    # Volatility: 3-10% range ideal for short-term trading
    if Decimal("3") <= range_pct <= Decimal("10"):
        volatility = Decimal("1")
    elif range_pct < Decimal("3"):
        volatility = range_pct / Decimal("3")
    else:
        volatility = max(
            Decimal("0"), Decimal("1") - (range_pct - Decimal("10")) / Decimal("15")
        )

    # Pullback: 2-10% from high = healthy entry, 0% = buying at top
    if Decimal("2") <= dist_high <= Decimal("10"):
        pullback = Decimal("1")
    elif dist_high < Decimal("2"):
        pullback = dist_high / Decimal("2")
    else:
        pullback = max(
            Decimal("0"), Decimal("1") - (dist_high - Decimal("10")) / Decimal("15")
        )

    return (
        Decimal("0.25") * liquidity
        + Decimal("0.20") * spread_score
        + Decimal("0.20") * momentum
        + Decimal("0.15") * volatility
        + Decimal("0.20") * pullback
    )


def _safe_decimal(value: object, default: Decimal = Decimal("0")) -> Decimal:
    try:
        if value is None:
            return default
        return Decimal(str(value))
    except Exception:
        return default


async def _refresh_dynamic_symbols(components: dict[str, Any]) -> list[str]:
    global _dynamic_symbols_cache, _dynamic_symbols_cache_ts, _previous_universe_selections

    settings: Settings = components["settings"]
    exchange_adapter = components.get("exchange_adapter")
    broker = components.get("broker")

    if not settings.dynamic_symbol_selection_enabled:
        return list(settings.trading_symbols)

    now = time.time()
    if _dynamic_symbols_cache and now - _dynamic_symbols_cache_ts < float(
        settings.dynamic_symbol_refresh_sec
    ):
        settings.trading_symbols = list(_dynamic_symbols_cache)
        return list(_dynamic_symbols_cache)

    if exchange_adapter is None or not hasattr(
        exchange_adapter, "get_tradeable_markets"
    ):
        return list(settings.trading_symbols)

    markets_raw = await exchange_adapter.get_tradeable_markets()
    markets = [m for m in markets_raw if isinstance(m, str)]
    if not markets:
        return list(settings.trading_symbols)

    tickers: dict[str, dict[str, object]] = {}
    batch_size = max(1, int(settings.dynamic_symbol_batch_size))
    for chunk in _chunk_symbols(markets, batch_size):
        part = await _fetch_batch_tickers(exchange_adapter, chunk)
        tickers.update(part)

    min_turnover = Decimal(
        str(
            settings.dynamic_symbol_min_turnover_24h
            if settings.exchange == ExchangeName.BINANCE
            else settings.dynamic_symbol_min_krw_24h
        )
    )
    eligible_turnovers: dict[str, Decimal] = {}
    for symbol, ticker in tickers.items():
        turnover = Decimal(str(ticker.get("turnover_24h", 0)))
        if turnover < min_turnover:
            continue
        eligible_turnovers[symbol] = turnover

    top_k = max(1, int(settings.dynamic_symbol_top_k))
    candidate_k = _compute_universe_candidate_k(top_k)

    held_symbols: list[str] = []
    held_positions_data: list[dict[str, object]] = []
    if broker is not None and hasattr(broker, "fetch_positions"):
        try:
            held = await broker.fetch_positions()
            for p in held:
                if getattr(p, "quantity", Decimal("0")) <= 0:
                    continue
                held_symbols.append(p.symbol)
                ticker_data = tickers.get(p.symbol, {})
                cur_price = _safe_decimal(ticker_data.get("price", 0))
                entry_price = getattr(p, "average_entry_price", Decimal("0"))
                pnl_pct = Decimal("0")
                if entry_price > 0 and cur_price > 0:
                    pnl_pct = (cur_price - entry_price) / entry_price * Decimal("100")
                held_positions_data.append(
                    {
                        "symbol": p.symbol,
                        "unrealized_pnl_pct": str(round(pnl_pct, 2)),
                        "entry_price": str(entry_price),
                        "current_price": str(cur_price),
                    }
                )
        except Exception:
            pass

    always_keep = [s for s in settings.get_always_keep_symbols() if s.endswith("/KRW")]
    forced_symbols: list[str] = []
    for symbol in [*always_keep, *held_symbols]:
        if symbol not in forced_symbols:
            forced_symbols.append(symbol)

    max_symbols = max(1, int(settings.dynamic_symbol_max_symbols))
    llm_slots = max(0, max_symbols - len(forced_symbols))

    # Fetch orderbooks for real spread data (batch)
    orderbooks: dict[str, dict[str, object]] = {}
    if hasattr(exchange_adapter, "get_orderbooks"):
        eligible_list = list(eligible_turnovers.keys())
        for chunk in _chunk_symbols(eligible_list, batch_size):
            try:
                part = await exchange_adapter.get_orderbooks(chunk)
                orderbooks.update(part)
            except Exception:
                pass

    def _candidate_metrics(
        symbol: str,
        ticker: dict[str, object],
        orderbook: dict[str, object] | None = None,
    ) -> dict[str, object]:
        trade_price = _safe_decimal(ticker.get("price", 0))
        opening_price = _safe_decimal(ticker.get("open", 0))
        high_price = _safe_decimal(ticker.get("high", trade_price), trade_price)
        low_price = _safe_decimal(ticker.get("low", trade_price), trade_price)
        change_rate = _safe_decimal(ticker.get("change_rate", 0)) * Decimal("100")
        range_pct = Decimal("0")
        if trade_price > 0:
            range_pct = (high_price - low_price) / trade_price * Decimal("100")
        dist_from_high_pct = Decimal("0")
        if high_price > 0:
            dist_from_high_pct = (
                (high_price - trade_price) / high_price * Decimal("100")
            )

        # Extract best bid/ask from orderbook (canonical: bids/asks as [[price,qty],...])
        bid = trade_price
        ask = trade_price
        if orderbook:
            bids = orderbook.get("bids")
            asks = orderbook.get("asks")
            if isinstance(bids, list) and bids:
                top_bid = bids[0]
                if isinstance(top_bid, (list, tuple)) and top_bid:
                    bid = _safe_decimal(top_bid[0], trade_price)
            if isinstance(asks, list) and asks:
                top_ask = asks[0]
                if isinstance(top_ask, (list, tuple)) and top_ask:
                    ask = _safe_decimal(top_ask[0], trade_price)
        spread_bps = Decimal("0")
        if ask > 0:
            spread_bps = abs(ask - bid) / ask * Decimal("10000")

        return {
            "symbol": symbol,
            "change_24h_pct": str(round(change_rate, 3)),
            "intraday_range_pct": str(round(range_pct, 3)),
            "distance_from_high_pct": str(round(dist_from_high_pct, 3)),
            "spread_bps": str(round(spread_bps, 2)),
            "volume_24h": str(ticker.get("volume_24h", 0)),
            "price": str(trade_price),
            "open": str(opening_price),
        }

    spread_limit = settings.dynamic_symbol_max_spread_bps
    abs_change_limit = settings.dynamic_symbol_max_abs_change_24h_pct
    range_limit = settings.dynamic_symbol_max_intraday_range_pct
    max_turnover = (
        max(eligible_turnovers.values()) if eligible_turnovers else Decimal("1")
    )

    scored: list[tuple[str, Decimal]] = []
    for symbol, turnover in eligible_turnovers.items():
        ticker = tickers.get(symbol, {})
        metrics = _candidate_metrics(symbol, ticker, orderbooks.get(symbol))
        spread_bps = _safe_decimal(metrics["spread_bps"])
        abs_change = abs(_safe_decimal(metrics["change_24h_pct"]))
        intraday_range = _safe_decimal(metrics["intraday_range_pct"])
        if spread_bps > spread_limit:
            continue
        if abs_change > abs_change_limit:
            continue
        if intraday_range > range_limit:
            continue
        score = _compute_candidate_score(metrics, turnover, max_turnover)
        scored.append((symbol, score))
    scored.sort(key=lambda item: item[1], reverse=True)

    llm_rank_base = [symbol for symbol, _ in scored if symbol not in forced_symbols][
        :candidate_k
    ]
    if not llm_rank_base:
        fallback = sorted(eligible_turnovers.items(), key=lambda x: x[1], reverse=True)
        llm_rank_base = [s for s, _ in fallback[:top_k] if s not in forced_symbols]

    candidate_payload: list[dict[str, object]] = []
    for symbol in llm_rank_base[:candidate_k]:
        ticker = tickers.get(symbol, {})
        payload = _candidate_metrics(symbol, ticker, orderbooks.get(symbol))
        payload["turnover_24h_krw"] = str(_safe_decimal(ticker.get("turnover_24h", 0)))
        payload["is_held"] = symbol in held_symbols
        payload["was_active"] = symbol in _dynamic_symbols_cache
        candidate_payload.append(payload)

    active_non_core = [
        s for s in _dynamic_symbols_cache if s not in forced_symbols and s in tickers
    ]
    active_payload = [
        _candidate_metrics(s, tickers[s], orderbooks.get(s)) for s in active_non_core
    ]

    llm_decision: Any | None = None
    llm_advisor = components.get("llm_advisor")
    if (
        llm_slots > 0
        and llm_advisor is not None
        and hasattr(llm_advisor, "get_symbol_universe")
        and candidate_payload
    ):
        # Fetch BTC trend for market regime context
        btc_trend: dict[str, object] | None = None
        try:
            strategy = components.get("strategy")
            if strategy is not None and exchange_adapter is not None:
                btc_trend = await _compute_btc_trend(exchange_adapter, strategy, "")
        except Exception:
            pass

        try:
            llm_decision = await llm_advisor.get_symbol_universe(
                top_n=llm_slots,
                candidates=candidate_payload,
                active_symbols=active_payload,
                held_symbols=held_symbols,
                core_symbols=forced_symbols,
                btc_trend=btc_trend,
                held_positions=held_positions_data,
                previous_selections=_previous_universe_selections,
            )
        except Exception:
            logger.warning("llm_universe_call_failed", exc_info=True)

    selected: list[str] = list(forced_symbols)
    dynamic_source = llm_decision.selected_symbols if llm_decision else llm_rank_base
    for symbol in dynamic_source:
        if symbol in selected:
            continue
        selected.append(symbol)
        if len(selected) >= max_symbols:
            break

    # Track selection history for future LLM context
    _previous_universe_selections.append(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "selected": selected,
            "reasoning": llm_decision.reasoning if llm_decision else "",
        }
    )
    if len(_previous_universe_selections) > _MAX_UNIVERSE_HISTORY:
        _previous_universe_selections = _previous_universe_selections[
            -_MAX_UNIVERSE_HISTORY:
        ]

    log_event(
        "symbol_decision",
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "top_n": top_k,
            "candidate_k": candidate_k,
            "raw_eligible_count": len(eligible_turnovers),
            "scored_count": len(scored),
            "filtered_count": len(llm_rank_base),
            "llm_used": llm_decision is not None,
            "forced_symbols": forced_symbols,
            "active_symbols_before": _dynamic_symbols_cache,
            "llm_selected_symbols": (
                llm_decision.selected_symbols if llm_decision else []
            ),
            "llm_reasoning": llm_decision.reasoning if llm_decision else "",
            "llm_prompt": llm_decision.prompt if llm_decision else "",
            "selected_symbols": selected,
        },
    )

    if not selected:
        selected = list(settings.trading_symbols)

    _dynamic_symbols_cache = selected
    _dynamic_symbols_cache_ts = now
    settings.trading_symbols = list(selected)
    return selected


async def _run_tick(
    components: dict[str, Any],
    symbol: str,
    prefetched_ticker: dict[str, object] | None = None,
) -> None:
    engine: ExecutionEngine = components["engine"]
    strategy = components["strategy"]
    broker = components["broker"]
    exchange_adapter = components["exchange_adapter"]
    anomaly_monitor: AnomalyMonitor = components["anomaly_monitor"]
    risk_manager: RiskManager = components["risk_manager"]
    kill_switch: KillSwitch = components["kill_switch"]
    store: StateStore = components["store"]
    settings: Settings = components["settings"]
    notifier = components.get("notifier")

    if kill_switch.is_active():
        return

    # 1. Fetch market data
    ticker = prefetched_ticker
    if ticker is None:
        try:
            ticker = await exchange_adapter.get_ticker(symbol)
            anomaly_monitor.record_api_success()
        except Exception as e:
            err_text = str(e)
            if "429" in err_text or "Too Many Requests" in err_text:
                logger.warning("exchange_rate_limited", symbol=symbol, error=err_text)
                return
            event = anomaly_monitor.record_api_failure()
            if event:
                logger.error("api_failure_threshold", symbol=symbol, error=str(e))
                store.save_safety_event(event)
                kill_switch.activate(f"API failure threshold reached: {e}")
                if notifier:
                    await notifier.send_alert(
                        "API Failure",
                        f"Consecutive API failures on {symbol}",
                        "critical",
                    )
            return
    else:
        anomaly_monitor.record_api_success()

    now = datetime.now(timezone.utc)
    trade_price = Decimal(str(ticker.get("price", 0)))
    if trade_price <= 0:
        return

    md = MarketData(
        exchange=settings.exchange,
        symbol=symbol,
        timestamp=now,
        open=Decimal(str(ticker.get("open", trade_price))),
        high=Decimal(str(ticker.get("high", trade_price))),
        low=Decimal(str(ticker.get("low", trade_price))),
        close=trade_price,
        volume=Decimal(str(ticker.get("volume_24h", 0))),
        bid=Decimal(str(ticker.get("bid", 0))) or None,
        ask=Decimal(str(ticker.get("ask", 0))) or None,
    )

    price_event = anomaly_monitor.check_price_anomaly(md)
    if price_event:
        store.save_safety_event(price_event)
        logger.warning("price_anomaly", symbol=symbol, desc=price_event.description)
        if notifier:
            await notifier.send_alert("Price Anomaly", price_event.description, "high")

    prev_close = Decimal(str(ticker.get("prev_close", 0)))
    if prev_close > 0:
        price_change_pct = abs(trade_price - prev_close) / prev_close * Decimal("100")
        if risk_manager.check_circuit_breaker(price_change_pct):
            logger.warning(
                "circuit_breaker", symbol=symbol, change_pct=f"{price_change_pct:.1f}%"
            )
            if notifier:
                await notifier.send_alert(
                    "Circuit Breaker",
                    f"{symbol} price changed {price_change_pct:.1f}%",
                    "critical",
                )
            return

    base_candles = _base_candle_cache.get(symbol, [])
    if not base_candles or _should_refresh_candles(symbol, settings):
        try:
            base_candles = await exchange_adapter.get_candles(
                symbol, interval="1h", count=200
            )
            _base_candle_cache[symbol] = base_candles
            _symbol_candle_refresh_ts[symbol] = time.time()
        except Exception:
            pass

    # Inject synthetic in-progress candle so indicators reflect current price
    synthetic = _build_synthetic_candle(
        ticker, base_candles[-1] if base_candles else None
    )
    strategy.update_candles(symbol, base_candles + [synthetic])

    # 5. Run strategy
    signals = await strategy.on_tick(md)

    # 6. Build state for risk checks
    balances = await broker.fetch_balances()
    positions = await broker.fetch_positions()
    store.save_balance_snapshot(balances)

    total_balance = (
        balances.total_value_quote or balances.total_value_krw or Decimal("0")
    )
    current_symbol_value = Decimal("0")
    for p in positions:
        if p.symbol != symbol or p.quantity <= 0:
            continue
        pos_price = (
            trade_price if p.symbol == symbol else (p.current_price or Decimal("0"))
        )
        current_symbol_value += p.quantity * pos_price

    state: dict[str, Any] = {
        "total_balance": total_balance,
        "position_count": len(positions),
        "current_symbol_value": current_symbol_value,
        "today_pnl": Decimal("0"),
        "market_price": trade_price,
    }

    # 7. LLM advisory (runs before position protection for soft stop / take-profit queries)
    llm_advisor = components.get("llm_advisor")
    advice = None
    _skip_llm = False
    if llm_advisor is not None:
        last_price = _last_llm_prices.get(symbol)
        if last_price is not None and last_price > 0:
            change_pct = abs(trade_price - last_price) / last_price * 100
            elapsed = time.time() - _last_llm_times.get(symbol, 0.0)
            if change_pct < _LLM_SKIP_THRESHOLD_PCT and elapsed < _LLM_MAX_SKIP_SECONDS:
                _skip_llm = True

        # Force LLM when position is in soft stop or take-profit zone
        if _skip_llm:
            for pos in positions:
                if pos.symbol == symbol and pos.quantity > 0:
                    pos_with = pos.model_copy(update={"current_price": trade_price})
                    pnl = pos_with.unrealized_pnl_pct
                    if pnl is not None and (
                        pnl <= -abs(settings.risk.soft_stop_loss_pct)
                        or pnl >= settings.risk.take_profit_pct
                    ):
                        _skip_llm = False
                    break

    if llm_advisor is not None and not _skip_llm:
        try:
            change_24h_pct = Decimal("0")
            if prev_close > 0:
                change_24h_pct = (
                    (trade_price - prev_close) / prev_close * Decimal("100")
                )

            market_summary: dict[str, object] = {
                "symbol": symbol,
                "timeframe": "1h",
                "current_time": now.isoformat().replace("+00:00", "Z"),
                "price": str(trade_price),
                "open": str(md.open),
                "high": str(md.high),
                "low": str(md.low),
                "volume": str(md.volume),
                "change_24h_pct": str(round(change_24h_pct, 2)),
            }
            strategy_signal_data = [
                {
                    "signal_type": s.signal_type.value,
                    "confidence": str(s.confidence),
                    "metadata": s.metadata,
                }
                for s in signals
            ]

            indicators = strategy.compute_indicators(symbol)
            indicators_dict = indicators.to_llm_dict() if indicators else None
            if indicators_dict:
                indicators_dict.pop("current_price", None)

            position_data = []
            for p in positions:
                if p.quantity <= 0:
                    continue
                if p.symbol == symbol:
                    pos_price = trade_price
                    pos_with_price = p.model_copy(update={"current_price": trade_price})
                else:
                    pos_price = p.current_price or Decimal("0")
                    pos_with_price = p
                position_data.append(
                    {
                        "symbol": p.symbol,
                        "quantity": str(p.quantity),
                        "average_entry_price": str(p.average_entry_price),
                        "current_price": str(pos_price),
                        "unrealized_pnl_pct": str(
                            pos_with_price.unrealized_pnl_pct or 0
                        ),
                    }
                )

            portfolio_exposure = _compute_portfolio_exposure(positions, total_balance)

            quote = (
                settings.quote_currency
                if hasattr(settings, "quote_currency")
                else "KRW"
            )
            available_cash = balances.balances.get(quote, Decimal("0"))
            balance_data: dict[str, object] = {
                f"total_portfolio_value_{quote.lower()}": str(total_balance),
                f"available_cash_{quote.lower()}": str(available_cash),
                "position_count": len(positions),
            }

            recent_orders_data = []
            for o in store.get_all_orders(limit=settings.llm_recent_orders_count):
                ago_sec = (now - o.created_at).total_seconds() if o.created_at else 0
                if ago_sec < 3600:
                    ago_str = f"{int(ago_sec // 60)}min ago"
                elif ago_sec < 86400:
                    ago_str = f"{ago_sec / 3600:.1f}h ago"
                else:
                    ago_str = f"{ago_sec / 86400:.1f}d ago"
                recent_orders_data.append(
                    {
                        "symbol": o.symbol,
                        "side": o.side.value,
                        "quantity": str(o.quantity),
                        "price": str(o.price),
                        "status": o.status.value,
                        "ago": ago_str,
                    }
                )

            candles = strategy._candle_history.get(symbol, [])
            recent_candles_data = [
                {
                    "open": str(c.get("open", 0)),
                    "high": str(c.get("high", 0)),
                    "low": str(c.get("low", 0)),
                    "close": str(c.get("close", 0)),
                    "volume": str(c.get("volume", 0)),
                }
                for c in candles[-settings.llm_recent_candles_count :]
            ]

            btc_trend = await _compute_btc_trend(exchange_adapter, strategy, symbol)

            atr_stop_distance: str | None = None
            if indicators and indicators.atr > 0:
                atr_stop_distance = str(
                    round(indicators.atr * float(settings.risk.atr_stop_multiplier), 2)
                )

            risk_context: dict[str, object] = {
                "atr_stop_distance": atr_stop_distance,
                "max_single_coin_exposure_pct": str(
                    settings.risk.max_single_coin_exposure_pct
                ),
                "max_alt_total_exposure_pct": str(
                    settings.risk.max_alt_total_exposure_pct
                ),
            }

            current_symbol_position: dict[str, object] | None = None
            for p in positions:
                if p.symbol == symbol and p.quantity > 0:
                    pos_price = trade_price
                    pos_with = p.model_copy(update={"current_price": trade_price})
                    pos_value = p.quantity * pos_price
                    pos_pct = (
                        float(pos_value / total_balance * Decimal("100"))
                        if total_balance > 0
                        else 0.0
                    )
                    current_symbol_position = {
                        "quantity": str(p.quantity),
                        "average_entry_price": str(p.average_entry_price),
                        "unrealized_pnl_pct": str(pos_with.unrealized_pnl_pct or 0),
                        "position_value_pct_of_portfolio": round(pos_pct, 2),
                    }
                    break

            symbol_candles = strategy._candle_history.get(symbol, [])
            recent_structure = _compute_recent_structure(symbol_candles)
            volume_context = _compute_volume_context(symbol_candles)

            advice = await llm_advisor.get_advice(
                symbol,
                market_summary,
                strategy_signal_data,
                technical_indicators=indicators_dict,
                positions=position_data,
                balance_info=balance_data,
                recent_orders=recent_orders_data,
                recent_candles=recent_candles_data,
                previous_decisions=_recent_decisions.get(symbol, []),
                btc_trend=btc_trend,
                portfolio_exposure=portfolio_exposure,
                risk_context=risk_context,
                current_symbol_position=current_symbol_position,
                recent_structure=recent_structure,
                volume_context=volume_context,
            )
            _last_llm_prices[symbol] = trade_price
            _last_llm_times[symbol] = time.time()
        except Exception:
            pass

    # 8. Position protection: hard stop-loss, soft stop-loss (LLM), take-profit (LLM), trailing stop
    risk_limits = settings.risk
    protection_sold = False
    for pos in positions:
        if pos.symbol != symbol or pos.quantity <= 0:
            continue

        is_short = pos.side == PositionSide.SHORT
        pos_with_price = pos.model_copy(update={"current_price": trade_price})
        pnl_pct = pos_with_price.unrealized_pnl_pct
        if pnl_pct is None:
            continue

        # For shorts: exit_side=BUY (cover), LLM action=COVER_CONSIDER
        # For longs: exit_side=SELL, LLM action=SELL_CONSIDER
        exit_side = OrderSide.BUY if is_short else OrderSide.SELL
        llm_exit_action = "COVER_CONSIDER" if is_short else "SELL_CONSIDER"
        exit_label = "COVER" if is_short else "SELL"

        close_reason: str | None = None
        use_market_order = False

        if pnl_pct <= -abs(risk_limits.stop_loss_pct):
            close_reason = (
                f"hard_stop_loss: pnl={pnl_pct:.2f}% <= -{risk_limits.stop_loss_pct}%"
            )
            use_market_order = True

        elif pnl_pct <= -abs(risk_limits.soft_stop_loss_pct):
            if advice is not None and advice.action == llm_exit_action:
                close_reason = (
                    f"soft_stop_loss (LLM approved): pnl={pnl_pct:.2f}%, "
                    f"LLM={advice.action}({advice.confidence})"
                )
            else:
                log_event(
                    "decision",
                    {
                        "symbol": symbol,
                        "final_action": "HOLD",
                        "trigger": "soft_stop_loss_held",
                        "position_side": pos.side.value,
                        "pnl_pct": str(pnl_pct),
                        "llm_action": advice.action if advice else "none",
                        "llm_reasoning": advice.reasoning if advice else "no LLM",
                        "price": str(trade_price),
                    },
                )

        elif pnl_pct >= risk_limits.take_profit_pct:
            # 익절 구간: LLM 판단에 위임. SELL_CONSIDER만 매도, 그 외는 유지.
            # 트레일링 스탑이 하락 반전을 잡아주므로 LLM이 상승 지속 여부를 판단.
            if advice is not None and advice.action == llm_exit_action:
                close_reason = (
                    f"take_profit (LLM sell): pnl={pnl_pct:.2f}% >= {risk_limits.take_profit_pct}%, "
                    f"LLM={advice.action}({advice.confidence})"
                )
            else:
                log_event(
                    "decision",
                    {
                        "symbol": symbol,
                        "final_action": "HOLD",
                        "trigger": "take_profit_held",
                        "position_side": pos.side.value,
                        "pnl_pct": str(pnl_pct),
                        "llm_action": advice.action if advice else "none",
                        "llm_reasoning": advice.reasoning if advice else "no LLM",
                        "price": str(trade_price),
                    },
                )

        elif risk_limits.trailing_stop_enabled:
            if is_short:
                # Short trailing: track price going DOWN, exit on bounce UP
                lw = _low_watermarks.get(symbol, pos.average_entry_price)
                if trade_price < lw:
                    _low_watermarks[symbol] = trade_price
                    lw = trade_price
                if lw < pos.average_entry_price and lw > 0:
                    bounce_from_low = (trade_price - lw) / lw * Decimal("100")
                    if bounce_from_low >= risk_limits.trailing_stop_pct:
                        close_reason = (
                            f"trailing_stop_short: bounce={bounce_from_low:.2f}% "
                            f"from low={lw}, threshold={risk_limits.trailing_stop_pct}%"
                        )
            else:
                # Long trailing: track price going UP, exit on drop DOWN
                hw = _high_watermarks.get(symbol, pos.average_entry_price)
                if trade_price > hw:
                    _high_watermarks[symbol] = trade_price
                    hw = trade_price
                if hw > pos.average_entry_price:
                    drop_from_high = (hw - trade_price) / hw * Decimal("100")
                    if drop_from_high >= risk_limits.trailing_stop_pct:
                        close_reason = (
                            f"trailing_stop: drop={drop_from_high:.2f}% "
                            f"from high={hw}, threshold={risk_limits.trailing_stop_pct}%"
                        )

        if close_reason is not None:
            log_event(
                "decision",
                {
                    "symbol": symbol,
                    "final_action": exit_label,
                    "trigger": "position_protection",
                    "position_side": pos.side.value,
                    "reason": close_reason,
                    "pnl_pct": str(pnl_pct),
                    "price": str(trade_price),
                },
            )
            intent = OrderIntent(
                signal_id=uuid4(),
                exchange=settings.exchange,
                symbol=symbol,
                side=exit_side,
                order_type=OrderType.MARKET if use_market_order else OrderType.LIMIT,
                quantity=pos.quantity,
                price=trade_price,
                reason=close_reason,
                reduce_only=True,
                position_side=pos.side if is_short else None,
                timestamp=now,
            )
            order = await engine.execute(intent, state)
            if order:
                log_event(
                    "order",
                    {
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "status": order.status.value,
                        "quantity": str(order.quantity),
                        "price": str(order.price),
                        "trigger": "position_protection",
                        "position_side": pos.side.value,
                    },
                )
                if notifier and settings.is_live_mode():
                    side_label = "COVER" if is_short else "SELL"
                    await notifier.send_alert(
                        f"포지션 보호 {side_label} 체결 요청",
                        (
                            f"{order.symbol} {side_label} {order.quantity} @ {order.price}"
                            f"\nstatus={order.status.value}, trigger=position_protection"
                        ),
                        "high",
                    )
                if is_short:
                    _low_watermarks.pop(symbol, None)
                else:
                    _high_watermarks.pop(symbol, None)
                protection_sold = True

    # Skip combined decision if position protection already executed a sell
    if protection_sold:
        return

    # 9. Combined decision: strategy signals + LLM advice
    final_action = _resolve_action(signals, advice, settings)
    if not _skip_llm:
        log_event(
            "decision",
            {
                "symbol": symbol,
                "final_action": final_action,
                "strategy_signal": signals[0].signal_type.value if signals else "none",
                "strategy_confidence": str(signals[0].confidence) if signals else "0",
                "llm_action": advice.action if advice else "none",
                "llm_confidence": str(advice.confidence) if advice else "0",
                "llm_reasoning": advice.reasoning if advice else "",
                "llm_risk_notes": advice.risk_notes if advice else "",
                "llm_prompt": advice._prompt if advice else "",
                "price": str(trade_price),
            },
        )

    if advice is not None:
        hist = _recent_decisions.setdefault(symbol, [])
        hist.append(
            {
                "action": advice.action,
                "confidence": str(advice.confidence),
                "reasoning": advice.reasoning[:100],
                "price": str(trade_price),
            }
        )
        if len(hist) > _MAX_DECISION_HISTORY:
            _recent_decisions[symbol] = hist[-_MAX_DECISION_HISTORY:]

    if final_action == "BUY":
        # Per-symbol buy cooldown: prevent rapid repeated buys on the same coin
        _cooldown_elapsed = time.time() - _last_buy_ts.get(symbol, 0.0)
        if _cooldown_elapsed < _BUY_COOLDOWN_SECONDS:
            _remaining = int(_BUY_COOLDOWN_SECONDS - _cooldown_elapsed)
            logger.info(
                "buy_cooldown_active",
                symbol=symbol,
                remaining_sec=_remaining,
            )
            return

        buy_pct = settings.risk.max_position_size_pct
        if advice is not None and advice.buy_pct is not None:
            buy_pct = max(
                Decimal("0"), min(advice.buy_pct, settings.risk.max_position_size_pct)
            )
        order_price = trade_price
        if advice is not None and advice.target_price is not None:
            order_price = advice.target_price
        order_value = total_balance * buy_pct / Decimal("100")

        if order_value < _MIN_ORDER_VALUE_KRW:
            logger.info(
                "buy_below_minimum",
                symbol=symbol,
                order_value=str(order_value),
                minimum=str(_MIN_ORDER_VALUE_KRW),
            )
            return

        if order_value > 0:
            use_market = advice is not None and advice.order_type == "market"
            intent_order_type = OrderType.MARKET if use_market else OrderType.LIMIT
            intent_price = order_price  # always pass price for proper KRW sizing
            reason_parts = [
                (
                    f"Strategy: {signals[0].metadata.get('reason', 'buy')}"
                    if signals
                    else ""
                )
            ]
            if advice:
                reason_parts.append(f"LLM: {advice.reasoning[:100]}")
            intent = OrderIntent(
                signal_id=signals[0].signal_id if signals else uuid4(),
                exchange=settings.exchange,
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=intent_order_type,
                quote_quantity=order_value,
                price=intent_price,
                reason=" | ".join(p for p in reason_parts if p),
                timestamp=now,
            )
            order = await engine.execute(intent, state)
            if order:
                _last_buy_ts[symbol] = time.time()  # record buy for cooldown
                log_event(
                    "order",
                    {
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "status": order.status.value,
                        "quantity": str(order.quantity),
                        "price": str(order.price),
                        "sizing_pct": str(buy_pct),
                    },
                )
                if notifier and settings.is_live_mode():
                    await notifier.send_alert(
                        "실제 매수 체결 요청",
                        (
                            f"{order.symbol} BUY {order.quantity} @ {order.price}"
                            f"\nstatus={order.status.value}, sizing_pct={buy_pct}"
                        ),
                        "medium",
                    )

    elif final_action == "SELL":
        # Quick-flip protection: don't sell within cooldown period of buying
        # unless it's a position protection sell (those happen in section 8, not here)
        _sell_since_buy = time.time() - _last_buy_ts.get(symbol, 0.0)
        if _sell_since_buy < _BUY_COOLDOWN_SECONDS:
            logger.info(
                "quick_flip_blocked",
                symbol=symbol,
                seconds_since_buy=int(_sell_since_buy),
                cooldown=int(_BUY_COOLDOWN_SECONDS),
            )
            return

        for pos in positions:
            if pos.symbol == symbol and pos.quantity > 0:
                sell_qty = pos.quantity
                sell_pct = Decimal("100")
                if advice is not None and advice.sell_pct is not None:
                    sell_pct = max(Decimal("0"), min(advice.sell_pct, Decimal("100")))
                    sell_qty = pos.quantity * sell_pct / Decimal("100")
                if sell_qty <= 0:
                    continue

                order_price = trade_price
                if advice is not None and advice.target_price is not None:
                    order_price = advice.target_price
                use_market = advice is not None and advice.order_type == "market"
                intent_order_type = OrderType.MARKET if use_market else OrderType.LIMIT
                intent_price = None if use_market else order_price
                reason_parts = [
                    (
                        f"Strategy: {signals[0].metadata.get('reason', 'sell')}"
                        if signals
                        else ""
                    )
                ]
                if advice:
                    reason_parts.append(f"LLM: {advice.reasoning[:100]}")
                intent = OrderIntent(
                    signal_id=signals[0].signal_id if signals else uuid4(),
                    exchange=settings.exchange,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=intent_order_type,
                    quantity=sell_qty,
                    price=intent_price,
                    reason=" | ".join(p for p in reason_parts if p),
                    timestamp=now,
                )
                order = await engine.execute(intent, state)
                if order:
                    log_event(
                        "order",
                        {
                            "order_id": order.order_id,
                            "symbol": order.symbol,
                            "side": order.side.value,
                            "status": order.status.value,
                            "quantity": str(order.quantity),
                            "price": str(order.price),
                            "sizing_pct": str(sell_pct),
                        },
                    )
                    if notifier and settings.is_live_mode():
                        await notifier.send_alert(
                            "실제 매도 체결 요청",
                            (
                                f"{order.symbol} SELL {order.quantity} @ {order.price}"
                                f"\nstatus={order.status.value}, sizing_pct={sell_pct}"
                            ),
                            "high",
                        )

    elif final_action == "SHORT" and settings.risk.futures_enabled:
        short_pct = settings.risk.max_position_size_pct
        if advice is not None and advice.short_pct is not None:
            short_pct = max(
                Decimal("0"), min(advice.short_pct, settings.risk.max_position_size_pct)
            )
        order_price = trade_price
        if advice is not None and advice.target_price is not None:
            order_price = advice.target_price
        order_value = total_balance * short_pct / Decimal("100")
        if order_value > 0:
            use_market = advice is not None and advice.order_type == "market"
            intent_order_type = OrderType.MARKET if use_market else OrderType.LIMIT
            intent_price = None if use_market else order_price
            reason_parts = [
                (
                    f"Strategy: {signals[0].metadata.get('reason', 'short')}"
                    if signals
                    else ""
                )
            ]
            if advice:
                reason_parts.append(f"LLM: {advice.reasoning[:100]}")
            intent = OrderIntent(
                signal_id=signals[0].signal_id if signals else uuid4(),
                exchange=settings.exchange,
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=intent_order_type,
                quote_quantity=order_value,
                price=intent_price,
                reason=" | ".join(p for p in reason_parts if p),
                position_side=PositionSide.SHORT,
                timestamp=now,
            )
            order = await engine.execute(intent, state)
            if order:
                log_event(
                    "order",
                    {
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "status": order.status.value,
                        "quantity": str(order.quantity),
                        "price": str(order.price),
                        "sizing_pct": str(short_pct),
                        "position_side": "short",
                    },
                )
                if notifier and settings.is_live_mode():
                    await notifier.send_alert(
                        "숏 포지션 진입 요청",
                        (
                            f"{order.symbol} SHORT {order.quantity} @ {order.price}"
                            f"\nstatus={order.status.value}, sizing_pct={short_pct}"
                        ),
                        "medium",
                    )

    elif final_action == "COVER" and settings.risk.futures_enabled:
        for pos in positions:
            if (
                pos.symbol == symbol
                and pos.quantity > 0
                and pos.side == PositionSide.SHORT
            ):
                cover_qty = pos.quantity
                cover_pct = Decimal("100")
                if advice is not None and advice.sell_pct is not None:
                    cover_pct = max(Decimal("0"), min(advice.sell_pct, Decimal("100")))
                    cover_qty = pos.quantity * cover_pct / Decimal("100")
                if cover_qty <= 0:
                    continue

                use_market = advice is not None and advice.order_type == "market"
                intent_order_type = OrderType.MARKET if use_market else OrderType.LIMIT
                intent_price = trade_price  # always pass price for proper sizing
                reason_parts = [
                    (
                        f"Strategy: {signals[0].metadata.get('reason', 'cover')}"
                        if signals
                        else ""
                    )
                ]
                if advice:
                    reason_parts.append(f"LLM: {advice.reasoning[:100]}")
                intent = OrderIntent(
                    signal_id=signals[0].signal_id if signals else uuid4(),
                    exchange=settings.exchange,
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=intent_order_type,
                    quantity=cover_qty,
                    price=intent_price,
                    reason=" | ".join(p for p in reason_parts if p),
                    reduce_only=True,
                    position_side=PositionSide.SHORT,
                    timestamp=now,
                )
                order = await engine.execute(intent, state)
                if order:
                    log_event(
                        "order",
                        {
                            "order_id": order.order_id,
                            "symbol": order.symbol,
                            "side": order.side.value,
                            "status": order.status.value,
                            "quantity": str(order.quantity),
                            "price": str(order.price),
                            "sizing_pct": str(cover_pct),
                            "position_side": "short",
                        },
                    )
                    if notifier and settings.is_live_mode():
                        await notifier.send_alert(
                            "숏 포지션 커버 요청",
                            (
                                f"{order.symbol} COVER {order.quantity} @ {order.price}"
                                f"\nstatus={order.status.value}, sizing_pct={cover_pct}"
                            ),
                            "high",
                        )
                _low_watermarks.pop(symbol, None)


async def _main_loop(settings: Settings, once: bool = False) -> None:
    """Main trading loop."""
    components = _build_system(settings, install_signal_handlers=True)
    store: StateStore = components["store"]
    engine: ExecutionEngine = components["engine"]
    broker = components["broker"]
    notifier = components.get("notifier")

    logger.info(
        "trader_started",
        mode=settings.trading_mode.value,
        exchange=settings.exchange.value,
        symbols=settings.trading_symbols,
        live=settings.is_live_mode(),
    )

    try:
        while True:
            try:
                cancelled = await engine.cancel_stale_orders(
                    settings.stale_order_timeout_sec
                )
                for order in cancelled:
                    log_event(
                        "order",
                        {
                            "order_id": order.order_id,
                            "symbol": order.symbol,
                            "side": order.side.value,
                            "status": "cancelled",
                            "trigger": "stale_order_timeout",
                            "quantity": str(order.quantity),
                            "price": str(order.price),
                        },
                    )
                    if notifier:
                        await notifier.send_alert(
                            "Stale Order Cancelled",
                            f"{order.symbol} {order.side.value} order cancelled (timeout)",
                            "medium",
                        )
            except Exception as e:
                logger.error("stale_order_cleanup_error", error=str(e))

            # Sync order statuses (submitted -> filled/cancelled) from exchange
            try:
                open_orders = store.get_open_orders()
                if open_orders and hasattr(broker, "sync_order_statuses"):
                    updated = await broker.sync_order_statuses(open_orders)
                    for order in updated:
                        store.save_order(order)
                    if updated:
                        logger.info("order_status_synced", count=len(updated))
            except Exception as e:
                logger.error("order_sync_error", error=str(e))

            active_symbols = await _refresh_dynamic_symbols(components)
            batched_tickers = await _fetch_batch_tickers(
                exchange_adapter=components["exchange_adapter"], symbols=active_symbols
            )

            for symbol in active_symbols:
                try:
                    await _run_tick(
                        components,
                        symbol,
                        prefetched_ticker=batched_tickers.get(symbol),
                    )
                except Exception as e:
                    logger.error("tick_error", symbol=symbol, error=str(e))

            if once:
                break

            await asyncio.sleep(settings.market_data_interval_sec)

    except KeyboardInterrupt:
        logger.info("trader_stopped", reason="keyboard_interrupt")
    finally:
        adapter = components.get("exchange_adapter")
        if adapter and hasattr(adapter, "close"):
            await adapter.close()
        llm = components.get("llm_advisor")
        if llm and hasattr(llm, "close"):
            await llm.close()
        notifier = components.get("notifier")
        if notifier and hasattr(notifier, "close"):
            await notifier.close()
        store.close()
        close_logging()


@click.group()
def cli() -> None:
    """Coin Trader - Safety-first automated crypto trading system."""
    pass


@cli.command()
@click.option("--mode", type=click.Choice(["paper", "live"]), default=None)
@click.option("--once", is_flag=True, help="Run a single tick and exit")
def run(mode: str | None, once: bool) -> None:
    """Start the trading bot."""
    settings = Settings.load_safe()

    if mode is not None:
        settings.trading_mode = TradingMode(mode)

    if settings.trading_mode == TradingMode.LIVE:
        if not settings.is_live_mode():
            click.echo("ERROR: LIVE MODE NOT ARMED")
            click.echo(
                f"Create {settings.live_mode_token_path} with content 'ARMED' to enable."
            )
            sys.exit(1)

        if not once:
            click.echo("WARNING: You are about to start LIVE trading.")
            confirmation = click.prompt(
                "Type 'I_UNDERSTAND_LIVE_TRADING' to confirm", type=str
            )
            if confirmation != "I_UNDERSTAND_LIVE_TRADING":
                click.echo("Aborted.")
                sys.exit(1)

    click.echo(f"Starting trader in {settings.trading_mode.value} mode...")
    asyncio.run(_main_loop(settings, once=once))


@cli.command()
@click.option("--close-positions", is_flag=True, help="Also close all open positions")
def kill(close_positions: bool) -> None:
    """Activate kill switch - halt all trading."""
    settings = Settings.load_safe()
    kill_switch = KillSwitch(settings.kill_switch_file)

    kill_switch.activate("Manual kill via CLI")
    click.echo("Kill switch ACTIVATED. All new orders blocked.")

    if close_positions:
        click.echo("Cancelling open orders...")

        async def _cancel() -> None:
            components = _build_system(settings)
            engine: ExecutionEngine = components["engine"]
            cancelled = await engine.cancel_all_open_orders()
            click.echo(f"Cancelled {cancelled} orders.")
            components["store"].close()

        asyncio.run(_cancel())

    click.echo("Done. To resume, delete the kill switch file or restart.")


@cli.command()
def selftest() -> None:
    """Run basic self-test to verify system integrity."""
    click.echo("Running self-test...")
    errors: list[str] = []

    # 1. Config loads
    try:
        settings = Settings.load_safe()
        assert (
            settings.trading_mode == TradingMode.PAPER
        ), "Default mode should be paper"
        click.echo("  [OK] Config loads (paper mode)")
    except Exception as e:
        errors.append(f"Config: {e}")

    # 2. State store
    try:
        store = StateStore(db_path=Path("/tmp/coin_trader_selftest.db"))
        store.close()
        Path("/tmp/coin_trader_selftest.db").unlink(missing_ok=True)
        click.echo("  [OK] State store initializes")
    except Exception as e:
        errors.append(f"StateStore: {e}")

    # 3. Risk limits immutable
    try:
        limits = RiskLimits()
        try:
            limits.max_positions = 999  # type: ignore[misc]
            errors.append("RiskLimits: should be immutable (frozen dataclass)")
        except AttributeError:
            click.echo("  [OK] Risk limits immutable")
    except Exception as e:
        errors.append(f"RiskLimits: {e}")

    # 4. Kill switch
    try:
        ks_path = Path("/tmp/coin_trader_selftest_ks")
        ks = KillSwitch(ks_path)
        assert not ks.is_active()
        ks.activate("test")
        assert ks.is_active()
        ks.deactivate()
        assert not ks.is_active()
        ks_path.unlink(missing_ok=True)
        click.echo("  [OK] Kill switch works")
    except Exception as e:
        errors.append(f"KillSwitch: {e}")

    # 5. Redaction
    try:
        from coin_trader.logging.redaction import redact_dict

        test = {"api_key": "secret123", "name": "test"}
        redacted = redact_dict(test)
        assert "secret123" not in str(redacted), "API key not redacted"
        click.echo("  [OK] Sensitive data redaction")
    except Exception as e:
        errors.append(f"Redaction: {e}")

    # 6. Models
    try:
        from coin_trader.core.models import (
            Order,
            OrderStatus,
            OrderType,
            ExchangeName,
            OrderSide,
        )

        order = Order(
            client_order_id="test_001",
            intent_id=uuid4(),
            exchange=ExchangeName.UPBIT,
            symbol="BTC/KRW",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000000"),
            status=OrderStatus.PENDING,
        )
        assert order.remaining_quantity == Decimal("0.001")
        click.echo("  [OK] Domain models validate")
    except Exception as e:
        errors.append(f"Models: {e}")

    if errors:
        click.echo(f"\nFAILED - {len(errors)} error(s):")
        for err in errors:
            click.echo(f"  - {err}")
        sys.exit(1)
    else:
        click.echo("\nOK - All self-tests passed.")
        sys.exit(0)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8000, type=int, help="Bind port")
def web(host: str, port: int) -> None:
    """Start the web dashboard server."""
    import uvicorn

    click.echo(f"Starting web dashboard at http://{host}:{port}")
    click.echo("Press Ctrl+C to stop.")
    uvicorn.run(
        "coin_trader.web.api:app",
        host=host,
        port=port,
        reload=False,
        timeout_graceful_shutdown=3,
        log_level="warning",
    )


@cli.command("encrypt-keys")
@click.option("--exchange", type=click.Choice(["upbit", "binance"]), required=True)
@click.option("--master-key", prompt=True, hide_input=True, confirmation_prompt=True)
def encrypt_keys(exchange: str, master_key: str) -> None:
    """Encrypt API keys for secure storage."""
    from coin_trader.security.key_manager import KeyManager

    api_key = click.prompt("API Key", hide_input=True)
    api_secret = click.prompt("API Secret", hide_input=True)

    settings = Settings.load_safe()
    if exchange == "upbit":
        key_file = settings.upbit_key_file
    else:
        key_file = settings.binance_key_file

    KeyManager.encrypt_keys(api_key, api_secret, key_file, master_key)
    click.echo(f"Keys encrypted and saved to {key_file}")


if __name__ == "__main__":
    cli()
