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
    SignalType,
    TradingMode,
)
from coin_trader.execution.engine import ExecutionEngine
from coin_trader.execution.idempotency import IdempotencyManager
from coin_trader.logging.logger import close_logging, get_logger, log_event, setup_logging
from coin_trader.risk.manager import RiskManager
from coin_trader.safety.kill_switch import KillSwitch
from coin_trader.safety.monitor import AnomalyMonitor
from coin_trader.state.store import StateStore

logger = get_logger("main")

_high_watermarks: dict[str, Decimal] = {}
_last_llm_prices: dict[str, Decimal] = {}
_last_llm_times: dict[str, float] = {}
_LLM_SKIP_THRESHOLD_PCT = Decimal("1.0")
_LLM_MAX_SKIP_SECONDS = 1800.0
_recent_decisions: dict[str, list[dict[str, str]]] = {}
_MAX_DECISION_HISTORY = 5

_btc_trend_cache: dict[str, object] = {}
_btc_trend_cache_ts: float = 0.0
_BTC_TREND_CACHE_TTL = 60.0
_BTC_SYMBOL = "BTC/KRW"


def _build_system(settings: Settings, *, install_signal_handlers: bool = False) -> dict[str, Any]:
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

    if settings.exchange == ExchangeName.UPBIT:
        if settings.is_live_mode():
            from coin_trader.security.key_manager import KeyManager

            access_key, secret_key = KeyManager.decrypt_keys(
                settings.upbit_key_file, settings.upbit_master_key
            )
            from coin_trader.exchange.upbit import UpbitAdapter

            exchange_adapter = UpbitAdapter(access_key=access_key, secret_key=secret_key)

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

    strategy = ConservativeStrategy()

    llm_advisor = None
    if settings.llm_enabled:
        from coin_trader.llm.advisory import LLMAdvisor

        if settings.llm_auth_mode == "oauth":
            llm_advisor = LLMAdvisor.create_oauth(
                auth_file=settings.llm_oauth_auth_file,
                model=settings.llm_oauth_model,
                open_browser=settings.llm_oauth_open_browser,
                force_login=settings.llm_oauth_force_login,
            )
        elif settings.llm_api_key:
            llm_advisor = LLMAdvisor(
                api_key=settings.llm_api_key,
                provider=settings.llm_provider,
                model=settings.llm_model,
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
        return "HOLD"

    llm_action = advice.action
    llm_conf = float(advice.confidence)

    if llm_conf < settings.llm_min_confidence:
        return "HOLD"

    if llm_action == "BUY_CONSIDER":
        return "BUY"
    if llm_action == "SELL_CONSIDER":
        return "SELL"

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
    highs = [float(str(c.get("high_price", c.get("high", 0)) or 0)) for c in recent]
    lows = [float(str(c.get("low_price", c.get("low", 0)) or 0)) for c in recent]

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

    volumes = [
        float(str(c.get("candle_acc_trade_volume", c.get("volume", 0)) or 0)) for c in candles
    ]
    recent = volumes[-window:]
    avg = sum(recent) / len(recent) if recent else 1.0
    current = volumes[-1] if volumes else 0.0
    ratio = round(current / avg, 2) if avg > 0 else 0.0

    first_half = sum(recent[: window // 2]) / (window // 2) if window >= 2 else avg
    second_half = sum(recent[window // 2 :]) / (window - window // 2) if window >= 2 else avg
    threshold = 0.1
    if second_half > first_half * (1 + threshold):
        trend = "increasing"
    elif second_half < first_half * (1 - threshold):
        trend = "decreasing"
    else:
        trend = "flat"

    return {"volume_vs_avg_ratio": ratio, "volume_trend": trend}


def _compute_decision_context(
    symbol: str,
    positions: list[Any],
    trade_price: Decimal,
    risk_limits: Any,
) -> str:
    for p in positions:
        if p.symbol != symbol or p.quantity <= 0:
            continue
        pos_with_price = p.model_copy(update={"current_price": trade_price})
        pnl_pct = pos_with_price.unrealized_pnl_pct
        if pnl_pct is None:
            return "add_position"
        if pnl_pct <= -abs(risk_limits.soft_stop_loss_pct):
            return "risk_management"
        if pnl_pct >= risk_limits.take_profit_pct:
            return "reduce_position"
        return "add_position"
    return "new_entry"


async def _compute_btc_trend(
    exchange_adapter: Any,
    strategy: Any,
    current_symbol: str,
) -> dict[str, object] | None:
    global _btc_trend_cache, _btc_trend_cache_ts

    now = time.time()
    if now - _btc_trend_cache_ts < _BTC_TREND_CACHE_TTL and _btc_trend_cache:
        return _btc_trend_cache  # type: ignore[return-value]

    try:
        if _BTC_SYMBOL not in strategy._candle_history:
            btc_candles = await exchange_adapter.get_candles(
                _BTC_SYMBOL, interval="minutes/60", count=200
            )
            strategy.update_candles(_BTC_SYMBOL, btc_candles)

        btc_indicators = strategy.compute_indicators(_BTC_SYMBOL)
        if btc_indicators is None:
            return None

        result: dict[str, object] = {
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


async def _run_tick(
    components: dict[str, Any],
    symbol: str,
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
    try:
        ticker = await exchange_adapter.get_ticker(symbol)
        anomaly_monitor.record_api_success()
    except Exception as e:
        err_text = str(e)
        if "429" in err_text or "Too Many Requests" in err_text:
            logger.warning("upbit_rate_limited", symbol=symbol, error=err_text)
            return
        event = anomaly_monitor.record_api_failure()
        if event:
            logger.error("api_failure_threshold", symbol=symbol, error=str(e))
            store.save_safety_event(event)
            kill_switch.activate(f"API failure threshold reached: {e}")
            if notifier:
                await notifier.send_alert(
                    "API Failure", f"Consecutive API failures on {symbol}", "critical"
                )
        return

    now = datetime.now(timezone.utc)
    trade_price = Decimal(str(ticker.get("trade_price", 0)))
    if trade_price <= 0:
        return

    md = MarketData(
        exchange=settings.exchange,
        symbol=symbol,
        timestamp=now,
        open=Decimal(str(ticker.get("opening_price", trade_price))),
        high=Decimal(str(ticker.get("high_price", trade_price))),
        low=Decimal(str(ticker.get("low_price", trade_price))),
        close=trade_price,
        volume=Decimal(str(ticker.get("acc_trade_volume_24h", 0))),
        bid=Decimal(str(ticker.get("highest_bid", 0))) or None,
        ask=Decimal(str(ticker.get("lowest_ask", 0))) or None,
    )

    price_event = anomaly_monitor.check_price_anomaly(md)
    if price_event:
        store.save_safety_event(price_event)
        logger.warning("price_anomaly", symbol=symbol, desc=price_event.description)
        if notifier:
            await notifier.send_alert("Price Anomaly", price_event.description, "high")

    prev_close = Decimal(str(ticker.get("prev_closing_price", 0)))
    if prev_close > 0:
        price_change_pct = abs(trade_price - prev_close) / prev_close * Decimal("100")
        if risk_manager.check_circuit_breaker(price_change_pct):
            logger.warning("circuit_breaker", symbol=symbol, change_pct=f"{price_change_pct:.1f}%")
            if notifier:
                await notifier.send_alert(
                    "Circuit Breaker",
                    f"{symbol} price changed {price_change_pct:.1f}%",
                    "critical",
                )
            return

    try:
        candles = await exchange_adapter.get_candles(symbol, interval="minutes/60", count=200)
        strategy.update_candles(symbol, candles)
    except Exception:
        pass

    # 5. Run strategy
    signals = await strategy.on_tick(md)

    # 6. Build state for risk checks
    balances = await broker.fetch_balances()
    positions = await broker.fetch_positions()
    store.save_balance_snapshot(balances)

    total_balance = balances.total_value_krw or Decimal("0")
    current_symbol_value = Decimal("0")
    for p in positions:
        if p.symbol != symbol or p.quantity <= 0:
            continue
        pos_price = trade_price if p.symbol == symbol else (p.current_price or Decimal("0"))
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

    if llm_advisor is not None and not _skip_llm:
        try:
            change_24h_pct = Decimal("0")
            if prev_close > 0:
                change_24h_pct = (trade_price - prev_close) / prev_close * Decimal("100")

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
                        "unrealized_pnl_pct": str(pos_with_price.unrealized_pnl_pct or 0),
                    }
                )

            portfolio_exposure = _compute_portfolio_exposure(positions, total_balance)

            balance_data: dict[str, object] = {
                "total_balance_krw": str(total_balance),
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
                    "open": str(c.get("opening_price", c.get("open", 0))),
                    "high": str(c.get("high_price", c.get("high", 0))),
                    "low": str(c.get("low_price", c.get("low", 0))),
                    "close": str(c.get("trade_price", c.get("close", 0))),
                    "volume": str(c.get("candle_acc_trade_volume", c.get("volume", 0))),
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
                "max_single_coin_exposure_pct": str(settings.risk.max_single_coin_exposure_pct),
                "max_alt_total_exposure_pct": str(settings.risk.max_alt_total_exposure_pct),
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
            decision_context = _compute_decision_context(
                symbol, positions, trade_price, settings.risk
            )

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
                decision_context=decision_context,
            )
            _last_llm_prices[symbol] = trade_price
            _last_llm_times[symbol] = time.time()
        except Exception:
            pass

    # 8. Position protection: hard stop-loss, soft stop-loss (LLM), take-profit (LLM), trailing stop
    risk_limits = settings.risk
    for pos in positions:
        if pos.symbol != symbol or pos.quantity <= 0:
            continue

        pos_with_price = pos.model_copy(update={"current_price": trade_price})
        pnl_pct = pos_with_price.unrealized_pnl_pct
        if pnl_pct is None:
            continue

        sell_reason: str | None = None
        use_market_order = False

        if pnl_pct <= -abs(risk_limits.stop_loss_pct):
            sell_reason = f"hard_stop_loss: pnl={pnl_pct:.2f}% <= -{risk_limits.stop_loss_pct}%"
            use_market_order = True

        elif pnl_pct <= -abs(risk_limits.soft_stop_loss_pct):
            if advice is not None and advice.action == "SELL_CONSIDER":
                sell_reason = (
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
                        "pnl_pct": str(pnl_pct),
                        "llm_action": advice.action if advice else "none",
                        "llm_reasoning": advice.reasoning if advice else "no LLM",
                        "price": str(trade_price),
                    },
                )

        elif pnl_pct >= risk_limits.take_profit_pct:
            if advice is not None and advice.action == "HOLD":
                log_event(
                    "decision",
                    {
                        "symbol": symbol,
                        "final_action": "HOLD",
                        "trigger": "take_profit_held_by_llm",
                        "pnl_pct": str(pnl_pct),
                        "llm_action": advice.action,
                        "llm_reasoning": advice.reasoning,
                        "price": str(trade_price),
                    },
                )
            else:
                sell_reason = (
                    f"take_profit (LLM ok): pnl={pnl_pct:.2f}% >= {risk_limits.take_profit_pct}%, "
                    f"LLM={advice.action if advice else 'none'}"
                )

        elif risk_limits.trailing_stop_enabled:
            hw = _high_watermarks.get(symbol, pos.average_entry_price)
            if trade_price > hw:
                _high_watermarks[symbol] = trade_price
                hw = trade_price
            if hw > pos.average_entry_price:
                drop_from_high = (hw - trade_price) / hw * Decimal("100")
                if drop_from_high >= risk_limits.trailing_stop_pct:
                    sell_reason = (
                        f"trailing_stop: drop={drop_from_high:.2f}% "
                        f"from high={hw}, threshold={risk_limits.trailing_stop_pct}%"
                    )

        if sell_reason is not None:
            log_event(
                "decision",
                {
                    "symbol": symbol,
                    "final_action": "SELL",
                    "trigger": "position_protection",
                    "reason": sell_reason,
                    "pnl_pct": str(pnl_pct),
                    "price": str(trade_price),
                },
            )
            intent = OrderIntent(
                signal_id=uuid4(),
                exchange=settings.exchange,
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET if use_market_order else OrderType.LIMIT,
                quantity=pos.quantity,
                price=trade_price,
                reason=sell_reason,
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
                    },
                )
                if notifier and settings.is_live_mode():
                    await notifier.send_alert(
                        "실제 매도 체결 요청",
                        (
                            f"{order.symbol} SELL {order.quantity} @ {order.price}"
                            f"\nstatus={order.status.value}, trigger=position_protection"
                        ),
                        "high",
                    )
                _high_watermarks.pop(symbol, None)

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
        buy_pct = settings.risk.max_position_size_pct
        if advice is not None and advice.buy_pct is not None:
            buy_pct = max(Decimal("0"), min(advice.buy_pct, settings.risk.max_position_size_pct))
        order_value = total_balance * buy_pct / Decimal("100")
        if order_value > 0:
            reason_parts = [
                f"Strategy: {signals[0].metadata.get('reason', 'buy')}" if signals else ""
            ]
            if advice:
                reason_parts.append(f"LLM: {advice.reasoning[:100]}")
            intent = OrderIntent(
                signal_id=signals[0].signal_id if signals else uuid4(),
                exchange=settings.exchange,
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quote_quantity=order_value,
                price=trade_price,
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
        for pos in positions:
            if pos.symbol == symbol and pos.quantity > 0:
                sell_qty = pos.quantity
                sell_pct = Decimal("100")
                if advice is not None and advice.sell_pct is not None:
                    sell_pct = max(Decimal("0"), min(advice.sell_pct, Decimal("100")))
                    sell_qty = pos.quantity * sell_pct / Decimal("100")
                if sell_qty <= 0:
                    continue

                reason_parts = [
                    f"Strategy: {signals[0].metadata.get('reason', 'sell')}" if signals else ""
                ]
                if advice:
                    reason_parts.append(f"LLM: {advice.reasoning[:100]}")
                intent = OrderIntent(
                    signal_id=signals[0].signal_id if signals else uuid4(),
                    exchange=settings.exchange,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=sell_qty,
                    price=trade_price,
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
                cancelled = await engine.cancel_stale_orders(settings.stale_order_timeout_sec)
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

            for symbol in settings.trading_symbols:
                try:
                    await _run_tick(components, symbol)
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
            click.echo(f"Create {settings.live_mode_token_path} with content 'ARMED' to enable.")
            sys.exit(1)

        if not once:
            click.echo("WARNING: You are about to start LIVE trading.")
            confirmation = click.prompt("Type 'I_UNDERSTAND_LIVE_TRADING' to confirm", type=str)
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
        assert settings.trading_mode == TradingMode.PAPER, "Default mode should be paper"
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
        from coin_trader.core.models import Order, OrderStatus, OrderType, ExchangeName, OrderSide

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
