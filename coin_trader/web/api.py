from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from coin_trader.config.settings import Settings
from coin_trader.core.models import TradingMode, OrderIntent, OrderSide, OrderType
from coin_trader.safety.kill_switch import KillSwitch
from coin_trader.state.store import StateStore

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

_components: dict[str, Any] = {}
_trading_task: asyncio.Task[None] | None = None
_started_at: float = 0.0
_trading_mode: str = "paper"
_is_trading: bool = False

_SESSION_COOKIE = "ct_session"
_valid_sessions: set[str] = set()

_AUTH_BYPASS_PATHS = frozenset({"/login", "/api/auth/login"})


def _decimal_default(obj: object) -> object:
    if isinstance(obj, Decimal):
        f = float(obj)
        if f == int(f) and abs(f) < 1e15:
            return int(f)
        return f
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _model_to_dict(m: object) -> dict[str, Any]:
    model_dump = getattr(m, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    return dict(m) if isinstance(m, dict) else {"value": str(m)}


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    global _components, _started_at, _is_trading, _trading_task, _trading_mode
    settings = Settings.load_safe()

    from coin_trader.main import _build_system
    import coin_trader.logging.logger as _logger_mod

    _components.update(_build_system(settings))
    _started_at = time.time()

    if settings.is_live_mode():
        _trading_mode = "live"
        _logger_mod.set_trading_mode("live")
        _is_trading = True
        _trading_task = asyncio.create_task(_trading_loop())
    yield
    if _trading_task and not _trading_task.done():
        _trading_task.cancel()
        try:
            await asyncio.wait_for(_trading_task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    adapter = _components.get("exchange_adapter")
    if adapter and hasattr(adapter, "close"):
        try:
            await asyncio.wait_for(adapter.close(), timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            pass
    store = _components.get("store")
    if store:
        store.close()


app = FastAPI(title="Coin Trader Dashboard", lifespan=lifespan)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Any:
        settings: Settings | None = _components.get("settings")
        master_code = settings.web_master_code if settings else ""

        if not master_code:
            return await call_next(request)

        path = request.url.path
        if path in _AUTH_BYPASS_PATHS or path.startswith("/static"):
            return await call_next(request)

        session_id = request.cookies.get(_SESSION_COOKIE)
        if session_id and session_id in _valid_sessions:
            return await call_next(request)

        if path.startswith("/api/"):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        return RedirectResponse(url="/login", status_code=302)


app.add_middleware(AuthMiddleware)


@app.get("/login", response_model=None)
async def login_page(request: Request):  # type: ignore[no-untyped-def]
    settings: Settings | None = _components.get("settings")
    if not settings or not settings.web_master_code:
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": ""})


@app.post("/api/auth/login")
async def auth_login(request: Request) -> JSONResponse:
    settings: Settings | None = _components.get("settings")
    if not settings or not settings.web_master_code:
        return JSONResponse({"ok": True})

    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    code = body.get("code", "") if isinstance(body, dict) else ""

    if not hmac.compare_digest(str(code), settings.web_master_code):
        return JSONResponse(
            {"ok": False, "error": "인증 코드가 올바르지 않습니다"}, status_code=401
        )

    session_id = secrets.token_urlsafe(32)
    _valid_sessions.add(session_id)

    response = JSONResponse({"ok": True})
    response.set_cookie(
        key=_SESSION_COOKIE,
        value=session_id,
        httponly=True,
        samesite="lax",
        max_age=86400,
    )
    return response


@app.post("/api/auth/logout")
async def auth_logout(request: Request) -> JSONResponse:
    session_id = request.cookies.get(_SESSION_COOKIE)
    if session_id:
        _valid_sessions.discard(session_id)
    response = JSONResponse({"ok": True})
    response.delete_cookie(key=_SESSION_COOKIE)
    return response


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/status")
async def get_status() -> JSONResponse:
    kill_switch: KillSwitch | None = _components.get("kill_switch")
    settings: Settings | None = _components.get("settings")

    ks_active = kill_switch.is_active() if kill_switch else False
    ks_reason = kill_switch.get_reason() if kill_switch and ks_active else None

    uptime = int(time.time() - _started_at) if _started_at else 0

    return JSONResponse(
        {
            "trading_active": _is_trading,
            "trading_mode": _trading_mode,
            "kill_switch_active": ks_active,
            "kill_switch_reason": ks_reason,
            "uptime_seconds": uptime,
            "exchange": settings.exchange.value if settings else "unknown",
            "symbols": settings.trading_symbols if settings else [],
        }
    )


# ---------------------------------------------------------------------------
# Trading control
# ---------------------------------------------------------------------------
async def _trading_loop() -> None:
    global _is_trading
    from coin_trader.main import _run_tick

    settings: Settings = _components["settings"]
    engine: Any = _components.get("engine")
    notifier: Any = _components.get("notifier")
    try:
        while _is_trading:
            if engine is not None:
                try:
                    cancelled = await engine.cancel_stale_orders(settings.stale_order_timeout_sec)
                    for order in cancelled:
                        if notifier and hasattr(notifier, "send_alert"):
                            await notifier.send_alert(
                                "Stale Order Cancelled",
                                f"{order.symbol} {order.side.value} order cancelled (timeout)",
                                "medium",
                            )
                except Exception:
                    pass

            for symbol in settings.trading_symbols:
                if not _is_trading:
                    break
                try:
                    await _run_tick(_components, symbol)
                except Exception:
                    pass
            if _is_trading:
                await asyncio.sleep(settings.market_data_interval_sec)
    except asyncio.CancelledError:
        pass
    finally:
        _is_trading = False


@app.post("/api/trading/start")
async def start_trading(request: Request) -> JSONResponse:
    global _trading_task, _is_trading, _trading_mode

    if _is_trading:
        return JSONResponse({"ok": False, "error": "이미 거래가 실행 중입니다"}, status_code=409)

    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    mode = body.get("mode", "paper") if isinstance(body, dict) else "paper"

    settings: Settings = _components["settings"]

    if mode == "live":
        settings.trading_mode = TradingMode.LIVE
        if not settings.is_live_mode():
            settings.trading_mode = TradingMode.PAPER
            return JSONResponse(
                {
                    "ok": False,
                    "error": "라이브 모드가 ARMED 상태가 아닙니다. RUN/live_mode_token.txt 파일을 확인하세요.",
                },
                status_code=403,
            )
    else:
        settings.trading_mode = TradingMode.PAPER

    from coin_trader.main import _build_system
    import coin_trader.logging.logger as _logger_mod

    _components.update(_build_system(settings))
    _logger_mod.set_trading_mode(mode)

    _trading_mode = mode
    _is_trading = True
    _trading_task = asyncio.create_task(_trading_loop())

    return JSONResponse({"ok": True, "mode": mode})


@app.post("/api/trading/stop")
async def stop_trading() -> JSONResponse:
    global _is_trading, _trading_task

    if not _is_trading:
        return JSONResponse({"ok": False, "error": "거래가 실행 중이 아닙니다"}, status_code=409)

    _is_trading = False
    if _trading_task and not _trading_task.done():
        _trading_task.cancel()
        try:
            await _trading_task
        except asyncio.CancelledError:
            pass
    _trading_task = None

    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------
@app.post("/api/kill-switch/activate")
async def activate_kill_switch(request: Request) -> JSONResponse:
    global _is_trading, _trading_task

    kill_switch: KillSwitch | None = _components.get("kill_switch")
    if not kill_switch:
        return JSONResponse({"ok": False, "error": "Kill switch not initialized"}, status_code=500)

    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    reason = (
        body.get("reason", "웹 대시보드에서 수동 활성화")
        if isinstance(body, dict)
        else "웹 대시보드에서 수동 활성화"
    )

    kill_switch.activate(reason)

    # Also stop trading
    _is_trading = False
    if _trading_task and not _trading_task.done():
        _trading_task.cancel()
        try:
            await _trading_task
        except asyncio.CancelledError:
            pass
    _trading_task = None

    return JSONResponse({"ok": True, "reason": reason})


@app.post("/api/kill-switch/deactivate")
async def deactivate_kill_switch() -> JSONResponse:
    kill_switch: KillSwitch | None = _components.get("kill_switch")
    if not kill_switch:
        return JSONResponse({"ok": False, "error": "Kill switch not initialized"}, status_code=500)

    kill_switch.deactivate()
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# Balances
# ---------------------------------------------------------------------------
@app.get("/api/balances")
async def get_balances() -> JSONResponse:
    broker = _components.get("broker")
    if not broker:
        return JSONResponse({"balances": {}, "total_value_krw": 0})

    try:
        snapshot = await broker.fetch_balances()
        return JSONResponse(_model_to_dict(snapshot))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------
@app.get("/api/positions")
async def get_positions() -> JSONResponse:
    broker = _components.get("broker")
    if not broker:
        return JSONResponse({"positions": []})

    try:
        positions = await broker.fetch_positions()
        return JSONResponse({"positions": [_model_to_dict(p) for p in positions]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------
@app.get("/api/orders")
async def get_orders() -> JSONResponse:
    store: StateStore | None = _components.get("store")
    if not store:
        return JSONResponse({"orders": []})

    orders = store.get_all_orders(limit=100)
    return JSONResponse({"orders": [_model_to_dict(o) for o in orders]})


@app.get("/api/orders/open")
async def get_open_orders() -> JSONResponse:
    broker = _components.get("broker")
    if not broker:
        return JSONResponse({"orders": []})

    try:
        orders = await broker.fetch_open_orders()
        return JSONResponse({"orders": [_model_to_dict(o) for o in orders]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Safety events
# ---------------------------------------------------------------------------
@app.get("/api/safety-events")
async def get_safety_events() -> JSONResponse:
    store: StateStore | None = _components.get("store")
    if not store:
        return JSONResponse({"events": []})

    events = store.get_safety_events(limit=50)
    return JSONResponse({"events": [_model_to_dict(e) for e in events]})


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------
def _read_decision_logs(symbol: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    import json as _json

    settings: Settings | None = _components.get("settings")
    log_dir = settings.log_dir if settings else Path("logs")

    if symbol:
        safe_name = symbol.replace("/", "_").lower()
        files = [log_dir / f"decisions_{safe_name}.jsonl"]
    else:
        files = sorted(log_dir.glob("decisions_*.jsonl"))

    entries: list[dict[str, Any]] = []
    for f in files:
        if not f.exists():
            continue
        try:
            lines = f.read_text(encoding="utf-8").strip().splitlines()
            for line in lines[-limit:]:
                try:
                    entries.append(_json.loads(line))
                except _json.JSONDecodeError:
                    pass
        except Exception:
            pass

    entries.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return entries[:limit]


@app.get("/api/decisions")
async def get_decisions() -> JSONResponse:
    return JSONResponse({"decisions": _read_decision_logs(limit=100)})


@app.get("/api/decisions/{symbol_path:path}")
async def get_decisions_by_symbol(symbol_path: str) -> JSONResponse:
    symbol = symbol_path.replace("_", "/").upper()
    return JSONResponse({"decisions": _read_decision_logs(symbol=symbol, limit=100)})


# ---------------------------------------------------------------------------
# P&L summary
# ---------------------------------------------------------------------------
@app.get("/api/pnl")
async def get_pnl() -> JSONResponse:
    broker = _components.get("broker")
    store: StateStore | None = _components.get("store")
    settings: Settings | None = _components.get("settings")

    if not broker:
        return JSONResponse({"error": "broker not initialized"}, status_code=500)

    try:
        balances = await broker.fetch_balances()
        positions = await broker.fetch_positions()

        total_value = float(balances.total_value_krw or 0)

        is_paper = settings and settings.trading_mode == TradingMode.PAPER
        if is_paper:
            initial_balance = 1_000_000.0
        else:
            try:
                initial_balance = float(await broker.get_net_deposits())
            except Exception:
                initial_balance = total_value  # fallback

        total_pnl = total_value - initial_balance
        total_pnl_pct = (total_pnl / initial_balance * 100) if initial_balance > 0 else 0.0

        position_pnls = []
        for p in positions:
            pd = _model_to_dict(p)
            position_pnls.append(
                {
                    "symbol": pd.get("symbol", ""),
                    "quantity": pd.get("quantity", 0),
                    "average_entry_price": pd.get("average_entry_price", 0),
                    "current_price": pd.get("current_price", 0),
                    "unrealized_pnl": pd.get("unrealized_pnl", 0),
                    "unrealized_pnl_pct": pd.get("unrealized_pnl_pct", 0),
                }
            )

        return JSONResponse(
            {
                "total_value_krw": total_value,
                "initial_balance_krw": initial_balance,
                "total_pnl_krw": round(total_pnl, 2),
                "total_pnl_pct": round(total_pnl_pct, 4),
                "positions": position_pnls,
                "is_paper": is_paper,
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Manual orders
# ---------------------------------------------------------------------------
@app.post("/api/orders/manual-buy")
async def manual_buy(request: Request) -> JSONResponse:
    from coin_trader.execution.engine import ExecutionEngine

    engine: ExecutionEngine | None = _components.get("engine")
    broker = _components.get("broker")
    exchange_adapter = _components.get("exchange_adapter")
    settings: Settings | None = _components.get("settings")

    if not engine or not broker or not exchange_adapter or not settings:
        return JSONResponse(
            {"ok": False, "error": "시스템 컴포넌트가 초기화되지 않았습니다"}, status_code=500
        )

    try:
        body = await request.json()
        symbol = body.get("symbol")
        amount_krw = body.get("amount_krw")

        if not symbol or not amount_krw:
            return JSONResponse({"ok": False, "error": "심볼과 금액을 입력하세요"}, status_code=400)

        if symbol not in settings.trading_symbols:
            return JSONResponse(
                {"ok": False, "error": f"{symbol}은(는) 거래 가능한 심볼이 아닙니다"},
                status_code=400,
            )

        ticker = await exchange_adapter.get_ticker(symbol)
        current_price = Decimal(str(ticker.get("trade_price", 0)))

        if current_price <= 0:
            return JSONResponse(
                {"ok": False, "error": "현재가를 가져올 수 없습니다"}, status_code=500
            )

        balances = await broker.fetch_balances()
        positions = await broker.fetch_positions()

        state: dict[str, object] = {
            "total_balance": balances.total_value_krw or Decimal("0"),
            "position_count": len(positions),
            "today_pnl": Decimal("0"),
            "market_price": current_price,
        }

        intent = OrderIntent(
            signal_id=uuid4(),
            exchange=settings.exchange,
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quote_quantity=Decimal(str(amount_krw)),
            price=current_price,
            reason="manual_buy via web dashboard",
            timestamp=datetime.now(timezone.utc),
        )

        order = await engine.execute(intent, state)

        if order:
            return JSONResponse({"ok": True, "order": _model_to_dict(order)})
        else:
            return JSONResponse(
                {"ok": False, "error": "주문이 리스크 체크에서 거부되었습니다"}, status_code=400
            )

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/orders/manual-sell")
async def manual_sell(request: Request) -> JSONResponse:
    from coin_trader.execution.engine import ExecutionEngine

    engine: ExecutionEngine | None = _components.get("engine")
    broker = _components.get("broker")
    exchange_adapter = _components.get("exchange_adapter")
    settings: Settings | None = _components.get("settings")

    if not engine or not broker or not exchange_adapter or not settings:
        return JSONResponse(
            {"ok": False, "error": "시스템 컴포넌트가 초기화되지 않았습니다"}, status_code=500
        )

    try:
        body = await request.json()
        symbol = body.get("symbol")
        quantity = body.get("quantity")

        if not symbol:
            return JSONResponse({"ok": False, "error": "심볼을 입력하세요"}, status_code=400)

        positions = await broker.fetch_positions()
        position = next((p for p in positions if p.symbol == symbol), None)

        if not position:
            return JSONResponse(
                {"ok": False, "error": f"{symbol} 포지션이 없습니다"}, status_code=400
            )

        sell_qty = Decimal(str(quantity)) if quantity else position.quantity

        if sell_qty <= 0 or sell_qty > position.quantity:
            return JSONResponse(
                {"ok": False, "error": "매도 수량이 유효하지 않습니다"}, status_code=400
            )

        ticker = await exchange_adapter.get_ticker(symbol)
        current_price = Decimal(str(ticker.get("trade_price", 0)))

        if current_price <= 0:
            return JSONResponse(
                {"ok": False, "error": "현재가를 가져올 수 없습니다"}, status_code=500
            )

        balances = await broker.fetch_balances()

        state: dict[str, object] = {
            "total_balance": balances.total_value_krw or Decimal("0"),
            "position_count": len(positions),
            "today_pnl": Decimal("0"),
            "market_price": current_price,
        }

        intent = OrderIntent(
            signal_id=uuid4(),
            exchange=settings.exchange,
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=sell_qty,
            price=current_price,
            reason="manual_sell via web dashboard",
            timestamp=datetime.now(timezone.utc),
        )

        order = await engine.execute(intent, state)

        if order:
            return JSONResponse({"ok": True, "order": _model_to_dict(order)})
        else:
            return JSONResponse(
                {"ok": False, "error": "주문이 리스크 체크에서 거부되었습니다"}, status_code=400
            )

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
