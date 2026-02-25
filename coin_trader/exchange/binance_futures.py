"""Binance USDT-M perpetual futures exchange adapter.

API docs: https://binance-docs.github.io/apidocs/futures/en/
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from decimal import Decimal
from typing import TypeAlias, cast
from urllib.parse import urlencode

import httpx
from typing_extensions import override

from coin_trader.exchange.base import BaseExchangeAdapter

_log = logging.getLogger(__name__)

# Binance USDT-M futures: 2400 weight/minute limit
_WEIGHT_LIMIT = 2400
_WEIGHT_WARN = 2000   # start throttling above this (~83%)
_WEIGHT_HARD = 2300   # heavy throttle above this (~96%)

BINANCE_FUTURES_PROD_URL: str = "https://fapi.binance.com"
BINANCE_FUTURES_TESTNET_URL: str = "https://testnet.binancefuture.com"

QueryParams: TypeAlias = dict[str, str]
JsonObject: TypeAlias = dict[str, object]
JsonArray: TypeAlias = list[object]


def _as_str_object_dict(value: object) -> JsonObject | None:
    if not isinstance(value, dict):
        return None
    raw = cast(dict[object, object], value)
    for key in raw.keys():
        if not isinstance(key, str):
            return None
    return cast(JsonObject, raw)


def _to_decimal(value: object, default: Decimal = Decimal("0")) -> Decimal:
    """Convert any value to Decimal safely."""
    try:
        return Decimal(str(value))
    except Exception:
        return default


def _normalize_ticker(raw: JsonObject) -> JsonObject:
    """Convert Binance 24hr ticker response to canonical field names."""
    price = _to_decimal(raw.get("lastPrice", "0"))
    open_price = _to_decimal(raw.get("openPrice", "0"))
    high = _to_decimal(raw.get("highPrice", "0"))
    low = _to_decimal(raw.get("lowPrice", "0"))
    volume = _to_decimal(raw.get("volume", "0"))
    bid = _to_decimal(raw.get("bidPrice", "0"))
    ask = _to_decimal(raw.get("askPrice", "0"))
    prev_close = _to_decimal(raw.get("prevClosePrice", "0"))
    turnover = _to_decimal(raw.get("quoteVolume", "0"))
    change_rate = _to_decimal(raw.get("priceChangePercent", "0")) / Decimal("100")

    return {
        "price": price,
        "open": open_price,
        "high": high,
        "low": low,
        "volume_24h": volume,
        "bid": bid,
        "ask": ask,
        "prev_close": prev_close,
        "turnover_24h": turnover,
        "change_rate": change_rate,
    }


def _normalize_candle(raw: list[object]) -> JsonObject:
    """Convert Binance kline array to canonical OHLCV dict.

    Kline format: [openTime, open, high, low, close, volume, closeTime, quoteVolume, ...]
    """
    # Guard against short arrays
    if len(raw) < 6:  # noqa: PLR2004
        return {
            "open": Decimal("0"),
            "high": Decimal("0"),
            "low": Decimal("0"),
            "close": Decimal("0"),
            "volume": Decimal("0"),
        }
    return {
        "open": _to_decimal(raw[1]),
        "high": _to_decimal(raw[2]),
        "low": _to_decimal(raw[3]),
        "close": _to_decimal(raw[4]),
        "volume": _to_decimal(raw[5]),
    }


def _normalize_orderbook(raw: JsonObject) -> JsonObject:
    """Convert Binance depth response to canonical [[price, qty], ...] format."""
    bids_raw = raw.get("bids", [])
    asks_raw = raw.get("asks", [])

    bids: list[list[Decimal]] = []
    asks: list[list[Decimal]] = []

    if isinstance(bids_raw, list):
        for entry in bids_raw:
            if isinstance(entry, list) and len(entry) >= 2:  # noqa: PLR2004
                bids.append([_to_decimal(entry[0]), _to_decimal(entry[1])])

    if isinstance(asks_raw, list):
        for entry in asks_raw:
            if isinstance(entry, list) and len(entry) >= 2:  # noqa: PLR2004
                asks.append([_to_decimal(entry[0]), _to_decimal(entry[1])])

    return {"bids": bids, "asks": asks}


class BinanceFuturesAdapter(BaseExchangeAdapter):
    """Binance USDT-M perpetual futures exchange adapter.

    Supports both production and testnet endpoints.
    All authenticated endpoints require api_key and api_secret.
    """

    _api_key: str
    _api_secret: str
    _testnet: bool

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        rate_limit_delay: float = 0.1,
        timeout: float = 30.0,
    ) -> None:
        base_url = BINANCE_FUTURES_TESTNET_URL if testnet else BINANCE_FUTURES_PROD_URL
        super().__init__(
            base_url=base_url, rate_limit_delay=rate_limit_delay, timeout=timeout
        )
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        # (used_weight, monotonic_time_when_read)
        self._weight_state: tuple[int, float] | None = None

    @override
    def _post_response_hook(self, url: str, headers: httpx.Headers) -> None:
        raw = headers.get("x-mbx-used-weight-1m", "")
        if raw:
            try:
                self._weight_state = (int(raw), asyncio.get_running_loop().time())
            except (ValueError, RuntimeError):
                pass

    @override
    async def _pre_request_hook(self, url: str) -> None:
        if self._weight_state is None:
            return
        used, read_at = self._weight_state
        now = asyncio.get_running_loop().time()
        elapsed = now - read_at
        if used >= _WEIGHT_HARD:
            # Very close to limit: wait until the 1-minute window resets
            wait = max(0.0, 60.0 - elapsed)
            if wait > 0.1:
                _log.warning("binance_weight_hard used=%d/%d wait=%.1fs", used, _WEIGHT_LIMIT, wait)
                await asyncio.sleep(wait)
        elif used >= _WEIGHT_WARN:
            # Proportional throttle: 0s at 2000, up to 5s at 2300
            wait = (used - _WEIGHT_WARN) / (_WEIGHT_HARD - _WEIGHT_WARN) * 5.0
            _log.info("binance_weight_warn used=%d/%d wait=%.2fs", used, _WEIGHT_LIMIT, wait)
            await asyncio.sleep(wait)

    # ------------------------------------------------------------------
    # Symbol conversion
    # ------------------------------------------------------------------

    @override
    def normalize_symbol(self, symbol: str) -> str:
        """Convert Binance symbol to canonical form: 'BTCUSDT' -> 'BTC/USDT'."""
        if "/" in symbol:
            return symbol
        # Binance USDT-M perps always end in USDT
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            return f"{base}/USDT"
        # Fallback: return as-is
        return symbol

    @override
    def denormalize_symbol(self, symbol: str) -> str:
        """Convert canonical symbol to Binance form: 'BTC/USDT' -> 'BTCUSDT'."""
        if "/" in symbol:
            base, quote = symbol.split("/", 1)
            return f"{base}{quote}"
        return symbol

    # ------------------------------------------------------------------
    # Authentication helpers
    # ------------------------------------------------------------------

    def _sign(self, params: QueryParams) -> QueryParams:
        """Add timestamp and HMAC-SHA256 signature to params dict (mutates in place)."""
        params["timestamp"] = str(int(time.time() * 1000))
        query_string = urlencode(params)
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _auth_headers(self) -> dict[str, str]:
        """Return headers required for authenticated endpoints."""
        return {"X-MBX-APIKEY": self._api_key}

    # ------------------------------------------------------------------
    # Public market data
    # ------------------------------------------------------------------

    async def get_ticker(self, symbol: str) -> JsonObject:
        """GET /fapi/v1/ticker/24hr — single symbol."""
        binance_symbol = self.denormalize_symbol(symbol)
        result = await self._request(
            "GET",
            "/fapi/v1/ticker/24hr",
            params={"symbol": binance_symbol},
        )
        raw = _as_str_object_dict(result) or {}
        return _normalize_ticker(raw)

    async def get_tickers(self, symbols: list[str]) -> dict[str, JsonObject]:
        """GET /fapi/v1/ticker/24hr — batch (all symbols, then filter)."""
        if not symbols:
            return {}
        # Binance returns all tickers when no symbol param is given
        result = await self._request("GET", "/fapi/v1/ticker/24hr")
        if not isinstance(result, list):
            return {}

        wanted = {self.denormalize_symbol(s) for s in symbols}
        out: dict[str, JsonObject] = {}
        for item in result:
            t = _as_str_object_dict(item)
            if t is None:
                continue
            binance_sym = t.get("symbol")
            if isinstance(binance_sym, str) and binance_sym in wanted:
                out[self.normalize_symbol(binance_sym)] = _normalize_ticker(t)
        return out

    async def get_candles(
        self,
        symbol: str,
        interval: str = "1h",
        count: int = 200,
    ) -> list[JsonObject]:
        """GET /fapi/v1/klines — OHLCV candles in chronological order.

        Binance kline interval strings (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h,
        6h, 8h, 12h, 1d, 3d, 1w, 1M) are passed through directly.
        """
        binance_symbol = self.denormalize_symbol(symbol)
        result = await self._request(
            "GET",
            "/fapi/v1/klines",
            params={
                "symbol": binance_symbol,
                "interval": interval,
                "limit": str(min(count, 1500)),  # Binance max is 1500
            },
        )
        if not isinstance(result, list):
            return []

        candles: list[JsonObject] = []
        for item in result:
            if isinstance(item, list):
                candles.append(_normalize_candle(item))
        # Binance returns klines in ascending (chronological) order already
        return candles

    async def get_orderbook(self, symbol: str) -> JsonObject:
        """GET /fapi/v1/depth — order book."""
        binance_symbol = self.denormalize_symbol(symbol)
        result = await self._request(
            "GET",
            "/fapi/v1/depth",
            params={"symbol": binance_symbol, "limit": "20"},
        )
        raw = _as_str_object_dict(result) or {}
        return _normalize_orderbook(raw)

    async def get_orderbooks(self, symbols: list[str]) -> dict[str, JsonObject]:
        """Sequential calls to get_orderbook for each symbol."""
        if not symbols:
            return {}
        out: dict[str, JsonObject] = {}
        for symbol in symbols:
            out[symbol] = await self.get_orderbook(symbol)
        return out

    async def get_tradeable_markets(self) -> list[str]:
        """GET /fapi/v1/exchangeInfo — perpetual USDT markets with TRADING status."""
        result = await self._request("GET", "/fapi/v1/exchangeInfo")
        raw = _as_str_object_dict(result) or {}
        symbols_raw = raw.get("symbols", [])
        if not isinstance(symbols_raw, list):
            return []

        markets: list[str] = []
        for item in symbols_raw:
            info = _as_str_object_dict(item)
            if info is None:
                continue
            contract_type = info.get("contractType")
            quote_asset = info.get("quoteAsset")
            status = info.get("status")
            if (
                contract_type == "PERPETUAL"
                and quote_asset == "USDT"
                and status == "TRADING"
            ):
                binance_sym = info.get("symbol")
                if isinstance(binance_sym, str):
                    markets.append(self.normalize_symbol(binance_sym))
        return markets

    async def get_funding_rate(self, symbol: str) -> JsonObject:
        """GET /fapi/v1/fundingRate — latest funding rate for symbol."""
        binance_symbol = self.denormalize_symbol(symbol)
        result = await self._request(
            "GET",
            "/fapi/v1/fundingRate",
            params={"symbol": binance_symbol, "limit": "1"},
        )
        if isinstance(result, list) and result:
            raw = _as_str_object_dict(result[0]) or {}
        else:
            raw = _as_str_object_dict(result) or {}
        return raw

    async def get_mark_price(self, symbol: str) -> JsonObject:
        """GET /fapi/v1/premiumIndex — mark price and index price."""
        binance_symbol = self.denormalize_symbol(symbol)
        result = await self._request(
            "GET",
            "/fapi/v1/premiumIndex",
            params={"symbol": binance_symbol},
        )
        raw = _as_str_object_dict(result) or {}
        # Normalize mark price to Decimal for precision
        if "markPrice" in raw:
            raw = dict(raw)
            raw["markPrice"] = _to_decimal(raw["markPrice"])
        if "indexPrice" in raw:
            raw = dict(raw)
            raw["indexPrice"] = _to_decimal(raw["indexPrice"])
        return raw

    # ------------------------------------------------------------------
    # Authenticated endpoints
    # ------------------------------------------------------------------

    async def get_balances(self) -> dict[str, Decimal]:
        """GET /fapi/v2/balance — account balances (USDT and asset balances)."""
        params = self._sign({})
        result = await self._request(
            "GET",
            "/fapi/v2/balance",
            params=params,
            headers=self._auth_headers(),
        )
        balances: dict[str, Decimal] = {}
        if not isinstance(result, list):
            return balances
        for item in result:
            acct = _as_str_object_dict(item)
            if acct is None:
                continue
            asset = acct.get("asset")
            if not isinstance(asset, str):
                continue
            wallet_balance = _to_decimal(acct.get("balance", "0"))
            if wallet_balance > Decimal("0"):
                balances[asset] = wallet_balance
        return balances

    async def get_open_orders(self, symbol: str | None = None) -> list[JsonObject]:
        """GET /fapi/v1/openOrders — open orders, optionally filtered by symbol."""
        params: QueryParams = {}
        if symbol:
            params["symbol"] = self.denormalize_symbol(symbol)
        self._sign(params)
        result = await self._request(
            "GET",
            "/fapi/v1/openOrders",
            params=params,
            headers=self._auth_headers(),
        )
        if not isinstance(result, list):
            return []
        orders: list[JsonObject] = []
        for item in result:
            order = _as_str_object_dict(item)
            if order is not None:
                orders.append(order)
        return orders

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal | None = None,
    ) -> JsonObject:
        """POST /fapi/v1/order — place a new futures order.

        Args:
            symbol: Canonical symbol, e.g. 'BTC/USDT'.
            side: 'buy' or 'sell' (normalized to 'BUY'/'SELL').
            order_type: 'limit' or 'market' (normalized to 'LIMIT'/'MARKET').
            quantity: Order quantity in base asset.
            price: Limit price (required for limit orders).
        """
        binance_symbol = self.denormalize_symbol(symbol)
        binance_side = side.upper()
        binance_type = order_type.upper()

        params: QueryParams = {
            "symbol": binance_symbol,
            "side": binance_side,
            "type": binance_type,
            "quantity": str(quantity),
        }
        if binance_type == "LIMIT" and price is not None:
            params["price"] = str(price)
            params["timeInForce"] = "GTC"

        self._sign(params)
        result = await self._request(
            "POST",
            "/fapi/v1/order",
            params=params,
            headers=self._auth_headers(),
        )
        return _as_str_object_dict(result) or {}

    async def cancel_order(self, order_id: str, symbol: str) -> JsonObject:
        """DELETE /fapi/v1/order — cancel an open order."""
        binance_symbol = self.denormalize_symbol(symbol)
        params: QueryParams = {
            "symbol": binance_symbol,
            "orderId": order_id,
        }
        self._sign(params)
        result = await self._request(
            "DELETE",
            "/fapi/v1/order",
            params=params,
            headers=self._auth_headers(),
        )
        return _as_str_object_dict(result) or {}

    async def get_positions(self) -> list[JsonObject]:
        """GET /fapi/v2/positionRisk — all position information."""
        params = self._sign({})
        result = await self._request(
            "GET",
            "/fapi/v2/positionRisk",
            params=params,
            headers=self._auth_headers(),
        )
        if not isinstance(result, list):
            return []
        positions: list[JsonObject] = []
        for item in result:
            pos = _as_str_object_dict(item)
            if pos is not None:
                positions.append(pos)
        return positions

    async def set_leverage(self, symbol: str, leverage: int) -> JsonObject:
        """POST /fapi/v1/leverage — set leverage for a symbol."""
        binance_symbol = self.denormalize_symbol(symbol)
        params: QueryParams = {
            "symbol": binance_symbol,
            "leverage": str(leverage),
        }
        self._sign(params)
        result = await self._request(
            "POST",
            "/fapi/v1/leverage",
            params=params,
            headers=self._auth_headers(),
        )
        return _as_str_object_dict(result) or {}

    async def set_margin_mode(self, symbol: str, mode: str) -> JsonObject:
        """POST /fapi/v1/marginType — set margin mode for a symbol.

        Args:
            symbol: Canonical symbol, e.g. 'BTC/USDT'.
            mode: 'ISOLATED' or 'CROSSED'.
        """
        binance_symbol = self.denormalize_symbol(symbol)
        params: QueryParams = {
            "symbol": binance_symbol,
            "marginType": mode.upper(),
        }
        self._sign(params)
        result = await self._request(
            "POST",
            "/fapi/v1/marginType",
            params=params,
            headers=self._auth_headers(),
        )
        return _as_str_object_dict(result) or {}
