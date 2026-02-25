"""Upbit exchange adapter - official API: https://docs.upbit.com/reference"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import uuid
from collections.abc import Mapping
from decimal import Decimal
from typing import Protocol, TypeAlias, cast
from urllib.parse import unquote, urlencode

import httpx
import jwt  # PyJWT
from typing_extensions import override

from coin_trader.exchange.base import BaseExchangeAdapter

_log = logging.getLogger(__name__)

# Maps URL path prefixes to Upbit rate-limit groups.
# market group: 10 req/sec (public market data)
# order/trade groups: separate quotas but less frequently hit
_URL_GROUP: list[tuple[str, str]] = [
    ("/candles/", "market"),
    ("/ticker", "market"),
    ("/orderbook", "market"),
    ("/market/", "market"),
    ("/trades/", "market"),
    ("/orders/open", "order"),   # must come before /orders
    ("/orders", "trade"),        # POST/DELETE order placement
    ("/order", "order"),         # GET single order by uuid
    ("/accounts", "default"),
]


class _JwtEncode(Protocol):
    def __call__(
        self,
        payload: Mapping[str, object],
        key: str,
        algorithm: str,
    ) -> str: ...


_jwt_encode = cast(_JwtEncode, jwt.encode)


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


_UPBIT_INTERVAL_MAP: dict[str, str] = {
    "1m": "minutes/1",
    "3m": "minutes/3",
    "5m": "minutes/5",
    "15m": "minutes/15",
    "30m": "minutes/30",
    "1h": "minutes/60",
    "4h": "minutes/240",
    "1d": "days",
    "1w": "weeks",
    "1M": "months",
}


def _to_upbit_interval(canonical: str) -> str:
    """Convert canonical interval (e.g. '1h') to Upbit path segment."""
    return _UPBIT_INTERVAL_MAP.get(canonical, canonical)


def _normalize_ticker(raw: JsonObject) -> JsonObject:
    """Convert Upbit ticker response to canonical field names."""
    return {
        "price": raw.get("trade_price", 0),
        "open": raw.get("opening_price", 0),
        "high": raw.get("high_price", 0),
        "low": raw.get("low_price", 0),
        "volume_24h": raw.get("acc_trade_volume_24h", 0),
        "bid": raw.get("highest_bid", 0),
        "ask": raw.get("lowest_ask", 0),
        "prev_close": raw.get("prev_closing_price", 0),
        "turnover_24h": raw.get("acc_trade_price_24h", 0),
        "change_rate": raw.get("signed_change_rate", 0),
    }


def _normalize_candle(raw: JsonObject) -> JsonObject:
    """Convert Upbit candle response to canonical field names."""
    return {
        "open": raw.get("opening_price", 0),
        "high": raw.get("high_price", 0),
        "low": raw.get("low_price", 0),
        "close": raw.get("trade_price", 0),
        "volume": raw.get("candle_acc_trade_volume", 0),
    }


def _normalize_orderbook(raw: JsonObject) -> JsonObject:
    """Convert Upbit orderbook response to canonical [[price, qty], ...] format."""
    units = raw.get("orderbook_units")
    bids: list[list[object]] = []
    asks: list[list[object]] = []
    if isinstance(units, list):
        for unit in units:
            if not isinstance(unit, dict):
                continue
            bid_p = unit.get("bid_price", 0)
            bid_s = unit.get("bid_size", 0)
            ask_p = unit.get("ask_price", 0)
            ask_s = unit.get("ask_size", 0)
            bids.append([bid_p, bid_s])
            asks.append([ask_p, ask_s])
    return {"bids": bids, "asks": asks}


class UpbitAdapter(BaseExchangeAdapter):
    UPBIT_API_URL: str = "https://api.upbit.com/v1"

    _access_key: str
    _secret_key: str

    def __init__(
        self, access_key: str, secret_key: str, rate_limit_delay: float = 0.12
    ) -> None:
        super().__init__(base_url=self.UPBIT_API_URL, rate_limit_delay=rate_limit_delay)
        self._access_key = access_key
        self._secret_key = secret_key
        # group -> (sec_remaining, monotonic_time_when_read)
        self._group_remaining: dict[str, tuple[int, float]] = {}

    def _url_to_group(self, url: str) -> str:
        for prefix, group in _URL_GROUP:
            if url.startswith(prefix):
                return group
        return "default"

    @override
    def _post_response_hook(self, url: str, headers: httpx.Headers) -> None:
        raw = headers.get("Remaining-Req", "")
        if not raw:
            return
        g = re.search(r"group=(\w+)", raw)
        s = re.search(r"sec=(\d+)", raw)
        if g and s:
            self._group_remaining[g.group(1)] = (int(s.group(1)), asyncio.get_running_loop().time())

    @override
    async def _pre_request_hook(self, url: str) -> None:
        group = self._url_to_group(url)
        state = self._group_remaining.get(group)
        if state is None:
            return
        remaining_sec, read_at = state
        if remaining_sec <= 1:
            elapsed = asyncio.get_running_loop().time() - read_at
            wait = 1.0 - elapsed
            if wait > 0.01:
                _log.warning("upbit_rate_throttle group=%s sec_remaining=%d wait=%.3fs", group, remaining_sec, wait)
                await asyncio.sleep(wait)

    def _create_token(self, query_params: QueryParams | None = None) -> str:
        payload: dict[str, str] = {
            "access_key": self._access_key,
            "nonce": str(uuid.uuid4()),
        }
        if query_params:
            query_string = unquote(urlencode(query_params, doseq=True))
            m = hashlib.sha512()
            m.update(query_string.encode())
            payload["query_hash"] = m.hexdigest()
            payload["query_hash_alg"] = "SHA512"

        token = _jwt_encode(payload, self._secret_key, algorithm="HS256")
        return f"Bearer {token}"

    def _auth_headers(self, query_params: QueryParams | None = None) -> dict[str, str]:
        return {"Authorization": self._create_token(query_params)}

    @override
    def normalize_symbol(self, symbol: str) -> str:
        # "KRW-BTC" -> "BTC/KRW"
        if "-" in symbol:
            quote, base = symbol.split("-", 1)
            return f"{base}/{quote}"
        return symbol

    @override
    def denormalize_symbol(self, symbol: str) -> str:
        # "BTC/KRW" -> "KRW-BTC"
        if "/" in symbol:
            base, quote = symbol.split("/", 1)
            return f"{quote}-{base}"
        return symbol

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal | None = None,
    ) -> JsonObject:
        market = self.denormalize_symbol(symbol)
        upbit_side = "bid" if side == "buy" else "ask"

        body: dict[str, str] = {"market": market, "side": upbit_side}

        if order_type == "limit":
            body["ord_type"] = "limit"
            body["volume"] = str(quantity)
            if price is not None:
                body["price"] = str(int(price))  # Upbit KRW prices are integers
        elif side == "buy":
            # Market buy: specify total KRW amount
            body["ord_type"] = "price"
            if price is not None:
                body["price"] = str(int(quantity * price))
            else:
                body["price"] = str(int(quantity))
        else:
            # Market sell: specify volume
            body["ord_type"] = "market"
            body["volume"] = str(quantity)

        headers = self._auth_headers(body)
        result = await self._request("POST", "/orders", json=body, headers=headers)
        return _as_str_object_dict(result) or {}

    async def cancel_order(self, order_id: str, _symbol: str) -> None:
        params: QueryParams = {"uuid": order_id}
        headers = self._auth_headers(params)
        _ = await self._request("DELETE", "/order", params=params, headers=headers)

    async def get_open_orders(self, symbol: str | None = None) -> list[JsonObject]:
        params: QueryParams = {"state": "wait"}
        if symbol:
            params["market"] = self.denormalize_symbol(symbol)
        headers = self._auth_headers(params)
        result = await self._request(
            "GET", "/orders/open", params=params, headers=headers
        )
        if not isinstance(result, list):
            return []
        orders: list[JsonObject] = []
        for item in result:
            order = _as_str_object_dict(item)
            if order is not None:
                orders.append(order)
        return orders

    async def get_balances(self) -> dict[str, Decimal]:
        headers = self._auth_headers()
        result = await self._request("GET", "/accounts", headers=headers)
        balances: dict[str, Decimal] = {}
        if isinstance(result, list):
            for item in result:
                acct = _as_str_object_dict(item)
                if acct is None:
                    continue

                currency_obj = acct.get("currency")
                if not isinstance(currency_obj, str):
                    continue

                balance = Decimal(str(acct.get("balance", "0")))
                locked = Decimal(str(acct.get("locked", "0")))
                total = balance + locked
                if total > 0:
                    balances[currency_obj] = total
        return balances

    async def get_accounts_raw(self) -> list[JsonObject]:
        headers = self._auth_headers()
        result = await self._request("GET", "/accounts", headers=headers)
        if not isinstance(result, list):
            return []
        accounts: list[JsonObject] = []
        for item in result:
            acct = _as_str_object_dict(item)
            if acct is not None:
                accounts.append(acct)
        return accounts

    async def get_ticker(self, symbol: str) -> JsonObject:
        market = self.denormalize_symbol(symbol)
        result = await self._request("GET", "/ticker", params={"markets": market})
        if not isinstance(result, list) or not result:
            return {}
        raw = _as_str_object_dict(result[0]) or {}
        return _normalize_ticker(raw)

    async def get_tickers(self, symbols: list[str]) -> dict[str, JsonObject]:
        if not symbols:
            return {}
        markets = ",".join(self.denormalize_symbol(s) for s in symbols)
        result = await self._request("GET", "/ticker", params={"markets": markets})
        if not isinstance(result, list):
            return {}
        out: dict[str, JsonObject] = {}
        for item in result:
            t = _as_str_object_dict(item)
            if t is None:
                continue
            market_str = t.get("market")
            if isinstance(market_str, str):
                out[self.normalize_symbol(market_str)] = _normalize_ticker(t)
        return out

    async def get_krw_markets(self) -> list[str]:
        return await self.get_tradeable_markets()

    async def get_tradeable_markets(self) -> list[str]:
        result = await self._request(
            "GET", "/market/all", params={"isDetails": "false"}
        )
        if not isinstance(result, list):
            return []

        symbols: list[str] = []
        for item in result:
            m = _as_str_object_dict(item)
            if m is None:
                continue
            market = m.get("market")
            if not isinstance(market, str):
                continue
            if not market.startswith("KRW-"):
                continue
            symbols.append(self.normalize_symbol(market))
        return symbols

    async def get_order(self, order_id: str) -> JsonObject:
        params: QueryParams = {"uuid": order_id}
        headers = self._auth_headers(params)
        result = await self._request("GET", "/order", params=params, headers=headers)
        return _as_str_object_dict(result) or {}

    async def get_orderbook(self, symbol: str) -> JsonObject:
        market = self.denormalize_symbol(symbol)
        result = await self._request("GET", "/orderbook", params={"markets": market})
        if not isinstance(result, list) or not result:
            return {}
        raw = _as_str_object_dict(result[0]) or {}
        return _normalize_orderbook(raw)

    async def get_orderbooks(self, symbols: list[str]) -> dict[str, JsonObject]:
        """Batch orderbook fetch. Returns {symbol: orderbook_data}."""
        if not symbols:
            return {}
        markets = ",".join(self.denormalize_symbol(s) for s in symbols)
        result = await self._request("GET", "/orderbook", params={"markets": markets})
        if not isinstance(result, list):
            return {}
        out: dict[str, JsonObject] = {}
        for item in result:
            ob = _as_str_object_dict(item)
            if ob is None:
                continue
            market_str = ob.get("market")
            if isinstance(market_str, str):
                out[self.normalize_symbol(market_str)] = _normalize_orderbook(ob)
        return out

    async def get_candles(
        self, symbol: str, interval: str = "1h", count: int = 200
    ) -> list[JsonObject]:
        """Get candle data. Accepts canonical intervals (1h, 1d) or Upbit native (minutes/60, days)."""
        upbit_interval = _to_upbit_interval(interval)
        market = self.denormalize_symbol(symbol)
        result = await self._request(
            "GET",
            f"/candles/{upbit_interval}",
            params={"market": market, "count": str(min(count, 200))},
        )
        if not isinstance(result, list):
            return []
        candles: list[JsonObject] = []
        for item in result:
            raw = _as_str_object_dict(item)
            if raw is not None:
                candles.append(_normalize_candle(raw))
        # Upbit returns candles in descending order (newest first).
        # Reverse to chronological (oldest first) for correct indicator computation.
        candles.reverse()
        return candles

    async def get_deposits(self, limit: int = 100) -> list[JsonObject]:
        """Get deposit history. Upbit API: GET /v1/deposits"""
        params: QueryParams = {"limit": str(limit), "order_by": "asc"}
        headers = self._auth_headers(params)
        result = await self._request("GET", "/deposits", params=params, headers=headers)
        if not isinstance(result, list):
            return []
        deposits: list[JsonObject] = []
        for item in result:
            d = _as_str_object_dict(item)
            if d is not None:
                deposits.append(d)
        return deposits

    async def get_withdraws(self, limit: int = 100) -> list[JsonObject]:
        """Get withdrawal history. Upbit API: GET /v1/withdraws"""
        params: QueryParams = {"limit": str(limit), "order_by": "asc"}
        headers = self._auth_headers(params)
        result = await self._request(
            "GET", "/withdraws", params=params, headers=headers
        )
        if not isinstance(result, list):
            return []
        withdraws: list[JsonObject] = []
        for item in result:
            w = _as_str_object_dict(item)
            if w is not None:
                withdraws.append(w)
        return withdraws
