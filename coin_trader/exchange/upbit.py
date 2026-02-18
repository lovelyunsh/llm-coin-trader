"""Upbit exchange adapter - official API: https://docs.upbit.com/reference"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Mapping
from decimal import Decimal
from typing import Protocol, TypeAlias, cast
from urllib.parse import unquote, urlencode

import jwt  # PyJWT
from typing_extensions import override

from coin_trader.exchange.base import BaseExchangeAdapter


class _JwtEncode(Protocol):
    def __call__(
        self,
        payload: Mapping[str, object],
        key: str,
        algorithm: str,
    ) -> str:
        ...


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


class UpbitAdapter(BaseExchangeAdapter):
    UPBIT_API_URL: str = "https://api.upbit.com/v1"

    _access_key: str
    _secret_key: str

    def __init__(self, access_key: str, secret_key: str, rate_limit_delay: float = 0.2) -> None:
        super().__init__(base_url=self.UPBIT_API_URL, rate_limit_delay=rate_limit_delay)
        self._access_key = access_key
        self._secret_key = secret_key

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
        result = await self._request("GET", "/orders/open", params=params, headers=headers)
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
        return _as_str_object_dict(result[0]) or {}

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
                out[self.normalize_symbol(market_str)] = t
        return out

    async def get_orderbook(self, symbol: str) -> JsonObject:
        market = self.denormalize_symbol(symbol)
        result = await self._request("GET", "/orderbook", params={"markets": market})
        if not isinstance(result, list) or not result:
            return {}
        return _as_str_object_dict(result[0]) or {}

    async def get_candles(
        self, symbol: str, interval: str = "minutes/60", count: int = 200
    ) -> list[JsonObject]:
        """Get candle data. interval examples: minutes/1, minutes/60, days"""
        market = self.denormalize_symbol(symbol)
        result = await self._request(
            "GET",
            f"/candles/{interval}",
            params={"market": market, "count": str(min(count, 200))},
        )
        if not isinstance(result, list):
            return []
        candles: list[JsonObject] = []
        for item in result:
            candle = _as_str_object_dict(item)
            if candle is not None:
                candles.append(candle)
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
        result = await self._request("GET", "/withdraws", params=params, headers=headers)
        if not isinstance(result, list):
            return []
        withdraws: list[JsonObject] = []
        for item in result:
            w = _as_str_object_dict(item)
            if w is not None:
                withdraws.append(w)
        return withdraws
