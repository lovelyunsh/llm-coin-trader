"""Base exchange adapter with httpx + tenacity retry + rate limiting"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from types import TracebackType
from typing import Self, TypeAlias, cast

_log = logging.getLogger(__name__)

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_API_METRICS: dict[str, object] = {
    "total_requests": 0,
    "total_429": 0,
    "last_429_ts": None,
    "per_endpoint": {},
}


def _to_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def get_api_metrics() -> dict[str, object]:
    per_endpoint_raw = _API_METRICS.get("per_endpoint", {})
    per_endpoint = dict(per_endpoint_raw) if isinstance(per_endpoint_raw, dict) else {}
    return {
        "total_requests": _to_int(_API_METRICS.get("total_requests", 0), 0),
        "total_429": _to_int(_API_METRICS.get("total_429", 0), 0),
        "last_429_ts": _API_METRICS.get("last_429_ts"),
        "per_endpoint": per_endpoint,
    }


class BaseExchangeAdapter:
    JsonObject: TypeAlias = dict[str, object]
    JsonArray: TypeAlias = list[object]
    Json: TypeAlias = JsonObject | JsonArray

    base_url: str
    rate_limit_delay: float
    _client: httpx.AsyncClient
    _rate_limit_lock: asyncio.Lock
    _last_request_time: float

    def __init__(
        self, base_url: str, rate_limit_delay: float = 0.15, timeout: float = 30.0
    ) -> None:
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self._rate_limit_lock = asyncio.Lock()
        self._last_request_time = 0.0

    async def _pre_request_hook(self, url: str) -> None:
        """Override in subclasses to wait based on rate-limit headers before a request."""

    def _post_response_hook(self, url: str, headers: httpx.Headers) -> None:
        """Override in subclasses to update rate-limit state from response headers."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
        reraise=True,
    )
    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        json: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Json:
        async with self._rate_limit_lock:
            response: httpx.Response | None = None
            for attempt in range(3):
                now = asyncio.get_running_loop().time()
                wait_for = self.rate_limit_delay - (now - self._last_request_time)
                if wait_for > 0:
                    await asyncio.sleep(wait_for)

                await self._pre_request_hook(url)

                response = await self._client.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    headers=headers,
                )
                _API_METRICS["total_requests"] = (
                    _to_int(_API_METRICS.get("total_requests", 0), 0) + 1
                )
                per_endpoint_raw = _API_METRICS.get("per_endpoint", {})
                per_endpoint: dict[str, int]
                if isinstance(per_endpoint_raw, dict):
                    per_endpoint = cast(dict[str, int], per_endpoint_raw)
                else:
                    per_endpoint = {}
                endpoint_key = f"{method.upper()} {url}"
                per_endpoint[endpoint_key] = per_endpoint.get(endpoint_key, 0) + 1
                _API_METRICS["per_endpoint"] = per_endpoint
                self._last_request_time = asyncio.get_running_loop().time()
                self._post_response_hook(url, response.headers)

                if response.status_code != 429:
                    response.raise_for_status()
                    return cast(BaseExchangeAdapter.Json, response.json())

                _API_METRICS["total_429"] = (
                    _to_int(_API_METRICS.get("total_429", 0), 0) + 1
                )
                _API_METRICS["last_429_ts"] = asyncio.get_running_loop().time()

                retry_after = response.headers.get("Retry-After")
                retry_wait = 1.0
                if retry_after:
                    try:
                        retry_wait = max(float(retry_after), 0.5)
                    except ValueError:
                        retry_wait = 1.0
                retry_wait = max(retry_wait, 0.5 * (2**attempt))
                _log.warning(
                    "exchange_rate_limited_429 url=%s attempt=%d retry_in=%.1fs",
                    url,
                    attempt,
                    retry_wait,
                )
                await asyncio.sleep(retry_wait)

            if response is None:
                raise RuntimeError("HTTP request failed without response")
            response.raise_for_status()
            return cast(BaseExchangeAdapter.Json, response.json())

    def normalize_symbol(self, symbol: str) -> str:
        return symbol

    def denormalize_symbol(self, symbol: str) -> str:
        return symbol

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()
