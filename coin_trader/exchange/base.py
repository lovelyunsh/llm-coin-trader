"""Base exchange adapter with httpx + tenacity retry + rate limiting"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import TracebackType
from typing import Self, TypeAlias, cast

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class BaseExchangeAdapter:
    JsonObject: TypeAlias = dict[str, object]
    JsonArray: TypeAlias = list[object]
    Json: TypeAlias = JsonObject | JsonArray

    base_url: str
    rate_limit_delay: float
    _client: httpx.AsyncClient
    _rate_limit_lock: asyncio.Lock
    _last_request_time: float

    def __init__(self, base_url: str, rate_limit_delay: float = 0.1, timeout: float = 30.0) -> None:
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self._rate_limit_lock = asyncio.Lock()
        self._last_request_time = 0.0

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

                response = await self._client.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    headers=headers,
                )
                self._last_request_time = asyncio.get_running_loop().time()

                if response.status_code != 429:
                    response.raise_for_status()
                    return cast(BaseExchangeAdapter.Json, response.json())

                retry_after = response.headers.get("Retry-After")
                retry_wait = 1.0
                if retry_after:
                    try:
                        retry_wait = max(float(retry_after), 0.5)
                    except ValueError:
                        retry_wait = 1.0
                retry_wait = max(retry_wait, 0.5 * (2**attempt))
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
