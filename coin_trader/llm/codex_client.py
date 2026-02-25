from __future__ import annotations

import asyncio
import json
import logging
from typing import cast

import httpx

logger = logging.getLogger("llm")

_CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"

_rate_lock: asyncio.Lock = asyncio.Lock()
_last_call_ts: float = 0.0


def _extract_output_text_from_output(output: object) -> str | None:
    output_list = _as_list(output)
    if output_list is None:
        return None

    parts: list[str] = []
    for out_obj in output_list:
        out = _as_dict(out_obj)
        if out is None:
            continue
        content_list = _as_list(out.get("content"))
        if content_list is None:
            continue
        for item_obj in content_list:
            item = _as_dict(item_obj)
            if item is None:
                continue
            if item.get("type") != "output_text":
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)

    if not parts:
        return None
    return "".join(parts)


def parse_sse_text(raw: str) -> str:
    delta_parts: list[str] = []
    final_text: str | None = None

    for line in raw.split("\n"):
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue

        try:
            event_obj = cast(object, json.loads(payload))
        except json.JSONDecodeError:
            continue

        event = _as_dict(event_obj)
        if event is None:
            continue

        if event.get("type") == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                delta_parts.append(delta)

        output_text = _extract_output_text_from_output(event.get("output"))
        if isinstance(output_text, str) and output_text:
            final_text = output_text

    reply = "".join(delta_parts) if delta_parts else (final_text or "")
    return reply.strip()


async def query_codex(
    access_token: str,
    account_id: str,
    model: str,
    system_prompt: str,
    user_message: str,
    timeout: float = 60.0,
    min_interval_seconds: float = 0.0,
) -> str | None:
    global _last_call_ts

    if not access_token or not account_id or not model:
        return None

    headers = {
        "Authorization": f"Bearer {access_token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": "opencode",
        "accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    body: dict[str, object] = {
        "model": model,
        "stream": True,
        "store": False,
        "instructions": system_prompt,
        "reasoning": {"effort": "medium", "summary": "auto"},
        "text": {"verbosity": "medium"},
        "include": ["reasoning.encrypted_content"],
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_message}],
            }
        ],
    }

    try:
        async with _rate_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            delay = min_interval_seconds - (now - _last_call_ts)
            if delay > 0:
                await asyncio.sleep(delay)

            delta_parts: list[str] = []
            final_text: str | None = None

            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                async with client.stream(
                    "POST", _CODEX_URL, headers=headers, json=body
                ) as resp:
                    _ = resp.raise_for_status()

                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload or payload == "[DONE]":
                            continue

                        try:
                            event_obj = cast(object, json.loads(payload))
                        except json.JSONDecodeError:
                            continue

                        event = _as_dict(event_obj)
                        if event is None:
                            continue

                        if event.get("type") == "response.output_text.delta":
                            delta = event.get("delta")
                            if isinstance(delta, str) and delta:
                                delta_parts.append(delta)

                        output_text = _extract_output_text_from_output(
                            event.get("output")
                        )
                        if isinstance(output_text, str) and output_text:
                            final_text = output_text

            _last_call_ts = asyncio.get_running_loop().time()

        reply = "".join(delta_parts) if delta_parts else (final_text or "")
        reply = reply.strip()
        return reply or None
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        raw_text = ""
        try:
            raw_text = e.response.text
        except Exception:
            raw_text = ""
        logger.error(
            "codex_http_status_error status=%s body=%s", str(status), raw_text[:1000]
        )
        return None
    except Exception:
        logger.exception("codex_query_failed")
        return None


def _as_list(obj: object) -> list[object] | None:
    if not isinstance(obj, list):
        return None
    return cast(list[object], obj)


def _as_dict(obj: object) -> dict[str, object] | None:
    if not isinstance(obj, dict):
        return None
    d = cast(dict[object, object], obj)
    out: dict[str, object] = {}
    for k, v in d.items():
        if isinstance(k, str):
            out[k] = v
    return out
