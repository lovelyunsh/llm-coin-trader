"""Slack webhook notification for alerts."""

from __future__ import annotations

import asyncio
import http.client
import json
import logging
import urllib.request
from typing import cast

logger = logging.getLogger("notify")

SEVERITY_EMOJI = {
    "low": ":information_source:",
    "medium": ":warning:",
    "high": ":rotating_light:",
    "critical": ":sos:",
}


class SlackNotifier:
    _url: str
    _timeout: float

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url
        self._timeout = 10.0

    async def send_alert(
        self, title: str, message: str, severity: str = "medium"
    ) -> None:
        if not self._url:
            logger.debug("slack_skip reason=no_webhook_url")
            return

        emoji = SEVERITY_EMOJI.get(severity, ":bell:")
        payload = {
            "text": f"{emoji} *[{severity.upper()}] {title}*\n{message}",
        }

        try:
            body = json.dumps(payload).encode("utf-8")
            status = await asyncio.to_thread(self._post_sync, body)
            if status != 200:
                logger.warning("slack_send_failed status=%s", status)
        except Exception as e:
            logger.error("slack_error error=%s", str(e))

    async def close(self) -> None:
        return

    def _post_sync(self, body: bytes) -> int:
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = cast(
            http.client.HTTPResponse, urllib.request.urlopen(req, timeout=self._timeout)
        )
        with resp:
            status = getattr(resp, "status", 0)
            return int(status or 0)
