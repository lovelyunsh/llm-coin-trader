"""Coinness Telegram channel parser.

Parses the public preview page of the Coinness KR Telegram channel
(https://t.me/s/coinnesskr) to extract recent news items.
"""

from __future__ import annotations

import dataclasses
import logging
import re

import httpx

logger = logging.getLogger("news")

_CHANNEL_URL = "https://t.me/s/coinnesskr"
_TIMEOUT = 15.0

# Regex patterns for telegram public preview HTML
_MSG_BLOCK_RE = re.compile(
    r'<div class="tgme_widget_message_wrap[^"]*"[^>]*>'
    r'.*?data-post="coinnesskr/(\d+)"'
    r".*?</div>\s*</div>\s*</div>",
    re.DOTALL,
)
_MSG_TEXT_RE = re.compile(
    r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>',
    re.DOTALL,
)
_MSG_TIME_RE = re.compile(
    r'<time[^>]*datetime="([^"]+)"',
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclasses.dataclass(frozen=True, slots=True)
class NewsItem:
    """A single news message from Coinness Telegram."""

    message_id: str
    content: str
    published_at: str  # ISO 8601


def _strip_html(html: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    text = _HTML_TAG_RE.sub(" ", html)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    return _WHITESPACE_RE.sub(" ", text).strip()


async def fetch_news() -> list[NewsItem]:
    """Fetch recent news from Coinness Telegram public preview.

    Returns a list of NewsItem sorted by message_id ascending (oldest first).
    """
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; CoinTrader/1.0)",
                "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.5",
            },
        ) as client:
            resp = await client.get(_CHANNEL_URL)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        logger.error("coinness_fetch_error error=%s", str(e))
        return []

    items: list[NewsItem] = []

    # Split by message blocks using data-post attribute
    for match in re.finditer(
        r'data-post="coinnesskr/(\d+)"(.*?)(?=data-post="coinnesskr/|$)',
        html,
        re.DOTALL,
    ):
        msg_id = match.group(1)
        block = match.group(2)

        # Extract text
        text_match = _MSG_TEXT_RE.search(block)
        if not text_match:
            continue
        content = _strip_html(text_match.group(1))
        if not content:
            continue

        # Extract time
        time_match = _MSG_TIME_RE.search(block)
        published_at = time_match.group(1) if time_match else ""

        items.append(
            NewsItem(
                message_id=msg_id,
                content=content,
                published_at=published_at,
            )
        )

    # Sort oldest first
    items.sort(key=lambda x: int(x.message_id))

    logger.info("coinness_parsed count=%d", len(items))
    return items
