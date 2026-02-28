"""News summarizer using existing LLM client.

Generates a concise market news summary from recent news items
for injection into the trading LLM prompt.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coin_trader.llm.advisory import LLMAdvisor

logger = logging.getLogger("news")

_SUMMARIZER_SYSTEM_PROMPT = (
    "You are a crypto market news analyst. "
    "Summarize the given news items into 3-5 lines in Korean. "
    "Rules:\n"
    "- Include timestamps (MM-DD HH:MM KST format) for each key event\n"
    "- Remove duplicate/redundant items\n"
    "- Prioritize by market impact (geopolitical > regulatory > whale moves > technical)\n"
    "- Focus on events that could move BTC/altcoin prices\n"
    "- Output plain text lines, numbered 1-5\n"
    "- Be concise, each line under 80 chars"
)


async def summarize_news(
    advisor: LLMAdvisor,
    news_items: list[dict[str, str]],
) -> str | None:
    """Summarize recent news items using the existing LLM client.

    Args:
        advisor: LLMAdvisor instance (reuses auth credentials).
        news_items: List of dicts with 'published_at' and 'content' keys.

    Returns:
        Summary text or None on failure.
    """
    if not news_items:
        return None

    _KST = timezone(timedelta(hours=9))
    lines: list[str] = []
    for item in news_items:
        ts = item.get("published_at", "")
        # Convert UTC ISO timestamp to KST "MM-DD HH:MM" format
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                dt_kst = dt.astimezone(_KST)
                ts = dt_kst.strftime("%m-%d %H:%M")
            except (ValueError, TypeError):
                if len(ts) >= 16:
                    ts = ts[5:16].replace("T", " ")
        content = item.get("content", "")
        lines.append(f"[{ts} KST] {content}")

    user_prompt = (
        f"다음은 최근 코인니스 뉴스 {len(news_items)}건입니다. "
        "시장에 영향이 큰 순서대로 3~5줄로 요약해주세요.\n\n"
        + "\n".join(lines)
    )

    try:
        result = await advisor.complete_generic(
            system_prompt=_SUMMARIZER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=400,
        )
        if result:
            logger.info("news_summarized input_count=%d", len(news_items))
        return result
    except Exception as e:
        logger.error("news_summarize_error error=%s", str(e))
        return None
