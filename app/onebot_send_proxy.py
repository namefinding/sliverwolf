from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import re

from local_agent.app.onebot_contacts import OneBotContactMatch


@dataclass(frozen=True)
class OneBotProxySendRequest:
    recipient_query: str
    message_body: str
    intent_label: str = "send_message"
    source_text: str = ""


@dataclass
class PendingProxySelection:
    requester_id: str
    session_id: str
    request: OneBotProxySendRequest
    candidates: tuple[OneBotContactMatch, ...]
    created_at: datetime

    def is_expired(self, *, ttl_seconds: int = 600) -> bool:
        return datetime.now(UTC) - self.created_at > timedelta(seconds=ttl_seconds)


def is_assistant_recipient_query(query: str, aliases: tuple[str, ...] | list[str] | set[str] | None = None) -> bool:
    normalized_query = _normalize_recipient_label(query)
    if not normalized_query:
        return False
    return any(normalized_query == _normalize_recipient_label(str(alias)) for alias in aliases or ())


def build_proxy_selection_prompt(request: OneBotProxySendRequest, matches: list[OneBotContactMatch]) -> str:
    lines = [
        "\u627e\u5230\u591a\u4e2a\u53ef\u80fd\u7684\u8054\u7cfb\u4eba\uff0c"
        f"\u8981\u628a\u300c{request.message_body}\u300d\u53d1\u7ed9\u8c01\uff1f"
    ]
    for index, match in enumerate(matches, start=1):
        kind_label = "\u7fa4" if match.contact.kind == "group" else "\u597d\u53cb"
        lines.append(f"{index}. {match.contact.name}\uff08{kind_label} {match.contact.target_id}\uff09")
    return "\n".join(lines)


def resolve_proxy_selection(pending: PendingProxySelection, text: str) -> OneBotContactMatch | None:
    normalized = str(text or "").strip()
    if not normalized:
        return None
    if normalized.isdigit():
        index = int(normalized) - 1
        if 0 <= index < len(pending.candidates):
            return pending.candidates[index]
        return None

    lowered = normalized.lower()
    for match in pending.candidates:
        if lowered == match.contact.name.lower():
            return match
        if lowered == str(match.contact.target_id):
            return match
    return None


def _normalize_recipient_label(value: str) -> str:
    return re.sub(r"[\s`'\"\u201c\u201d\u2018\u2019\u300c\u300d\uff0c\u3002\uff01\uff1f?.!?:\uff1a\uff1b\uff08\uff09()\[\]{}_-]+", "", str(value or "").strip().lower())
