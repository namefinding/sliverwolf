from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Any, Literal


ContactKind = Literal["friend", "group"]


@dataclass(frozen=True)
class OneBotContact:
    kind: ContactKind
    target_id: int
    name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class OneBotContactMatch:
    contact: OneBotContact
    score: float
    reason: str


def build_contacts(
    friend_payload: list[dict[str, Any]] | None,
    group_payload: list[dict[str, Any]] | None,
) -> tuple[OneBotContact, ...]:
    contacts: list[OneBotContact] = []

    if isinstance(friend_payload, list):
        for item in friend_payload:
            if not isinstance(item, dict):
                continue
            user_id = item.get("user_id")
            nickname = _clean_label(item.get("nickname"))
            remark = _clean_label(item.get("remark"))
            if user_id is None or (not nickname and not remark):
                continue
            aliases = tuple(alias for alias in (remark, nickname) if alias)
            contacts.append(
                OneBotContact(
                    kind="friend",
                    target_id=int(user_id),
                    name=remark or nickname or str(user_id),
                    aliases=aliases,
                )
            )

    if isinstance(group_payload, list):
        for item in group_payload:
            if not isinstance(item, dict):
                continue
            group_id = item.get("group_id")
            group_name = _clean_label(item.get("group_name"))
            if group_id is None or not group_name:
                continue
            contacts.append(
                OneBotContact(
                    kind="group",
                    target_id=int(group_id),
                    name=group_name,
                    aliases=(group_name,),
                )
            )

    return tuple(contacts)


def match_contacts(
    query: str,
    contacts: tuple[OneBotContact, ...],
    *,
    limit: int = 5,
) -> list[OneBotContactMatch]:
    normalized_query = _normalize_label(query)
    if not normalized_query:
        return []

    matches: list[OneBotContactMatch] = []
    for contact in contacts:
        score, reason = _score_contact(contact, query, normalized_query)
        if score <= 0:
            continue
        matches.append(OneBotContactMatch(contact=contact, score=score, reason=reason))

    matches.sort(key=lambda item: (-item.score, item.contact.kind, item.contact.name))
    return matches[:limit]


def is_confident_unique_match(matches: list[OneBotContactMatch]) -> bool:
    if not matches:
        return False
    best = matches[0]
    second_score = matches[1].score if len(matches) > 1 else -1.0
    if best.reason in {"exact_id", "exact_name"}:
        return True
    return best.score >= 6.0 and (len(matches) == 1 or (best.score - second_score) >= 1.2)


def _score_contact(contact: OneBotContact, original_query: str, normalized_query: str) -> tuple[float, str]:
    all_labels = [contact.name, *contact.aliases, str(contact.target_id)]
    best_score = 0.0
    best_reason = ""
    wants_group = "群" in original_query or "group" in normalized_query
    wants_friend = any(token in original_query for token in ("好友", "私聊")) or "friend" in normalized_query

    for label in all_labels:
        normalized_label = _normalize_label(label)
        if not normalized_label:
            continue

        score = 0.0
        reason = "fuzzy"
        if normalized_query == str(contact.target_id):
            score = 10.0
            reason = "exact_id"
        elif normalized_query == normalized_label:
            score = 9.0
            reason = "exact_name"
        elif normalized_query in normalized_label or normalized_label in normalized_query:
            score = 7.0
            reason = "substring"
        else:
            ratio = SequenceMatcher(None, normalized_query, normalized_label).ratio()
            overlap = len(set(normalized_query) & set(normalized_label))
            score = ratio * 5.0 + min(overlap, 6) * 0.3

        if contact.kind == "group" and wants_group:
            score += 0.8
        if contact.kind == "friend" and wants_friend:
            score += 0.8

        if score > best_score:
            best_score = score
            best_reason = reason

    return best_score, best_reason


def _clean_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_label(value: str) -> str:
    lowered = value.lower().strip()
    lowered = lowered.replace("qq群", "群").replace("群聊", "群")
    return re.sub(r"[\s`'\"，。！？,.!?:：；（）()\[\]{}_\-]+", "", lowered)
