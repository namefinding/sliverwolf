from __future__ import annotations

from typing import Any


def build_onebot_access_policy(
    sender_id: str | None,
    full_access_user_ids: list[str],
    owner_user_ids: list[str] | None = None,
    owner_display_name: str = "主人",
) -> dict[str, Any]:
    normalized_sender = "" if sender_id is None else str(sender_id).strip()
    normalized_allowlist = {str(item).strip() for item in full_access_user_ids if str(item).strip()}
    normalized_owner_ids = {
        str(item).strip()
        for item in (owner_user_ids or [])
        if str(item).strip()
    }
    is_full_access = bool(normalized_sender and normalized_sender in normalized_allowlist)
    is_owner = bool(normalized_sender and normalized_sender in normalized_owner_ids)
    address_as = str(owner_display_name or "主人").strip() or "主人"
    return {
        "sender_id": normalized_sender or None,
        "role": "full_access" if is_full_access else "restricted",
        "allow_local_tools": is_full_access,
        "allow_web_tools": True,
        "allow_chat": True,
        "is_owner": is_owner,
        "address_as": address_as if is_owner else None,
    }
