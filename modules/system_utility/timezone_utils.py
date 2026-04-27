from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo

_FIXED_TIMEZONE_FALLBACKS: dict[str, tzinfo] = {
    "Asia/Shanghai": timezone(timedelta(hours=8), name="Asia/Shanghai"),
    "Asia/Chongqing": timezone(timedelta(hours=8), name="Asia/Chongqing"),
    "Asia/Hong_Kong": timezone(timedelta(hours=8), name="Asia/Hong_Kong"),
    "UTC": UTC,
}


def resolve_timezone(
    timezone_name: str | None,
    *,
    default_timezone: str = "Asia/Shanghai",
) -> tzinfo:
    candidates: list[str] = []
    for candidate in (timezone_name, default_timezone, "UTC"):
        normalized = str(candidate or "").strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    for candidate in candidates:
        try:
            return ZoneInfo(candidate)
        except Exception:
            fallback = _FIXED_TIMEZONE_FALLBACKS.get(candidate)
            if fallback is not None:
                return fallback

    return datetime.now().astimezone().tzinfo or UTC
