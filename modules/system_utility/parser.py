from __future__ import annotations

import re
from datetime import datetime, timedelta

from .timezone_utils import resolve_timezone

_CN_FULLWIDTH_COLON = "\uff1a"
_CN_POINT_VARIANT = "\u9ede"
_CN_POINT = "\u70b9"
_CN_MINUTES_VARIANT = "\u5206\u9418"
_CN_MINUTES = "\u5206\u949f"
_CN_HOURS_VARIANT = "\u5c0f\u6642"
_CN_HOURS = "\u5c0f\u65f6"
_CN_COUNT_VARIANT = "\u500b"
_CN_COUNT = "\u4e2a"

_RELATIVE_MINUTES_PATTERN = r"(\d+)\s*\u5206\u949f\u540e"
_RELATIVE_HOURS_PATTERN = r"(\d+)\s*(?:\u4e2a)?\u5c0f\u65f6\u540e"
_HALF_HOUR_PHRASE = "\u534a\u5c0f\u65f6\u540e"
_TOMORROW = "\u660e\u5929"
_DAY_AFTER_TOMORROW = "\u540e\u5929"

_MERIDIEM_PATTERN = (
    "\u51cc\u6668|"
    "\u65e9\u4e0a|"
    "\u4e0a\u5348|"
    "\u4e2d\u5348|"
    "\u4e0b\u5348|"
    "\u665a\u4e0a|"
    "\u508d\u665a"
)
_COLON_TIME_PATTERN = rf"({_MERIDIEM_PATTERN})?\s*(\d{{1,2}})\s*:\s*(\d{{1,2}})"
_POINT_TIME_PATTERN = rf"({_MERIDIEM_PATTERN})?\s*(\d{{1,2}})\s*\u70b9(?:\s*(\d{{1,2}})\s*\u5206?)?"


def parse_when_text(
    *,
    when_text: str,
    now: datetime,
    timezone_name: str | None = None,
) -> datetime | None:
    text = str(when_text or "").strip()
    if not text:
        return None

    tz = now.tzinfo or resolve_timezone(timezone_name)
    normalized = (
        text.replace(_CN_FULLWIDTH_COLON, ":")
        .replace(_CN_POINT_VARIANT, _CN_POINT)
        .replace(_CN_MINUTES_VARIANT, _CN_MINUTES)
        .replace(_CN_HOURS_VARIANT, _CN_HOURS)
        .replace(_CN_COUNT_VARIANT, _CN_COUNT)
    )

    minute_match = re.search(_RELATIVE_MINUTES_PATTERN, normalized)
    if minute_match:
        return now + timedelta(minutes=int(minute_match.group(1)))

    if _HALF_HOUR_PHRASE in normalized:
        return now + timedelta(minutes=30)

    hour_match = re.search(_RELATIVE_HOURS_PATTERN, normalized)
    if hour_match:
        return now + timedelta(hours=int(hour_match.group(1)))

    day_offset = 0
    if _TOMORROW in normalized:
        day_offset = 1
    elif _DAY_AFTER_TOMORROW in normalized:
        day_offset = 2

    hour = None
    minute = 0

    colon_match = re.search(_COLON_TIME_PATTERN, normalized)
    point_match = re.search(_POINT_TIME_PATTERN, normalized)
    matched = colon_match or point_match

    if matched:
        meridiem = matched.group(1)
        hour = int(matched.group(2))
        if matched.group(3):
            minute = int(matched.group(3))

        if meridiem in {"\u4e0b\u5348", "\u665a\u4e0a", "\u508d\u665a"} and 1 <= hour <= 11:
            hour += 12
        elif meridiem == "\u4e2d\u5348" and 1 <= hour <= 10:
            hour += 12
        elif meridiem == "\u51cc\u6668" and hour == 12:
            hour = 0

    if hour is None:
        return None

    target_date = (now + timedelta(days=day_offset)).date()
    scheduled = datetime(
        year=target_date.year,
        month=target_date.month,
        day=target_date.day,
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
        tzinfo=tz,
    )
    if day_offset == 0 and scheduled <= now:
        scheduled += timedelta(days=1)
    return scheduled
