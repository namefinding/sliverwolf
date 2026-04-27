from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OneBotAudioAttachment:
    local_path: str | None = None
    remote_url: str | None = None
    mime_type: str | None = None


@dataclass(frozen=True)
class OneBotImageAttachment:
    local_path: str | None = None
    remote_url: str | None = None
    file_id: str | None = None
    mime_type: str | None = None
    summary: str | None = None
    image_type: str | None = None
    is_emoji: bool = False


@dataclass(frozen=True)
class OneBotTarget:
    message_type: str
    user_id: int | None = None
    group_id: int | None = None


@dataclass(frozen=True)
class OneBotInboundMessage:
    session_id: str
    text: str
    sender_id: str
    mode: str
    scope_root: str | None
    target: OneBotTarget
    metadata: dict[str, Any]
    audio_attachments: tuple[OneBotAudioAttachment, ...] = ()
    image_attachments: tuple[OneBotImageAttachment, ...] = ()
