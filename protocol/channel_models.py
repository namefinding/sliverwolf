from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ChannelUser(BaseModel):
    user_id: str
    display_name: str | None = None


class ChannelMessage(BaseModel):
    channel: str
    text: str
    session_id: str | None = None
    scope_root: str | None = None
    mode: str = "auto"
    runtime_settings: dict[str, Any] | None = None
    progress_callback: Any | None = None
    sender: ChannelUser | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChannelReply(BaseModel):
    channel: str
    session_id: str
    mode: str
    used_agent: bool
    response: str
    speech_text: str
    tts_dispatched: bool
    scope_root: str | None = None
    overall_task_goal: dict[str, Any] | None = None
    completed_outputs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
