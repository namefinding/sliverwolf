from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChatTurnResult:
    session_id: str
    mode: str
    response: str
    speech_text: str
    tts_dispatched: bool
    used_agent: bool
    scope_root: str | None = None
    overall_task_goal: dict | None = None
    completed_outputs: list[str] | None = None
    pending_task: dict | None = None
    metadata: dict | None = None
