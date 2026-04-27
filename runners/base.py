from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

from local_agent.app.session_store import SessionState

if TYPE_CHECKING:
    from local_agent.app.chat_models import ChatTurnResult
    from local_agent.app.chat_service import ChatService


@dataclass
class RunnerContext:
    service: "ChatService"
    session: SessionState
    text: str
    progress_callback: Callable[[str, str, dict[str, Any]], None] | None = None


class TurnRunner(Protocol):
    name: str

    def run(self, context: RunnerContext) -> "ChatTurnResult":
        ...
