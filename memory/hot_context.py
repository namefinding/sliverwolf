from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass
class HotContextState:
    session_summary: str = ""
    active_task_summary: str = ""
    last_compacted_at: datetime | None = None
    last_archived_digest: str = ""
    last_archived_message_count: int = 0
    summarized_visible_count: int = 0

    def touch(self) -> None:
        self.last_compacted_at = datetime.now(UTC)
