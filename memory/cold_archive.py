from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Iterable

from local_agent.protocol.models import MemoryRecord
from local_agent.storage.memory_store import SQLiteMemoryStore


COLD_MEMORY_TYPES = {"archive", "project_archive", "episodic_archive"}
_HISTORY_TOKENS = (
    "\u4e4b\u524d",
    "\u4ee5\u524d",
    "\u4e0a\u6b21",
    "\u5386\u53f2",
    "\u8fd8\u8bb0\u5f97",
    "\u66fe\u7ecf",
    "earlier",
    "previous",
    "history",
    "before",
)


class ColdArchiveService:
    def __init__(self, store: SQLiteMemoryStore) -> None:
        self.store = store

    def remember_archive(
        self,
        content: str,
        *,
        scope: str = "user",
        tags: Iterable[str] | None = None,
        importance: float = 0.65,
        memory_type: str = "archive",
    ) -> None:
        record = MemoryRecord(
            memory_type=memory_type,
            scope=scope,
            content=content.strip(),
            importance=importance,
            tags=list(tags or []),
            created_at=datetime.now(UTC),
        )
        self.store.remember(record)

    def archive_session_summary(
        self,
        summary: str,
        *,
        scope: str = "session",
        tags: Iterable[str] | None = None,
        importance: float = 0.58,
    ) -> str:
        normalized = " ".join(summary.split()).strip()
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        if not normalized:
            return digest

        archive_tags = list(tags or [])
        archive_tags.extend(["session-archive", f"digest:{digest}"])
        self.remember_archive(
            normalized,
            scope=scope,
            tags=archive_tags,
            importance=importance,
            memory_type="episodic_archive",
        )
        return digest

    def recall_for_text(
        self,
        user_text: str,
        *,
        scope: str = "user",
        limit: int = 3,
    ) -> list[MemoryRecord]:
        normalized = " ".join(user_text.split()).strip()
        if not normalized:
            return []
        return self.store.recall_structured(
            normalized,
            limit=limit,
            memory_types=COLD_MEMORY_TYPES,
            scopes={scope, "session", "global"},
        )

    @staticmethod
    def should_recall(user_text: str) -> bool:
        normalized = user_text.lower()
        return any(token in normalized for token in _HISTORY_TOKENS)

    @staticmethod
    def format_for_prompt(records: list[MemoryRecord]) -> str:
        if not records:
            return ""
        lines = ["\u76f8\u5173\u51b7\u8bb0\u5fc6:"]
        for index, record in enumerate(records, start=1):
            lines.append(f"{index}. {record.content}")
        return "\n".join(lines)
