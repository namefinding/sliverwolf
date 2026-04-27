from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from local_agent.protocol.models import MemoryRecord, OutputKind, ToolManifest
from local_agent.storage.memory_store import SQLiteMemoryStore


class RememberInput(BaseModel):
    memory_type: str = "episodic"
    scope: str = "user"
    content: str
    importance: float = 0.6
    tags: list[str] = Field(default_factory=list)


class RecallInput(BaseModel):
    query: str
    limit: int = 5


class MemoryModule:
    def __init__(self, store: SQLiteMemoryStore) -> None:
        self.store = store

    def manifests(self) -> list[ToolManifest]:
        return [
            ToolManifest(
                tool_name="memory.remember",
                module="memory",
                description="Persist a memory record into the local memory store.",
                side_effect=True,
                idempotent=False,
                produces=[OutputKind.MEMORY_SAVED],
                input_schema=RememberInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"saved": {"type": "boolean"}}},
            ),
            ToolManifest(
                tool_name="memory.recall",
                module="memory",
                description="Recall memory records related to a query.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.MEMORY_ITEMS],
                input_schema=RecallInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"items": {"type": "array"}}},
            ),
        ]

    def executor_map(self) -> dict[str, Any]:
        return {
            "memory.remember": self.remember,
            "memory.recall": self.recall,
        }

    def remember(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = RememberInput.model_validate(arguments)
        record = MemoryRecord(
            memory_type=payload.memory_type,
            scope=payload.scope,
            content=payload.content,
            importance=payload.importance,
            tags=payload.tags,
        )
        self.store.remember(record)
        return {"saved": True, "content": payload.content}

    def recall(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = RecallInput.model_validate(arguments)
        results = self.store.recall(payload.query, payload.limit)
        return {"items": [item.model_dump(mode="json") for item in results]}
