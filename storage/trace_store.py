from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class JsonlTraceStore:
    def __init__(self, trace_path: str) -> None:
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_jsonable(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {str(key): JsonlTraceStore._safe_jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [JsonlTraceStore._safe_jsonable(item) for item in value]
        if isinstance(value, tuple):
            return [JsonlTraceStore._safe_jsonable(item) for item in value]
        if isinstance(value, set):
            return [JsonlTraceStore._safe_jsonable(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "event_type": event_type,
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": self._safe_jsonable(payload),
        }
        with self.trace_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")