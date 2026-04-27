from __future__ import annotations

import time
from threading import Lock
from typing import Any


class InMemoryWebCache:
    def __init__(self, ttl_seconds: int = 300, max_items: int = 128) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._items: dict[str, tuple[float, Any]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at < time.time():
                self._items.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._items) >= self.max_items:
                oldest_key = min(self._items, key=lambda item_key: self._items[item_key][0])
                self._items.pop(oldest_key, None)
            self._items[key] = (time.time() + self.ttl_seconds, value)
