from __future__ import annotations

from threading import Lock
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class QQRuntime(Protocol):
    def get_current_context(self) -> dict[str, Any]:
        ...

    def get_recent_messages(
        self,
        *,
        session_id: str | None = None,
        limit: int = 8,
        include_assistant: bool = True,
    ) -> list[dict[str, Any]]:
        ...

    def get_last_reply(
        self,
        *,
        session_id: str | None = None,
        contact_query: str | None = None,
    ) -> dict[str, Any] | None:
        ...

    def search_history(
        self,
        *,
        session_id: str | None = None,
        contact_query: str | None = None,
        query: str | None = None,
        limit: int = 5,
        reply_after_last_outbound: bool = False,
    ) -> list[dict[str, Any]]:
        ...

    def get_recent_attachments(
        self,
        *,
        session_id: str | None = None,
        contact_query: str | None = None,
        kind: str = "any",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        ...

    def search_contacts(
        self,
        query: str,
        *,
        target_kind: str = "any",
        limit: int = 5,
        exclude_sender: bool = False,
        sender_id: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def send_text(
        self,
        message: str,
        *,
        target_kind: str = "current",
        target_id: int | None = None,
        current_target: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...

    def send_file(
        self,
        file_path: str,
        *,
        target_kind: str = "current",
        target_id: int | None = None,
        current_target: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...

    def send_voice(
        self,
        *,
        speech_text: str | None = None,
        audio_path: str | None = None,
        target_kind: str = "current",
        target_id: int | None = None,
        current_target: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


class QQRuntimeRegistry:
    _lock = Lock()
    _runtimes: dict[str, QQRuntime] = {}

    @classmethod
    def register(cls, runtime_id: str, runtime: QQRuntime) -> None:
        with cls._lock:
            cls._runtimes[runtime_id] = runtime

    @classmethod
    def unregister(cls, runtime_id: str) -> None:
        with cls._lock:
            cls._runtimes.pop(runtime_id, None)

    @classmethod
    def get(cls, runtime_id: str | None) -> QQRuntime | None:
        if not runtime_id:
            return None
        with cls._lock:
            return cls._runtimes.get(runtime_id)

    @classmethod
    def get_any(cls) -> QQRuntime | None:
        with cls._lock:
            for runtime in cls._runtimes.values():
                return runtime
        return None
