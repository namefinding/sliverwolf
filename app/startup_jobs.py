from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock, Thread
from typing import Any, Callable


@dataclass
class StartupSnapshot:
    status: str = "idle"
    phase: str = "not_started"
    message: str = "Startup jobs have not started yet."
    started_at: str | None = None
    finished_at: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "phase": self.phase,
            "message": self.message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "summary": self.summary,
        }


class StartupJobsRunner:
    def __init__(
        self,
        *,
        has_index: Callable[[], bool],
        rebuild_index: Callable[[], dict[str, Any]],
        sync_index: Callable[[], dict[str, Any]],
    ) -> None:
        self.has_index = has_index
        self.rebuild_index = rebuild_index
        self.sync_index = sync_index
        self._lock = Lock()
        self._thread: Thread | None = None
        self._snapshot = StartupSnapshot()

    def start_async(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = Thread(target=self.run_now, name="local-agent-startup", daemon=True)
            self._thread.start()

    def run_now(self) -> None:
        self._update(
            status="running",
            phase="startup",
            message="I am checking the local index and preparing background services.",
            started_at=datetime.now(UTC).isoformat(),
            finished_at=None,
            summary={},
        )
        try:
            if self.has_index():
                self._update(status="running", phase="sync_index", message="I found an existing local index and am syncing changes.")
                summary = self.sync_index()
            else:
                self._update(status="running", phase="build_index", message="I did not find a local index, so I am building one in the background.")
                summary = self.rebuild_index()
            self._update(
                status="ready",
                phase="complete",
                message="Startup preparation finished. The agent is ready and the local index is usable.",
                finished_at=datetime.now(UTC).isoformat(),
                summary=summary,
            )
        except Exception as exc:  # noqa: BLE001
            self._update(
                status="failed",
                phase="failed",
                message=f"Startup preparation hit an error: {exc}",
                finished_at=datetime.now(UTC).isoformat(),
                summary={"error": str(exc)},
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot.to_dict()

    def mark_skipped(self, message: str) -> None:
        self._update(
            status="skipped",
            phase="skipped",
            message=message,
            finished_at=datetime.now(UTC).isoformat(),
        )

    def _update(
        self,
        *,
        status: str | None = None,
        phase: str | None = None,
        message: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            if status is not None:
                self._snapshot.status = status
            if phase is not None:
                self._snapshot.phase = phase
            if message is not None:
                self._snapshot.message = message
            if started_at is not None:
                self._snapshot.started_at = started_at
            if finished_at is not None:
                self._snapshot.finished_at = finished_at
            if summary is not None:
                self._snapshot.summary = summary
