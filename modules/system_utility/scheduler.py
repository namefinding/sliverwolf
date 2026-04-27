from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Callable

from .reminder_store import ReminderStore


class ReminderScheduler:
    def __init__(
        self,
        reminder_store: ReminderStore,
        notify_callback: Callable[[dict], None],
        dispatch_callback: Callable[[dict], None] | None = None,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        self.reminder_store = reminder_store
        self.notify_callback = notify_callback
        self.dispatch_callback = dispatch_callback
        self.poll_interval_seconds = poll_interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="reminder-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                due = self.reminder_store.get_due_reminders(now_utc=datetime.now(UTC), limit=20)
                for task in due:
                    reminder_id = str(task.get("reminder_id") or "").strip()
                    if not reminder_id:
                        continue

                    # 先标记 fired，避免重复触发
                    fired = self.reminder_store.mark_fired(reminder_id)
                    if fired is None:
                        continue

                    task_type = str(fired.get("task_type") or "notify").strip().lower()

                    if task_type == "deferred_agent_task":
                        if self.dispatch_callback is not None:
                            self.dispatch_callback(fired)
                        else:
                            print(
                                "[reminder-scheduler] deferred task skipped because dispatch_callback is missing:",
                                fired.get("reminder_id"),
                            )
                    else:
                        self.notify_callback(fired)

            except Exception as exc:
                print(f"[reminder-scheduler] error: {exc}")

            self._stop_event.wait(self.poll_interval_seconds)