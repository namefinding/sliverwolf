from __future__ import annotations

from datetime import UTC, datetime, timedelta
from threading import Lock

from local_agent.protocol.models import PendingTask, TaskProgressEvent, TaskRun, TaskStatus


RESPONSE_READY_STATUSES = {
    TaskStatus.WAITING_FOR_CLARIFICATION,
    TaskStatus.WAITING_FOR_SELECTION,
    TaskStatus.WAITING_FOR_CONFIRMATION,
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
}


class InMemoryTaskStore:
    def __init__(self, ttl_minutes: int = 240, max_tasks: int = 500) -> None:
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_tasks = max_tasks
        self._tasks: dict[str, TaskRun] = {}
        self._lock = Lock()

    def create(self, task: TaskRun) -> TaskRun:
        with self._lock:
            self._cleanup_locked()
            if len(self._tasks) >= self.max_tasks:
                self._evict_oldest_locked()
            self._tasks[task.task_id] = task
            return task

    def get(self, task_id: str) -> TaskRun | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self, session_id: str | None = None) -> list[TaskRun]:
        with self._lock:
            self._cleanup_locked()
            tasks = list(self._tasks.values())
            if session_id is not None:
                tasks = [task for task in tasks if task.session_id == session_id]
            return sorted(tasks, key=lambda item: item.updated_at, reverse=True)

    def add_event(self, task_id: str, event: TaskProgressEvent) -> TaskRun | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.events.append(event)
            task.progress_message = event.message
            task.updated_at = datetime.now(UTC)
            return task

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        final_response: str | None = None,
        speech_text: str | None = None,
        completed_outputs=None,
        overall_task_goal=None,
        tts_dispatched: bool | None = None,
        error: str | None = None,
        needs_confirmation: bool | None = None,
        pending_task: PendingTask | None = None,
    ) -> TaskRun | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.status = status
            task.updated_at = datetime.now(UTC)
            if status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
                task.completed_at = task.updated_at
            if status in RESPONSE_READY_STATUSES:
                task.response_ready_at = task.updated_at
                task.elapsed_ms = max(0, int((task.response_ready_at - task.created_at).total_seconds() * 1000))
            if final_response is not None:
                task.final_response = final_response
            if speech_text is not None:
                task.speech_text = speech_text
            if completed_outputs is not None:
                task.completed_outputs = list(completed_outputs)
            if overall_task_goal is not None:
                task.overall_task_goal = overall_task_goal
            if tts_dispatched is not None:
                task.tts_dispatched = tts_dispatched
            if error is not None:
                task.error = error
            if needs_confirmation is not None:
                task.needs_confirmation = needs_confirmation
            task.pending_task = pending_task
            return task

    def acknowledge(self, task_id: str) -> TaskRun | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.acknowledged = True
            task.updated_at = datetime.now(UTC)
            return task

    def cancel(self, task_id: str) -> TaskRun | None:
        return self.update_status(task_id, TaskStatus.CANCELLED, error="Cancelled by user.")

    def status_of(self, task_id: str) -> TaskStatus | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return None if task is None else task.status

    def _cleanup_locked(self) -> None:
        now = datetime.now().astimezone()
        expired = [
            task_id
            for task_id, task in self._tasks.items()
            if now - task.updated_at > self.ttl
        ]
        for task_id in expired:
            self._tasks.pop(task_id, None)

    def _evict_oldest_locked(self) -> None:
        if not self._tasks:
            return
        oldest_task_id = min(self._tasks, key=lambda task_id: self._tasks[task_id].updated_at)
        self._tasks.pop(oldest_task_id, None)
