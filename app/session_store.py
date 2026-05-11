from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import inspect
import json
from threading import Lock
from typing import Callable

from local_agent.kernel.agent_kernel import AgentKernel
from local_agent.memory.hot_context import HotContextState
from local_agent.protocol.models import ContextTaskRecord, FollowUpAssessment, PendingTask


@dataclass
class SessionState:
    session_id: str
    kernel: AgentKernel
    scope_root: str | None = None
    runtime_settings: dict | None = None
    settings_signature: str = ""
    last_mode: str = "chat"
    pending_task: PendingTask | None = None
    context_tasks: list[ContextTaskRecord | PendingTask] = field(default_factory=list)
    active_context_task: ContextTaskRecord | None = None
    pending_follow_up_assessment: FollowUpAssessment | None = None
    hot_context: HotContextState = field(default_factory=HotContextState)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_active_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_finalized_turn_id: str | None = None
    last_finalized_turn_text: str = ""
    recent_turn_texts: list[str] = field(default_factory=list)
    recent_artifacts: list[dict] = field(default_factory=list)
    last_memory_update: dict | None = None
    last_route_decision: dict | None = None

    def touch(self, mode: str) -> None:
        self.last_mode = mode
        self.last_active_at = datetime.now(UTC)


class InMemorySessionStore:
    def __init__(self, ttl_minutes: int = 60, max_sessions: int = 100) -> None:
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_sessions = max_sessions
        self._sessions: dict[str, SessionState] = {}
        self._lock = Lock()

    def record_finalized_turn(self, session_id: str, turn_id: str, raw_user_turn_text: str) -> SessionState | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.last_finalized_turn_id = turn_id
            session.last_finalized_turn_text = raw_user_turn_text
            session.recent_turn_texts.append(raw_user_turn_text)
            if len(session.recent_turn_texts) > 12:
                session.recent_turn_texts = session.recent_turn_texts[-12:]
            return session

    def get_or_create(
        self,
        session_id: str,
        factory: Callable[[str | None, dict | None], AgentKernel],
        scope_root: str | None = None,
        runtime_settings: dict | None = None,
    ) -> SessionState:
        with self._lock:
            self.cleanup_locked()
            session = self._sessions.get(session_id)
            normalized_scope = scope_root
            settings_signature = json.dumps(runtime_settings or {}, ensure_ascii=False, sort_keys=True)
            if session is None or session.scope_root != normalized_scope:
                if len(self._sessions) >= self.max_sessions:
                    self._evict_oldest_locked()
                last_mode = session.last_mode if session is not None else "chat"
                pending_task = session.pending_task if session is not None else None
                context_tasks = list(session.context_tasks) if session is not None else []
                active_context_task = session.active_context_task if session is not None else None
                pending_follow_up_assessment = session.pending_follow_up_assessment if session is not None else None
                last_finalized_turn_id = session.last_finalized_turn_id if session is not None else None
                last_finalized_turn_text = session.last_finalized_turn_text if session is not None else ""
                recent_turn_texts = list(session.recent_turn_texts) if session is not None else []
                recent_artifacts = [dict(item) for item in session.recent_artifacts] if session is not None else []
                last_memory_update = dict(session.last_memory_update) if session is not None and isinstance(session.last_memory_update, dict) else None
                last_route_decision = dict(session.last_route_decision) if session is not None and isinstance(session.last_route_decision, dict) else None
                session = SessionState(
                    session_id=session_id,
                    kernel=self._invoke_factory(factory, normalized_scope, runtime_settings),
                    scope_root=normalized_scope,
                    runtime_settings=runtime_settings or {},
                    settings_signature=settings_signature,
                    last_mode=last_mode,
                    pending_task=pending_task,
                    context_tasks=context_tasks,
                    active_context_task=active_context_task,
                    pending_follow_up_assessment=pending_follow_up_assessment,
                    last_finalized_turn_id=last_finalized_turn_id,
                    last_finalized_turn_text=last_finalized_turn_text,
                    recent_turn_texts=recent_turn_texts,
                    recent_artifacts=recent_artifacts,
                    last_memory_update=last_memory_update,
                    last_route_decision=last_route_decision,
                )
                self._sessions[session_id] = session
            return session

    def list_session_ids(self) -> list[str]:
        with self._lock:
            self.cleanup_locked()
            return list(self._sessions.keys())

    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            self.cleanup_locked()
            return self._sessions.get(session_id)

    def set_pending_task(self, session_id: str, pending_task: PendingTask | None, *, mode: str = "agent") -> SessionState | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.pending_task = pending_task
            if pending_task is None:
                session.active_context_task = None
            session.touch(mode)
            return session

    def remove(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def cleanup(self) -> None:
        with self._lock:
            self.cleanup_locked()

    def cleanup_locked(self) -> None:
        now = datetime.now().astimezone()
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_active_at > self.ttl
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def _evict_oldest_locked(self) -> None:
        if not self._sessions:
            return
        oldest_session_id = min(
            self._sessions,
            key=lambda session_id: self._sessions[session_id].last_active_at,
        )
        self._sessions.pop(oldest_session_id, None)

    @staticmethod
    def _invoke_factory(
        factory: Callable[[str | None, dict | None], AgentKernel],
        scope_root: str | None,
        runtime_settings: dict | None,
    ) -> AgentKernel:
        try:
            parameters = list(inspect.signature(factory).parameters.values())
        except (TypeError, ValueError):
            parameters = []

        positional = [
            parameter
            for parameter in parameters
            if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ]
        has_varargs = any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters)

        if has_varargs or len(positional) >= 2:
            return factory(scope_root, runtime_settings)
        if len(positional) == 1:
            return factory(scope_root)
        return factory()
