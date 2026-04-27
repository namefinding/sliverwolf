from __future__ import annotations

import inspect
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from local_agent.app.chat_models import ChatTurnResult
from local_agent.app.context_tasks import ContextTaskManager
from local_agent.app.session_store import InMemorySessionStore
from local_agent.app.task_store import InMemoryTaskStore
from local_agent.kernel.agent_kernel import AgentKernel
from local_agent.protocol.models import CandidateState, TaskProgressEvent, TaskRun, TaskStatus, TurnArtifacts


class TaskService:
    def __init__(
        self,
        session_store: InMemorySessionStore,
        task_store: InMemoryTaskStore,
        kernel_factory: Callable[[str | None, dict | None], AgentKernel],
        max_workers: int = 4,
    ) -> None:
        self.session_store = session_store
        self.task_store = task_store
        self.kernel_factory = kernel_factory
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="local-agent-task")

    def submit(
        self,
        *,
        user_text: str,
        session_id: str,
        scope_root: str | None = None,
        runtime_settings: dict | None = None,
    ) -> TaskRun:
        session = self.session_store.get_or_create(
            session_id,
            self.kernel_factory,
            scope_root=scope_root,
            runtime_settings=runtime_settings,
        )
        task = TaskRun(
            task_id=f"task_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            user_text=user_text,
            scope_root=str(Path(session.scope_root).resolve()) if session.scope_root else None,
            runtime_settings=runtime_settings,
            status=TaskStatus.QUEUED,
            progress_message="任务已经进入队列，马上开始处理。",
        )
        self.task_store.create(task)
        self.task_store.add_event(
            task.task_id,
            TaskProgressEvent(
                event_id=f"evt_{uuid.uuid4().hex[:10]}",
                task_id=task.task_id,
                stage="queued",
                message="我先把任务收下了，准备开始处理。",
            ),
        )
        self.executor.submit(self._run_task, session_id, task.task_id, user_text, session.scope_root, runtime_settings)
        return task

    def list_tasks(self, session_id: str | None = None) -> list[TaskRun]:
        return self.task_store.list(session_id=session_id)

    def get_task(self, task_id: str) -> TaskRun | None:
        return self.task_store.get(task_id)

    def acknowledge_task(self, task_id: str) -> TaskRun | None:
        return self.task_store.acknowledge(task_id)

    def cancel_task(self, task_id: str) -> TaskRun | None:
        return self.task_store.cancel(task_id)

    def select_candidate(
        self,
        task_id: str,
        *,
        candidate_id: str | None = None,
        candidate_path: str | None = None,
    ) -> TaskRun | None:
        task = self.task_store.get(task_id)
        if task is None or task.pending_task is None:
            return None

        selected_candidate = None
        for candidate in task.pending_task.selection_candidates:
            if candidate_id and candidate.candidate_id == candidate_id:
                selected_candidate = candidate
                break
            if candidate_path and candidate.path == candidate_path:
                selected_candidate = candidate
                break
        if selected_candidate is None:
            return None

        resumed_pending_task = task.pending_task.model_copy(deep=True)
        resumed_pending_task.collected_slots["selected_candidate_path"] = selected_candidate.path
        resumed_pending_task.collected_slots["selected_candidate_name"] = selected_candidate.name

        self.session_store.set_pending_task(task.session_id, None, mode="agent")
        self.task_store.update_status(
            task_id,
            TaskStatus.RUNNING,
            final_response="",
            speech_text="",
            error=None,
            needs_confirmation=False,
            pending_task=None,
        )
        self._emit(
            task_id,
            "selection_received",
            f"我收到你的选择了，接下来按 {selected_candidate.name} 继续处理。",
            {
                "candidate_id": selected_candidate.candidate_id,
                "candidate_path": selected_candidate.path,
            },
        )
        self.executor.submit(
            self._run_selected_candidate,
            task.session_id,
            task_id,
            task.scope_root,
            task.runtime_settings,
            resumed_pending_task,
            selected_candidate.path,
            selected_candidate.name,
            selected_candidate.kind,
        )
        return self.task_store.get(task_id)

    def _run_task(
        self,
        session_id: str,
        task_id: str,
        user_text: str,
        scope_root: str | None,
        runtime_settings: dict | None,
    ) -> None:
        self.session_store.get_or_create(
            session_id,
            self.kernel_factory,
            scope_root=scope_root,
            runtime_settings=runtime_settings,
        )
        kernel = self._invoke_factory(scope_root, runtime_settings)
        self.task_store.update_status(task_id, TaskStatus.RUNNING)
        self._emit(task_id, "running", "我已经开始动手了，先规划一下怎么做。")

        def progress(stage: str, message: str, payload: dict | None = None) -> None:
            if self.task_store.status_of(task_id) == TaskStatus.CANCELLED:
                raise RuntimeError("Task cancelled by user.")
            self._emit(task_id, stage, message, payload or {})

        try:
            artifacts = kernel.handle_user_input(user_text, progress_callback=progress)
            if self.task_store.status_of(task_id) == TaskStatus.CANCELLED:
                self.session_store.set_pending_task(session_id, None, mode="agent")
                self._emit(task_id, "cancelled", "我已经按你的意思停下这条后台任务了。")
                return
            self._finalize(session_id, task_id, artifacts)
        except Exception as exc:  # noqa: BLE001
            if self.task_store.status_of(task_id) == TaskStatus.CANCELLED:
                self.session_store.set_pending_task(session_id, None, mode="agent")
                self._emit(task_id, "cancelled", "我已经按你的意思停下这条后台任务了。")
                return
            self.session_store.set_pending_task(session_id, None, mode="agent")
            self._emit(task_id, "failed", f"任务中途出了点问题：{exc}")
            self.task_store.update_status(task_id, TaskStatus.FAILED, error=str(exc), needs_confirmation=False)

    def _run_selected_candidate(
        self,
        session_id: str,
        task_id: str,
        scope_root: str | None,
        runtime_settings: dict | None,
        pending_task,
        selected_path: str,
        selected_name: str,
        selected_kind: str,
    ) -> None:
        self.session_store.get_or_create(
            session_id,
            self.kernel_factory,
            scope_root=scope_root,
            runtime_settings=runtime_settings,
        )
        kernel = self._invoke_factory(scope_root, runtime_settings)
        seed_candidate_state = CandidateState(
            query=selected_name,
            target_kind=selected_kind or "file",
            path_scope=scope_root or ".",
            query_terms=[],
            candidate_paths=[selected_path],
            candidate_names=[selected_name],
            source_tool="user_selection",
            confidence=1.0,
            confidence_reason="user_selected_candidate",
        )
        resumed_request = (
            f"{pending_task.original_user_request}\n"
            f"用户已确认候选文件：{selected_name}\n"
            f"确认路径：{selected_path}\n"
            "请直接基于这个已确认候选继续完成原任务。"
        )
        self._emit(
            task_id,
            "running",
            f"我已经锁定 {selected_name}，现在继续往下处理。",
            {"selected_candidate_path": selected_path},
        )

        def progress(stage: str, message: str, payload: dict | None = None) -> None:
            if self.task_store.status_of(task_id) == TaskStatus.CANCELLED:
                raise RuntimeError("Task cancelled by user.")
            self._emit(task_id, stage, message, payload or {})

        try:
            artifacts = kernel.handle_user_input(
                resumed_request,
                progress_callback=progress,
                seed_candidate_state=seed_candidate_state,
                seed_overall_task_goal=pending_task.overall_task_goal,
            )
            if self.task_store.status_of(task_id) == TaskStatus.CANCELLED:
                self.session_store.set_pending_task(session_id, None, mode="agent")
                self._emit(task_id, "cancelled", "我已经按你的意思停下这条后台任务了。")
                return
            self._finalize(session_id, task_id, artifacts)
        except Exception as exc:  # noqa: BLE001
            if self.task_store.status_of(task_id) == TaskStatus.CANCELLED:
                self.session_store.set_pending_task(session_id, None, mode="agent")
                self._emit(task_id, "cancelled", "我已经按你的意思停下这条后台任务了。")
                return
            self.session_store.set_pending_task(session_id, None, mode="agent")
            self._emit(task_id, "failed", f"任务中途出了点问题：{exc}")
            self.task_store.update_status(task_id, TaskStatus.FAILED, error=str(exc), needs_confirmation=False)

    def _finalize(self, session_id: str, task_id: str, artifacts: TurnArtifacts) -> None:
        if artifacts.pending_task is not None:
            linked_pending_task = artifacts.pending_task.model_copy(update={"task_id": task_id})
            self.session_store.set_pending_task(session_id, linked_pending_task, mode="agent")
            waiting_status = (
                TaskStatus.WAITING_FOR_SELECTION
                if linked_pending_task.selection_candidates
                else TaskStatus.WAITING_FOR_CLARIFICATION
            )
            waiting_stage = "waiting_for_selection" if linked_pending_task.selection_candidates else "waiting_for_clarification"
            waiting_payload = {}
            if linked_pending_task.selection_candidates:
                waiting_payload = {
                    "selection_candidates": [
                        candidate.model_dump(mode="json")
                        for candidate in linked_pending_task.selection_candidates
                    ]
                }
            self._emit(
                task_id,
                waiting_stage,
                artifacts.final_response or "我还差一点信息，等你补一句我就能继续。",
                waiting_payload,
            )
            self.task_store.update_status(
                task_id,
                waiting_status,
                final_response=artifacts.final_response,
                speech_text=artifacts.speech_text,
                completed_outputs=artifacts.completed_outputs,
                overall_task_goal=artifacts.overall_task_goal,
                tts_dispatched=artifacts.tts_dispatched,
                needs_confirmation=False,
                pending_task=linked_pending_task,
            )
            return

        self.session_store.set_pending_task(session_id, None, mode="agent")
        self._emit(task_id, "completed", "任务跑完了，你可以点开看看结果。")
        self.task_store.update_status(
            task_id,
            TaskStatus.COMPLETED,
            final_response=artifacts.final_response,
            speech_text=artifacts.speech_text,
            completed_outputs=artifacts.completed_outputs,
            overall_task_goal=artifacts.overall_task_goal,
            tts_dispatched=artifacts.tts_dispatched,
            needs_confirmation=True,
            pending_task=None,
        )
        self._register_completed_context_task(session_id, task_id, artifacts)

    def _register_completed_context_task(
        self,
        session_id: str,
        task_id: str,
        artifacts: TurnArtifacts,
    ) -> None:
        session = self.session_store.get(session_id)
        task = self.task_store.get(task_id)
        if session is None or task is None:
            return
        if artifacts.pending_task is not None:
            return

        turn_result = ChatTurnResult(
            session_id=session_id,
            mode="agent",
            response=artifacts.final_response,
            speech_text=artifacts.speech_text or artifacts.final_response,
            tts_dispatched=artifacts.tts_dispatched,
            used_agent=True,
            scope_root=task.scope_root,
            overall_task_goal=(
                None
                if artifacts.overall_task_goal is None
                else artifacts.overall_task_goal.model_dump(mode="json")
            ),
            completed_outputs=[item.value for item in artifacts.completed_outputs],
            pending_task=None,
            metadata={
                "trace_id": artifacts.trace_id,
                "execution_summary": artifacts.execution_summary,
            },
        )
        record = ContextTaskManager.build_context_task_record_from_result(task.user_text, turn_result)
        if record is None:
            return
        record.task_id = f"context_{task_id}"
        record.source_turn_id = task_id
        record.metadata["source"] = "background_task"
        ContextTaskManager.remember_context_task(session, record)

    def _emit(self, task_id: str, stage: str, message: str, payload: dict | None = None) -> None:
        event = TaskProgressEvent(
            event_id=f"evt_{uuid.uuid4().hex[:10]}",
            task_id=task_id,
            stage=stage,
            message=message,
            payload=payload or {},
            created_at=datetime.now(UTC),
        )
        self.task_store.add_event(task_id, event)

    def _invoke_factory(self, scope_root: str | None, runtime_settings: dict | None) -> AgentKernel:
        try:
            parameters = list(inspect.signature(self.kernel_factory).parameters.values())
        except (TypeError, ValueError):
            parameters = []

        positional = [
            parameter
            for parameter in parameters
            if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ]
        has_varargs = any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters)

        if has_varargs or len(positional) >= 2:
            return self.kernel_factory(scope_root, runtime_settings)
        if len(positional) == 1:
            return self.kernel_factory(scope_root)
        return self.kernel_factory()
