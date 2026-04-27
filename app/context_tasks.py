from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from local_agent.app.chat_models import ChatTurnResult
from local_agent.app.session_store import SessionState
from local_agent.kernel.agent_kernel import AgentKernel
from local_agent.protocol.models import (
    CandidateState,
    ContextTaskRecord,
    FollowUpAssessment,
    OutputKind,
    PendingTask,
    SelectionCandidate,
    TaskGoal,
    WorkflowState,
)


class ContextTaskManager:
    _DEFAULT_CONTEXT_TASK_TTL_MINUTES = 45

    @staticmethod
    def _as_pending_task(task: ContextTaskRecord | PendingTask) -> PendingTask:
        if isinstance(task, ContextTaskRecord):
            return task.to_pending_task()
        return task

    @staticmethod
    def _is_context_task_expired(task: ContextTaskRecord | PendingTask) -> bool:
        if not isinstance(task, ContextTaskRecord) or task.expires_at is None:
            return False
        return datetime.now(UTC) >= task.expires_at

    @staticmethod
    def classify_follow_up(kernel: AgentKernel, pending_task: PendingTask, text: str) -> FollowUpAssessment:
        selection_assessment = ContextTaskManager.resolve_selection_follow_up(pending_task, text)
        if selection_assessment is not None:
            return selection_assessment
        try:
            llm_assessment = kernel.llm_client.classify_follow_up(pending_task=pending_task, user_text=text)
            if llm_assessment.action in {"resume", "resume_with_correction"} and not llm_assessment.merged_user_request:
                llm_assessment.merged_user_request = ContextTaskManager.merge_pending_request(
                    pending_task,
                    text,
                    correction=llm_assessment.action == "resume_with_correction",
                )
                if not llm_assessment.rationale:
                    llm_assessment.rationale = (
                        "llm_resume_correction_follow_up"
                        if llm_assessment.action == "resume_with_correction"
                        else "llm_resume_follow_up"
                    )
            return llm_assessment
        except Exception:
            return FollowUpAssessment(action="new_request", rationale="llm_follow_up_unavailable")

    @staticmethod
    def resolve_selection_follow_up(pending_task: PendingTask, text: str) -> FollowUpAssessment | None:
        if not pending_task.selection_candidates:
            return None

        normalized = text.strip()
        if not normalized:
            return None

        selected = None
        if normalized.isdigit():
            for candidate in pending_task.selection_candidates:
                if candidate.candidate_id == normalized:
                    selected = candidate
                    break

        if selected is None:
            lowered = normalized.lower()
            for candidate in pending_task.selection_candidates:
                if lowered == candidate.name.lower() or lowered == candidate.path.lower():
                    selected = candidate
                    break

        if selected is None:
            return None

        merged_user_request = (
            f"{pending_task.original_user_request}\n"
            f"User selected candidate file: {selected.name}\n"
            f"Confirmed path: {selected.path}\n"
            "Continue the original task using this confirmed candidate."
        )
        return FollowUpAssessment(
            action="resume",
            rationale="explicit_selection_candidate",
            slot_updates={
                "selected_candidate_path": selected.path,
                "selected_candidate_name": selected.name,
            },
            merged_user_request=merged_user_request,
        )

    @staticmethod
    def merge_pending_request(
        pending_task: PendingTask,
        text: str,
        *,
        correction: bool = False,
    ) -> str:
        if correction:
            return (
                f"{pending_task.original_user_request}\n"
                f"用户更正/澄清: {text}\n"
                "请以这条最新更正为准，覆盖之前可能错误的人物、对象、动作理解后继续执行。"
            )
        return (
            f"{pending_task.original_user_request}\n"
            f"补充信息: {text}\n"
            "请基于原始任务和这条补充信息继续执行。"
        )

    @staticmethod
    def activate_context_task_for_message(session: SessionState, text: str) -> PendingTask | None:
        raw_candidates = ContextTaskManager.recent_context_task_records(session)
        candidates = [ContextTaskManager._as_pending_task(task) for task in raw_candidates]
        if not candidates:
            return None

        exact_match = ContextTaskManager.match_explicit_follow_up_candidate(candidates, text)
        if exact_match is not None:
            task, assessment = exact_match
            session.pending_follow_up_assessment = assessment
            session.active_context_task = ContextTaskManager.find_context_task_record_by_id(raw_candidates, task.task_id)
            ContextTaskManager.drop_context_task(session, task.task_id)
            return task

        try:
            attribution = session.kernel.llm_client.classify_task_attribution(
                task_candidates=candidates,
                user_text=text,
            )
        except Exception:
            attribution = None

        if attribution is None or attribution.action == "new_request":
            return None

        if attribution.action in {"resume", "resume_with_correction", "cancel"}:
            matched = ContextTaskManager.find_context_task_by_id(candidates, attribution.target_task_id)
            if matched is None and len(candidates) == 1 and attribution.action == "cancel":
                matched = candidates[0]
            if matched is None:
                return None
            if attribution.action in {"resume", "resume_with_correction"} and not attribution.merged_user_request:
                attribution.merged_user_request = ContextTaskManager.merge_pending_request(
                    matched,
                    text,
                    correction=attribution.action == "resume_with_correction",
                )
                if not attribution.rationale:
                    attribution.rationale = (
                        "llm_task_attribution_resume_with_correction"
                        if attribution.action == "resume_with_correction"
                        else "llm_task_attribution_resume"
                    )
            session.pending_follow_up_assessment = attribution
            session.active_context_task = ContextTaskManager.find_context_task_record_by_id(raw_candidates, matched.task_id)
            ContextTaskManager.drop_context_task(session, matched.task_id)
            return matched

        return None

    @staticmethod
    def recent_context_task_records(session: SessionState) -> list[ContextTaskRecord | PendingTask]:
        tasks = [
            task
            for task in reversed(session.context_tasks)
            if not ContextTaskManager._is_context_task_expired(task)
            and ContextTaskManager._as_pending_task(task).state_kind in {"selection_follow_up", "task_follow_up", "clarification"}
        ]
        return tasks[:5]

    @staticmethod
    def recent_context_tasks(session: SessionState) -> list[PendingTask]:
        return [ContextTaskManager._as_pending_task(task) for task in ContextTaskManager.recent_context_task_records(session)]

    @staticmethod
    def format_context_task_board(session: SessionState, *, limit: int = 3) -> str:
        records = ContextTaskManager.recent_context_task_records(session)[:limit]
        if not records:
            return ""
        lines = ["Recent resumable tasks:"]
        for index, raw_task in enumerate(records, start=1):
            pending = ContextTaskManager._as_pending_task(raw_task)
            candidate_names = [candidate.name for candidate in pending.selection_candidates[:3] if candidate.name]
            parts = [
                f"{index}. id={pending.task_id}",
                f"intent={pending.intent}",
                f"summary={pending.summary}",
            ]
            if isinstance(raw_task, ContextTaskRecord):
                parts.append(f"workflow={raw_task.workflow_family}")
                missing = [item.value for item in raw_task.missing_outputs[:4]]
                if missing:
                    parts.append(f"missing={','.join(missing)}")
            if candidate_names:
                parts.append("candidates=" + " | ".join(candidate_names))
            if pending.resume_hint:
                parts.append(f"hint={pending.resume_hint}")
            lines.append("; ".join(parts))
        return "\n".join(lines)

    @staticmethod
    def find_context_task_by_id(tasks: list[PendingTask], task_id: str | None) -> PendingTask | None:
        if not task_id:
            return None
        for task in tasks:
            if task.task_id == task_id:
                return task
        return None

    @staticmethod
    def find_context_task_record_by_id(
        tasks: list[ContextTaskRecord | PendingTask],
        task_id: str | None,
    ) -> ContextTaskRecord | None:
        if not task_id:
            return None
        for task in tasks:
            if isinstance(task, ContextTaskRecord) and task.task_id == task_id:
                return task
        return None

    @staticmethod
    def match_explicit_follow_up_candidate(
        tasks: list[PendingTask],
        text: str,
    ) -> tuple[PendingTask, FollowUpAssessment] | None:
        for task in tasks:
            assessment = ContextTaskManager.resolve_selection_follow_up(task, text)
            if assessment is not None:
                return task, assessment
        return None

    @staticmethod
    def drop_context_task(session: SessionState, task_id: str) -> None:
        session.context_tasks = [
            task
            for task in session.context_tasks
            if ContextTaskManager._as_pending_task(task).task_id != task_id
        ]

    @staticmethod
    def build_context_task_from_result(user_text: str, turn_result: ChatTurnResult) -> PendingTask | None:
        if not turn_result.used_agent or turn_result.pending_task is not None:
            return None
        metadata = turn_result.metadata or {}
        execution_summary = metadata.get("execution_summary")
        if not isinstance(execution_summary, dict):
            return None
        task_kind = str((execution_summary.get("task_classification") or {}).get("task_kind", "") or "").lower()
        if task_kind not in {"delivery", "summarize", "document_summary", "lookup", "local_lookup", "file_lookup", "document_edit"}:
            return None

        candidate_state = None
        candidate_payload = execution_summary.get("candidate_state")
        if isinstance(candidate_payload, dict):
            try:
                candidate_state = CandidateState.model_validate(candidate_payload)
            except Exception:
                candidate_state = None

        workflow_state = None
        workflow_payload = execution_summary.get("workflow_state")
        if isinstance(workflow_payload, dict):
            try:
                workflow_state = WorkflowState.model_validate(workflow_payload)
            except Exception:
                workflow_state = None

        candidate_paths: list[str] = []
        candidate_names: list[str] = []
        if candidate_state is not None:
            candidate_paths = list(candidate_state.candidate_paths)
            candidate_names = list(candidate_state.candidate_names)
        elif workflow_state is not None:
            candidate_paths = [candidate.path_or_ref for candidate in workflow_state.candidates if candidate.path_or_ref][:5]
            candidate_names = [candidate.display_name for candidate in workflow_state.candidates if candidate.path_or_ref][:5]

        goal = None
        if isinstance(turn_result.overall_task_goal, dict):
            try:
                goal = TaskGoal.model_validate(turn_result.overall_task_goal)
            except Exception:
                goal = None

        collected_slots: dict[str, str] = {}
        if candidate_state is not None:
            collected_slots["query"] = candidate_state.query
            collected_slots["path_scope"] = candidate_state.path_scope
            type_constraints = candidate_state.metadata.get("type_constraints", [])
            if isinstance(type_constraints, list):
                collected_slots["type_constraints"] = ",".join(str(item) for item in type_constraints if item)
        if workflow_state is not None:
            collected_slots["workflow_family"] = workflow_state.workflow_family
            if workflow_state.primary_target_ref:
                collected_slots["primary_target_ref"] = workflow_state.primary_target_ref
        if candidate_paths:
            collected_slots.setdefault("selected_candidate_path", candidate_paths[0])
        if candidate_names:
            collected_slots.setdefault("selected_candidate_name", candidate_names[0])
        for action in execution_summary.get("successful_actions", []):
            if not isinstance(action, dict) or action.get("tool_name") != "qq.send_file":
                continue
            data = action.get("data")
            if isinstance(data, dict) and isinstance(data.get("path"), str):
                collected_slots["last_sent_file"] = data["path"]

        missing_outputs = {
            str(getattr(item, "value", item))
            for item in execution_summary.get("missing_outputs", [])
            if item
        }
        if workflow_state is not None:
            missing_outputs.update(
                str(getattr(item, "value", item))
                for item in workflow_state.missing_outputs
                if item
            )
        if (
            task_kind == "document_edit"
            and candidate_paths
            and missing_outputs & {"file_written", "object_details"}
        ):
            selection_candidates: list[SelectionCandidate] = []
            for index, path in enumerate(candidate_paths[:1], start=1):
                if not path:
                    continue
                name = (
                    candidate_names[index - 1]
                    if index - 1 < len(candidate_names) and candidate_names[index - 1]
                    else Path(path).name
                )
                selection_candidates.append(
                    SelectionCandidate(
                        candidate_id=str(index),
                        path=path,
                        name=name,
                    )
                )
            return PendingTask(
                task_id=f"latent_{uuid.uuid4().hex[:12]}",
                intent="document_edit_follow_up",
                summary="Resume the grounded document edit with supplemental instructions.",
                original_user_request=user_text,
                state_kind="task_follow_up",
                selection_candidates=selection_candidates,
                overall_task_goal=goal,
                missing_slots=["supplemental_instruction"],
                collected_slots=collected_slots,
                resume_hint="You can add or correct the exact content to write; the target file is already grounded.",
            )
        keep_empty_follow_up = not candidate_paths and bool(
            missing_outputs & {"object_candidates", "message_sent", "file_contents"}
        )
        if len(candidate_paths) < 2 and not keep_empty_follow_up:
            return None

        selection_candidates: list[SelectionCandidate] = []
        for index, path in enumerate(candidate_paths[:5], start=1):
            if not path:
                continue
            name = (
                candidate_names[index - 1]
                if index - 1 < len(candidate_names) and candidate_names[index - 1]
                else Path(path).name
            )
            selection_candidates.append(
                SelectionCandidate(
                    candidate_id=str(index),
                    path=path,
                    name=name,
                )
            )
        if len(selection_candidates) < 2 and not keep_empty_follow_up:
            return None

        return PendingTask(
            task_id=f"latent_{uuid.uuid4().hex[:12]}",
            intent="selection_follow_up",
            summary="保留上一轮候选结果，允许用户继续用年份、月份或关键词缩小范围。",
            original_user_request=user_text,
            state_kind="task_follow_up",
            selection_candidates=selection_candidates,
            overall_task_goal=goal,
            missing_slots=["selected_candidate_path"] if selection_candidates else ["candidate_refinement"],
            collected_slots=collected_slots,
            resume_hint="如果你要补充年份、月份、文件格式或关键词，我可以基于上一轮候选继续缩小范围。",
        )

    @staticmethod
    @staticmethod
    def build_context_task_record_from_result(
        user_text: str,
        turn_result: ChatTurnResult,
    ) -> ContextTaskRecord | None:
        pending_task = ContextTaskManager.build_context_task_from_result(user_text, turn_result)
        if pending_task is None:
            return None

        metadata = turn_result.metadata or {}
        execution_summary = metadata.get("execution_summary")
        if not isinstance(execution_summary, dict):
            execution_summary = {}

        candidate_state = None
        candidate_payload = execution_summary.get("candidate_state")
        if isinstance(candidate_payload, dict):
            try:
                candidate_state = CandidateState.model_validate(candidate_payload)
            except Exception:
                candidate_state = None

        workflow_state = None
        workflow_payload = execution_summary.get("workflow_state")
        if isinstance(workflow_payload, dict):
            try:
                workflow_state = WorkflowState.model_validate(workflow_payload)
            except Exception:
                workflow_state = None

        task_classification = execution_summary.get("task_classification")
        task_kind = ""
        if isinstance(task_classification, dict):
            task_kind = str(task_classification.get("task_kind") or "").strip()
        if not task_kind:
            task_kind = pending_task.intent

        workflow_family = str(execution_summary.get("workflow_family") or "").strip()
        if not workflow_family and workflow_state is not None:
            workflow_family = workflow_state.workflow_family

        return ContextTaskRecord(
            task_id=pending_task.task_id,
            original_user_request=pending_task.original_user_request,
            summary=pending_task.summary,
            task_kind=task_kind,
            workflow_family=workflow_family or "generic",
            state_kind=pending_task.state_kind,
            selection_candidates=list(pending_task.selection_candidates),
            candidate_state=candidate_state,
            workflow_state=workflow_state,
            overall_task_goal=pending_task.overall_task_goal,
            completed_outputs=ContextTaskManager._coerce_output_kinds(execution_summary.get("completed_outputs")),
            missing_outputs=ContextTaskManager._coerce_output_kinds(execution_summary.get("missing_outputs")),
            collected_slots=dict(pending_task.collected_slots),
            resume_hint=pending_task.resume_hint,
            confidence=0.86 if candidate_state is not None or workflow_state is not None else 0.68,
            expires_at=datetime.now(UTC) + timedelta(minutes=ContextTaskManager._DEFAULT_CONTEXT_TASK_TTL_MINUTES),
            metadata={
                "source": "turn_result",
                "trace_id": metadata.get("trace_id"),
                "stop_reason": execution_summary.get("stop_reason"),
            },
        )

    @staticmethod
    def _coerce_output_kinds(values) -> list[OutputKind]:
        output_values = {item.value: item for item in OutputKind}
        result: list[OutputKind] = []
        raw_items = values if isinstance(values, list) else []
        for item in raw_items:
            raw = str(getattr(item, "value", item))
            output = output_values.get(raw)
            if output is not None and output not in result:
                result.append(output)
        return result

    @staticmethod
    def remember_context_task(session: SessionState, task: ContextTaskRecord | PendingTask) -> None:
        pending_task = ContextTaskManager._as_pending_task(task)
        deduped = [
            existing
            for existing in session.context_tasks
            if not (
                ContextTaskManager._as_pending_task(existing).original_user_request == pending_task.original_user_request
                and ContextTaskManager._as_pending_task(existing).intent == pending_task.intent
                and ContextTaskManager._as_pending_task(existing).state_kind == pending_task.state_kind
            )
            and not ContextTaskManager._is_context_task_expired(existing)
        ]
        deduped.append(task)
        session.context_tasks = deduped[-6:]

    @staticmethod
    def update_context_task_registry(session: SessionState, user_text: str, turn_result: ChatTurnResult) -> None:
        if turn_result.pending_task is not None:
            session.pending_follow_up_assessment = None
            session.active_context_task = None
            return
        task = ContextTaskManager.build_context_task_record_from_result(user_text, turn_result)
        if task is not None:
            ContextTaskManager.remember_context_task(session, task)
