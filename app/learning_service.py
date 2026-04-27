from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from local_agent.app.chat_models import ChatTurnResult
from local_agent.memory.warm_memory import WarmMemoryService
from local_agent.storage.memory_store import SQLiteMemoryStore
from local_agent.storage.trace_store import JsonlTraceStore


_MEMORY_INSTRUCTION_TERMS = (
    "记住",
    "以后",
    "默认",
    "不要再",
    "不要把",
    "remember",
    "default",
    "do not",
    "don't",
)


@dataclass(frozen=True)
class LearnedMemory:
    memory_type: str
    content: str


class RealTurnLearningService:
    _TRANSIENT_ERROR_TERMS = (
        "database is locked",
        "busy timeout",
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
        "timed out",
        "timeout",
        "try again",
    )

    def __init__(
        self,
        memory_store: SQLiteMemoryStore,
        *,
        trace_store: JsonlTraceStore | None = None,
        llm_client: Any | None = None,
        reflection_enabled: bool = True,
    ) -> None:
        self._memory_store = memory_store
        self._warm_memory = WarmMemoryService(memory_store)
        self._trace_store = trace_store
        self._llm_client = llm_client
        self._reflection_enabled = reflection_enabled

    def learn_from_turn(
        self,
        *,
        user_text: str,
        turn_result: ChatTurnResult,
        scope: str = "user",
    ) -> list[LearnedMemory]:
        normalized = " ".join(str(user_text or "").split()).strip()
        if len(normalized) < 6:
            return []
        if not turn_result.used_agent:
            return []
        if self._is_memory_instruction(normalized):
            return []

        execution_summary = {}
        if isinstance(turn_result.metadata, dict):
            execution_summary = turn_result.metadata.get("execution_summary") or {}

        learned: list[LearnedMemory] = []
        drafted = self._build_memories(normalized, turn_result, execution_summary)
        reflected = self._build_reflection_memories(normalized, turn_result, execution_summary, drafted)
        for memory in self._dedupe_memories([*drafted, *reflected]):
            self._persist_memory(memory, scope=scope, tags=self._build_tags(turn_result, execution_summary))
            learned.append(memory)

        digest = self._warm_memory.compact_learning_memories(scope=scope)
        if digest:
            learned.append(LearnedMemory(memory_type="lesson_digest", content=digest))

        if learned and self._trace_store is not None:
            self._trace_store.append(
                "runtime_learning",
                {
                    "user_text": normalized,
                    "trace_id": turn_result.metadata.get("trace_id") if isinstance(turn_result.metadata, dict) else "",
                    "context": self._build_trace_context(execution_summary, turn_result),
                    "memories": [memory.__dict__ for memory in learned],
                },
            )
        return learned

    def _build_memories(
        self,
        user_text: str,
        turn_result: ChatTurnResult,
        execution_summary: dict,
    ) -> list[LearnedMemory]:
        learned: list[LearnedMemory] = []
        request = self._shorten_text(user_text, limit=48)
        tool_names = self._tool_names(execution_summary)
        completed_outputs = list(turn_result.completed_outputs or [])
        failed_actions = execution_summary.get("failed_actions") or []
        task_status = str(execution_summary.get("task_status", "") or "")
        missing_outputs = [str(item).strip() for item in execution_summary.get("missing_outputs") or [] if str(item).strip()]
        context_suffix = self._memory_context_suffix(execution_summary, tool_names=tool_names)

        if self._looks_like_scope_leak(user_text, execution_summary):
            learned.append(
                LearnedMemory(
                    memory_type="failure_pattern",
                    content="真实执行经验：当用户明确提到桌面时，先优先检查桌面根目录候选，再考虑 testing 等子目录里的相似文件。",
                )
            )

        if self._is_true_success(turn_result, execution_summary):
            tool_text = " -> ".join(tool_names[:4]) if tool_names else "直接完成"
            output_text = ", ".join(completed_outputs[:4]) if completed_outputs else "无显式输出"
            learned.append(
                LearnedMemory(
                    memory_type="success_pattern",
                    content=(
                        f"真实成功模式：处理类似“{request}”的请求时，优先走 agent；"
                        f"关键步骤通常是 {tool_text}；目标输出通常包括 {output_text}。"
                    ),
                )
            )

        for action in failed_actions[:2]:
            tool_name = str(action.get("tool_name", "")).strip()
            error_message = self._shorten_text(str(action.get("error", "")).strip(), limit=80)
            if not tool_name:
                continue
            if self._is_transient_failure(tool_name, error_message):
                continue
            learned.append(
                LearnedMemory(
                    memory_type="failure_pattern",
                    content=(
                        f"真实执行经验：处理类似“{request}”的任务时，若调用 {tool_name}，"
                        f"要先规避这类报错：{error_message or '未知错误'}。"
                    ),
                )
            )

        if task_status != "completed" and missing_outputs:
            output_text = ", ".join(missing_outputs[:4])
            learned.append(
                LearnedMemory(
                    memory_type="workflow_lesson",
                    content=f"真实执行经验：处理类似“{request}”的任务时，在拿到 {output_text} 之前不要提前结束。",
                )
            )

        if (
            task_status == "partial"
            and not failed_actions
            and execution_summary.get("candidate_paths")
            and "object_candidates" in completed_outputs
            and missing_outputs
        ):
            learned.append(
                LearnedMemory(
                    memory_type="eval_lesson",
                    content=(
                        f"真实执行经验：处理类似“{request}”的任务时，"
                        "找到候选文件后通常还需要继续推进，不要只停在候选列表。"
                    ),
                )
            )

        if context_suffix:
            learned = [
                LearnedMemory(memory_type=item.memory_type, content=f"{item.content}{context_suffix}")
                for item in learned
            ]
        return self._dedupe_memories(learned)

    def _build_reflection_memories(
        self,
        user_text: str,
        turn_result: ChatTurnResult,
        execution_summary: dict,
        drafted: list[LearnedMemory],
    ) -> list[LearnedMemory]:
        if not self._reflection_enabled or self._llm_client is None:
            return []
        if not self._should_request_reflection(user_text, turn_result, execution_summary):
            return []
        reflect = getattr(self._llm_client, "reflect_runtime_learning", None)
        if not callable(reflect):
            return []
        try:
            raw_memories = reflect(
                user_text=user_text,
                turn_result={
                    "mode": turn_result.mode,
                    "response": turn_result.response,
                    "completed_outputs": list(turn_result.completed_outputs or []),
                    "overall_task_goal": turn_result.overall_task_goal,
                    "pending_task": turn_result.pending_task,
                },
                execution_summary=execution_summary,
                existing_memories=[memory.__dict__ for memory in drafted],
            )
        except Exception:
            return []

        reflected: list[LearnedMemory] = []
        for item in raw_memories or []:
            if not isinstance(item, dict):
                continue
            memory_type = str(item.get("memory_type", "")).strip()
            content = str(item.get("content", "")).strip()
            if memory_type not in {"failure_pattern", "workflow_lesson", "eval_lesson", "success_pattern"}:
                continue
            if not content:
                continue
            reflected.append(LearnedMemory(memory_type=memory_type, content=content))
        return self._dedupe_memories(reflected)

    def _persist_memory(self, memory: LearnedMemory, *, scope: str, tags: list[str]) -> None:
        if memory.memory_type == "success_pattern":
            self._warm_memory.remember_success_pattern(memory.content, scope=scope, tags=tags)
        elif memory.memory_type == "failure_pattern":
            self._warm_memory.remember_failure_pattern(memory.content, scope=scope, tags=tags)
        elif memory.memory_type == "workflow_lesson":
            self._warm_memory.remember_workflow_lesson(memory.content, scope=scope, tags=tags)
        else:
            self._warm_memory.remember_eval_lesson(memory.content, scope=scope, tags=tags)

    @staticmethod
    def _tool_names(execution_summary: dict) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for collection_name in ("successful_actions", "failed_actions"):
            for item in execution_summary.get(collection_name, []) or []:
                tool_name = str(item.get("tool_name", "")).strip()
                if tool_name and tool_name not in seen:
                    seen.add(tool_name)
                    names.append(tool_name)
        return names

    @staticmethod
    def _build_tags(turn_result: ChatTurnResult, execution_summary: dict) -> list[str]:
        tags: list[str] = ["runtime_learning", "agent"]
        tags.extend(str(item).strip() for item in (turn_result.completed_outputs or [])[:4] if str(item).strip())
        tags.extend(tool.replace(".", "_") for tool in RealTurnLearningService._tool_names(execution_summary)[:4])
        task_classification = execution_summary.get("task_classification") or {}
        workflow_state = execution_summary.get("workflow_state") or {}
        decision_stats = execution_summary.get("decision_stats") or {}
        domain = str(task_classification.get("domain", "") or "").strip()
        task_kind = str(task_classification.get("task_kind", "") or "").strip()
        workflow_family = str(workflow_state.get("workflow_family", "") or "").strip()
        workflow_stage = str(workflow_state.get("workflow_stage", "") or "").strip()
        stop_reason = str(execution_summary.get("stop_reason", "") or "").strip()
        if domain:
            tags.append(domain)
        if task_kind:
            tags.append(task_kind)
        if workflow_family:
            tags.append(workflow_family)
        if workflow_stage:
            tags.append(f"stage_{workflow_stage}")
        if stop_reason:
            tags.append(f"stop_{stop_reason}")
        if int(decision_stats.get("planner_bypass_count") or 0) > 0:
            tags.append("state_machine_direct")
        if execution_summary.get("candidate_paths"):
            tags.append("has_candidates")
        return list(dict.fromkeys(tags))

    @staticmethod
    def _shorten_text(text: str, *, limit: int) -> str:
        normalized = " ".join(str(text or "").split()).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(8, limit - 1)].rstrip() + "…"

    @staticmethod
    def _dedupe_memories(memories: list[LearnedMemory]) -> list[LearnedMemory]:
        deduped: list[LearnedMemory] = []
        seen: set[tuple[str, str]] = set()
        for item in memories:
            key = (item.memory_type, item.content)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _is_memory_instruction(user_text: str) -> bool:
        lowered = user_text.lower()
        return any(term in user_text or term in lowered for term in _MEMORY_INSTRUCTION_TERMS)

    @staticmethod
    def _normalize_outputs(payload: Any) -> set[str]:
        if not isinstance(payload, (list, tuple, set)):
            return set()
        return {str(item).strip() for item in payload if str(item).strip()}

    @staticmethod
    def _memory_context_suffix(execution_summary: dict, *, tool_names: list[str]) -> str:
        task_classification = execution_summary.get("task_classification") or {}
        workflow_state = execution_summary.get("workflow_state") or {}
        decision_stats = execution_summary.get("decision_stats") or {}
        parts: list[str] = []
        domain = str(task_classification.get("domain", "") or "").strip()
        task_kind = str(task_classification.get("task_kind", "") or "").strip()
        if domain or task_kind:
            parts.append(f"领域={domain or 'unknown'}/{task_kind or 'unknown'}")
        workflow_family = str(workflow_state.get("workflow_family", "") or "").strip()
        workflow_stage = str(workflow_state.get("workflow_stage", "") or "").strip()
        if workflow_family or workflow_stage:
            parts.append(f"workflow={workflow_family or 'unknown'}/{workflow_stage or 'unknown'}")
        stop_reason = str(execution_summary.get("stop_reason", "") or "").strip()
        if stop_reason:
            parts.append(f"stop_reason={stop_reason}")
        if tool_names:
            parts.append(f"工具链={' -> '.join(tool_names[:4])}")
        bypass_count = int(decision_stats.get("planner_bypass_count") or 0)
        planner_invocations = int(decision_stats.get("planner_invocations") or 0)
        if bypass_count > 0:
            parts.append(f"状态机直推={bypass_count}")
        elif planner_invocations > 0:
            parts.append(f"planner调用={planner_invocations}")
        if not parts:
            return ""
        return "（" + "；".join(parts) + "）"

    @staticmethod
    def _build_trace_context(execution_summary: dict, turn_result: ChatTurnResult) -> dict[str, Any]:
        task_classification = execution_summary.get("task_classification") or {}
        workflow_state = execution_summary.get("workflow_state") or {}
        decision_stats = execution_summary.get("decision_stats") or {}
        return {
            "task_status": str(execution_summary.get("task_status", "") or ""),
            "domain": str(task_classification.get("domain", "") or ""),
            "task_kind": str(task_classification.get("task_kind", "") or ""),
            "workflow_family": str(workflow_state.get("workflow_family", "") or ""),
            "workflow_stage": str(workflow_state.get("workflow_stage", "") or ""),
            "stop_reason": str(execution_summary.get("stop_reason", "") or ""),
            "completed_outputs": list(turn_result.completed_outputs or []),
            "missing_outputs": [str(item).strip() for item in execution_summary.get("missing_outputs") or [] if str(item).strip()],
            "tool_names": RealTurnLearningService._tool_names(execution_summary),
            "decision_stats": {
                "planner_invocations": int(decision_stats.get("planner_invocations") or 0),
                "critic_invocations": int(decision_stats.get("critic_invocations") or 0),
                "planner_bypass_count": int(decision_stats.get("planner_bypass_count") or 0),
            },
        }

    def _is_true_success(self, turn_result: ChatTurnResult, execution_summary: dict) -> bool:
        failed_actions = execution_summary.get("failed_actions") or []
        task_status = str(execution_summary.get("task_status", "") or "")
        missing_outputs = self._normalize_outputs(execution_summary.get("missing_outputs") or [])
        completed_outputs = self._normalize_outputs(turn_result.completed_outputs or [])
        task_kind = str((execution_summary.get("task_classification") or {}).get("task_kind", "") or "").lower()
        required_outputs = self._normalize_outputs(
            ((execution_summary.get("overall_task_goal") or {}).get("required_outputs") or [])
        )
        completion_review = execution_summary.get("completion_review") or {}
        grounding_review = execution_summary.get("grounding_review") or {}

        if task_status != "completed" or failed_actions or missing_outputs:
            return False
        if completion_review.get("approved") is False or grounding_review.get("approved") is False:
            return False
        if required_outputs and not required_outputs.issubset(completed_outputs):
            return False
        if self._looks_like_scope_leak("", execution_summary):
            return False
        if task_kind in {"summarize", "document_summary"} and "file_contents" not in completed_outputs:
            return False
        if task_kind == "delivery" and not ({"message_sent", "file_written"} & completed_outputs):
            return False
        if task_kind in {"document_edit", "edit", "rewrite", "transform"} and "file_written" not in completed_outputs:
            return False
        if completed_outputs == {"object_candidates"}:
            return task_kind in {"lookup", "file_lookup", "local_lookup"}
        return bool(completed_outputs or self._tool_names(execution_summary))

    @staticmethod
    def _looks_like_scope_leak(user_text: str, execution_summary: dict) -> bool:
        lowered = user_text.lower()
        if "桌面" not in user_text and "desktop" not in lowered:
            return False
        candidate_paths = [str(item) for item in execution_summary.get("candidate_paths") or [] if str(item).strip()]
        if len(candidate_paths) < 2:
            return False
        top_candidate = candidate_paths[0].lower()
        has_root_desktop_candidate = any(
            "\\desktop\\" in path.lower() and "\\desktop\\testing\\" not in path.lower()
            for path in candidate_paths[:4]
        )
        return "\\desktop\\testing\\" in top_candidate and has_root_desktop_candidate

    @classmethod
    def _is_transient_failure(cls, tool_name: str, error_message: str) -> bool:
        lowered = f"{tool_name} {error_message}".lower()
        return any(term in lowered for term in cls._TRANSIENT_ERROR_TERMS)

    def _should_request_reflection(
        self,
        user_text: str,
        turn_result: ChatTurnResult,
        execution_summary: dict,
    ) -> bool:
        task_status = str(execution_summary.get("task_status", "") or "")
        if task_status in {"partial", "failed"}:
            return True
        if execution_summary.get("failed_actions"):
            return True
        completion_review = execution_summary.get("completion_review") or {}
        grounding_review = execution_summary.get("grounding_review") or {}
        if completion_review.get("approved") is False or grounding_review.get("approved") is False:
            return True
        if self._looks_like_scope_leak(user_text, execution_summary):
            return True
        response = str(turn_result.response or "")
        if "候选" in response or "确认" in response or "请选" in response:
            return True
        return False
