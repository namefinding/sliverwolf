from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Any
from local_agent.app.scope_resolver import infer_scope_root
from local_agent.artifacts.output_planner import OutputArtifactPlanner
from local_agent.intent.models import WorkflowProposal
from local_agent.intent.service import IntentService
from local_agent.kernel.completion_judge import CompletionJudge
from local_agent.kernel.context_builder import ContextBuilder
from local_agent.kernel.decision_critic import DecisionCritic
from local_agent.kernel.decision_validator import DecisionValidator
from local_agent.kernel.execution_critic import ExecutionCritic
from local_agent.kernel.file_retrieval_strategy import FileRetrievalStrategy
from local_agent.kernel.guardrails import Guardrails
from local_agent.kernel.loop_controller import LoopController
from local_agent.kernel.request_intent_analyzer import RequestIntentAnalyzer
from local_agent.kernel.web_retrieval_strategy import WebRetrievalStrategy
from local_agent.kernel.workflow_argument_planner import WorkflowArgumentPlanner
from local_agent.kernel.workflow_selector import WorkflowSelector
from local_agent.llm.ollama_client import OllamaClient
from local_agent.memory.warm_memory import WarmMemoryService
from local_agent.modules.base import ToolRegistry
from local_agent.protocol.models import (
    AgentConfig,
    CandidateState,
    DecisionReview,
    DecisionType,
    ExecutionReview,
    MemoryRecord,
    Message,
    OutputKind,
    PendingTask,
    RiskLevel,
    Role,
    SelectionCandidate,
    TaskGoal,
    ToolCallResult,
    ToolDecision,
    TurnArtifacts,
    WorkflowCandidate,
    WorkflowNodeSpec,
    WorkflowSpec,
    WorkflowState,
)
from local_agent.storage.memory_store import SQLiteMemoryStore
from local_agent.storage.trace_store import JsonlTraceStore
from local_agent.utils.file_query_normalizer import FileQueryNormalizer
from local_agent.utils.target_resolver import resolve_target_reference
from local_agent.utils.workspace_path import WorkspacePathNormalizer
from local_agent.voice.gptsovits import GPTSoVITSAdapter
from local_agent.workflows.local_collection_workflow import LocalCollectionWorkflow


class AgentKernel:
    _TRUSTED_STATE_MACHINE_TOOLS = {
        "retrieval.search_local_objects",
        "retrieval.inspect_local_candidate",
        "file.search_by_name",
        "file.search_text",
        "file.list",
        "file.read",
        "file.extract_text",
        "file.metadata",
        "file.preview",
        "document_agent.summarize",
        "document_agent.read",
        "document_agent.inspect",
        "document_agent.edit",
        "qq.get_current_context",
        "qq.get_recent_messages",
        "qq.get_last_reply",
        "qq.search_history",
        "qq.get_recent_attachments",
        "qq.search_contacts",
        "web.search",
        "web.research",
        "web.fetch",
    }
    _WORKFLOW_SPEED_LESSON = (
        "\u6267\u884c\u7b56\u7565\u53c2\u8003\uff1a\u5bf9\u5355\u6b65\u3001\u4f4e\u98ce\u9669\u3001"
        "\u53c2\u6570\u5df2\u5b8c\u6574\u7684\u4efb\u52a1\uff08\u5982\u5f53\u524d\u65f6\u95f4\u3001QQ\u5386\u53f2\u3001"
        "\u5929\u6c14\u6216\u666e\u901a\u7f51\u9875\u67e5\u8be2\uff09\uff0c\u4f18\u5148\u4f7f\u7528\u7ed3\u6784\u5316 intent "
        "\u548c\u72b6\u6001\u673a\u5019\u9009\uff1b\u53ea\u5728\u8de8\u6a21\u5757\u3001\u591a\u9636\u6bb5\u6216\u53c2\u6570\u4e0d\u5b8c\u6574\u65f6"
        "\u518d\u542f\u7528\u5168\u91cf workflow spec \u6216\u53c2\u6570\u89c4\u5212\u3002\u8fd9\u662f\u8f6f\u53c2\u8003\uff0c"
        "\u4e0d\u4f5c\u4e3a\u56fa\u5b9a\u77ed\u8bed\u6216\u6b7b\u6d41\u7a0b\u3002"
    )

    def __init__(
        self,
        config: AgentConfig,
        llm_client: OllamaClient,
        registry: ToolRegistry,
        memory_store: SQLiteMemoryStore,
        trace_store: JsonlTraceStore,
        voice_adapter: GPTSoVITSAdapter,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.registry = registry
        self.memory_store = memory_store
        self.trace_store = trace_store
        self.voice_adapter = voice_adapter
        self.guardrails = Guardrails()
        self.critic = DecisionCritic(llm_client)
        self.execution_critic = ExecutionCritic()
        self.file_retrieval_strategy = FileRetrievalStrategy()
        self.web_retrieval_strategy = WebRetrievalStrategy()
        self.request_intent_analyzer = RequestIntentAnalyzer(llm_client)
        self.intent_service = IntentService(self.request_intent_analyzer)
        self.validator = DecisionValidator(config.workspace_root, registry)
        self.workflow_argument_planner = WorkflowArgumentPlanner(llm_client, registry)
        self.path_normalizer = WorkspacePathNormalizer(config.workspace_root)
        self.local_collection_workflow = LocalCollectionWorkflow(config.workspace_root, llm_client=llm_client)
        self.completion_judge = CompletionJudge()
        self.loop_controller = LoopController()
        self._remember_runtime_workflow_speed_lesson()
        self.hot_context_summary: str = ""
        self.active_task_summary: str = ""
        self.user_memory_summary: str = ""
        self.learning_memory_summary: str = ""
        self.warm_memory_summary: str = ""
        self.cold_memory_summary: str = ""
        self.history: list[Message] = [
            Message(
                role=Role.SYSTEM,
                content=(
                    "You are a local intelligent agent. "
                    "Understand the user's request, decide whether to respond directly or call tools, "
                    "and continue the workflow step by step until the task is completed. "
                    "Return the final result in natural Chinese."
                ),
            )
        ]

    def _remember_runtime_workflow_speed_lesson(self) -> None:
        try:
            WarmMemoryService(self.memory_store).remember_workflow_lesson(
                self._WORKFLOW_SPEED_LESSON,
                scope="global",
                tags=["workflow_speed", "state_machine", "web_lookup"],
                importance=0.74,
            )
        except Exception:  # noqa: BLE001
            return


    @staticmethod
    def _extract_scheduled_task_fire_context(runtime_channel_context: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(runtime_channel_context, dict):
            return None
        scheduled_task = runtime_channel_context.get("scheduled_task")
        if not isinstance(scheduled_task, dict):
            return None
        phase = str(scheduled_task.get("phase") or "").strip().lower()
        if not phase:
            return None
        return {
            **scheduled_task,
            "phase": phase,
            "task_type": str(scheduled_task.get("task_type") or "").strip().lower() or None,
            "message": str(scheduled_task.get("message") or "").strip(),
        }

    def _should_schedule_request(
        self,
        task_classification,
        *,
        runtime_channel_context: dict[str, Any] | None = None,
    ) -> bool:
        if task_classification is None:
            return False
        fire_context = self._extract_scheduled_task_fire_context(runtime_channel_context)
        if fire_context is not None and fire_context.get("phase") == "fired":
            return False
        run_mode = str(getattr(task_classification, "run_mode", "immediate") or "immediate").strip().lower()
        return run_mode == "scheduled"

    def _build_scheduled_task_goal(
            self,
            *,
            user_text: str,
            task_type: str,
    ) -> TaskGoal:
        summary = f"Create a scheduled task for: {user_text}"
        if task_type == "notify":
            return TaskGoal(
                summary=summary,
                required_outputs=[],
                completion_mode="success",
            )
        return TaskGoal(
            summary=summary,
            required_outputs=[],
            completion_mode="success",
        )

    @staticmethod
    def _emit_progress(
        progress_callback: Callable[[str, str, dict], None] | None,
        stage: str,
        message: str,
        payload: dict | None = None,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(stage, message, payload or {})

    def _scripted_planner_decision_count(self) -> int:
        decisions = getattr(self.llm_client, "decisions", None)
        return len(decisions) if isinstance(decisions, list) else 0

    def _context_messages(self) -> list[Message]:
        return ContextBuilder.build_prompt_messages(
            self.history,
            session_summary=getattr(self, "hot_context_summary", ""),
            active_task_summary=getattr(self, "active_task_summary", ""),
            warm_memory_summary=getattr(self, "warm_memory_summary", ""),
            learning_memory_summary=getattr(self, "learning_memory_summary", ""),
            cold_memory_summary=getattr(self, "cold_memory_summary", ""),
        )

    def _local_scope_hints(self) -> dict[str, str]:
        hints: dict[str, str] = {}
        workspace_root = str(getattr(self.config, "workspace_root", "") or "").strip()
        if workspace_root:
            hints["workspace"] = workspace_root

        home = Path.home()
        for alias, folder in (
            ("desktop", "Desktop"),
            ("downloads", "Downloads"),
            ("documents", "Documents"),
            ("pictures", "Pictures"),
        ):
            candidate = (home / folder).resolve()
            if candidate.exists():
                hints[alias] = str(candidate)
        return hints

    def _recent_conversation_text(self, limit: int = 6) -> str:
        visible_messages = [
            f"{message.role.value}: {message.content}"
            for message in self.history
            if message.role in {Role.SYSTEM, Role.USER, Role.ASSISTANT}
        ]
        return "\n".join(visible_messages[-limit:])

    @staticmethod
    def _build_channel_context_summary(
        runtime_channel: str | None,
        runtime_channel_context: dict[str, Any] | None,
    ) -> str:
        channel_name = str(runtime_channel or "").strip()
        if not channel_name and not isinstance(runtime_channel_context, dict):
            return ""

        parts: list[str] = []
        if channel_name:
            parts.append(f"channel={channel_name}")

        if isinstance(runtime_channel_context, dict):
            session_id = str(runtime_channel_context.get("session_id") or "").strip()
            if session_id:
                parts.append(f"session_id={session_id}")

            sender_id = str(runtime_channel_context.get("sender_id") or "").strip()
            if sender_id:
                parts.append(f"sender_id={sender_id}")

            sender_name = str(runtime_channel_context.get("sender_name") or "").strip()
            if sender_name:
                parts.append(f"sender_name={sender_name}")

            mode = str(runtime_channel_context.get("mode") or "").strip()
            if mode:
                parts.append(f"mode={mode}")

            current_target = runtime_channel_context.get("current_target")
            if isinstance(current_target, dict):
                message_type = str(current_target.get("message_type") or "").strip().lower()
                if message_type:
                    parts.append(f"target_type={message_type}")
                user_id = current_target.get("user_id")
                if isinstance(user_id, int):
                    parts.append(f"target_user_id={user_id}")
                group_id = current_target.get("group_id")
                if isinstance(group_id, int):
                    parts.append(f"target_group_id={group_id}")

            finalized_segments = runtime_channel_context.get("finalized_turn_segments")
            if isinstance(finalized_segments, list):
                compact_segments = [
                    str(item).strip()
                    for item in finalized_segments[:6]
                    if str(item).strip()
                ]
                if compact_segments:
                    parts.append(
                        "finalized_turn_segments:\n"
                        + "\n".join(f"- {segment}" for segment in compact_segments)
                    )

            recent_user_messages = runtime_channel_context.get("recent_user_messages")
            if isinstance(recent_user_messages, list):
                rendered_messages: list[str] = []
                for item in recent_user_messages[-6:]:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("text") or "").strip()
                    if not text:
                        continue
                    created_at = str(item.get("created_at") or "").strip()
                    rendered_messages.append(
                        f"- {created_at}: {text}" if created_at else f"- {text}"
                    )
                if rendered_messages:
                    parts.append("recent_user_messages:\n" + "\n".join(rendered_messages))

        return "\n".join(parts)

    @staticmethod
    def _merge_goals(
        planner_goal: TaskGoal | None,
        review_goal: TaskGoal | None,
    ) -> TaskGoal | None:
        if planner_goal is None:
            return review_goal
        if review_goal is None:
            return planner_goal

        merged_outputs = list(planner_goal.required_outputs)
        for output_kind in review_goal.required_outputs:
            if output_kind not in merged_outputs:
                merged_outputs.append(output_kind)
        return TaskGoal(
            summary=planner_goal.summary or review_goal.summary,
            required_outputs=merged_outputs,
        )

    def _resolve_reviewed_decision(self, planner_decision: ToolDecision, review: DecisionReview) -> ToolDecision:
        if review.approved or review.suggested_decision is None:
            return planner_decision

        reviewed = review.suggested_decision.model_copy(deep=True)
        reviewed.overall_task_goal = self._merge_goals(
            planner_decision.overall_task_goal,
            review.suggested_decision.overall_task_goal,
        )
        selected_tool = reviewed.selected_tool or planner_decision.selected_tool
        if selected_tool and self.registry.has_tool(selected_tool):
            allowed_step_outputs = list(self.registry.get_manifest(selected_tool).produces)
            filtered_outputs = [output_kind for output_kind in reviewed.expected_step_outputs if output_kind in allowed_step_outputs]
            if filtered_outputs:
                reviewed.expected_step_outputs = filtered_outputs
            elif planner_decision.expected_step_outputs:
                reviewed.expected_step_outputs = [
                    output_kind
                    for output_kind in planner_decision.expected_step_outputs
                    if output_kind in allowed_step_outputs
                ]
            else:
                reviewed.expected_step_outputs = allowed_step_outputs
        elif not reviewed.expected_step_outputs:
            reviewed.expected_step_outputs = list(planner_decision.expected_step_outputs)
        return reviewed

    @staticmethod
    def _is_effectively_empty_argument(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False

    @classmethod
    def _planner_tool_call_is_less_grounded_than_state_machine_repair(
        cls,
        planner_decision: ToolDecision,
        state_machine_repair: ToolDecision | None,
    ) -> bool:
        if state_machine_repair is None:
            return False
        if planner_decision.decision != DecisionType.TOOL_CALL:
            return False
        if state_machine_repair.decision != DecisionType.TOOL_CALL:
            return False
        if not planner_decision.selected_tool or planner_decision.selected_tool != state_machine_repair.selected_tool:
            return False
        if not state_machine_repair.arguments:
            return False
        for key, repair_value in state_machine_repair.arguments.items():
            if cls._is_effectively_empty_argument(repair_value):
                continue
            if cls._is_effectively_empty_argument(planner_decision.arguments.get(key)):
                return True
        return False

    @staticmethod
    def _validated_state_machine_repair(
            state_machine_repair: ToolDecision | None,
            *,
            request_signatures: set[str],
            request_signature: Callable[[ToolDecision], str],
            validate_decision: Callable[[ToolDecision], None],
            validate_task_state: Callable[[ToolDecision], None],
            validate_guardrails: Callable[[ToolDecision], None],
    ) -> ToolDecision | None:
        if state_machine_repair is None:
            return None
        if (
                state_machine_repair.decision == DecisionType.TOOL_CALL
                and request_signature(state_machine_repair) in request_signatures
        ):
            return None
        try:
            validate_decision(state_machine_repair)
            validate_task_state(state_machine_repair)
            validate_guardrails(state_machine_repair)
        except Exception:  # noqa: BLE001
            return None
        return state_machine_repair

    @classmethod
    def _planner_response_should_use_state_machine_repair(
        cls,
        planner_decision: ToolDecision,
        state_machine_repair: ToolDecision | None,
        completed_outputs: list[OutputKind],
        *,
        has_prior_progress: bool,
    ) -> bool:
        if state_machine_repair is None:
            return False
        if cls._planner_tool_call_is_less_grounded_than_state_machine_repair(
            planner_decision,
            state_machine_repair,
        ):
            return True
        if planner_decision.decision != DecisionType.RESPOND or planner_decision.selected_tool:
            return False
        if not has_prior_progress:
            return cls._planner_response_is_placeholder(planner_decision)
        required_outputs = list(state_machine_repair.expected_step_outputs)
        if state_machine_repair.overall_task_goal is not None:
            for output in state_machine_repair.overall_task_goal.required_outputs:
                if output not in required_outputs:
                    required_outputs.append(output)
        if not required_outputs:
            return True
        return any(output not in completed_outputs for output in required_outputs)

    @staticmethod
    def _planner_response_is_placeholder(planner_decision: ToolDecision) -> bool:
        response_hint = str(planner_decision.response_hint or "").strip()
        reason = str(planner_decision.reason or "").strip().lower()
        intent = str(planner_decision.intent or "").strip().lower()
        if response_hint:
            return False
        if planner_decision.overall_task_goal is not None:
            return False
        if planner_decision.expected_step_outputs:
            return False
        return intent in {"", "noop"} or any(
            marker in reason
            for marker in (
                "noop",
                "bypass",
                "placeholder",
                "planner should",
            )
        )

    @staticmethod
    def _planner_decision_within_allowed_actions(
        decision: ToolDecision,
        *,
        allowed_actions: set[str],
        overall_task_goal: TaskGoal | None,
        completed_outputs: list[OutputKind],
    ) -> bool:
        if not allowed_actions:
            return True

        normalized_allowed = {str(item).strip().lower() for item in allowed_actions if str(item).strip()}
        decision_type = decision.decision

        if decision_type == DecisionType.TOOL_CALL:
            return "tool_call" in normalized_allowed
        if decision_type == DecisionType.CLARIFY:
            return "clarify" in normalized_allowed
        if decision_type == DecisionType.RESPOND:
            return "respond" in normalized_allowed
        if decision_type == DecisionType.FINISH:
            if "finish" in normalized_allowed:
                return True
            if "respond" not in normalized_allowed:
                return False
            if overall_task_goal is None:
                return True
            required_outputs = set(overall_task_goal.required_outputs)
            completed = set(completed_outputs)
            return required_outputs.issubset(completed)
        return False

    @staticmethod
    def _format_state_machine_candidate_observation(decision: ToolDecision) -> str:
        expected_outputs = [output.value for output in decision.expected_step_outputs]
        required_outputs = []
        if decision.overall_task_goal is not None:
            required_outputs = [output.value for output in decision.overall_task_goal.required_outputs]
        return (
            "State machine candidate next step: "
            f"decision={decision.decision.value!r} "
            f"selected_tool={decision.selected_tool!r} "
            f"intent={decision.intent!r} "
            f"reason={decision.reason!r} "
            f"expected_step_outputs={expected_outputs} "
            f"required_outputs={required_outputs}"
        )

    @staticmethod
    def _apply_execution_review(
        completion,
        review: ExecutionReview,
    ):
        if review.approved:
            return completion
        missing_outputs = list(completion.missing_outputs)
        for output_kind in review.missing_outputs:
            if output_kind not in missing_outputs:
                missing_outputs.append(output_kind)
        return completion.model_copy(
            update={
                "done": False,
                "reason": review.summary or completion.reason,
                "missing_outputs": missing_outputs,
                "should_render_response": False,
            }
        )

    @staticmethod
    def _latest_search_candidate_paths(tool_results: list[ToolCallResult]) -> list[str]:
        for result in reversed(tool_results):
            if result.status == "success" and result.tool_name in {"retrieval.search_local_objects", "file.search_by_name"}:
                return [
                    candidate.get("path", "")
                    for candidate in result.data.get("candidates", [])
                    if candidate.get("path")
                ]
        return []

    @staticmethod
    def _latest_web_search_result_url(tool_results: list[ToolCallResult]) -> str | None:
        for result in reversed(tool_results):
            if result.status != "success" or result.tool_name not in {"web.search", "web.research"}:
                continue
            items = result.data.get("results", [])
            for item in items:
                if not isinstance(item, dict):
                    continue
                url = item.get("url")
                if isinstance(url, str) and url.strip():
                    return url.strip()
        return None

    @classmethod
    def _normalize_network_lookup_decision(
        cls,
        decision: ToolDecision,
        *,
        user_text: str,
    ) -> ToolDecision:
        if decision.decision != DecisionType.TOOL_CALL or decision.selected_tool != "web.search":
            return decision

        normalized = decision.model_copy(deep=True)
        normalized.selected_tool = "web.research"
        if normalized.intent == "search_web_topic":
            normalized.intent = "research_web_topic"
        if normalized.reason:
            normalized.reason = f"{normalized.reason} Route web lookups through bounded web research."

        arguments = dict(normalized.arguments or {})
        query = str(arguments.get("query") or "").strip()
        weather_query = WebRetrievalStrategy._is_weather_or_forecast_query(query) or WebRetrievalStrategy._is_weather_or_forecast_query(user_text)
        arguments.setdefault("max_results", 5)
        arguments.setdefault("max_pages", 1 if weather_query else 2)
        arguments.setdefault("prefer_browser", not weather_query)
        normalized.arguments = arguments

        expected_outputs = list(normalized.expected_step_outputs)
        for output_kind in (OutputKind.SEARCH_RESULTS, OutputKind.WEB_CONTENT):
            if output_kind not in expected_outputs:
                expected_outputs.append(output_kind)
        normalized.expected_step_outputs = expected_outputs

        if normalized.overall_task_goal is not None:
            required_outputs = list(normalized.overall_task_goal.required_outputs)
            for output_kind in (OutputKind.SEARCH_RESULTS, OutputKind.WEB_CONTENT):
                if output_kind not in required_outputs:
                    required_outputs.append(output_kind)
            normalized.overall_task_goal = normalized.overall_task_goal.model_copy(
                update={"required_outputs": required_outputs}
            )
        return normalized

    @staticmethod
    def _primary_task_graph_subtask(task_graph):
        primary_task_id = str(getattr(task_graph, "primary_task_id", "") or "").strip()
        subtasks = list(getattr(task_graph, "subtasks", []) or [])
        for subtask in subtasks:
            subtask_id = str(getattr(subtask, "task_id", "") or "").strip()
            if primary_task_id and subtask_id == primary_task_id:
                return subtask
        return subtasks[0] if subtasks else None

    @staticmethod
    def _tool_family_for_selected_tool(selected_tool: str) -> str | None:
        tool = str(selected_tool or "").strip()
        if not tool:
            return None
        if tool in {"web.search", "web.research", "web.fetch"}:
            return "web_lookup"
        if tool == "web.open_page":
            return "web_target"
        if tool.startswith("qq.send_"):
            return "delivery"
        if tool.startswith("qq."):
            return "qq_history"
        if tool.startswith("system.") or tool.startswith("time.") or tool.startswith("calendar."):
            return "system_utility"
        if tool in {"retrieval.search_local_objects", "file.search_by_name", "file.list", "file.read", "file.extract_text", "file.extract_structure"}:
            return "local_lookup"
        if tool in {"file.metadata", "file.preview", "file.open_path", "file.reveal_in_explorer"}:
            return "file_lookup"
        if tool in {"document_agent.summarize", "document_agent.read", "document_agent.inspect"}:
            return "document_summary"
        if tool in {"document_agent.edit", "file.edit_docx", "file.write", "file.write_many", "file.append", "file.append_many"}:
            return "document_operation"
        return None

    def _enforce_upstream_constraints_on_decision(
        self,
        decision: ToolDecision,
        *,
        task_envelope,
        task_graph,
        overall_task_goal: TaskGoal | None,
        completed_outputs: list[OutputKind],
        state_machine_repair: ToolDecision | None,
    ) -> tuple[ToolDecision, dict[str, Any] | None]:
        primary_subtask = self._primary_task_graph_subtask(task_graph)
        completed_output_set = set(completed_outputs or [])
        goal_missing_outputs = [
            output_kind
            for output_kind in ((overall_task_goal.required_outputs if overall_task_goal is not None else []) or [])
            if output_kind not in completed_output_set
        ]
        if primary_subtask is not None:
            status = str(getattr(primary_subtask, "status", "") or "").strip().lower()
            missing_slots = [
                str(slot).strip()
                for slot in getattr(primary_subtask, "missing_slots", []) or []
                if str(slot).strip()
            ]
            if status == "waiting_for_input" and decision.decision == DecisionType.TOOL_CALL:
                pending_task = self._build_task_graph_pending_task(
                    user_text=str(getattr(task_envelope, "primary_objective", "") or ""),
                    task_graph=task_graph,
                    overall_task_goal=overall_task_goal,
                )
                if pending_task is not None:
                    followup_text = str(getattr(pending_task, "clarification_prompt", "") or "").strip()
                    clarify = ToolDecision(
                        decision=DecisionType.CLARIFY,
                        intent="clarify_missing_task_slots",
                        reason=f"Primary subtask is still waiting for input: {', '.join(missing_slots) or 'missing context'}.",
                        response_hint=followup_text or "I still need one key detail before I continue.",
                        risk_level=RiskLevel.LOW,
                        overall_task_goal=overall_task_goal,
                    )
                    return clarify, {
                        "reason": "task_graph_waiting_for_input",
                        "missing_slots": missing_slots,
                    }

        primary_kind = str(getattr(primary_subtask, "kind", "") or "").strip().lower() if primary_subtask is not None else ""
        envelope_mode = str(getattr(task_envelope, "mode", "") or "").strip().lower()
        required_outputs = set((overall_task_goal.required_outputs if overall_task_goal is not None else []) or [])
        requires_grounding_outputs = bool(
            {OutputKind.SEARCH_RESULTS, OutputKind.WEB_CONTENT} & required_outputs
        ) or envelope_mode == "grounded_lookup" or primary_kind in {"web_lookup", "web_target"}
        has_grounding_outputs = bool(
            {OutputKind.SEARCH_RESULTS, OutputKind.WEB_CONTENT} & completed_output_set
        )
        if decision.decision == DecisionType.RESPOND and not decision.selected_tool and goal_missing_outputs:
            allowed_families = {
                str(item).strip()
                for item in getattr(task_envelope, "allowed_families", []) or []
                if str(item).strip()
            }
            blocked_families = {
                str(item).strip()
                for item in getattr(task_envelope, "blocked_families", []) or []
                if str(item).strip()
            }
            fallback_family = self._tool_family_for_selected_tool(getattr(state_machine_repair, "selected_tool", "") or "")
            if (
                state_machine_repair is not None
                and state_machine_repair.decision == DecisionType.TOOL_CALL
                and fallback_family is not None
                and (not allowed_families or fallback_family in allowed_families)
                and fallback_family not in blocked_families
            ):
                return state_machine_repair, {
                    "reason": "required_outputs_missing_before_respond",
                    "fallback_tool": state_machine_repair.selected_tool,
                    "fallback_family": fallback_family,
                    "missing_outputs": [item.value for item in goal_missing_outputs],
                }
            clarify = ToolDecision(
                decision=DecisionType.CLARIFY,
                intent="clarify_before_required_outputs",
                reason="The task still has required outputs pending before a direct response is safe.",
                response_hint="我先把这一步该拿到的信息补齐，再给你完整答复。",
                risk_level=RiskLevel.LOW,
                overall_task_goal=overall_task_goal,
            )
            return clarify, {
                "reason": "required_outputs_missing_before_respond",
                "missing_outputs": [item.value for item in goal_missing_outputs],
            }
        if (
            decision.decision == DecisionType.RESPOND
            and not decision.selected_tool
            and requires_grounding_outputs
            and not has_grounding_outputs
        ):
            allowed_families = {
                str(item).strip()
                for item in getattr(task_envelope, "allowed_families", []) or []
                if str(item).strip()
            }
            blocked_families = {
                str(item).strip()
                for item in getattr(task_envelope, "blocked_families", []) or []
                if str(item).strip()
            }
            fallback_family = self._tool_family_for_selected_tool(getattr(state_machine_repair, "selected_tool", "") or "")
            if (
                state_machine_repair is not None
                and state_machine_repair.decision == DecisionType.TOOL_CALL
                and fallback_family is not None
                and (not allowed_families or fallback_family in allowed_families)
                and fallback_family not in blocked_families
            ):
                return state_machine_repair, {
                    "reason": "grounding_required_before_respond",
                    "fallback_tool": state_machine_repair.selected_tool,
                    "fallback_family": fallback_family,
                    "completed_outputs": [item.value for item in completed_output_set],
                }
            clarify = ToolDecision(
                decision=DecisionType.CLARIFY,
                intent="clarify_before_grounded_response",
                reason="This verification task still needs grounded search outputs before responding.",
                response_hint="我先去核实一下这件事，再给你一个靠谱的答复。",
                risk_level=RiskLevel.LOW,
                overall_task_goal=overall_task_goal,
            )
            return clarify, {
                "reason": "grounding_required_before_respond",
                "completed_outputs": [item.value for item in completed_output_set],
            }

        if decision.decision != DecisionType.TOOL_CALL:
            return decision, None

        selected_tool = str(decision.selected_tool or "").strip()
        family = self._tool_family_for_selected_tool(selected_tool)
        allowed_families = {
            str(item).strip()
            for item in getattr(task_envelope, "allowed_families", []) or []
            if str(item).strip()
        }
        blocked_families = {
            str(item).strip()
            for item in getattr(task_envelope, "blocked_families", []) or []
            if str(item).strip()
        }
        if family is None:
            return decision, None

        violates_allowed = bool(allowed_families) and family not in allowed_families
        violates_blocked = family in blocked_families
        if not violates_allowed and not violates_blocked:
            return decision, None

        fallback_family = self._tool_family_for_selected_tool(getattr(state_machine_repair, "selected_tool", "") or "")
        if (
            state_machine_repair is not None
            and state_machine_repair.decision == DecisionType.TOOL_CALL
            and fallback_family is not None
            and (not allowed_families or fallback_family in allowed_families)
            and fallback_family not in blocked_families
        ):
            return state_machine_repair, {
                "reason": "task_envelope_family_override",
                "selected_tool": selected_tool,
                "selected_family": family,
                "fallback_tool": state_machine_repair.selected_tool,
                "fallback_family": fallback_family,
            }

        clarify = ToolDecision(
            decision=DecisionType.CLARIFY,
            intent="clarify_after_invalid_family",
            reason=f"Selected tool family {family} conflicts with upstream orchestration constraints.",
            response_hint="我先按前面的任务目标收住一下。这一步被带偏了，我需要重新按正确方向继续处理。",
            risk_level=RiskLevel.LOW,
            overall_task_goal=overall_task_goal,
        )
        return clarify, {
            "reason": "task_envelope_family_conflict",
            "selected_tool": selected_tool,
            "selected_family": family,
            "allowed_families": sorted(allowed_families),
            "blocked_families": sorted(blocked_families),
        }

    @classmethod
    def _should_auto_approve_write_followup(
        cls,
        decision: ToolDecision,
        tool_results: list[ToolCallResult],
    ) -> bool:
        if decision.decision != DecisionType.TOOL_CALL or decision.selected_tool != "file.write":
            return False

        content = decision.arguments.get("content")
        path = decision.arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            return False
        if not isinstance(content, str) or not content.strip():
            return False
        if content.strip() in {"{paths}", "{candidates}", "{results}", "{candidate_paths}"}:
            return False

        candidate_paths = cls._latest_search_candidate_paths(tool_results)
        if not candidate_paths:
            return False

        content_lines = [line.strip().strip('"') for line in content.splitlines() if line.strip()]
        if not content_lines:
            return False

        normalized_candidates = {candidate.strip().strip('"') for candidate in candidate_paths}
        return all(line in normalized_candidates for line in content_lines)

    @classmethod
    def _should_auto_approve_candidate_followup(
        cls,
        decision: ToolDecision,
        tool_results: list[ToolCallResult],
    ) -> bool:
        if decision.decision != DecisionType.TOOL_CALL:
            return False

        candidate_paths = cls._latest_search_candidate_paths(tool_results)
        if not candidate_paths:
            return False

        normalized_candidates = {candidate.strip().strip('"') for candidate in candidate_paths}
        selected_tool = decision.selected_tool or ""
        if selected_tool == "retrieval.inspect_local_candidate":
            raw_path = decision.arguments.get("path")
            return isinstance(raw_path, str) and raw_path.strip().strip('"') in normalized_candidates

        if selected_tool == "file.read":
            paths = decision.arguments.get("paths", [])
            if not isinstance(paths, list) or not paths:
                return False
            normalized_paths = {value.strip().strip('"') for value in paths if isinstance(value, str) and value.strip()}
            return bool(normalized_paths) and normalized_paths.issubset(normalized_candidates)

        if selected_tool == "file.extract_text":
            paths = decision.arguments.get("paths", [])
            if not isinstance(paths, list) or not paths:
                return False
            normalized_paths = {value.strip().strip('"') for value in paths if isinstance(value, str) and value.strip()}
            return bool(normalized_paths) and normalized_paths.issubset(normalized_candidates)

        if selected_tool == "document_agent.summarize":
            raw_path = decision.arguments.get("source_path")
            return isinstance(raw_path, str) and raw_path.strip().strip('"') in normalized_candidates

        if selected_tool == "document_agent.read":
            raw_path = decision.arguments.get("source_path")
            return isinstance(raw_path, str) and raw_path.strip().strip('"') in normalized_candidates

        if selected_tool == "document_agent.inspect":
            raw_path = decision.arguments.get("source_path")
            return isinstance(raw_path, str) and raw_path.strip().strip('"') in normalized_candidates

        if selected_tool == "document_agent.edit":
            raw_path = decision.arguments.get("source_path")
            return isinstance(raw_path, str) and raw_path.strip().strip('"') in normalized_candidates

        if selected_tool in {"file.metadata", "file.preview", "file.open_path", "file.reveal_in_explorer"}:
            raw_path = decision.arguments.get("path")
            return isinstance(raw_path, str) and raw_path.strip().strip('"') in normalized_candidates

        return False

    @classmethod
    def _should_short_circuit_state_machine_repair(
        cls,
        state_machine_repair: ToolDecision | None,
        *,
        has_prior_progress: bool,
        request_signatures: set[str],
        request_signature: Callable[[ToolDecision], str],
    ) -> bool:
        if state_machine_repair is None:
            return False
        if state_machine_repair.decision != DecisionType.TOOL_CALL:
            return False
        if not state_machine_repair.selected_tool:
            return False
        if state_machine_repair.selected_tool not in cls._TRUSTED_STATE_MACHINE_TOOLS:
            return False
        if not has_prior_progress and not cls._has_complete_initial_state_machine_arguments(state_machine_repair):
            return False
        if state_machine_repair.memory_write:
            return False
        if state_machine_repair.risk_level != RiskLevel.LOW:
            return False
        if request_signature(state_machine_repair) in request_signatures:
            return False
        return True

    @staticmethod
    def _has_complete_initial_state_machine_arguments(decision: ToolDecision) -> bool:
        selected_tool = decision.selected_tool or ""
        arguments = decision.arguments if isinstance(decision.arguments, dict) else {}
        if selected_tool in {"web.search", "web.research"}:
            return bool(str(arguments.get("query") or "").strip())
        if selected_tool == "web.fetch":
            return bool(str(arguments.get("url") or "").strip())
        if selected_tool.startswith("qq."):
            return bool(arguments) or selected_tool in {"qq.get_current_context", "qq.get_last_reply"}
        return False

    @classmethod
    def _should_use_locked_workflow_decision(
        cls,
        workflow_decision: ToolDecision | None,
        *,
        workflow_family: str,
        request_signatures: set[str],
        request_signature: Callable[[ToolDecision], str],
    ) -> bool:
        if workflow_decision is None:
            return False
        if workflow_family not in {"document_operation", "document_summary", "file_lookup", "local_lookup", "llm_workflow_spec"}:
            return False
        if workflow_decision.decision != DecisionType.TOOL_CALL:
            return False
        if not workflow_decision.selected_tool:
            return False
        if workflow_decision.selected_tool not in cls._TRUSTED_STATE_MACHINE_TOOLS:
            return False
        if workflow_decision.memory_write:
            return False
        if workflow_decision.risk_level != RiskLevel.LOW:
            return False
        if request_signature(workflow_decision) in request_signatures:
            return False
        return True

    @staticmethod
    def _should_use_state_machine_repair_after_planner_exception(
            *,
            has_prior_progress: bool,
            knowledge_type: str,
    ) -> bool:
        if has_prior_progress:
            return True
        return knowledge_type in {"local_workspace", "qq_history", "system_utility"}

    @staticmethod
    def _initial_workflow_spec_skip_reason(intent_bundle) -> str | None:
        task_classification = getattr(intent_bundle, "task_classification", None)
        if task_classification is None:
            return None
        run_mode = str(getattr(task_classification, "run_mode", "") or "").strip().lower()
        if run_mode == "scheduled":
            return "scheduled_task"

        try:
            confidence = float(getattr(task_classification, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0.70:
            return "low_confidence_defer_to_decision_loop"

        document_delivery = getattr(intent_bundle, "document_delivery", None)
        if document_delivery is not None and (
            bool(getattr(document_delivery, "wants_document", False))
            or bool(getattr(document_delivery, "save_output", False))
        ):
            return None

        preferred_families = [
            str(item).strip()
            for item in getattr(task_classification, "preferred_families", []) or []
            if str(item).strip()
        ]
        if confidence >= 0.88 and len(preferred_families) == 1:
            return "llm_classified_single_family_high_confidence"
        return None

    def _plan_initial_workflow_spec(
        self,
        *,
        user_text: str,
        intent_bundle,
        recent_context: str,
        trace_id: str,
    ) -> WorkflowSpec | None:
        if not hasattr(self.llm_client, "plan_workflow_spec"):
            return None
        task_classification = getattr(intent_bundle, "task_classification", None)
        if task_classification is None:
            return None
        if str(getattr(task_classification, "run_mode", "") or "") == "scheduled":
            return None
        skip_reason = self._initial_workflow_spec_skip_reason(intent_bundle)
        if skip_reason is not None:
            self.trace_store.append(
                "workflow_spec_skipped",
                {
                    "trace_id": trace_id,
                    "reason": skip_reason,
                    "policy": "structured_intent_state_machine_reference",
                },
            )
            return None
        try:
            spec = self.llm_client.plan_workflow_spec(
                messages=self._context_messages(),
                tool_manifests=self.registry.list_manifests(),
                intent_bundle=intent_bundle.model_dump(mode="json"),
                recent_context=recent_context,
                max_nodes=6,
            )
        except Exception as exc:  # noqa: BLE001
            self.trace_store.append("workflow_spec_error", {"trace_id": trace_id, "error": str(exc)})
            return None
        review = self._review_workflow_spec(spec)
        self.trace_store.append(
            "workflow_spec_review",
            {
                "trace_id": trace_id,
                "workflow_spec": spec.model_dump(mode="json"),
                "review": review,
            },
        )
        if not review.get("approved"):
            return None
        return spec

    def _review_workflow_spec(self, spec: WorkflowSpec) -> dict[str, object]:
        issues: list[str] = []
        if not spec.nodes:
            issues.append("workflow_spec_empty")
        if len(spec.nodes) > 8:
            issues.append("workflow_spec_too_long")

        available_tools = {manifest.tool_name for manifest in self.registry.list_manifests()}
        produced: set[OutputKind] = set()
        seen_ids: set[str] = set()
        side_effect_nodes = 0
        for index, node in enumerate(spec.nodes):
            node_id = str(node.node_id or "").strip()
            if not node_id:
                issues.append(f"node_{index}_missing_id")
            elif node_id in seen_ids:
                issues.append(f"node_{index}_duplicate_id")
            seen_ids.add(node_id)

            if node.tool is None:
                if index != len(spec.nodes) - 1:
                    issues.append(f"{node_id or index}_response_node_not_last")
                continue
            if node.tool not in available_tools:
                issues.append(f"{node_id or index}_unknown_tool:{node.tool}")
                continue

            missing_requires = [item.value for item in node.requires if item not in produced]
            if missing_requires:
                issues.append(f"{node_id or index}_requires_missing:{','.join(missing_requires)}")

            manifest = self.registry.get_manifest(node.tool)
            if manifest.side_effect:
                side_effect_nodes += 1
                if side_effect_nodes > 2:
                    issues.append("workflow_spec_too_many_side_effects")
                if node.tool.startswith("document_agent.") and OutputKind.OBJECT_CANDIDATES not in produced:
                    issues.append(f"{node_id or index}_document_agent_without_target_candidates")

            for output in node.produces or manifest.produces:
                produced.add(output)

        if spec.goal is not None:
            missing_goal_outputs = [item.value for item in spec.goal.required_outputs if item not in produced]
            if missing_goal_outputs:
                issues.append(f"goal_outputs_not_produced:{','.join(missing_goal_outputs)}")

        return {
            "approved": not issues,
            "issues": issues,
            "summary": "Workflow spec passed deterministic graph review." if not issues else "Workflow spec was rejected by deterministic graph review.",
        }

    def _workflow_spec_node_to_decision(
        self,
        *,
        spec: WorkflowSpec,
        node: WorkflowNodeSpec,
    ) -> ToolDecision | None:
        if node.tool is None:
            return None
        manifest = self.registry.get_manifest(node.tool)
        produces = list(node.produces or manifest.produces)
        return ToolDecision(
            decision=DecisionType.TOOL_CALL,
            intent=node.intent or node.node_id,
            reason=node.reason or f"Run locked workflow node {node.node_id}.",
            selected_tool=node.tool,
            arguments={},
            risk_level=RiskLevel.LOW,
            overall_task_goal=spec.goal,
            expected_step_outputs=produces,
        )

    @staticmethod
    def _looks_like_document_summary_request(user_text: str) -> bool:
        lowered = user_text.lower()
        summary_terms = (
            "summarize",
            "summary",
            "content of",
            "explain",
            "提炼",
            "主要说",
            "主要讲",
            "说了什么",
            "讲了什么",
            "写了什么",
            "总结",
            "概括",
            "内容",
            "讲了什么",
            "写了什么",
            "栏目",
            "结构",
            "重点",
            "核心观点",
            "有哪些",
            "哪些部分",
        )
        doc_terms = (
            "document",
            "doc",
            "log",
            "markdown",
            "word",
            "ppt",
            "presentation",
            "excel",
            "spreadsheet",
            "文档",
            "日志",
            "开发日志",
            "汇报",
            "报告",
            "模板",
            "表格",
            ".docx",
            ".pptx",
            ".xlsx",
            ".md",
            ".txt",
        )
        negative_terms = ("write", "save", "export", "append", "create", "列出", "写入", "保存", "导出", "创建")
        if any(term in lowered for term in negative_terms):
            return False
        return any(term in lowered for term in summary_terms) and any(term in lowered for term in doc_terms)

    @staticmethod
    def _looks_like_document_structure_request(user_text: str) -> bool:
        lowered = user_text.lower()
        structure_terms = (
            "structure",
            "outline",
            "sections",
            "headings",
            "blocks",
            "栏目",
            "结构",
            "大纲",
            "标题",
            "章节",
            "版式",
            "格式",
            "段落",
        )
        doc_terms = ("word", "docx", "document", "template", "文档", "文件", "word里", "word中", "模板")
        return any(term in lowered for term in structure_terms) and any(term in lowered for term in doc_terms)

    @staticmethod
    def _looks_like_document_block_search_request(user_text: str) -> bool:
        lowered = user_text.lower()
        block_terms = (
            "about",
            "related to",
            "mentions",
            "which part",
            "which section",
            "search",
            "find the part",
            "关于",
            "有关",
            "提到",
            "哪部分",
            "哪一段",
            "哪几段",
            "搜索",
            "查找",
            "定位",
            "相关内容",
        )
        doc_terms = ("word", "docx", "document", "文档", "文件", "word里", "word中")
        return any(term in lowered for term in block_terms) and any(term in lowered for term in doc_terms)

    @staticmethod
    def _looks_like_original_file_request(user_text: str) -> bool:
        lowered = user_text.lower()
        return any(
            term in lowered
            for term in (
                "原文件",
                "原始文件",
                "原文",
                "source file",
                "original file",
                "original document",
            )
        )

    @classmethod
    def _extract_document_block_query(cls, user_text: str) -> str:
        cleaned = user_text.strip()
        for phrase in (
            "帮我",
            "请帮我",
            "请",
            "把",
            "这个",
            "那个",
            "这份",
            "那份",
            "word里",
            "word中",
            "文档里",
            "文档中",
            "文件里",
            "文件中",
            "关于",
            "有关",
            "提到",
            "搜索",
            "查找",
            "定位",
            "哪部分",
            "哪一段",
            "哪几段",
            "相关内容",
            "内容",
        ):
            cleaned = re.sub("[,.!?:;()\\[\\]{}\"'`?????????????]+", " ", cleaned)
        cleaned = cls._finalize_local_lookup_query(cleaned)
        return cleaned or user_text.strip()


    @staticmethod
    def _latest_docx_structure_result(tool_results: list[ToolCallResult], path: str) -> dict | None:
        normalized_path = str(path).strip().lower()
        for result in reversed(tool_results):
            if result.status != "success" or result.tool_name not in {"file.extract_structure", "file.search_blocks"}:
                continue
            result_path = str(result.data.get("path", "")).strip().lower()
            if result_path == normalized_path:
                return result.data
        return None

    @staticmethod
    def _pick_docx_append_anchor_block(structure_result: dict) -> str | None:
        blocks = structure_result.get("blocks", [])
        last_paragraph_block_id: str | None = None
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("block_id", "")).strip()
            block_type = str(block.get("block_type", "")).strip().lower()
            if not block_id:
                continue
            if block_type == "paragraph":
                last_paragraph_block_id = block_id
        if last_paragraph_block_id:
            return last_paragraph_block_id

        matches = structure_result.get("matches", [])
        for block in reversed(matches):
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("block_id", "")).strip()
            if block_id:
                return block_id
        return None


    @staticmethod
    def _looks_like_document_path(path: str) -> bool:
        return Path(path).suffix.lower() in {".docx", ".pptx", ".xlsx", ".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".log", ".pdf"}

    @staticmethod
    def _looks_like_document_agent_editable_path(path: str) -> bool:
        return Path(path).suffix.lower() in {".docx", ".md", ".txt"}

    @staticmethod
    def _looks_like_image_path(path: str) -> bool:
        return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

    @classmethod
    def _document_task_expected_outputs(
        cls,
        *,
        overall_task_goal: TaskGoal | None,
        fallback: list[OutputKind],
    ) -> list[OutputKind]:
        if overall_task_goal is None or not overall_task_goal.required_outputs:
            return list(fallback)
        supported = {OutputKind.OBJECT_DETAILS, OutputKind.FILE_CONTENTS, OutputKind.FILE_WRITTEN}
        selected = [output for output in overall_task_goal.required_outputs if output in supported]
        return selected or list(fallback)

    @classmethod
    def _build_document_agent_grounded_inputs(
        cls,
        *,
        user_text: str,
        path: str,
        overall_task_goal: TaskGoal | None,
        recent_context: str = "",
    ) -> dict[str, object]:
        now_local = datetime.now(timezone(timedelta(hours=8), name="Asia/Shanghai"))
        grounded_inputs: dict[str, object] = {
            "target_path": path,
            "target_name": Path(path).name,
            "user_request": user_text.strip(),
            "current_date": now_local.date().isoformat(),
            "current_date_mmdd": now_local.strftime("%m%d"),
            "current_time_iso": now_local.isoformat(),
            "timezone": "Asia/Shanghai",
        }
        if overall_task_goal is not None:
            if overall_task_goal.summary:
                grounded_inputs["task_summary"] = overall_task_goal.summary
            if overall_task_goal.required_outputs:
                grounded_inputs["required_outputs"] = [output.value for output in overall_task_goal.required_outputs]
        compact_recent_context = str(recent_context or "").strip()
        if compact_recent_context:
            grounded_inputs["recent_context_excerpt"] = compact_recent_context[:600]
        return grounded_inputs

    @classmethod
    def _build_document_agent_arguments(
        cls,
        *,
        user_text: str,
        path: str,
        overall_task_goal: TaskGoal | None,
        recent_context: str = "",
        output_path: str | None = None,
        max_chars: int = 12000,
        max_blocks: int | None = None,
        max_chars_per_block: int | None = None,
        max_matches: int | None = None,
        allow_overwrite: bool | None = None,
        preserve_structure: bool | None = None,
        preserve_style: bool | None = None,
    ) -> dict[str, object]:
        grounded_inputs = cls._build_document_agent_grounded_inputs(
            user_text=user_text,
            path=path,
            overall_task_goal=overall_task_goal,
            recent_context=recent_context,
        )
        resolved_facts = {
            "target_path": grounded_inputs["target_path"],
            "target_name": grounded_inputs["target_name"],
            "current_date": grounded_inputs["current_date"],
            "current_date_mmdd": grounded_inputs["current_date_mmdd"],
            "current_time_iso": grounded_inputs["current_time_iso"],
            "timezone": grounded_inputs["timezone"],
        }
        constraints = {
            "allow_overwrite": True if allow_overwrite is None else allow_overwrite,
            "preserve_structure": True if preserve_structure is None else preserve_structure,
            "preserve_style": True if preserve_style is None else preserve_style,
            "document_agent_decides_edit_operations": True,
            "main_agent_does_not_select_blocks": True,
            "must_not_change_unrelated_content": True,
        }
        arguments: dict[str, object] = {
            "source_path": path,
            "instruction": user_text,
            "recent_context": recent_context,
            "grounded_inputs": grounded_inputs,
            "resolved_facts": resolved_facts,
            "source_materials": {},
            "constraints": constraints,
            "style_hints": {},
            "max_chars": max_chars,
        }
        if output_path is not None:
            arguments["output_path"] = output_path
        if max_blocks is not None:
            arguments["max_blocks"] = max_blocks
        if max_chars_per_block is not None:
            arguments["max_chars_per_block"] = max_chars_per_block
        if max_matches is not None:
            arguments["max_matches"] = max_matches
        if allow_overwrite is not None:
            arguments["allow_overwrite"] = allow_overwrite
        if preserve_structure is not None:
            arguments["preserve_structure"] = preserve_structure
        if preserve_style is not None:
            arguments["preserve_style"] = preserve_style
        return arguments

    @classmethod
    def _build_document_agent_summary_followup(
        cls,
        *,
        user_text: str,
        path: str,
        overall_task_goal: TaskGoal | None,
        reason: str,
        recent_context: str = "",
    ) -> ToolDecision:
        return ToolDecision(
            decision=DecisionType.TOOL_CALL,
            intent="document_agent_summary",
            reason=reason,
            selected_tool="document_agent.summarize",
            arguments=cls._build_document_agent_arguments(
                user_text=user_text,
                path=path,
                overall_task_goal=overall_task_goal,
                recent_context=recent_context,
                max_chars=12000,
            ),
            risk_level=RiskLevel.LOW,
            overall_task_goal=overall_task_goal,
            expected_step_outputs=cls._document_task_expected_outputs(
                overall_task_goal=overall_task_goal,
                fallback=[OutputKind.FILE_CONTENTS],
            ),
        )

    @classmethod
    def _build_document_agent_read_followup(
        cls,
        *,
        user_text: str,
        path: str,
        overall_task_goal: TaskGoal | None,
        reason: str,
        recent_context: str = "",
    ) -> ToolDecision:
        return ToolDecision(
            decision=DecisionType.TOOL_CALL,
            intent="document_agent_read",
            reason=reason,
            selected_tool="document_agent.read",
            arguments=cls._build_document_agent_arguments(
                user_text=user_text,
                path=path,
                overall_task_goal=overall_task_goal,
                recent_context=recent_context,
                max_chars=12000,
            ),
            risk_level=RiskLevel.LOW,
            overall_task_goal=overall_task_goal,
            expected_step_outputs=[OutputKind.FILE_CONTENTS],
        )

    @classmethod
    def _build_document_agent_inspect_followup(
        cls,
        *,
        user_text: str,
        path: str,
        overall_task_goal: TaskGoal | None,
        reason: str,
        recent_context: str = "",
    ) -> ToolDecision:
        return ToolDecision(
            decision=DecisionType.TOOL_CALL,
            intent="document_agent_inspect",
            reason=reason,
            selected_tool="document_agent.inspect",
            arguments=cls._build_document_agent_arguments(
                user_text=user_text,
                path=path,
                overall_task_goal=overall_task_goal,
                recent_context=recent_context,
                max_chars=12000,
                max_blocks=200,
                max_chars_per_block=1200,
                max_matches=8,
            ),
            risk_level=RiskLevel.LOW,
            overall_task_goal=overall_task_goal,
            expected_step_outputs=[OutputKind.OBJECT_DETAILS],
        )

    @classmethod
    def _build_document_agent_edit_followup(
        cls,
        *,
        user_text: str,
        path: str,
        overall_task_goal: TaskGoal | None,
        reason: str,
        recent_context: str = "",
    ) -> ToolDecision | None:
        if not cls._looks_like_document_agent_editable_path(path):
            return None
        return ToolDecision(
            decision=DecisionType.TOOL_CALL,
            intent="document_agent_edit",
            reason=reason,
            selected_tool="document_agent.edit",
            arguments=cls._build_document_agent_arguments(
                user_text=user_text,
                path=path,
                overall_task_goal=overall_task_goal,
                recent_context=recent_context,
                output_path=path,
                max_chars=12000,
                max_blocks=200,
                max_chars_per_block=1200,
                allow_overwrite=True,
                preserve_structure=True,
                preserve_style=True,
            ),
            risk_level=RiskLevel.LOW,
            overall_task_goal=overall_task_goal,
            expected_step_outputs=cls._document_task_expected_outputs(
                overall_task_goal=overall_task_goal,
                fallback=[OutputKind.OBJECT_DETAILS, OutputKind.FILE_CONTENTS, OutputKind.FILE_WRITTEN],
            ),
        )

    @staticmethod
    def _looks_like_image_request(user_text: str) -> bool:
        lowered = user_text.lower()
        image_terms = (
            "image",
            "picture",
            "photo",
            "screenshot",
            "png",
            "jpg",
            "jpeg",
            "webp",
            "gif",
            "bmp",
            "图片",
            "照片",
            "截图",
            "图像",
            "这张图",
            "那张图",
        )
        image_actions = (
            "what is in",
            "what's in",
            "describe",
            "inspect",
            "analyze",
            "analyse",
            "ocr",
            "看看",
            "看一下",
            "查看",
            "分析",
            "识别",
            "内容",
            "是什么",
            "图里",
            "图上",
            "图中",
        )
        return any(term in user_text or term in lowered for term in image_terms) and any(
            term in user_text or term in lowered for term in image_actions
        )

    @staticmethod
    def _looks_like_image_text_request(user_text: str) -> bool:
        lowered = user_text.lower()
        if not AgentKernel._looks_like_image_request(user_text):
            return False
        if re.search(r"\b(?:ocr|text|read text|read the text|words)\b", lowered):
            return True
        return any(
            term in user_text or term in lowered
            for term in (
                "文字",
                "文本",
                "读字",
                "识别文字",
                "提取文字",
                "图上的字",
                "图里文字",
                "写了什么",
                "上面写了什么",
            )
        )

    @classmethod
    def _infer_local_followup_outputs(cls, user_text: str) -> list[OutputKind]:
        inferred: list[OutputKind] = []
        if cls._looks_like_image_request(user_text):
            inferred.append(OutputKind.OBJECT_DETAILS)
        if cls._looks_like_image_text_request(user_text):
            inferred.append(OutputKind.FILE_CONTENTS)
        if cls._looks_like_candidate_path_write_request(user_text):
            inferred.append(OutputKind.FILE_WRITTEN)
        if cls._looks_like_metadata_request(user_text) or cls._looks_like_preview_request(user_text):
            if OutputKind.OBJECT_DETAILS not in inferred:
                inferred.append(OutputKind.OBJECT_DETAILS)
        if cls._looks_like_open_request(user_text) or cls._looks_like_reveal_request(user_text):
            inferred.append(OutputKind.PATH_OPENED)
        return inferred

    @staticmethod
    def _classification_implied_outputs(task_classification) -> list[OutputKind]:
        task_kind = ""
        if task_classification is not None:
            task_kind = str(getattr(task_classification, "task_kind", "") or "").strip().lower()
        mapping = {
            "inspect": [OutputKind.OBJECT_DETAILS],
            "summarize": [OutputKind.FILE_CONTENTS],
            "document_summary": [OutputKind.FILE_CONTENTS],
            "delivery": [OutputKind.MESSAGE_SENT],
            "document_edit": [OutputKind.OBJECT_DETAILS, OutputKind.FILE_WRITTEN],
            "edit": [OutputKind.OBJECT_DETAILS, OutputKind.FILE_WRITTEN],
            "rewrite": [OutputKind.OBJECT_DETAILS, OutputKind.FILE_WRITTEN],
            "transform": [OutputKind.OBJECT_DETAILS, OutputKind.FILE_WRITTEN],
            "collection_action": [OutputKind.FILE_WRITTEN],
        }
        return list(mapping.get(task_kind, []))

    @staticmethod
    def _goal_requires_document_write(overall_task_goal: TaskGoal | None) -> bool:
        return bool(
            overall_task_goal is not None
            and OutputKind.FILE_WRITTEN in ((overall_task_goal.required_outputs or []) or [])
        )

    @staticmethod
    def _goal_requires_document_details(overall_task_goal: TaskGoal | None) -> bool:
        return bool(
            overall_task_goal is not None
            and OutputKind.OBJECT_DETAILS in ((overall_task_goal.required_outputs or []) or [])
        )

    @staticmethod
    def _classification_preferred_families(task_classification) -> set[str]:
        if task_classification is None:
            return set()
        preferred = getattr(task_classification, "preferred_families", []) or []
        return {
            str(item).strip()
            for item in preferred
            if str(item).strip()
        }

    @classmethod
    def _compose_local_task_goal(
        cls,
        *,
        user_text: str,
        default_summary: str,
        overall_task_goal: TaskGoal | None = None,
        include_candidates: bool = False,
        supports_message_delivery: bool = False,
        task_classification=None,
    ) -> TaskGoal:
        required_outputs = [] if overall_task_goal is None else list(overall_task_goal.required_outputs)
        if include_candidates and OutputKind.OBJECT_CANDIDATES not in required_outputs:
            required_outputs.append(OutputKind.OBJECT_CANDIDATES)
        for output_kind in cls._classification_implied_outputs(task_classification):
            if output_kind not in required_outputs:
                required_outputs.append(output_kind)
        for output_kind in cls._infer_local_followup_outputs(user_text):
            if output_kind not in required_outputs:
                required_outputs.append(output_kind)
        if supports_message_delivery and cls._looks_like_file_delivery_request_clean(user_text):
            if OutputKind.MESSAGE_SENT not in required_outputs:
                required_outputs.append(OutputKind.MESSAGE_SENT)
        return TaskGoal(
            summary=default_summary if overall_task_goal is None or not overall_task_goal.summary else overall_task_goal.summary,
            required_outputs=required_outputs,
            completion_mode=(
                "outputs"
                if overall_task_goal is None or not getattr(overall_task_goal, "completion_mode", "")
                else overall_task_goal.completion_mode
            ),
        )

    @classmethod
    def _looks_like_candidate_path_write_request(cls, user_text: str) -> bool:
        if not cls._extract_requested_output_file(user_text):
            return False
        lowered = user_text.lower()
        return any(
            term in lowered
            for term in (
                "write",
                "save",
                "export",
                "append",
                "写入",
                "写进",
                "记到",
                "记录到",
                "保存到",
                "导出",
            )
        )

    @staticmethod
    def _looks_like_file_path(path: str) -> bool:
        candidate = Path(path)
        if candidate.suffix:
            return True
        try:
            return candidate.exists() and candidate.is_file()
        except OSError:
            return False

    @staticmethod
    def _looks_like_dir_path(path: str) -> bool:
        candidate = Path(path)
        if candidate.suffix:
            return False
        try:
            return candidate.exists() and candidate.is_dir()
        except OSError:
            return False

    @staticmethod
    def _looks_like_open_request(user_text: str) -> bool:
        lowered = user_text.lower()
        return any(term in lowered for term in ("open", "打开", "打开它", "open it", "open the", "打开这个"))

    @staticmethod
    def _looks_like_file_delivery_request(user_text: str) -> bool:
        lowered = user_text.lower()
        return any(term in lowered for term in ("send me", "send it", "attach it", "upload it", "发给我", "发我", "传给我", "传我"))

    @staticmethod
    def _looks_like_reveal_request(user_text: str) -> bool:
        lowered = user_text.lower()
        return any(
            term in lowered
            for term in ("reveal", "show in explorer", "locate in explorer", "在资源管理器", "在文件夹中显示", "定位到")
        )

    @staticmethod
    def _looks_like_metadata_request(user_text: str) -> bool:
        lowered = user_text.lower()
        return any(
            term in lowered
            for term in ("metadata", "details", "properties", "size", "modified", "mtime", "元数据", "属性", "信息", "大小", "修改时间")
        )

    @staticmethod
    def _looks_like_preview_request(user_text: str) -> bool:
        lowered = user_text.lower()
        return any(
            term in lowered
            for term in ("preview", "peek", "head", "first lines", "前几行", "预览", "先看看", "开头")
        )

    @classmethod
    def _infer_candidate_action(cls, user_text: str) -> str | None:
        if cls._looks_like_file_delivery_request(user_text):
            return None
        if cls._looks_like_image_text_request(user_text):
            return "read_image_text"
        if cls._looks_like_image_request(user_text):
            return "describe_image"
        if cls._looks_like_document_block_search_request(user_text):
            return "search_document_blocks"
        if cls._looks_like_document_structure_request(user_text):
            return "extract_document_structure"
        if cls._looks_like_reveal_request(user_text):
            return "reveal"
        if cls._looks_like_metadata_request(user_text):
            return "metadata"
        if cls._looks_like_preview_request(user_text):
            return "preview"
        if cls._looks_like_open_request(user_text):
            return "open"
        if cls._looks_like_document_summary_request(user_text):
            return "read"
        return None

    @classmethod
    def _select_candidate_path_for_action(
        cls,
        candidate_state: CandidateState,
        action: str,
    ) -> str | None:
        paths = [path for path in candidate_state.candidate_paths if path]
        if not paths:
            return None

        if action in {"describe_image", "inspect_image", "read_image_text"}:
            image_paths = [path for path in paths if cls._looks_like_image_path(path)]
            if image_paths:
                return image_paths[0]
            file_paths = [path for path in paths if cls._looks_like_file_path(path)]
            if file_paths:
                return file_paths[0]
            return paths[0]

        if action in {"read", "metadata", "preview", "extract_document_structure", "search_document_blocks"}:
            primary_path = paths[0]
            if action == "read" and cls._looks_like_document_path(primary_path):
                return primary_path
            document_paths = [path for path in paths if cls._looks_like_document_path(path)]
            if document_paths:
                return document_paths[0]
            file_paths = [path for path in paths if cls._looks_like_file_path(path)]
            if file_paths:
                return file_paths[0]
            return paths[0]

        if action in {"open", "reveal"}:
            if candidate_state.target_kind == "folder":
                dir_paths = [path for path in paths if cls._looks_like_dir_path(path)]
                if dir_paths:
                    return dir_paths[0]
            return paths[0]

        return paths[0]

    @classmethod
    def _build_document_details_followup(
        cls,
        *,
        path: str,
        user_text: str,
        overall_task_goal: TaskGoal | None,
        reason: str,
    ) -> ToolDecision:
        return cls._build_document_agent_inspect_followup(
            user_text=user_text,
            path=path,
            overall_task_goal=overall_task_goal,
            reason=reason,
        )

    @classmethod
    def _build_candidate_action_followup(
        cls,
        user_text: str,
        overall_task_goal: TaskGoal | None,
        completed_outputs: list[OutputKind],
        candidate_state: CandidateState | None,
        *,
        supports_message_delivery: bool = False,
    ) -> ToolDecision | None:
        if candidate_state is None:
            return None
        if not cls._has_actionable_candidates(candidate_state, overall_task_goal):
            return None

        action = cls._infer_goal_driven_candidate_action(
            user_text=user_text,
            overall_task_goal=overall_task_goal,
            completed_outputs=completed_outputs,
            supports_message_delivery=supports_message_delivery,
        )
        if action is None:
            action = cls._infer_candidate_action(user_text)
        if action is None:
            return None

        path = cls._select_candidate_path_for_action(candidate_state, action)
        if not path:
            return None

        if action == "reveal":
            if OutputKind.PATH_OPENED in completed_outputs:
                return None
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="reveal_candidate_path",
                reason="Reveal the most reliable candidate in Explorer after candidate selection.",
                selected_tool="file.reveal_in_explorer",
                arguments={"path": path},
                risk_level=RiskLevel.LOW,
                overall_task_goal=overall_task_goal,
                expected_step_outputs=[OutputKind.PATH_OPENED],
            )

        if action == "metadata":
            if OutputKind.OBJECT_DETAILS in completed_outputs:
                return None
            if cls._looks_like_document_path(path) and cls._goal_requires_document_write(overall_task_goal):
                edit_decision = cls._build_document_agent_edit_followup(
                    user_text=user_text,
                    path=path,
                    overall_task_goal=overall_task_goal,
                    reason="Delegate the grounded document edit task to the document sub-agent after candidate selection.",
                )
                if edit_decision is not None:
                    return edit_decision
            if cls._looks_like_document_path(path) and (
                cls._looks_like_document_structure_request(user_text)
                or cls._looks_like_document_block_search_request(user_text)
                or cls._goal_requires_document_details(overall_task_goal)
            ):
                return cls._build_document_details_followup(
                    path=path,
                    user_text=user_text,
                    overall_task_goal=overall_task_goal,
                    reason="Inspect the document structure before continuing the requested document operation.",
                )
            if cls._looks_like_image_path(path):
                return ToolDecision(
                    decision=DecisionType.TOOL_CALL,
                    intent="inspect_candidate_image",
                    reason="Inspect the most reliable image candidate after candidate selection.",
                    selected_tool="image.inspect",
                    arguments={"path": path, "include_ocr": True, "ocr_max_chars": 4000},
                    risk_level=RiskLevel.LOW,
                    overall_task_goal=overall_task_goal,
                    expected_step_outputs=[OutputKind.OBJECT_DETAILS],
                )
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="metadata_for_candidate",
                reason="Read metadata for the most reliable candidate after candidate selection.",
                selected_tool="file.metadata",
                arguments={"path": path},
                risk_level=RiskLevel.LOW,
                overall_task_goal=overall_task_goal,
                expected_step_outputs=[OutputKind.OBJECT_DETAILS],
            )

        if action in {"extract_document_structure", "search_document_blocks"}:
            if OutputKind.OBJECT_DETAILS in completed_outputs:
                return None
            return cls._build_document_details_followup(
                path=path,
                user_text=user_text,
                overall_task_goal=overall_task_goal,
                reason="Ground the requested document operation on structured blocks before editing or summarizing.",
            )

        if action == "preview":
            if OutputKind.OBJECT_DETAILS in completed_outputs:
                return None
            if cls._looks_like_image_path(path):
                return ToolDecision(
                    decision=DecisionType.TOOL_CALL,
                    intent="inspect_candidate_image",
                    reason="Inspect the most reliable image candidate after candidate selection.",
                    selected_tool="image.inspect",
                    arguments={"path": path, "include_ocr": True, "ocr_max_chars": 4000},
                    risk_level=RiskLevel.LOW,
                    overall_task_goal=overall_task_goal,
                    expected_step_outputs=[OutputKind.OBJECT_DETAILS],
                )
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="preview_candidate",
                reason="Preview the most reliable candidate after candidate selection.",
                selected_tool="file.preview",
                arguments={"path": path},
                risk_level=RiskLevel.LOW,
                overall_task_goal=overall_task_goal,
                expected_step_outputs=[OutputKind.OBJECT_DETAILS],
            )

        if action == "open":
            if OutputKind.PATH_OPENED in completed_outputs:
                return None
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="open_candidate_path",
                reason="Open the most reliable candidate after candidate selection.",
                selected_tool="file.open_path",
                arguments={"path": path},
                risk_level=RiskLevel.LOW,
                overall_task_goal=overall_task_goal,
                expected_step_outputs=[OutputKind.PATH_OPENED],
            )

        if action == "read":
            if OutputKind.FILE_CONTENTS in completed_outputs:
                return None
            if cls._looks_like_document_path(path):
                if cls._looks_like_document_summary_request(user_text):
                    return cls._build_document_agent_summary_followup(
                        user_text=user_text,
                        path=path,
                        overall_task_goal=overall_task_goal,
                        reason="Delegate document summarization to the document sub-agent after candidate selection.",
                    )
                return cls._build_document_agent_read_followup(
                    user_text=user_text,
                    path=path,
                    overall_task_goal=overall_task_goal,
                    reason="Delegate document content extraction to the document sub-agent after candidate selection.",
                )
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="read_candidate_file",
                reason="Read the most reliable candidate after candidate selection.",
                selected_tool="file.read",
                arguments={"paths": [path], "encoding": "utf-8", "max_bytes": 200000},
                risk_level=RiskLevel.LOW,
                overall_task_goal=overall_task_goal,
                expected_step_outputs=[OutputKind.FILE_CONTENTS],
            )

        if action == "describe_image":
            if OutputKind.OBJECT_DETAILS in completed_outputs:
                return None
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="describe_candidate_image",
                reason="Describe the semantic content of the most reliable image candidate after candidate selection.",
                selected_tool="image.describe",
                arguments={"path": path, "focus": "general", "include_ocr": True, "ocr_max_chars": 4000},
                risk_level=RiskLevel.LOW,
                overall_task_goal=overall_task_goal,
                expected_step_outputs=[OutputKind.OBJECT_DETAILS],
            )

        if action == "read_image_text":
            if OutputKind.FILE_CONTENTS in completed_outputs:
                return None
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="read_candidate_image_text",
                reason="Extract OCR text from the most reliable image candidate after candidate selection.",
                selected_tool="image.read_text",
                arguments={"paths": [path], "max_chars": 8000},
                risk_level=RiskLevel.LOW,
                overall_task_goal=overall_task_goal,
                expected_step_outputs=[OutputKind.FILE_CONTENTS],
            )

        if action == "send_file":
            if OutputKind.MESSAGE_SENT in completed_outputs:
                return None
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="send_candidate_file_to_current_session",
                reason="The current task still needs a delivered file, so send the strongest grounded candidate to the active session.",
                selected_tool="qq.send_file",
                arguments={"file_path": path, "target_kind": "current"},
                risk_level=RiskLevel.LOW,
                overall_task_goal=overall_task_goal,
                expected_step_outputs=[OutputKind.MESSAGE_SENT],
            )

        return None

    @classmethod
    def _build_state_transition_followup(
        cls,
        *,
        user_text: str,
        workflow_state: WorkflowState | None,
        candidate_state: CandidateState | None,
        overall_task_goal: TaskGoal | None,
        completed_outputs: list[OutputKind],
        supports_message_delivery: bool = False,
    ) -> ToolDecision | None:
        if workflow_state is None:
            return None
        if workflow_state.workflow_stage == "completed":
            return None
        if not workflow_state.next_allowed_actions:
            return None

        primary_path = str(workflow_state.primary_target_ref or "").strip()
        if not primary_path and candidate_state is not None:
            primary_path = cls._select_candidate_path_for_action(candidate_state, "read") or ""
        if not primary_path:
            return None

        if candidate_state is not None and not cls._has_actionable_candidates(candidate_state, overall_task_goal):
            if workflow_state.workflow_stage not in {"content_ready", "action_ready"}:
                return None

        actions = {str(action).strip().lower() for action in workflow_state.next_allowed_actions if str(action).strip()}
        missing_outputs = set(workflow_state.missing_outputs)

        if "deliver" in actions and OutputKind.MESSAGE_SENT not in completed_outputs:
            if not supports_message_delivery:
                return None
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="workflow_deliver_primary_target",
                reason="State transition: the observed workflow is action-ready and still needs message_sent.",
                selected_tool="qq.send_file",
                arguments={"file_path": primary_path, "target_kind": "current"},
                risk_level=RiskLevel.LOW,
                overall_task_goal=overall_task_goal,
                expected_step_outputs=[OutputKind.MESSAGE_SENT],
            )

        if cls._looks_like_document_path(primary_path):
            if (
                OutputKind.FILE_WRITTEN in missing_outputs
                and OutputKind.FILE_WRITTEN not in completed_outputs
            ):
                return cls._build_document_agent_edit_followup(
                    user_text=user_text,
                    path=primary_path,
                    overall_task_goal=overall_task_goal,
                    reason="State transition: the grounded document edit should now be handled by the document sub-agent.",
                )
            if (
                OutputKind.FILE_CONTENTS in missing_outputs
                and OutputKind.FILE_CONTENTS not in completed_outputs
                and cls._looks_like_document_summary_request(user_text)
            ):
                return cls._build_document_agent_summary_followup(
                    user_text=user_text,
                    path=primary_path,
                    overall_task_goal=overall_task_goal,
                    reason="State transition: the grounded document summary should now be handled by the document sub-agent.",
                )

        if "read" in actions and OutputKind.FILE_CONTENTS in missing_outputs and OutputKind.FILE_CONTENTS not in completed_outputs:
            if cls._looks_like_image_path(primary_path):
                return ToolDecision(
                    decision=DecisionType.TOOL_CALL,
                    intent="workflow_read_image_text",
                    reason="State transition: the observed workflow has an image candidate and still needs file_contents.",
                    selected_tool="image.read_text",
                    arguments={"paths": [primary_path], "max_chars": 8000},
                    risk_level=RiskLevel.LOW,
                    overall_task_goal=overall_task_goal,
                    expected_step_outputs=[OutputKind.FILE_CONTENTS],
                )
            if cls._looks_like_document_path(primary_path):
                return ToolDecision(
                    decision=DecisionType.TOOL_CALL,
                    intent="workflow_extract_document_text",
                    reason="State transition: the observed workflow has a document candidate and still needs file_contents.",
                    selected_tool="file.extract_text",
                    arguments={"paths": [primary_path], "max_chars": 12000, "max_rows_per_sheet": 10},
                    risk_level=RiskLevel.LOW,
                    overall_task_goal=overall_task_goal,
                    expected_step_outputs=[OutputKind.FILE_CONTENTS],
                )
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="workflow_read_file",
                reason="State transition: the observed workflow has a file candidate and still needs file_contents.",
                selected_tool="file.read",
                arguments={"paths": [primary_path], "encoding": "utf-8", "max_bytes": 12000},
                risk_level=RiskLevel.LOW,
                overall_task_goal=overall_task_goal,
                expected_step_outputs=[OutputKind.FILE_CONTENTS],
            )

        if "inspect" in actions and OutputKind.OBJECT_DETAILS in missing_outputs and OutputKind.OBJECT_DETAILS not in completed_outputs:
            if cls._looks_like_image_path(primary_path):
                return ToolDecision(
                    decision=DecisionType.TOOL_CALL,
                    intent="workflow_describe_image",
                    reason="State transition: the observed workflow has an image candidate and still needs object_details.",
                    selected_tool="image.describe",
                    arguments={"path": primary_path, "focus": "general", "include_ocr": True, "ocr_max_chars": 4000},
                    risk_level=RiskLevel.LOW,
                    overall_task_goal=overall_task_goal,
                    expected_step_outputs=[OutputKind.OBJECT_DETAILS],
                )
            if cls._looks_like_document_path(primary_path) and (
                cls._looks_like_document_structure_request(user_text)
                or cls._looks_like_document_block_search_request(user_text)
                or cls._goal_requires_document_details(overall_task_goal)
                or cls._goal_requires_document_write(overall_task_goal)
            ):
                return cls._build_document_details_followup(
                    path=primary_path,
                    user_text=user_text,
                    overall_task_goal=overall_task_goal,
                    reason="State transition: ground the document operation on structured content before continuing.",
                )
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="workflow_inspect_object",
                reason="State transition: the observed workflow has a candidate and still needs object_details.",
                selected_tool="file.metadata",
                arguments={"path": primary_path},
                risk_level=RiskLevel.LOW,
                overall_task_goal=overall_task_goal,
                expected_step_outputs=[OutputKind.OBJECT_DETAILS],
            )

        return None

    @staticmethod
    def _infer_goal_driven_candidate_action(
        *,
        user_text: str,
        overall_task_goal: TaskGoal | None,
        completed_outputs: list[OutputKind],
        supports_message_delivery: bool = False,
    ) -> str | None:
        if overall_task_goal is None:
            return None

        missing_outputs = [
            output_kind for output_kind in overall_task_goal.required_outputs if output_kind not in completed_outputs
        ]
        if OutputKind.FILE_WRITTEN in missing_outputs:
            if OutputKind.OBJECT_DETAILS not in completed_outputs:
                if AgentKernel._looks_like_document_block_search_request(user_text):
                    return "search_document_blocks"
                return "extract_document_structure"
            return None
        if OutputKind.FILE_CONTENTS in missing_outputs:
            if AgentKernel._looks_like_image_text_request(user_text):
                return "read_image_text"
            return "read"
        if OutputKind.OBJECT_DETAILS in missing_outputs:
            if AgentKernel._looks_like_image_request(user_text):
                return "describe_image"
            if AgentKernel._looks_like_document_block_search_request(user_text):
                return "search_document_blocks"
            if AgentKernel._looks_like_document_structure_request(user_text) or OutputKind.FILE_WRITTEN in missing_outputs:
                return "extract_document_structure"
            if AgentKernel._looks_like_preview_request(user_text):
                return "preview"
            return "metadata"
        if OutputKind.MESSAGE_SENT in missing_outputs:
            if supports_message_delivery:
                return "send_file"
            return None
        if OutputKind.PATH_OPENED in missing_outputs:
            if AgentKernel._looks_like_reveal_request(user_text):
                return "reveal"
            return "open"
        return None

    @classmethod
    def _build_document_summary_workflow(
        cls,
        user_text: str,
        completed_outputs: list[OutputKind],
        candidate_state: CandidateState | None,
        overall_task_goal: TaskGoal | None = None,
        knowledge_intent=None,
        task_classification=None,
        *,
        supports_message_delivery: bool = False,
    ) -> ToolDecision | None:
        knowledge_type = "" if knowledge_intent is None else str(getattr(knowledge_intent, "knowledge_type", "") or "")
        if knowledge_type and knowledge_type != "local_workspace":
            return None
        classified_kind = "" if task_classification is None else str(getattr(task_classification, "task_kind", "") or "").strip().lower()
        preferred_families = cls._classification_preferred_families(task_classification)
        if classified_kind not in {"summarize", "document_summary"} and "document_summary" not in preferred_families:
            return None
        if OutputKind.FILE_CONTENTS in completed_outputs:
            return None
        if candidate_state is None or not candidate_state.candidate_paths:
            return None

        if cls._has_actionable_candidates(candidate_state, overall_task_goal):
            chosen_paths = [path for path in candidate_state.candidate_paths if cls._looks_like_document_path(path)]
            if chosen_paths:
                return cls._build_document_agent_summary_followup(
                    user_text=user_text,
                    path=chosen_paths[0],
                    overall_task_goal=cls._compose_local_task_goal(
                        user_text=user_text,
                        default_summary="Summarize the matched document through the document sub-agent.",
                        overall_task_goal=overall_task_goal,
                        include_candidates=True,
                        supports_message_delivery=supports_message_delivery,
                        task_classification=task_classification,
                    ),
                    reason="Delegate document summarization to the document sub-agent after the target file is grounded.",
                )
        return None

    @classmethod
    def _build_document_operation_workflow(
        cls,
        user_text: str,
        completed_outputs: list[OutputKind],
        candidate_state: CandidateState | None,
        overall_task_goal: TaskGoal | None = None,
        knowledge_intent=None,
        task_classification=None,
        *,
        supports_message_delivery: bool = False,
    ) -> ToolDecision | None:
        knowledge_type = "" if knowledge_intent is None else str(getattr(knowledge_intent, "knowledge_type", "") or "")
        if knowledge_type and knowledge_type != "local_workspace":
            return None
        classified_kind = "" if task_classification is None else str(getattr(task_classification, "task_kind", "") or "").strip().lower()
        preferred_families = cls._classification_preferred_families(task_classification)
        is_document_operation = classified_kind in {"document_edit", "edit", "rewrite", "transform"} or "document_operation" in preferred_families
        if not is_document_operation:
            return None
        if OutputKind.FILE_WRITTEN in completed_outputs:
            return None

        if candidate_state is None or not candidate_state.candidate_paths:
            return None

        chosen_paths = [path for path in candidate_state.candidate_paths if cls._looks_like_document_path(path)]
        if cls._has_actionable_candidates(candidate_state, overall_task_goal) and chosen_paths:
            if classified_kind in {"document_edit", "edit", "rewrite", "transform"}:
                return cls._build_document_agent_edit_followup(
                    user_text=user_text,
                    path=chosen_paths[0],
                    overall_task_goal=cls._compose_local_task_goal(
                        user_text=user_text,
                        default_summary="Edit the matched document through the document sub-agent.",
                        overall_task_goal=overall_task_goal,
                        include_candidates=True,
                        supports_message_delivery=supports_message_delivery,
                        task_classification=task_classification,
                    ),
                    reason="Delegate the grounded document edit task to the document sub-agent.",
                )
            if OutputKind.OBJECT_DETAILS not in completed_outputs:
                return cls._build_document_details_followup(
                    path=chosen_paths[0],
                    user_text=user_text,
                    overall_task_goal=cls._compose_local_task_goal(
                        user_text=user_text,
                        default_summary="Inspect the matched document structure before continuing the requested document operation.",
                        overall_task_goal=overall_task_goal,
                        include_candidates=True,
                        supports_message_delivery=supports_message_delivery,
                        task_classification=task_classification,
                    ),
                    reason="Inspect the matched document structure before continuing the requested document operation.",
                )
        return None

    @staticmethod
    def _extract_document_lookup_query(user_text: str) -> str:
        cleaned = user_text.strip()
        for phrase in (
            "帮我",
            "请帮我",
            "请你",
            "给我",
            "替我",
            "总结一下",
            "总结",
            "概括一下",
            "概括",
            "看一下",
            "看看",
            "讲了什么",
            "内容",
            "这个文档",
            "这份文档",
            "这个文件",
            "这份文件",
            "那个文档",
            "那份文档",
            "那个文件",
            "那份文件",
            "桌面上的",
            "桌面的",
            "桌面",
            "下载里的",
            "下载中的",
            "我的",
            "文件",
            "文档",
            "图片",
            "照片",
            "截图",
            "图像",
            "read",
            "summarize",
            "summary",
            "content of",
            "content",
            "explain",
            "what is in",
            "what's in",
            "describe",
            "inspect",
            "ocr",
        ):
            cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
        return AgentKernel._finalize_local_lookup_query(cleaned)

    @staticmethod
    def _finalize_local_lookup_query(cleaned: str) -> str:
        cleaned = str(cleaned or "")
        cleaned = re.sub("[,.!?:;()\\[\\]{}\"'`?????????????]+", " ", cleaned)
        cleaned = re.sub(
            r"\b(send|attach|upload|deliver|it|me|the|to|open|show|find|search|look|check)\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        stop_tokens = {
            "帮我",
            "请帮我",
            "请你",
            "给我",
            "替我",
            "然后",
            "发我",
            "发给我",
            "发过来",
            "传我",
            "传给我",
            "传过来",
            "给我发",
            "给我传",
            "原文件",
            "原文",
            "原始文件",
            "send",
            "attach",
            "upload",
            "deliver",
            "it",
            "me",
            "the",
            "to",
            "open",
            "show",
            "find",
            "search",
            "look",
            "check",
            "file",
            "document",
        }
        tokens: list[str] = []
        for token in re.split(r"\s+", cleaned):
            normalized = token.strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in stop_tokens or normalized in stop_tokens:
                continue
            tokens.append(normalized)
        return " ".join(tokens).strip()

    @staticmethod
    def _document_extensions_for_query(query: str) -> list[str]:
        return AgentKernel._local_file_extensions_for_query(query)

    @classmethod
    def _local_file_extensions_for_query(cls, query: str, *, user_text: str = "") -> list[str]:
        lowered = f"{query} {user_text}".lower()
        if any(
            token in lowered
            for token in ("png", "jpg", "jpeg", "webp", "gif", "bmp", "图片", "照片", "截图", "图像", "image", "picture", "photo", "screenshot")
        ):
            return [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]
        if any(token in lowered for token in ("ppt", "presentation", ".pptx", "幻灯片", "演示")):
            return [".pptx"]
        if any(token in lowered for token in ("excel", "sheet", "spreadsheet", ".xlsx", "表格")):
            return [".xlsx", ".csv"]
        if any(token in lowered for token in ("word", ".docx", "文档")):
            return [".docx"]
        if any(token in lowered for token in ("markdown", ".md")):
            return [".md"]
        if ".pdf" in lowered or re.search(r"(?<![a-z0-9])pdf(?![a-z0-9])", lowered):
            return [".pdf"]
        if ".txt" in lowered or "文本文档" in lowered or re.search(r"(?<![a-z0-9])txt(?![a-z0-9])", lowered):
            return [".txt"]
        if ".log" in lowered or "日志文件" in lowered or "日志格式" in lowered or "log file" in lowered:
            return [".log"]
        return []

    @classmethod
    def _build_local_retrieval_plan(cls, query: str, *, user_text: str = "") -> dict[str, object]:
        normalized = FileQueryNormalizer.normalize(query)
        extensions = list(normalized.file_type_hints) or cls._local_file_extensions_for_query(query, user_text=user_text)
        planned_query = FileQueryNormalizer.strip_file_type_terms(query, extensions) or query
        tokenized_query = cls._tokenize_query(planned_query)
        if normalized.core_terms:
            query_terms = cls._dedupe_preserve_order(list(normalized.core_terms))
            if any("." in term or re.search(r"\d{4,}", term) for term in tokenized_query):
                query_terms = cls._dedupe_preserve_order([*tokenized_query, *query_terms])
        else:
            query_terms = cls._dedupe_preserve_order(tokenized_query)
        return {
            "query": planned_query,
            "query_terms": query_terms or cls._tokenize_query(planned_query),
            "alias_terms": normalized.alias_terms,
            "extensions": extensions,
            "type_constraints": extensions,
        }

    @classmethod
    def _extract_image_lookup_query(cls, user_text: str) -> str:
        explicit = re.search(
            r"([A-Za-z0-9_\-\u4e00-\u9fff]+\.(?:png|jpg|jpeg|webp|gif|bmp))",
            user_text,
            flags=re.IGNORECASE,
        )
        if explicit:
            return explicit.group(1).replace(".", " ")

        cleaned = user_text
        for phrase in (
            "help me",
            "please",
            "look at",
            "tell me",
            "read the",
            "read this",
            "read",
            "describe",
            "inspect",
            "analyze",
            "analyse",
            "ocr",
            "what is in",
            "what's in",
            "content of",
            "text in",
            "words in",
            "桌面上的",
            "桌面",
            "这张",
            "那张",
            "帮我",
            "请帮我",
            "帮我读一下",
            "帮我看看",
            "看看",
            "看一下",
            "查看",
            "分析",
            "识别",
            "读一下",
            "读取",
            "上面写了什么",
            "里有什么",
            "图里有什么",
            "图上写了什么",
            "里的文字",
            "内容",
            "是什么",
            "图片",
            "照片",
            "图像",
            "image",
            "picture",
            "photo",
        ):
            cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
        cleaned = cls._finalize_local_lookup_query(cleaned)
        return cleaned or "image"

    @classmethod
    def _extract_explicit_image_path(cls, user_text: str) -> str | None:
        for line in user_text.splitlines():
            lowered = line.lower()
            if "image path" in lowered and ":" in line:
                candidate = cls._clean_explicit_path_candidate(line.split(":", 1)[1])
                if cls._looks_like_image_path(candidate):
                    return candidate

        for match in re.finditer(r"([A-Za-z]:[\\/][^\r\n]+\.(?:png|jpg|jpeg|webp|gif|bmp))", user_text, flags=re.IGNORECASE):
            candidate = cls._clean_explicit_path_candidate(match.group(1))
            if cls._looks_like_image_path(candidate):
                return candidate
        return None

    @staticmethod
    def _clean_explicit_path_candidate(value: str) -> str:
        cleaned = str(value or "").strip().strip("\"'`")
        cleaned = re.sub(r"\s*(?:[)\]}>，。；;]+)?\s*$", "", cleaned).strip()
        return cleaned.strip("\"'`")

    @classmethod
    def _build_explicit_image_path_workflow(
        cls,
        user_text: str,
        completed_outputs: list[OutputKind],
        overall_task_goal: TaskGoal | None,
    ) -> ToolDecision | None:
        if not cls._looks_like_image_request(user_text):
            return None
        explicit_path = cls._extract_explicit_image_path(user_text)
        if not explicit_path:
            return None
        if cls._looks_like_image_text_request(user_text):
            if OutputKind.FILE_CONTENTS in completed_outputs:
                return None
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="read_explicit_image_text",
                reason="The request provides an explicit local image path and asks for OCR/text extraction.",
                selected_tool="image.read_text",
                arguments={"paths": [explicit_path], "max_chars": 8000},
                risk_level=RiskLevel.LOW,
                overall_task_goal=cls._compose_local_task_goal(
                    user_text=user_text,
                    default_summary=f"Read text from image {explicit_path}.",
                    overall_task_goal=overall_task_goal,
                    include_candidates=False,
                ),
                expected_step_outputs=[OutputKind.FILE_CONTENTS],
            )
        if OutputKind.OBJECT_DETAILS in completed_outputs:
            return None
        return ToolDecision(
            decision=DecisionType.TOOL_CALL,
            intent="describe_explicit_image",
            reason="The request provides an explicit local image path and asks for image understanding.",
            selected_tool="image.describe",
            arguments={"path": explicit_path, "focus": "general", "include_ocr": True, "ocr_max_chars": 4000},
            risk_level=RiskLevel.LOW,
            overall_task_goal=cls._compose_local_task_goal(
                user_text=user_text,
                default_summary=f"Describe image {explicit_path}.",
                overall_task_goal=overall_task_goal,
                include_candidates=False,
            ),
            expected_step_outputs=[OutputKind.OBJECT_DETAILS],
        )

    @staticmethod
    def _looks_like_file_delivery_request_clean(user_text: str) -> bool:
        lowered = user_text.lower()
        return any(
            term in user_text or term in lowered
            for term in (
                "发我",
                "发给我",
                "发过来",
                "传我",
                "传给我",
                "传过来",
                "给我发",
                "给我传",
                "原文件发我",
                "send me",
                "send it",
                "attach it",
                "upload it",
            )
        )

    @staticmethod
    def _extract_file_delivery_query_clean(user_text: str) -> str:
        cleaned = user_text
        for phrase in (
            "帮我",
            "请帮我",
            "请你",
            "你",
            "把",
            "的",
            "你能把",
            "你帮我",
            "有没有",
            "有没",
            "一个",
            "一份",
            "桌面有没有一个",
            "桌面上有没有一个",
            "桌面上的",
            "桌面的",
            "桌面",
            "也在桌面上",
            "也在桌面",
            "也在",
            "下载",
            "下载里的",
            "下载中的",
            "文件",
            "文档",
            "把原始文件发给我",
            "把原文件发给我",
            "把原文发给我",
            "把原始文件传给我",
            "把原文件传给我",
            "把原文传给我",
            "那个",
            "那份",
            "这份",
            "发给我",
            "发我",
            "发过来",
            "传给我",
            "传我",
            "传过来",
            "给我发",
            "给我传",
            "send me",
            "send it",
            "attach it",
            "upload it",
            "please",
            "然后",
            "原文件",
            "原文",
            "原始文件",
        ):
            cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
        return AgentKernel._finalize_local_lookup_query(cleaned)

    @classmethod
    def _build_file_delivery_workflow(
        cls,
        user_text: str,
        completed_outputs: list[OutputKind],
        candidate_state: CandidateState | None,
        overall_task_goal: TaskGoal | None,
        knowledge_intent=None,
        task_classification=None,
        *,
        supports_message_delivery: bool = False,
    ) -> ToolDecision | None:
        classified_kind = "" if task_classification is None else str(getattr(task_classification, "task_kind", "") or "").strip().lower()
        preferred_families = cls._classification_preferred_families(task_classification)
        if classified_kind != "delivery" and "file_delivery" not in preferred_families:
            return None
        knowledge_type = "" if knowledge_intent is None else str(getattr(knowledge_intent, "knowledge_type", "") or "")
        if knowledge_type and knowledge_type != "local_workspace":
            return None
        if candidate_state is None or not candidate_state.candidate_paths:
            return None
        return None

    def _build_llm_local_search_workflow(
        self,
        *,
        user_text: str,
        completed_outputs: list[OutputKind],
        candidate_state: CandidateState | None,
        overall_task_goal: TaskGoal | None,
        knowledge_intent=None,
        task_classification=None,
        supports_message_delivery: bool = False,
    ) -> ToolDecision | None:
        prior_source_tool = ""
        if candidate_state is not None and candidate_state.candidate_paths:
            return None
        if OutputKind.OBJECT_CANDIDATES in completed_outputs:
            return None

        knowledge_type = "" if knowledge_intent is None else str(getattr(knowledge_intent, "knowledge_type", "") or "")
        if knowledge_type and knowledge_type != "local_workspace":
            return None
        if self._build_explicit_image_path_workflow(
            user_text=user_text,
            completed_outputs=completed_outputs,
            overall_task_goal=overall_task_goal,
        ) is not None:
            return None
        if candidate_state is not None and not candidate_state.candidate_paths:
            prior_source_tool = str(candidate_state.source_tool or "").strip()
            if prior_source_tool == "retrieval.search_local_objects":
                return None
        if not hasattr(self.llm_client, "plan_local_search_step"):
            return self._build_legacy_initial_local_search_workflow(
                user_text=user_text,
                candidate_state=candidate_state,
                task_classification=task_classification,
                overall_task_goal=overall_task_goal,
                supports_message_delivery=supports_message_delivery,
            )

        task_kind = "" if task_classification is None else str(getattr(task_classification, "task_kind", "") or "").strip().lower()
        planner_family = "local_lookup"
        intent = "search_local_file_candidates"
        default_summary = "Find the requested local file."
        default_target_kind = "file"
        preferred_extensions: list[str] = []

        if task_kind in {"document_edit", "edit", "rewrite", "transform"}:
            planner_family = "document_operation"
            intent = "search_document_operation_candidates"
            default_summary = "Find the target local document before continuing the requested document operation."
            preferred_extensions = [".docx"]
        elif task_kind in {"summarize", "document_summary"}:
            planner_family = "document_summary"
            intent = "search_document_candidates"
            default_summary = "Find the target local document before summarizing it."
            preferred_extensions = [".docx", ".md", ".txt", ".pdf", ".pptx", ".xlsx"]
        elif task_kind == "delivery":
            planner_family = "file_delivery"
            intent = "search_file_for_delivery"
            default_summary = "Find the requested local file so it can be sent back to the user."
        elif task_kind in {"lookup", "file_lookup", "local_lookup", "inspect", ""}:
            planner_family = "local_lookup"
            intent = "search_local_file_candidates"
            default_summary = "Find the requested local file."

        if self._looks_like_image_request(user_text):
            preferred_extensions = [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]

        goal = self._compose_local_task_goal(
            user_text=user_text,
            default_summary=default_summary,
            overall_task_goal=overall_task_goal,
            include_candidates=True,
            supports_message_delivery=supports_message_delivery,
            task_classification=task_classification,
        )
        if planner_family == "file_delivery" and supports_message_delivery and OutputKind.MESSAGE_SENT not in goal.required_outputs:
            goal.required_outputs.append(OutputKind.MESSAGE_SENT)

        allowed_tools = [
            tool_name
            for tool_name in ("file.list", "file.search_by_name", "retrieval.search_local_objects")
            if self.registry.has_tool(tool_name)
        ]
        if prior_source_tool:
            retry_tools = [tool_name for tool_name in allowed_tools if tool_name != prior_source_tool]
            if retry_tools:
                if prior_source_tool == "file.search_by_name" and "retrieval.search_local_objects" in retry_tools:
                    retry_tools = [
                        "retrieval.search_local_objects",
                        *[tool_name for tool_name in retry_tools if tool_name != "retrieval.search_local_objects"],
                    ]
                allowed_tools = retry_tools
        if not allowed_tools:
            return None

        tool_schemas = {
            tool_name: self.registry.get_manifest(tool_name).input_schema
            for tool_name in allowed_tools
        }
        now_local = datetime.now().astimezone()
        try:
            plan = self.llm_client.plan_local_search_step(
                user_text=user_text,
                task_classification=None
                if task_classification is None
                else task_classification.model_dump(mode="json"),
                current_time_iso=now_local.isoformat(),
                timezone=str(now_local.tzinfo or "Asia/Shanghai"),
                scope_hints=self._local_scope_hints(),
                allowed_tools=allowed_tools,
                tool_schemas=tool_schemas,
                default_target_kind=default_target_kind,
                preferred_extensions=preferred_extensions,
                recent_context=self._recent_conversation_text(),
                hot_context_summary=self.hot_context_summary,
                warm_memory_summary=self.warm_memory_summary,
                cold_memory_summary=self.cold_memory_summary,
                active_task_summary=self.active_task_summary,
            )
        except Exception:
            return self._build_legacy_initial_local_search_workflow(
                user_text=user_text,
                candidate_state=candidate_state,
                task_classification=task_classification,
                overall_task_goal=overall_task_goal,
                supports_message_delivery=supports_message_delivery,
            )

        selected_tool = str(plan.get("selected_tool", "") or "").strip()
        arguments = plan.get("arguments", {})
        if selected_tool not in set(allowed_tools) or not isinstance(arguments, dict):
            return self._build_legacy_initial_local_search_workflow(
                user_text=user_text,
                candidate_state=candidate_state,
                task_classification=task_classification,
                overall_task_goal=overall_task_goal,
                supports_message_delivery=supports_message_delivery,
            )

        normalized_arguments = dict(arguments)
        if selected_tool == "file.list":
            path = str(normalized_arguments.get("path", "") or normalized_arguments.get("path_scope", "")).strip()
            query = str(normalized_arguments.get("query", "") or "").strip()
            if not path:
                return None
            normalized_arguments["path"] = path
            normalized_arguments.setdefault("recursive", True)
            normalized_arguments.setdefault("include_dirs", True)
            if preferred_extensions and not normalized_arguments.get("patterns"):
                normalized_arguments["patterns"] = [f"*{ext}" for ext in preferred_extensions]
            if query and not normalized_arguments.get("query_terms"):
                normalized_arguments["query_terms"] = self._tokenize_query(query)
            if query:
                normalized_arguments["query"] = query
            expected_outputs = [OutputKind.DIRECTORY_ENTRIES]
        elif selected_tool == "file.search_by_name":
            path = str(normalized_arguments.get("path", "") or normalized_arguments.get("path_scope", "")).strip()
            query = str(normalized_arguments.get("query", "") or "").strip()
            if not path or not query:
                return None
            normalized_arguments["path"] = path
            normalized_arguments["query"] = query
            normalized_arguments.setdefault("recursive", True)
            normalized_arguments.setdefault("include_dirs", True)
            normalized_arguments.setdefault("scope_mode", "subtree")
            normalized_arguments.setdefault("target_kind", default_target_kind)
            normalized_arguments.setdefault("top_k", 8)
            if preferred_extensions and not normalized_arguments.get("extensions"):
                normalized_arguments["extensions"] = preferred_extensions
            expected_outputs = [OutputKind.OBJECT_CANDIDATES]
        elif selected_tool == "retrieval.search_local_objects":
            path_scope = str(normalized_arguments.get("path_scope", "") or normalized_arguments.get("path", "")).strip()
            query = str(normalized_arguments.get("query", "") or "").strip()
            if not path_scope or not query:
                return None
            normalized_arguments["path_scope"] = path_scope
            normalized_arguments["query"] = query
            normalized_arguments.setdefault("scope_mode", "subtree")
            normalized_arguments.setdefault("target_kind", default_target_kind)
            normalized_arguments.setdefault("top_k", 8)
            normalized_arguments.setdefault("rebuild_if_missing", True)
            if preferred_extensions and not normalized_arguments.get("extensions"):
                normalized_arguments["extensions"] = preferred_extensions
            expected_outputs = [OutputKind.OBJECT_CANDIDATES]
        else:
            return self._build_legacy_initial_local_search_workflow(
                user_text=user_text,
                candidate_state=candidate_state,
                task_classification=task_classification,
                overall_task_goal=overall_task_goal,
                supports_message_delivery=supports_message_delivery,
            )

        return ToolDecision(
            decision=DecisionType.TOOL_CALL,
            intent=intent,
            reason=str(plan.get("reason", "") or "LLM planned the initial local search step.").strip()
            or "LLM planned the initial local search step.",
            selected_tool=selected_tool,
            arguments=normalized_arguments,
            risk_level=RiskLevel.LOW,
            overall_task_goal=goal,
            expected_step_outputs=expected_outputs,
        )

    @staticmethod
    def _local_search_planner_family(task_classification) -> str:
        task_kind = "" if task_classification is None else str(getattr(task_classification, "task_kind", "") or "").strip().lower()
        if task_kind in {"document_edit", "edit", "rewrite", "transform"}:
            return "document_operation"
        if task_kind in {"summarize", "document_summary"}:
            return "document_summary"
        if task_kind == "delivery":
            return "file_delivery"
        return "local_lookup"

    @classmethod
    def _build_legacy_initial_local_search_workflow(
        cls,
        *,
        user_text: str,
        candidate_state: CandidateState | None = None,
        task_classification=None,
        overall_task_goal: TaskGoal | None,
        supports_message_delivery: bool = False,
    ) -> ToolDecision | None:
        task_kind = "" if task_classification is None else str(getattr(task_classification, "task_kind", "") or "").strip().lower()
        prior_source_tool = ""
        if candidate_state is not None and not candidate_state.candidate_paths:
            prior_source_tool = str(candidate_state.source_tool or "").strip()
            if prior_source_tool == "retrieval.search_local_objects":
                return None

        if task_kind in {"document_edit", "edit", "rewrite", "transform"}:
            query = cls._extract_document_lookup_query(user_text)
            default_summary = f"Find the local document matching {query} before continuing the requested document operation."
            intent_by_name = "search_document_operation_candidates_by_name"
            intent_hybrid = "search_document_operation_candidates"
        elif task_kind in {"summarize", "document_summary"}:
            query = cls._extract_document_lookup_query(user_text)
            default_summary = f"Find and summarize the document matching {query}."
            intent_by_name = "search_document_candidates_by_name"
            intent_hybrid = "search_document_candidates"
        elif task_kind == "delivery":
            query = cls._extract_file_delivery_query_clean(user_text)
            if cls._looks_like_document_summary_request(user_text):
                document_query = cls._extract_document_lookup_query(user_text)
                if document_query:
                    query = document_query
            default_summary = f"Find the local file matching {query} so it can be sent back to the user."
            intent_by_name = "search_file_for_delivery_by_name"
            intent_hybrid = "search_file_for_delivery"
        else:
            if cls._looks_like_image_request(user_text):
                query = cls._extract_image_lookup_query(user_text)
            else:
                query = cls._extract_document_lookup_query(user_text)
                if not query and cls._looks_like_file_delivery_request_clean(user_text):
                    query = cls._extract_file_delivery_query_clean(user_text)
            default_summary = f"Find the local file matching {query}."
            intent_by_name = "search_local_file_candidates_by_name"
            intent_hybrid = "search_local_file_candidates"

        if not query:
            return None

        query = re.sub(r"^(找找|找一下|查找|找到)\s*", "", str(query).strip())
        retrieval_plan = cls._build_local_retrieval_plan(query, user_text=user_text)
        query = str(retrieval_plan["query"])
        query_terms = list(retrieval_plan["query_terms"])
        alias_terms = list(retrieval_plan["alias_terms"])
        extensions = list(retrieval_plan["extensions"])
        if task_kind in {"document_edit", "edit", "rewrite", "transform"} and ".docx" not in extensions:
            extensions = [*extensions, ".docx"]

        goal = cls._compose_local_task_goal(
            user_text=user_text,
            default_summary=default_summary,
            overall_task_goal=overall_task_goal,
            include_candidates=True,
            supports_message_delivery=supports_message_delivery,
            task_classification=task_classification,
        )
        if task_kind == "delivery" and supports_message_delivery and OutputKind.MESSAGE_SENT not in goal.required_outputs:
            goal.required_outputs.append(OutputKind.MESSAGE_SENT)

        if prior_source_tool == "file.search_by_name":
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent=intent_hybrid,
                reason="Fallback heuristic broadens to hybrid local retrieval after an empty name lookup.",
                selected_tool="retrieval.search_local_objects",
                arguments={
                    "query": query,
                    "target_kind": "file",
                    "path_scope": cls._preferred_local_path_scope(user_text),
                    "scope_mode": cls._preferred_local_scope_mode(user_text),
                    "extensions": extensions,
                    "query_terms": query_terms,
                    "alias_terms": alias_terms,
                    "top_k": 8,
                    "rebuild_if_missing": True,
                },
                risk_level=RiskLevel.LOW,
                overall_task_goal=goal,
                expected_step_outputs=[OutputKind.OBJECT_CANDIDATES],
            )

        if cls._looks_like_title_first_local_request(
            user_text,
            query,
        ) and cls._should_keep_name_first_with_type_constraints(user_text, query, extensions):
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent=intent_by_name,
                reason="Fallback heuristic chose direct file-name matching before broader local retrieval.",
                selected_tool="file.search_by_name",
                arguments={
                    "path": cls._preferred_local_path_scope(user_text),
                    "query": query,
                    "query_terms": query_terms,
                    "alias_terms": alias_terms,
                    "recursive": True,
                    "scope_mode": cls._preferred_local_scope_mode(user_text),
                    "target_kind": "file",
                    "extensions": extensions,
                    "include_dirs": True,
                    "top_k": 8,
                },
                risk_level=RiskLevel.LOW,
                overall_task_goal=goal,
                expected_step_outputs=[OutputKind.OBJECT_CANDIDATES],
            )

        return ToolDecision(
            decision=DecisionType.TOOL_CALL,
            intent=intent_hybrid,
            reason="Fallback heuristic chose hybrid local retrieval for the initial search step.",
            selected_tool="retrieval.search_local_objects",
            arguments={
                "query": query,
                "target_kind": "file",
                "path_scope": cls._preferred_local_path_scope(user_text),
                "scope_mode": cls._preferred_local_scope_mode(user_text),
                "extensions": extensions,
                "query_terms": query_terms,
                "alias_terms": alias_terms,
                "top_k": 8,
                "rebuild_if_missing": True,
            },
            risk_level=RiskLevel.LOW,
            overall_task_goal=goal,
            expected_step_outputs=[OutputKind.OBJECT_CANDIDATES],
        )

    @classmethod
    def _build_local_file_lookup_workflow(
        cls,
        user_text: str,
        completed_outputs: list[OutputKind],
        candidate_state: CandidateState | None,
        knowledge_intent=None,
        overall_task_goal: TaskGoal | None = None,
        task_classification=None,
    ) -> ToolDecision | None:
        if candidate_state is not None and candidate_state.candidate_paths:
            return None
        if OutputKind.OBJECT_CANDIDATES in completed_outputs:
            return None
        knowledge_type = "" if knowledge_intent is None else str(getattr(knowledge_intent, "knowledge_type", "") or "")
        if knowledge_type != "local_workspace":
            return None

        explicit_image_decision = cls._build_explicit_image_path_workflow(
            user_text=user_text,
            completed_outputs=completed_outputs,
            overall_task_goal=overall_task_goal,
        )
        if explicit_image_decision is not None:
            return explicit_image_decision
        return None

        if cls._looks_like_image_request(user_text):
            cleaned = cls._extract_image_lookup_query(user_text)
        else:
            cleaned = cls._extract_document_lookup_query(user_text)
            if not cleaned and cls._looks_like_file_delivery_request_clean(user_text):
                cleaned = cls._extract_file_delivery_query_clean(user_text)
        if not cleaned:
            return None
        cleaned = re.sub(r"^(找找|找一下|查找|找到)\s*", "", cleaned)

        retrieval_plan = cls._build_local_retrieval_plan(cleaned, user_text=user_text)
        cleaned = str(retrieval_plan["query"])
        query_terms = list(retrieval_plan["query_terms"])
        alias_terms = list(retrieval_plan["alias_terms"])
        extensions = list(retrieval_plan["extensions"])
        if candidate_state is not None and not candidate_state.candidate_paths:
            if candidate_state.source_tool == "retrieval.search_local_objects":
                return None
            if candidate_state.source_tool == "file.search_by_name":
                return ToolDecision(
                    decision=DecisionType.TOOL_CALL,
                    intent="search_local_file_candidates",
                    reason="Name lookup was empty, so broaden to hybrid semantic and title retrieval inside the current scope.",
                    selected_tool="retrieval.search_local_objects",
                    arguments={
                        "query": cleaned,
                        "target_kind": "file",
                        "path_scope": cls._preferred_local_path_scope(user_text),
                        "scope_mode": cls._preferred_local_scope_mode(user_text),
                        "extensions": extensions,
                        "query_terms": query_terms,
                        "alias_terms": alias_terms,
                        "top_k": 8,
                        "rebuild_if_missing": True,
                    },
                    risk_level=RiskLevel.LOW,
                    overall_task_goal=cls._compose_local_task_goal(
                        user_text=user_text,
                        default_summary=f"Find the local file matching {cleaned}.",
                        overall_task_goal=overall_task_goal,
                        include_candidates=True,
                        task_classification=task_classification,
                    ),
                    expected_step_outputs=[OutputKind.OBJECT_CANDIDATES],
                )
        if cls._looks_like_title_first_local_request(
            user_text,
            cleaned,
        ) and cls._should_keep_name_first_with_type_constraints(user_text, cleaned, extensions):
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="search_local_file_candidates_by_name",
                reason="This local request sounds title-like, so direct file-name matching should run before broader retrieval.",
                selected_tool="file.search_by_name",
                arguments={
                    "path": cls._preferred_local_path_scope(user_text),
                    "query": cleaned,
                    "query_terms": query_terms,
                    "alias_terms": alias_terms,
                    "recursive": True,
                    "scope_mode": cls._preferred_local_scope_mode(user_text),
                    "target_kind": "file",
                    "extensions": extensions,
                    "include_dirs": True,
                    "top_k": 8,
                },
                risk_level=RiskLevel.LOW,
                overall_task_goal=cls._compose_local_task_goal(
                    user_text=user_text,
                    default_summary=f"Find the local file matching {cleaned}.",
                    overall_task_goal=overall_task_goal,
                    include_candidates=True,
                    task_classification=task_classification,
                ),
                expected_step_outputs=[OutputKind.OBJECT_CANDIDATES],
            )

        return ToolDecision(
            decision=DecisionType.TOOL_CALL,
            intent="search_local_file_candidates",
            reason="This looks like a local file lookup, so use the same hybrid local retrieval path as the WebUI.",
            selected_tool="retrieval.search_local_objects",
            arguments={
                "query": cleaned,
                "target_kind": "file",
                "path_scope": cls._preferred_local_path_scope(user_text),
                "scope_mode": cls._preferred_local_scope_mode(user_text),
                "extensions": extensions,
                "query_terms": query_terms,
                "alias_terms": alias_terms,
                "top_k": 8,
                "rebuild_if_missing": True,
            },
            risk_level=RiskLevel.LOW,
            overall_task_goal=cls._compose_local_task_goal(
                user_text=user_text,
                default_summary=f"Find the local file matching {cleaned}.",
                overall_task_goal=overall_task_goal,
                include_candidates=True,
                task_classification=task_classification,
            ),
            expected_step_outputs=[OutputKind.OBJECT_CANDIDATES],
        )


    @classmethod
    def _preferred_local_path_scope(cls, user_text: str) -> str:
        resolved_scope = infer_scope_root(
            user_text,
            allow_default_fallback=False,
        )
        if resolved_scope:
            return resolved_scope
        return "."

    @classmethod
    def _preferred_local_scope_mode(cls, user_text: str) -> str:
        lowered = user_text.lower()
        if ("桌面" in user_text or "desktop" in lowered) and "testing" not in lowered and "测试" not in user_text:
            return "shallow_first"
        return "subtree"

    @classmethod
    def _looks_like_title_first_local_request(cls, user_text: str, query: str) -> bool:
        lowered = f"{user_text} {query}".lower()
        title_markers = (
            "模板",
            "template",
            "栏目",
            "结构",
            "年份",
            "year",
            "表格",
            "excel",
            "自测",
        )
        if any(marker in lowered for marker in title_markers):
            return True
        if cls._looks_like_image_request(user_text):
            query_terms = cls._tokenize_query(query)
            return any("." in term for term in query_terms) or any(re.search(r"\d{4,}", term) for term in query_terms)
        query_terms = cls._tokenize_query(query)
        year_context_markers = ("模板", "template", "汇报", "报告", "表格", "excel", "文档", "文件")
        return any(marker in lowered for marker in year_context_markers) and len(query_terms) <= 6 and any(
            term.isdigit() or re.fullmatch(r"20\d{2}", term) for term in query_terms
        )

    @classmethod
    def _should_keep_name_first_with_type_constraints(cls, user_text: str, query: str, extensions: list[str]) -> bool:
        if not extensions:
            return True
        if cls._looks_like_image_request(user_text):
            query_terms = cls._tokenize_query(query)
            return any("." in term for term in query_terms) or any(re.search(r"\d{4,}", term) for term in query_terms)
        return False

    @classmethod
    def _build_web_target_workflow(
        cls,
        user_text: str,
        completed_outputs: list[OutputKind],
        tool_results: list[ToolCallResult],
    ) -> ToolDecision | None:
        resolution = resolve_target_reference(user_text)
        if resolution.target_type != "web" or resolution.confidence < 0.55:
            return None

        target_value = (resolution.resolved_target or resolution.raw_target).strip()
        if not target_value:
            return None

        if resolution.action == "open":
            if OutputKind.PATH_OPENED in completed_outputs:
                return None
            if target_value.startswith(("http://", "https://")) and resolution.confidence >= 0.75:
                return ToolDecision(
                    decision=DecisionType.TOOL_CALL,
                    intent="open_web_page",
                    reason=f"Open the resolved website target directly: {target_value}.",
                    selected_tool="web.open_page",
                    arguments={"url": target_value},
                    risk_level=RiskLevel.LOW,
                    overall_task_goal=TaskGoal(
                        summary=f"Open the requested website {target_value}.",
                        required_outputs=[OutputKind.PATH_OPENED],
                    ),
                    expected_step_outputs=[OutputKind.PATH_OPENED],
                )

            top_result_url = cls._latest_web_search_result_url(tool_results)
            if top_result_url:
                return ToolDecision(
                    decision=DecisionType.TOOL_CALL,
                    intent="open_top_web_search_result",
                    reason="Open the top web search result for the resolved website request.",
                    selected_tool="web.open_page",
                    arguments={"url": top_result_url},
                    risk_level=RiskLevel.LOW,
                    overall_task_goal=TaskGoal(
                        summary=f"Search and open the website matching {target_value}.",
                        required_outputs=[OutputKind.SEARCH_RESULTS, OutputKind.PATH_OPENED],
                    ),
                    expected_step_outputs=[OutputKind.PATH_OPENED],
                )

            if OutputKind.SEARCH_RESULTS not in completed_outputs:
                return ToolDecision(
                    decision=DecisionType.TOOL_CALL,
                    intent="search_web_target",
                    reason="Research the web first so the agent can open the best matching public website.",
                    selected_tool="web.research",
                    arguments={"query": target_value, "max_results": 5, "max_pages": 2},
                    risk_level=RiskLevel.LOW,
                    overall_task_goal=TaskGoal(
                        summary=f"Search and open the website matching {target_value}.",
                        required_outputs=[OutputKind.SEARCH_RESULTS, OutputKind.PATH_OPENED],
                    ),
                    expected_step_outputs=[OutputKind.SEARCH_RESULTS, OutputKind.WEB_CONTENT],
                )
            return None

        if resolution.action == "search":
            if OutputKind.SEARCH_RESULTS in completed_outputs:
                return None
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="search_web_target",
                reason="Research the web for the resolved target.",
                selected_tool="web.research",
                arguments={"query": target_value, "max_results": 5, "max_pages": 2},
                risk_level=RiskLevel.LOW,
                overall_task_goal=TaskGoal(
                    summary=f"Search the web for {target_value}.",
                    required_outputs=[OutputKind.SEARCH_RESULTS],
                ),
                expected_step_outputs=[OutputKind.SEARCH_RESULTS, OutputKind.WEB_CONTENT],
            )

        if resolution.action == "research":
            if OutputKind.WEB_CONTENT in completed_outputs:
                return None
            if target_value.startswith(("http://", "https://")):
                return ToolDecision(
                    decision=DecisionType.TOOL_CALL,
                    intent="fetch_web_target",
                    reason="Fetch the resolved webpage directly before summarizing it.",
                    selected_tool="web.fetch",
                    arguments={"url": target_value, "max_chars": 4000, "prefer_browser": True},
                    risk_level=RiskLevel.LOW,
                    overall_task_goal=TaskGoal(
                        summary=f"Read the website content from {target_value}.",
                        required_outputs=[OutputKind.WEB_CONTENT],
                    ),
                    expected_step_outputs=[OutputKind.WEB_CONTENT],
                )
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="research_web_target",
                reason="Research the resolved web target before answering.",
                selected_tool="web.research",
                arguments={"query": target_value, "max_results": 5, "max_pages": 3, "prefer_browser": True},
                risk_level=RiskLevel.LOW,
                overall_task_goal=TaskGoal(
                    summary=f"Research the web target {target_value}.",
                    required_outputs=[OutputKind.SEARCH_RESULTS, OutputKind.WEB_CONTENT],
                ),
                expected_step_outputs=[OutputKind.SEARCH_RESULTS, OutputKind.WEB_CONTENT],
            )
        return None

    @staticmethod
    def _tokenize_query(text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{1,8}", text.lower())
        seen: set[str] = set()
        result: list[str] = []
        for token in tokens:
            normalized = token.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            normalized = str(value).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    @staticmethod
    def _is_code_like_path(path: str) -> bool:
        lowered = path.lower()
        return any(
            segment in lowered
            for segment in (
                "\\tests\\",
                "\\test\\",
                "\\tools\\",
                "\\responders\\",
                "\\confirmers\\",
                "\\planners\\",
                "\\agent\\",
                "\\capabilities\\",
                "\\core\\",
                "__pycache__",
            )
        )

    @staticmethod
    def _is_noise_like_path(path: str) -> bool:
        lowered = path.lower()
        return any(
            segment in lowered
            for segment in (
                "\\data\\",
                "debug",
                "trace",
                "smoke",
                "result",
                "results",
                "output",
                "fallback_hits",
                "questionnaire_hits",
                "travel_doc_hits",
            )
        )

    @classmethod
    def _query_looks_code_or_docs(cls, query: str, query_terms: list[str]) -> bool:
        lowered = query.lower()
        tokens = set(query_terms) | set(cls._tokenize_query(query))
        code_doc_tokens = {
            "代码",
            "python",
            "py",
            "markdown",
            "md",
            "文档",
            "doc",
            "docx",
            "readme",
            "planner",
            "router",
            "agent",
            "模块",
            "架构",
            "项目",
            "gpt",
            "sovits",
            "gpt-sovits",
        }
        return any(token in lowered or token in tokens for token in code_doc_tokens)

    @classmethod
    def _candidate_confidence_threshold(cls, overall_task_goal: TaskGoal | None) -> float:
        if overall_task_goal is not None and OutputKind.FILE_WRITTEN in overall_task_goal.required_outputs:
            return 0.6
        return 0.52

    @staticmethod
    def _looks_like_shallow_desktop_path(path: str) -> bool:
        try:
            target = Path(path).resolve()
            desktop = (Path.home() / "Desktop").resolve()
        except Exception:  # noqa: BLE001
            return False
        return target.parent == desktop or desktop in target.parents and len(target.relative_to(desktop).parts) <= 2

    @classmethod
    def _meaningful_query_terms(cls, candidate_state: CandidateState | None) -> list[str]:
        if candidate_state is None:
            return []
        stop_terms = {
            "帮我",
            "请帮我",
            "请",
            "读一下",
            "看一下",
            "看看",
            "那张",
            "那个",
            "这张",
            "这个",
            "上面写了什么",
            "里的文字",
            "文字",
            "提炼三条",
            "提炼三条重点",
            "提炼重点",
            "主要",
            "内容",
            "是什么",
        }
        terms: list[str] = []
        for term in candidate_state.query_terms:
            normalized = str(term).strip().lower()
            if not normalized or normalized in stop_terms:
                continue
            terms.append(normalized)
        return terms

    @classmethod
    def _top_candidate_matches_query(cls, candidate_state: CandidateState | None) -> bool:
        if candidate_state is None or not candidate_state.candidate_names:
            return False
        top_name = candidate_state.candidate_names[0].lower()
        terms = cls._meaningful_query_terms(candidate_state)
        if not terms:
            return False
        overlap = sum(1 for term in terms if cls._query_term_matches_candidate_name(term, top_name))
        if overlap >= 2:
            return True
        return overlap >= 1 and len(terms) <= 2

    @staticmethod
    def _query_term_matches_candidate_name(term: str, candidate_name: str) -> bool:
        normalized = str(term).strip().lower()
        if not normalized:
            return False
        if normalized in candidate_name:
            return True
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", normalized):
            stripped = normalized.strip("把帮请给找看读写讲说的了个那个这这份那份")
            if len(stripped) >= 3 and stripped in candidate_name:
                return True
            # Long Chinese chunks often include a leading verb from the request
            # because the tokenizer is intentionally lightweight.
            for start in range(1, min(3, len(normalized) - 2)):
                if normalized[start:] in candidate_name:
                    return True
        return False

    @classmethod
    def _candidate_quality_score(cls, candidate_state: CandidateState | None) -> float:
        if candidate_state is None or not candidate_state.candidate_paths:
            return 0.0
        score = float(candidate_state.confidence or 0.0) * 0.55
        score += float(candidate_state.top_score or 0.0) * 0.35
        score += min(float(candidate_state.score_gap or 0.0), 0.3) * 0.3
        if cls._top_candidate_matches_query(candidate_state):
            score += 0.12
        if candidate_state.source_tool == "retrieval.search_local_objects":
            score += 0.06
        if candidate_state.source_tool == "retrieval.inspect_local_candidate":
            score += 0.2
        return round(score, 4)

    @classmethod
    def _should_preserve_previous_candidates(
        cls,
        previous_state: CandidateState | None,
        new_state: CandidateState | None,
    ) -> bool:
        if previous_state is None or new_state is None:
            return False
        if not previous_state.candidate_paths or not new_state.candidate_paths:
            return False
        if previous_state.source_tool == "retrieval.inspect_local_candidate":
            return True
        if previous_state.source_tool != "retrieval.search_local_objects":
            return False
        if new_state.source_tool not in {"file.search_by_name", "file.search_text", "file.list"}:
            return False

        previous_quality = cls._candidate_quality_score(previous_state)
        new_quality = cls._candidate_quality_score(new_state)
        previous_query_match = cls._top_candidate_matches_query(previous_state)
        new_query_match = cls._top_candidate_matches_query(new_state)
        previous_top_path = previous_state.candidate_paths[0]
        new_top_path = new_state.candidate_paths[0]

        if previous_top_path == new_top_path:
            return False
        if previous_query_match and not new_query_match:
            return True
        if previous_quality >= new_quality + 0.08:
            return True
        if previous_quality >= 0.42 and cls._is_noise_like_path(new_top_path):
            return True
        return False

    @classmethod
    def _merge_candidate_states(
        cls,
        previous_state: CandidateState | None,
        new_state: CandidateState | None,
    ) -> CandidateState | None:
        if new_state is None:
            return previous_state
        if previous_state is None:
            return new_state
        if not cls._should_preserve_previous_candidates(previous_state, new_state):
            return new_state

        merged_paths: list[str] = []
        merged_names: list[str] = []
        for path, name in zip(previous_state.candidate_paths, previous_state.candidate_names):
            if path and path not in merged_paths:
                merged_paths.append(path)
                merged_names.append(name)
        for path, name in zip(new_state.candidate_paths, new_state.candidate_names):
            if path and path not in merged_paths:
                merged_paths.append(path)
                merged_names.append(name)

        return CandidateState(
            query=previous_state.query,
            target_kind=previous_state.target_kind,
            path_scope=previous_state.path_scope,
            query_terms=list(previous_state.query_terms),
            candidate_paths=merged_paths[:5],
            candidate_names=merged_names[:5],
            source_tool=previous_state.source_tool,
            confidence=previous_state.confidence,
            confidence_reason=previous_state.confidence_reason or "preserved_better_retrieval_candidates",
            top_score=previous_state.top_score,
            second_score=previous_state.second_score,
            score_gap=previous_state.score_gap,
            metadata=dict(previous_state.metadata),
        )

    @classmethod
    def _has_actionable_candidates(cls, candidate_state: CandidateState | None, overall_task_goal: TaskGoal | None) -> bool:
        if candidate_state is None or not candidate_state.candidate_paths:
            return False
        if cls._has_reliable_candidates(candidate_state, overall_task_goal):
            return True

        top_path = candidate_state.candidate_paths[0]
        top_score = float(candidate_state.top_score or 0.0)
        score_gap = float(candidate_state.score_gap or 0.0)
        confidence = float(candidate_state.confidence or 0.0)
        query_match = cls._top_candidate_matches_query(candidate_state)
        required_outputs = [] if overall_task_goal is None else list(overall_task_goal.required_outputs)

        if len(candidate_state.candidate_paths) == 1 and top_score >= 0.34 and confidence >= 0.3:
            return True
        if top_score >= 0.46 and score_gap >= 0.18 and confidence >= 0.34:
            return True
        if query_match and top_score >= 0.38 and score_gap >= 0.1 and confidence >= 0.32:
            return True
        if OutputKind.FILE_CONTENTS in required_outputs and cls._looks_like_document_path(top_path):
            if (
                query_match
                and top_score >= 0.18
                and confidence >= 0.18
                and cls._looks_like_shallow_desktop_path(top_path)
                and not cls._is_noise_like_path(top_path)
                and not cls._is_code_like_path(top_path)
            ):
                return True
            if top_score >= 0.4 and (score_gap >= 0.12 or query_match) and confidence >= 0.32:
                return True
            if (
                top_score >= 0.44
                and confidence >= 0.34
                and score_gap >= 0.04
                and not cls._is_noise_like_path(top_path)
                and not cls._is_code_like_path(top_path)
            ):
                return True
        if OutputKind.OBJECT_DETAILS in required_outputs and cls._looks_like_image_path(top_path):
            if top_score >= 0.4 and (score_gap >= 0.12 or query_match) and confidence >= 0.32:
                return True
            if (
                top_score >= 0.39
                and confidence >= 0.3
                and cls._looks_like_shallow_desktop_path(top_path)
                and not cls._is_noise_like_path(top_path)
            ):
                return True
        return False

    @classmethod
    def _has_reliable_candidates(cls, candidate_state: CandidateState | None, overall_task_goal: TaskGoal | None) -> bool:
        if candidate_state is None or not candidate_state.candidate_paths:
            return False
        return candidate_state.confidence >= cls._candidate_confidence_threshold(overall_task_goal)

    @staticmethod
    def _candidate_action_already_satisfied(completed_outputs: list[OutputKind]) -> bool:
        terminal_outputs = {
            OutputKind.FILE_CONTENTS,
            OutputKind.OBJECT_DETAILS,
            OutputKind.PATH_OPENED,
            OutputKind.FILE_WRITTEN,
            OutputKind.PATH_CREATED,
            OutputKind.PATH_UPDATED,
            OutputKind.PATH_DELETED,
        }
        return any(output_kind in completed_outputs for output_kind in terminal_outputs)

    @classmethod
    def _should_offer_candidate_selection(
        cls,
        user_text: str,
        candidate_state: CandidateState | None,
        overall_task_goal: TaskGoal | None,
        completed_outputs: list[OutputKind],
    ) -> bool:
        if candidate_state is None:
            return False
        if len(candidate_state.candidate_paths) < 2:
            return False
        if cls._has_actionable_candidates(candidate_state, overall_task_goal):
            return False
        if candidate_state.confidence < 0.18:
            return False
        if cls._candidate_action_already_satisfied(completed_outputs):
            return False
        if candidate_state.source_tool not in {
            "file.search_by_name",
            "retrieval.search_local_objects",
            "file.search_text",
            "file.list",
        }:
            return False
        action = cls._infer_candidate_action(user_text)
        required_outputs = [] if overall_task_goal is None else list(overall_task_goal.required_outputs)
        if OutputKind.MESSAGE_SENT in required_outputs and OutputKind.MESSAGE_SENT not in completed_outputs:
            return True
        if action in {
            "open",
            "reveal",
            "metadata",
            "preview",
            "read",
            "describe_image",
            "read_image_text",
            "extract_document_structure",
            "search_document_blocks",
        }:
            return True
        return cls._looks_like_document_summary_request(user_text)

    @classmethod
    def _build_candidate_selection_pending(
        cls,
        *,
        user_text: str,
        candidate_state: CandidateState,
        overall_task_goal: TaskGoal | None,
    ) -> PendingTask:
        candidates: list[SelectionCandidate] = []
        for index, path in enumerate(candidate_state.candidate_paths[:4]):
            if not path:
                continue
            name = candidate_state.candidate_names[index] if index < len(candidate_state.candidate_names) else Path(path).name
            subtitle = str(Path(path).parent)
            candidates.append(
                SelectionCandidate(
                    candidate_id=f"candidate_{index + 1}",
                    path=path,
                    name=name or Path(path).name,
                    kind=candidate_state.target_kind or "file",
                    subtitle=subtitle,
                )
            )
        type_constraints = candidate_state.metadata.get("type_constraints", [])
        return PendingTask(
            task_id=f"pending_{uuid.uuid4().hex[:10]}",
            intent="select_candidate",
            summary="Select the best-matching file candidate before continuing the task.",
            original_user_request=user_text,
            state_kind="task_follow_up",
            clarification_prompt="",
            selection_candidates=candidates,
            overall_task_goal=overall_task_goal,
            missing_slots=["selected_candidate_path"],
            collected_slots={
                "query": candidate_state.query,
                "path_scope": candidate_state.path_scope,
                "type_constraints": ",".join(str(item) for item in type_constraints) if isinstance(type_constraints, list) else "",
                "source_tool": candidate_state.source_tool,
                "confidence_reason": candidate_state.confidence_reason,
            },
            resume_hint="",
        )

    @classmethod
    def _derive_candidate_state(
        cls,
        decision: ToolDecision,
        result: ToolCallResult,
        previous_state: CandidateState | None,
    ) -> CandidateState | None:
        if result.status != "success" or decision.decision != DecisionType.TOOL_CALL:
            return previous_state

        tool_name = decision.selected_tool or ""
        if tool_name in {"retrieval.search_local_objects", "file.search_by_name"}:
            candidates = result.data.get("candidates", [])
            query = str(decision.arguments.get("query", ""))
            query_terms = cls._tokenize_query(query)
            top_score = float(candidates[0].get("score", 0.0)) if candidates else 0.0
            second_score = float(candidates[1].get("score", 0.0)) if len(candidates) > 1 else 0.0
            confidence = top_score + min(max(top_score - second_score, 0.0), 0.15)
            confidence_reason = "retrieval_top_score" if tool_name == "retrieval.search_local_objects" else "name_search_top_score"
            if candidates:
                top_candidate_path = str(candidates[0].get("path", ""))
                top_sources = candidates[0].get("score_sources", {}) or {}
                grounded_signal = float(top_sources.get("fts", 0.0)) + float(top_sources.get("name", 0.0)) + float(top_sources.get("intent", 0.0)) + float(top_sources.get("folder", 0.0))
                if tool_name == "retrieval.search_local_objects" and grounded_signal <= 0 and float(top_sources.get("embedding", 0.0)) > 0:
                    confidence *= 0.72
                    confidence_reason = "embedding_only_candidate"
                if (
                    str(decision.arguments.get("target_kind", "any")) == "folder"
                    and cls._is_code_like_path(top_candidate_path)
                    and not cls._query_looks_code_or_docs(query, query_terms)
                ):
                    confidence *= 0.45
                    confidence_reason = "code_like_folder_for_non_code_query"
                if cls._is_noise_like_path(top_candidate_path):
                    confidence *= 0.65
                    confidence_reason = "noise_like_top_candidate"
            metadata = {
                "path_scope": str(decision.arguments.get("path_scope", decision.arguments.get("path", "."))),
                "scope_mode": str(decision.arguments.get("scope_mode", "subtree")),
                "type_constraints": list(decision.arguments.get("extensions", []) or []),
                "alias_terms": list(decision.arguments.get("alias_terms", []) or []),
                "candidate_match_reasons": [
                    str(candidate.get("match_reason", "") or "") for candidate in candidates[:5]
                ],
                "top_match_reason": str(candidates[0].get("match_reason", "") or "") if candidates else "",
            }
            next_state = CandidateState(
                query=query,
                target_kind=str(decision.arguments.get("target_kind", "any")),
                path_scope=str(decision.arguments.get("path_scope", decision.arguments.get("path", "."))),
                query_terms=list(decision.arguments.get("query_terms", []) or query_terms),
                candidate_paths=[candidate.get("path", "") for candidate in candidates if candidate.get("path")][:5],
                candidate_names=[candidate.get("name", "") for candidate in candidates if candidate.get("name")][:5],
                workflow_stage="candidate_ready" if candidates else "searching",
                source_tool=tool_name,
                confidence=round(max(0.0, min(1.0, confidence if candidates else 0.0)), 4),
                confidence_reason=confidence_reason if candidates else "no_candidates",
                top_score=round(top_score, 4) if candidates else 0.0,
                second_score=round(second_score, 4) if len(candidates) > 1 else 0.0,
                score_gap=round(max(top_score - second_score, 0.0), 4) if candidates else 0.0,
                metadata=metadata,
            )
            return cls._merge_candidate_states(previous_state, next_state)

        if tool_name == "file.search_text":
            query_terms = list(previous_state.query_terms if previous_state is not None else [])
            if not query_terms:
                query_terms = cls._tokenize_query(" ".join(str(term) for term in decision.arguments.get("terms", [])))
            paths: list[str] = []
            for match in result.data.get("matches", []):
                path = match.get("path")
                if isinstance(path, str) and path and path not in paths:
                    paths.append(path)
            return CandidateState(
                query=previous_state.query if previous_state is not None else str(decision.arguments.get("query", "")),
                target_kind=previous_state.target_kind if previous_state is not None else "file",
                path_scope=str(decision.arguments.get("path", previous_state.path_scope if previous_state else ".")),
                query_terms=query_terms,
                candidate_paths=paths[:5],
                candidate_names=[Path(path).name for path in paths[:5]],
                workflow_stage="candidate_ready" if paths else "searching",
                source_tool=tool_name,
                confidence=0.72 if paths else 0.0,
                confidence_reason="search_text_matches" if paths else "no_matches",
                top_score=0.72 if paths else 0.0,
                second_score=0.0,
                score_gap=0.72 if paths else 0.0,
            )

        if tool_name == "file.list":
            entries = result.data.get("entries", [])
            path_scope = str(decision.arguments.get("path", previous_state.path_scope if previous_state else "."))
            target_kind = previous_state.target_kind if previous_state is not None else "any"
            query_terms = list(previous_state.query_terms if previous_state is not None else [])
            if not query_terms:
                query_terms = list(decision.arguments.get("query_terms", []) or [])
            if not query_terms:
                query_terms = cls._tokenize_query(str(decision.arguments.get("query", "") or ""))
            ranked_entries: list[tuple[float, dict]] = []
            for entry in entries:
                if target_kind == "folder" and not entry.get("is_dir"):
                    continue
                if target_kind == "file" and entry.get("is_dir"):
                    continue
                candidate_text = f"{entry.get('name', '')} {entry.get('path', '')}".lower()
                overlap = sum(1 for term in query_terms if term and term in candidate_text)
                if query_terms:
                    score = overlap / len(query_terms)
                else:
                    score = 0.0
                if target_kind == "folder" and entry.get("is_dir"):
                    score += 0.1
                if target_kind == "file" and not entry.get("is_dir"):
                    score += 0.05
                if score > 0:
                    ranked_entries.append((score, entry))
            ranked_entries.sort(key=lambda item: item[0], reverse=True)
            best_entries = [entry for _, entry in ranked_entries[:5]]
            return CandidateState(
                query=previous_state.query if previous_state is not None else str(decision.arguments.get("query", "") or ""),
                target_kind=target_kind,
                path_scope=path_scope,
                query_terms=query_terms,
                candidate_paths=[entry.get("path", "") for entry in best_entries if entry.get("path")],
                candidate_names=[entry.get("name", "") for entry in best_entries if entry.get("name")],
                workflow_stage="candidate_ready" if best_entries else "searching",
                source_tool=tool_name,
                confidence=ranked_entries[0][0] if ranked_entries else 0.0,
                confidence_reason="directory_name_overlap" if ranked_entries else "no_ranked_entries",
                top_score=ranked_entries[0][0] if ranked_entries else 0.0,
                second_score=ranked_entries[1][0] if len(ranked_entries) > 1 else 0.0,
                score_gap=max(ranked_entries[0][0] - ranked_entries[1][0], 0.0) if len(ranked_entries) > 1 else (ranked_entries[0][0] if ranked_entries else 0.0),
            )

        if tool_name == "retrieval.inspect_local_candidate":
            path = result.data.get("path")
            if isinstance(path, str) and path:
                return CandidateState(
                    query=previous_state.query if previous_state is not None else "",
                    target_kind=previous_state.target_kind if previous_state is not None else str(result.data.get("object_kind", "any")),
                    path_scope=previous_state.path_scope if previous_state is not None else ".",
                    query_terms=list(previous_state.query_terms if previous_state is not None else []),
                    candidate_paths=[path],
                    candidate_names=[Path(path).name],
                    workflow_stage="candidate_ready",
                    source_tool=tool_name,
                    confidence=1.0,
                    confidence_reason="inspected_candidate",
                    top_score=1.0,
                    second_score=0.0,
                    score_gap=1.0,
                )
        return previous_state

    @staticmethod
    def _extract_requested_output_file(user_text: str) -> str | None:
        match = re.search(r"([A-Za-z0-9_.\\/-]+\.(txt|md|json|csv|yaml|yml|docx|xlsx|pptx))", user_text)
        if match:
            return match.group(1)
        return None

    @classmethod
    def _resolve_requested_output_file(cls, user_text: str, delivery_intent) -> str | None:
        explicit_output = None if delivery_intent is None else getattr(delivery_intent, "output_file", None)
        if isinstance(explicit_output, str) and explicit_output.strip():
            return explicit_output.strip()
        fallback = cls._extract_requested_output_file(user_text)
        if fallback:
            return fallback
        if delivery_intent is None or not getattr(delivery_intent, "save_output", False):
            return None
        return cls._default_output_filename(delivery_intent)

    @staticmethod
    def _default_output_filename(delivery_intent) -> str:
        raw_title = getattr(delivery_intent, "title", None) or "agent-output"
        output_format = getattr(delivery_intent, "output_format", None) or "docx"
        safe_title = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff._-]+", "-", str(raw_title)).strip("-._") or "agent-output"
        return f"{safe_title}.{output_format}"

    @classmethod
    def _build_write_arguments_for_output(cls, *, output_file: str, content: str, delivery_intent) -> tuple[str, dict]:
        return OutputArtifactPlanner.build_write_arguments(
            output_file=output_file,
            content=content,
            delivery_intent=delivery_intent,
        )

    @classmethod
    def _build_candidate_write_followup(
        cls,
        user_text: str,
        overall_task_goal: TaskGoal | None,
        completed_outputs: list[OutputKind],
        candidate_state: CandidateState | None,
    ) -> ToolDecision | None:
        if overall_task_goal is None or OutputKind.FILE_WRITTEN not in overall_task_goal.required_outputs:
            return None
        if OutputKind.FILE_WRITTEN in completed_outputs:
            return None
        if not cls._has_reliable_candidates(candidate_state, overall_task_goal):
            return None

        output_file = cls._extract_requested_output_file(user_text)
        if not output_file:
            return None

        content = "\n".join(candidate_state.candidate_paths)
        if not content.strip():
            return None

        return ToolDecision(
            decision=DecisionType.TOOL_CALL,
            intent="write_candidate_paths",
            reason="Write the grounded candidate paths collected so far into the requested output file.",
            selected_tool="file.write",
            arguments={"path": output_file, "content": content, "overwrite": True},
            risk_level=RiskLevel.LOW,
            overall_task_goal=overall_task_goal,
            expected_step_outputs=[OutputKind.FILE_WRITTEN],
        )

    @classmethod
    def _build_docx_edit_followup(
        cls,
        user_text: str,
        overall_task_goal: TaskGoal | None,
        completed_outputs: list[OutputKind],
        candidate_state: CandidateState | None,
    ) -> ToolDecision | None:
        if overall_task_goal is None or OutputKind.FILE_WRITTEN not in overall_task_goal.required_outputs:
            return None
        if OutputKind.FILE_WRITTEN in completed_outputs:
            return None
        if candidate_state is None or not candidate_state.candidate_paths:
            return None
        source_path = next((path for path in candidate_state.candidate_paths if Path(path).suffix.lower() == ".docx"), None)
        if not source_path:
            return None
        return cls._build_document_agent_edit_followup(
            user_text=user_text,
            path=source_path,
            overall_task_goal=overall_task_goal,
            reason="Fallback: delegate the grounded docx append request to the document sub-agent.",
        )

    @staticmethod
    def _latest_web_research_bundle(tool_results: list[ToolCallResult]) -> dict[str, Any] | None:
        for result in reversed(tool_results):
            if result.status == "success" and result.tool_name in {"web.research", "web.fetch"}:
                return result.data
        return None

    @staticmethod
    def _latest_web_write_content(tool_results: list[ToolCallResult]) -> str:
        for result in reversed(tool_results):
            if result.status != "success":
                continue
            if result.tool_name == "web.research":
                content = result.data.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                snippets: list[str] = []
                for item in result.data.get("sources", [])[:3]:
                    if not isinstance(item, dict):
                        continue
                    title = str(item.get("title", "")).strip()
                    url = str(item.get("url", "")).strip()
                    excerpt = str(item.get("excerpt", "") or item.get("content", "")).strip()
                    line = "\n".join(part for part in (title, url, excerpt) if part)
                    if line:
                        snippets.append(line)
                if snippets:
                    return "\n\n".join(snippets)
            if result.tool_name == "web.fetch":
                content = result.data.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        return ""

    def _compose_web_write_content(
        self,
        *,
        user_text: str,
        tool_results: list[ToolCallResult],
        delivery_intent,
        recent_context: str = "",
    ) -> tuple[str, str | None]:
        fallback_content = self._latest_web_write_content(tool_results)
        bundle = self._latest_web_research_bundle(tool_results)
        if bundle is None or not hasattr(self.llm_client, "compose_web_research_document"):
            return fallback_content, None
        try:
            composed = self.llm_client.compose_web_research_document(
                user_text=user_text,
                title=None if delivery_intent is None else getattr(delivery_intent, "title", None),
                research_bundle=bundle,
                recent_context=recent_context,
                hot_context_summary=self.hot_context_summary,
                warm_memory_summary=self.warm_memory_summary,
                cold_memory_summary=self.cold_memory_summary,
                active_task_summary=self.active_task_summary,
            )
        except Exception:  # noqa: BLE001
            return fallback_content, None
        content = str(composed.get("content") or "").strip()
        title = str(composed.get("title") or "").strip() or None
        return (content or fallback_content), title

    def _build_web_write_followup(
        self,
        user_text: str,
        overall_task_goal: TaskGoal | None,
        completed_outputs: list[OutputKind],
        tool_results: list[ToolCallResult],
        delivery_intent,
        recent_context: str = "",
    ) -> ToolDecision | None:
        if overall_task_goal is None or OutputKind.FILE_WRITTEN not in overall_task_goal.required_outputs:
            return None
        if OutputKind.FILE_WRITTEN in completed_outputs:
            return None

        output_file = self._resolve_requested_output_file(user_text, delivery_intent)
        if not output_file:
            return None

        content, composed_title = self._compose_web_write_content(
            user_text=user_text,
            tool_results=tool_results,
            delivery_intent=delivery_intent,
            recent_context=recent_context,
        )
        if not content.strip():
            return None

        tool_name, arguments = self._build_write_arguments_for_output(
            output_file=output_file,
            content=content,
            delivery_intent=delivery_intent,
        )
        if composed_title and "title" in arguments:
            arguments["title"] = composed_title

        return ToolDecision(
            decision=DecisionType.TOOL_CALL,
            intent="write_web_research_result",
            reason="Write the grounded web research content into the requested output file.",
            selected_tool=tool_name,
            arguments=arguments,
            risk_level=RiskLevel.LOW,
            overall_task_goal=overall_task_goal,
            expected_step_outputs=[OutputKind.FILE_WRITTEN],
        )

    def _build_scheduled_task_workflow(
            self,
            *,
            user_text: str,
            task_classification,
            session_id: str | None = None,
            channel: str | None = None,
            channel_runtime: dict[str, Any] | None = None,
    ) -> ToolDecision | None:
        if task_classification is None:
            return None

        run_mode = str(getattr(task_classification, "run_mode", "immediate") or "immediate").strip().lower()
        if run_mode != "scheduled":
            return None

        scheduled_plan = self.llm_client.plan_scheduled_task_arguments(
            user_text=user_text,
            task_classification=task_classification.model_dump(mode="json"),
            current_time_iso=datetime.now().astimezone().isoformat(),
            timezone="Asia/Shanghai",
            recent_context=self._recent_conversation_text(),
            hot_context_summary=self.hot_context_summary,
            warm_memory_summary=self.warm_memory_summary,
            cold_memory_summary=self.cold_memory_summary,
            active_task_summary=self.active_task_summary,
        )

        if str(scheduled_plan.get("run_mode", "immediate") or "immediate").strip().lower() != "scheduled":
            return None

        task_type = str(scheduled_plan.get("task_type", "notify") or "notify").strip().lower()
        if task_type not in {"notify", "deferred_agent_task"}:
            task_type = "notify"

        when_iso = str(scheduled_plan.get("when_iso", "") or "").strip()
        timezone = str(scheduled_plan.get("timezone", "Asia/Shanghai") or "Asia/Shanghai").strip()
        message = str(scheduled_plan.get("message", "") or "").strip()

        task_payload = scheduled_plan.get("task_payload", {})
        if not isinstance(task_payload, dict):
            task_payload = {}

        if channel_runtime:
            task_payload = {
                **task_payload,
                "channel_runtime": channel_runtime,
            }

        arguments = {
            "task_type": task_type,
            "when_iso": when_iso,
            "timezone": timezone,
            "session_id": session_id,
            "channel": channel,
            "message": message,
            "task_payload": task_payload,
        }

        return ToolDecision(
            decision=DecisionType.TOOL_CALL,
            intent="create_scheduled_task",
            reason="The request is a scheduled future task, so create a deferred scheduled task instead of executing immediately.",
            selected_tool="system.create_scheduled_task",
            arguments=arguments,
            risk_level=RiskLevel.LOW,
            overall_task_goal=self._build_scheduled_task_goal(
                user_text=user_text,
                task_type=task_type,
            ),
            expected_step_outputs=[],
        )

    def _build_fired_scheduled_task_delivery_workflow(
        self,
        *,
        runtime_channel: str | None = None,
        runtime_channel_context: dict[str, Any] | None = None,
    ) -> ToolDecision | None:
        fire_context = self._extract_scheduled_task_fire_context(runtime_channel_context)
        if fire_context is None or fire_context.get("phase") != "fired":
            return None
        if fire_context.get("task_type") != "notify":
            return None
        if str(runtime_channel or "").strip() != "onebot_v11":
            return None
        if not self.registry.has_tool("qq.send_text"):
            return None

        message = str(fire_context.get("message") or "").strip()
        if not message:
            return None

        return ToolDecision(
            decision=DecisionType.TOOL_CALL,
            intent="deliver_fired_scheduled_notification",
            reason="This scheduled QQ notification has already fired, so deliver its prepared text to the current session now.",
            selected_tool="qq.send_text",
            arguments={
                "message": message,
                "target_kind": "current",
            },
            risk_level=RiskLevel.LOW,
            overall_task_goal=TaskGoal(
                summary="Deliver the fired scheduled QQ notification to the current session.",
                required_outputs=[OutputKind.MESSAGE_SENT],
                completion_mode="success",
            ),
            expected_step_outputs=[OutputKind.MESSAGE_SENT],
        )

    def handle_user_input(
            self,
            user_text: str,
            progress_callback: Callable[[str, str, dict], None] | None = None,
            seed_candidate_state: CandidateState | None = None,
            seed_overall_task_goal: TaskGoal | None = None,
            seed_workflow_state: WorkflowState | None = None,
            runtime_session_id: str | None = None,
            runtime_channel: str | None = None,
            runtime_channel_context: dict[str, Any] | None = None,
    ) -> TurnArtifacts:
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        session_id = f"session_{uuid.uuid4().hex[:10]}"
        artifacts = TurnArtifacts(trace_id=trace_id)
        tool_results: list[ToolCallResult] = []
        request_signatures: list[str] = []
        active_overall_goal: TaskGoal | None = seed_overall_task_goal
        active_candidate_state: CandidateState | None = seed_candidate_state
        observed_workflow_state: WorkflowState | None = seed_workflow_state
        completed_outputs: list[OutputKind] = []
        loop_stop_reason: str | None = None
        last_executed_decision: ToolDecision | None = None
        locked_workflow_family: str | None = None
        locked_workflow_nodes: list[dict[str, object]] = []
        locked_workflow_spec: WorkflowSpec | None = None
        locked_workflow_index = 0
        planner_invocations = 0
        critic_invocations = 0
        planner_bypass_count = 0
        decision_path: list[dict[str, object]] = []
        self.history.append(Message(role=Role.USER, content=user_text))
        recent_context = self._recent_conversation_text()
        channel_context_summary = self._build_channel_context_summary(runtime_channel, runtime_channel_context)
        intent_bundle = self.intent_service.analyze(
            user_text,
            recent_context=recent_context,
            hot_context_summary=self.hot_context_summary,
            warm_memory_summary=self.warm_memory_summary,
            learning_memory_summary=self.learning_memory_summary,
            cold_memory_summary=self.cold_memory_summary,
            active_task_summary=self.active_task_summary,
            channel_context_summary=channel_context_summary,
        )
        task_graph = getattr(intent_bundle, "task_graph", None)
        planner_user_text = (
            str(getattr(task_graph, "primary_task_text", "") or "").strip()
            or str(getattr(intent_bundle.task_envelope, "planning_focus_text", "") or "").strip()
            or user_text
        )
        document_delivery_intent = intent_bundle.document_delivery
        knowledge_request_intent = intent_bundle.knowledge_request
        site_search_intent = intent_bundle.site_search
        local_collection_request = self.local_collection_workflow.parse_request(
            planner_user_text,
            knowledge_request_intent,
            task_classification=intent_bundle.task_classification,
            recent_context=recent_context,
        )

        scheduled_task_decision = None
        if intent_bundle.task_classification is not None:
            if self._should_schedule_request(
                intent_bundle.task_classification,
                runtime_channel_context=runtime_channel_context,
            ):
                try:
                    scheduled_task_decision = self._build_scheduled_task_workflow(
                        user_text=user_text,
                        task_classification=intent_bundle.task_classification,
                        session_id=runtime_session_id,
                        channel=runtime_channel,
                        channel_runtime=runtime_channel_context,
                    )
                except Exception as exc:  # noqa: BLE001
                    self.trace_store.append(
                        "scheduled_task_planning_error",
                        {
                            "trace_id": trace_id,
                            "error": str(exc),
                        },
                    )
                    scheduled_task_decision = None


        scripted_planner_decision_count = self._scripted_planner_decision_count()
        scripted_planner_decisions_used = 0

        self.trace_store.append("user_input", {"trace_id": trace_id, "text": user_text, "planner_text": planner_user_text})
        self.trace_store.append(
            "intent_context",
            {
                "trace_id": trace_id,
                "recent_context": recent_context,
                "hot_context_summary": self.hot_context_summary,
                "user_memory_summary": self.user_memory_summary,
                "learning_memory_summary": self.learning_memory_summary,
                "warm_memory_summary": self.warm_memory_summary,
                "cold_memory_summary": self.cold_memory_summary,
                "active_task_summary": self.active_task_summary,
                "channel_context_summary": channel_context_summary,
            },
        )
        self.trace_store.append(
            "task_graph_intent",
            {
                "trace_id": trace_id,
                "intent": None if task_graph is None else task_graph.model_dump(mode="json"),
                "planner_text": planner_user_text,
            },
        )
        self.trace_store.append(
            "document_delivery_intent",
            {"trace_id": trace_id, "intent": document_delivery_intent.model_dump(mode="json")},
        )
        self.trace_store.append(
            "knowledge_request_intent",
            {"trace_id": trace_id, "intent": knowledge_request_intent.model_dump(mode="json")},
        )
        self.trace_store.append(
            "site_search_intent",
            {"trace_id": trace_id, "intent": site_search_intent.model_dump(mode="json")},
        )
        self.trace_store.append(
            "memory_candidate_intent",
            {
                "trace_id": trace_id,
                "intent": getattr(intent_bundle, "memory_candidate_intent", None).model_dump(mode="json")
                if getattr(intent_bundle, "memory_candidate_intent", None) is not None
                else None,
            },
        )
        self.trace_store.append(
            "task_classification",
            {
                "trace_id": trace_id,
                "intent": None
                if intent_bundle.task_classification is None
                else intent_bundle.task_classification.model_dump(mode="json"),
            },
        )
        self.trace_store.append(
            "answerability",
            {
                "trace_id": trace_id,
                "intent": getattr(intent_bundle, "answerability", None).model_dump(mode="json")
                if getattr(intent_bundle, "answerability", None) is not None
                else None,
            },
        )
        self.trace_store.append(
            "task_envelope",
            {
                "trace_id": trace_id,
                "intent": getattr(intent_bundle, "task_envelope", None).model_dump(mode="json")
                if getattr(intent_bundle, "task_envelope", None) is not None
                else None,
            },
        )
        self.trace_store.append(
            "local_collection_request",
            {
                "trace_id": trace_id,
                "intent": None if local_collection_request is None else local_collection_request.model_dump(mode="json"),
            },
        )
        locked_workflow_spec = self._plan_initial_workflow_spec(
            user_text=planner_user_text,
            intent_bundle=intent_bundle,
            recent_context=recent_context,
            trace_id=trace_id,
        )
        if locked_workflow_spec is not None:
            locked_workflow_family = "llm_workflow_spec"
            if locked_workflow_spec.goal is not None:
                active_overall_goal = locked_workflow_spec.goal
            self.trace_store.append(
                "workflow_lock_initialized",
                {
                    "trace_id": trace_id,
                    "workflow_family": locked_workflow_family,
                    "source": "llm_workflow_spec",
                    "workflow_spec": locked_workflow_spec.model_dump(mode="json"),
                },
            )
        self._emit_progress(progress_callback, "received", "我先记下你的需求，准备开始分析。", {"trace_id": trace_id})

        try:
            for step in range(self.config.max_steps):
                observations = ContextBuilder.build_observations(tool_results)

                if step == 0 and scheduled_task_decision is not None:
                    decision = scheduled_task_decision
                    artifacts.decision = decision
                    active_overall_goal = decision.overall_task_goal
                    artifacts.overall_task_goal = active_overall_goal
                    critic_invocations += 1
                    review = self.critic.review(
                        messages=self._context_messages(),
                        decision=decision,
                        tool_manifests=self.registry.list_manifests(),
                        observations=observations,
                    )
                    self.trace_store.append(
                        "decision_review",
                        {
                            "trace_id": trace_id,
                            "step": step,
                            "source": "scheduled_task_short_circuit",
                            "review": review.model_dump(mode="json"),
                        },
                    )
                    if not review.approved and review.suggested_decision is None:
                        raise ValueError(f"Scheduled task review rejected planner output: {review.summary or review.issues}")
                    decision = self._resolve_reviewed_decision(decision, review)
                    self.validator.validate(decision)
                    self.validator.validate_against_task_state(
                        decision,
                        overall_task_goal=active_overall_goal,
                        completed_outputs=completed_outputs,
                    )
                    self.guardrails.validate(decision)

                    self.trace_store.append(
                        "decision_effective",
                        {
                            "trace_id": trace_id,
                            "step": step,
                            "source": "scheduled_task_short_circuit",
                            "decision": decision.model_dump(mode="json"),
                            "active_overall_goal": None
                            if active_overall_goal is None
                            else active_overall_goal.model_dump(mode="json"),
                        },
                    )

                    request = self.registry.build_request(
                        trace_id=trace_id,
                        session_id=session_id,
                        tool_name=decision.selected_tool or "",
                        arguments=decision.arguments,
                    )
                    request_signatures.append(self.loop_controller.request_signature(decision))
                    self.trace_store.append(
                        "tool_request",
                        {"trace_id": trace_id, "step": step, "request": request.model_dump(mode="json")},
                    )

                    result = self.registry.execute(request)
                    last_executed_decision = decision
                    tool_results.append(result)
                    artifacts.tool_results.append(result)

                    if result.status == "success" and decision.selected_tool:
                        manifest = self.registry.get_manifest(decision.selected_tool)
                        effective_outputs = self.completion_judge.resolve_effective_outputs(
                            tool_name=decision.selected_tool,
                            produced_outputs=list(manifest.produces),
                            result=result,
                        )
                        for output_name in effective_outputs:
                            if output_name not in completed_outputs:
                                completed_outputs.append(output_name)

                    artifacts.completed_outputs = list(completed_outputs)
                    self.history.append(
                        Message(role=Role.TOOL, content=f"{decision.selected_tool}: {result.model_dump(mode='json')}")
                    )
                    self.trace_store.append(
                        "tool_result",
                        {"trace_id": trace_id, "step": step, "result": result.model_dump(mode="json")},
                    )

                    completion = self.completion_judge.assess(
                        overall_task_goal=active_overall_goal,
                        completed_outputs=completed_outputs,
                        tool_results=tool_results,
                    )
                    completion, _, completion_review = self._review_execution_state(
                        user_text=user_text,
                        overall_task_goal=active_overall_goal,
                        completed_outputs=completed_outputs,
                        tool_results=tool_results,
                        candidate_state=active_candidate_state,
                        completion=completion,
                        stop_reason=loop_stop_reason,
                        document_delivery_intent=document_delivery_intent,
                        knowledge_request_intent=knowledge_request_intent,
                        site_search_intent=site_search_intent,
                        task_classification=intent_bundle.task_classification,
                    )
                    self.trace_store.append(
                        "completion_check",
                        {
                            "trace_id": trace_id,
                            "step": step,
                            "assessment": completion.model_dump(mode="json"),
                            "review": completion_review.model_dump(mode="json"),
                        },
                    )
                    if completion.done:
                        loop_stop_reason = "completion_judge"
                        break

                self._emit_progress(
                    progress_callback,
                    "planning",
                    f"第 {step + 1} 步：我在判断下一步该直接回答，还是去调用工具。",
                    {"step": step},
                )

                if step == 0:
                    task_graph_pending = self._build_task_graph_pending_task(
                        user_text=user_text,
                        task_graph=task_graph,
                        overall_task_goal=active_overall_goal,
                    )
                    if task_graph_pending is not None:
                        artifacts.pending_task = task_graph_pending
                        loop_stop_reason = "task_graph_waiting_for_input"
                        self.trace_store.append(
                            "task_graph_pending",
                            {
                                "trace_id": trace_id,
                                "step": step,
                                "pending_task": task_graph_pending.model_dump(mode="json"),
                            },
                        )
                        break

                fallback_decision = None
                scripted_planner_has_pending_decision = scripted_planner_decisions_used < scripted_planner_decision_count
                allow_programmatic_lookup_workflow = (
                    scripted_planner_decision_count == 0
                    and str(getattr(knowledge_request_intent, "knowledge_type", "") or "") != "qq_history"
                )
                workflow_binding = self._bind_workflow_plan(
                    user_text=planner_user_text,
                    intent_bundle=intent_bundle,
                    overall_task_goal=active_overall_goal,
                    candidate_state=active_candidate_state,
                    completed_outputs=completed_outputs,
                    tool_results=tool_results,
                    recent_context=recent_context,
                    runtime_channel=runtime_channel,
                    runtime_channel_context=runtime_channel_context,
                )

                bound_workflow_family = str(workflow_binding.get("workflow_family") or "generic")
                bound_workflow_decision = workflow_binding.get("workflow_decision")
                bound_goal = workflow_binding.get("overall_task_goal")
                if (
                    locked_workflow_family is None
                    and isinstance(bound_workflow_decision, ToolDecision)
                    and bound_workflow_family != "generic"
                    and self._should_use_locked_workflow_decision(
                        bound_workflow_decision,
                        workflow_family=bound_workflow_family,
                        request_signatures=set(request_signatures),
                        request_signature=self.loop_controller.request_signature,
                    )
                ):
                    locked_workflow_family = bound_workflow_family
                    self.trace_store.append(
                        "workflow_lock_initialized",
                        {
                            "trace_id": trace_id,
                            "workflow_family": locked_workflow_family,
                            "goal": None if bound_goal is None else bound_goal.model_dump(mode="json"),
                            "first_node": {
                                "tool": bound_workflow_decision.selected_tool,
                                "intent": bound_workflow_decision.intent,
                                "expected_outputs": [item.value for item in bound_workflow_decision.expected_step_outputs],
                            },
                        },
                    )
                if locked_workflow_family is not None:
                    bound_workflow_family = locked_workflow_family
                workflow_spec_response_ready = False
                if locked_workflow_spec is not None:
                    completed_set = set(completed_outputs)
                    workflow_spec_decision: ToolDecision | None = None
                    while locked_workflow_index < len(locked_workflow_spec.nodes):
                        node = locked_workflow_spec.nodes[locked_workflow_index]
                        if node.tool is None:
                            workflow_spec_response_ready = True
                            locked_workflow_index = len(locked_workflow_spec.nodes)
                            break
                        if node.produces and all(output in completed_set for output in node.produces):
                            locked_workflow_index += 1
                            continue
                        workflow_spec_decision = self._workflow_spec_node_to_decision(
                            spec=locked_workflow_spec,
                            node=node,
                        )
                        locked_workflow_index += 1
                        break
                    if workflow_spec_response_ready:
                        self.trace_store.append(
                            "loop_stop",
                            {
                                "trace_id": trace_id,
                                "step": step,
                                "reason": "workflow_spec_response_node",
                                "details": "Locked workflow reached its response node.",
                            },
                        )
                        loop_stop_reason = "workflow_spec_response_node"
                        break
                    if workflow_spec_decision is not None:
                        bound_workflow_decision = workflow_spec_decision
                        bound_goal = workflow_spec_decision.overall_task_goal or bound_goal
                        workflow_decision = workflow_spec_decision
                        self.trace_store.append(
                            "workflow_node_selected",
                            {
                                "trace_id": trace_id,
                                "step": step,
                                "workflow_family": locked_workflow_family,
                                "source": "llm_workflow_spec",
                                "node": locked_workflow_spec.nodes[locked_workflow_index - 1].model_dump(mode="json"),
                                "decision": workflow_spec_decision.model_dump(mode="json"),
                            },
                        )

                if active_overall_goal is None and isinstance(bound_goal, TaskGoal):
                    active_overall_goal = bound_goal
                if (
                    active_overall_goal is None
                    and intent_bundle.task_envelope is not None
                    and getattr(intent_bundle.task_envelope, "required_outputs", None)
                ):
                    active_overall_goal = TaskGoal(
                        summary=str(getattr(intent_bundle.task_envelope, "primary_objective", "") or "").strip()
                        or "Complete the active task.",
                        required_outputs=list(getattr(intent_bundle.task_envelope, "required_outputs", []) or []),
                        completion_mode="outputs",
                    )

                allowed_actions = self._allowed_actions_for_goal(
                    workflow_family=bound_workflow_family,
                    overall_task_goal=active_overall_goal,
                    completed_outputs=completed_outputs,
                    candidate_state=active_candidate_state,
                )
                llm_local_search_decision = None if not allow_programmatic_lookup_workflow else self._build_llm_local_search_workflow(
                    user_text=planner_user_text,
                    completed_outputs=completed_outputs,
                    candidate_state=active_candidate_state,
                    overall_task_goal=active_overall_goal,
                    knowledge_intent=knowledge_request_intent,
                    task_classification=intent_bundle.task_classification,
                    supports_message_delivery=self._supports_message_delivery(),
                )
                shared_proposals = self._build_shared_workflow_proposals(
                    user_text=planner_user_text,
                    completed_outputs=completed_outputs,
                    candidate_state=active_candidate_state,
                    overall_task_goal=active_overall_goal,
                    task_classification=intent_bundle.task_classification,
                    knowledge_intent=knowledge_request_intent,
                    document_delivery_intent=document_delivery_intent,
                    site_search_intent=site_search_intent,
                    tool_results=tool_results,
                    local_search_decision=llm_local_search_decision,
                    allow_programmatic_lookup_workflow=allow_programmatic_lookup_workflow,
                    local_search_source="local_search_planner",
                    local_search_priority=98,
                    local_search_reason="LLM-planned initial local search workflow",
                    document_summary_priority=82,
                    document_summary_reason="Local document summary workflow",
                    document_operation_priority=90,
                    document_operation_reason="Local document structure and editing workflow",
                    file_delivery_priority=88,
                    file_delivery_reason="Local file delivery workflow",
                    local_lookup_priority=84,
                    local_lookup_reason="Hybrid local file lookup workflow",
                    web_target_priority=86,
                    web_target_reason="Explicit website open/search workflow",
                    web_lookup_priority=74,
                    web_lookup_reason="General web research workflow",
                )
                proposals: list[WorkflowProposal] = [
                    shared_proposals[0],
                    WorkflowProposal(
                        source="state_transition",
                        family="state_transition",
                        priority=99,
                        reason="Advance directly from the observed workflow state.",
                        decision=None if scripted_planner_has_pending_decision else self._build_state_transition_followup(
                            user_text=planner_user_text,
                            workflow_state=observed_workflow_state,
                            candidate_state=active_candidate_state,
                            overall_task_goal=active_overall_goal,
                            completed_outputs=completed_outputs,
                            supports_message_delivery=self.registry.has_tool("qq.send_file"),
                        ),
                    ),
                    WorkflowProposal(
                        source="local_collection",
                        family="local_collection",
                        priority=95,
                        reason="Structured local collection workflow",
                        decision=None if not allow_programmatic_lookup_workflow else self.local_collection_workflow.build_next_decision(
                            user_text=planner_user_text,
                            completed_outputs=completed_outputs,
                            tool_results=tool_results,
                            intent=local_collection_request,
                        ),
                    ),
                    WorkflowProposal(
                        source="file_strategy",
                        family="file_lookup",
                        priority=76,
                        reason="Conservative file lookup fallback",
                        decision=None if not allow_programmatic_lookup_workflow else self.file_retrieval_strategy.build_initial_lookup(
                            user_text=planner_user_text,
                            completed_outputs=completed_outputs,
                            candidate_state=active_candidate_state,
                            overall_task_goal=active_overall_goal,
                        ),
                    ),
                    *shared_proposals[1:],
                    WorkflowProposal(
                        source="candidate_followup",
                        family="candidate_followup",
                        priority=92,
                        reason="Continue from previously collected candidates",
                        decision=None if scripted_planner_has_pending_decision else self._build_candidate_action_followup(
                            user_text=planner_user_text,
                            overall_task_goal=active_overall_goal,
                            completed_outputs=completed_outputs,
                            candidate_state=active_candidate_state,
                            supports_message_delivery=self._supports_message_delivery(),
                        ),
                    ),
                ]
                workflow_decision = bound_workflow_decision
                if tool_results and last_executed_decision is not None:
                    fallback_decision = self._build_candidate_write_followup(
                        user_text=planner_user_text,
                        overall_task_goal=active_overall_goal,
                        completed_outputs=completed_outputs,
                        candidate_state=active_candidate_state,
                    )
                    if fallback_decision is None:
                        fallback_decision = self._build_docx_edit_followup(
                            user_text=planner_user_text,
                            overall_task_goal=active_overall_goal,
                            completed_outputs=completed_outputs,
                            candidate_state=active_candidate_state,
                        )
                    if fallback_decision is None:
                        fallback_decision = self._build_web_write_followup(
                            user_text=user_text,
                            overall_task_goal=active_overall_goal,
                            completed_outputs=completed_outputs,
                            tool_results=tool_results,
                            delivery_intent=document_delivery_intent,
                            recent_context=recent_context,
                        )
                    if fallback_decision is None:
                        fallback_decision = self.web_retrieval_strategy.build_empty_result_fallback(
                            user_text=user_text,
                            last_decision=last_executed_decision,
                            last_result=tool_results[-1],
                            delivery_intent=document_delivery_intent,
                            knowledge_intent=knowledge_request_intent,
                            site_search_intent=site_search_intent,
                        )
                    if fallback_decision is None:
                        fallback_decision = self.file_retrieval_strategy.build_empty_result_fallback(
                            user_text=user_text,
                            last_decision=last_executed_decision,
                            last_result=tool_results[-1],
                            candidate_state=active_candidate_state,
                            reliable_candidates=self._has_reliable_candidates(active_candidate_state, active_overall_goal),
                        )

                try:
                    state_machine_repair = workflow_decision or fallback_decision
                    has_prior_progress = bool(
                        tool_results
                        or active_candidate_state is not None
                        or observed_workflow_state is not None
                        or completed_outputs
                    )

                    planner_observations = [
                        *observations,
                        f"Bound workflow family: {bound_workflow_family}",
                        f"Allowed decision types for this step: {sorted(allowed_actions)}",
                    ]

                    if active_overall_goal is not None:
                        planner_observations.append(
                            f"Current overall task goal required outputs: {[item.value for item in active_overall_goal.required_outputs]}"
                        )

                    if state_machine_repair is not None:
                        planner_observations.append(
                            self._format_state_machine_candidate_observation(state_machine_repair)
                        )
                    state_machine_repair_summary = (
                        "State machine selected a grounded next step after planner output was unavailable or incomplete."
                        if workflow_decision is not None
                        else "State machine selected a grounded recovery step after planner output was unavailable or incomplete."
                    )
                    decision_source = "planner"
                    if (
                        locked_workflow_family is not None
                        and self._should_use_locked_workflow_decision(
                            workflow_decision,
                            workflow_family=bound_workflow_family,
                            request_signatures=set(request_signatures),
                            request_signature=self.loop_controller.request_signature,
                        )
                    ):
                        raw_decision = workflow_decision
                        review = DecisionReview(
                            approved=True,
                            issues=[],
                            summary="Trusted the locked workflow node without re-running the general planner.",
                            suggested_decision=None,
                        )
                        planner_bypass_count += 1
                        decision_source = "workflow_locked"
                        locked_workflow_nodes.append(
                            {
                                "step": step,
                                "tool": raw_decision.selected_tool,
                                "intent": raw_decision.intent,
                                "expected_outputs": [item.value for item in raw_decision.expected_step_outputs],
                            }
                        )
                        self.trace_store.append(
                            "workflow_node_selected",
                            {
                                "trace_id": trace_id,
                                "step": step,
                                "workflow_family": locked_workflow_family,
                                "node": locked_workflow_nodes[-1],
                                "decision": raw_decision.model_dump(mode="json"),
                            },
                        )
                    elif self._should_short_circuit_state_machine_repair(
                        state_machine_repair,
                        has_prior_progress=has_prior_progress,
                        request_signatures=request_signatures,
                        request_signature=self.loop_controller.request_signature,
                    ):
                        raw_decision = state_machine_repair
                        review = DecisionReview(
                            approved=True,
                            issues=[],
                            summary="Trusted the state machine's grounded low-risk next step without re-running the planner.",
                            suggested_decision=None,
                        )
                        planner_bypass_count += 1
                        decision_source = "state_machine_direct"
                        self.trace_store.append(
                            "decision_short_circuit",
                            {
                                "trace_id": trace_id,
                                "step": step,
                                "source": decision_source,
                                "decision": raw_decision.model_dump(mode="json"),
                            },
                        )
                    else:
                        try:
                            planner_invocations += 1
                            raw_decision = self.llm_client.decide(
                                messages=self._context_messages(),
                                tool_manifests=self.registry.list_manifests(),
                                observations=planner_observations,
                                allowed_decisions=sorted(allowed_actions),
                                bound_workflow_family=bound_workflow_family,
                            )
                            raw_decision = self._normalize_network_lookup_decision(raw_decision, user_text=user_text)
                            scripted_planner_decisions_used += 1
                            if not self._planner_decision_within_allowed_actions(
                                    raw_decision,
                                    allowed_actions=allowed_actions,
                                    overall_task_goal=active_overall_goal,
                                    completed_outputs=completed_outputs,
                            ):
                                if workflow_decision is not None:
                                    raw_decision = workflow_decision
                                    review = DecisionReview(
                                        approved=True,
                                        issues=["planner_outside_allowed_actions"],
                                        summary="Planner decision exceeded workflow/state constraints, so workflow decision was used.",
                                        suggested_decision=None,
                                    )
                                    decision_source = "workflow_policy_override"
                                else:
                                    raw_decision = ToolDecision(
                                        decision=DecisionType.CLARIFY,
                                        intent="clarify_missing_grounding",
                                        reason="The task still requires grounded outputs before a final response.",
                                        response_hint="我还需要先补齐必要信息，才能继续准确处理。",
                                        risk_level=RiskLevel.LOW,
                                        overall_task_goal=active_overall_goal,
                                    )
                                    review = DecisionReview(
                                        approved=True,
                                        issues=["planner_outside_allowed_actions_no_fallback"],
                                        summary="Planner decision exceeded workflow/state constraints and no safe workflow fallback was available.",
                                        suggested_decision=None,
                                    )
                                    decision_source = "workflow_policy_clarify"
                            else:
                                if self._should_auto_approve_write_followup(raw_decision, tool_results) or self._should_auto_approve_candidate_followup(raw_decision, tool_results):
                                    review = DecisionReview(
                                        approved=True,
                                        issues=[],
                                        summary="Auto-approved a grounded follow-up because it only operates on concrete paths returned by the latest retrieval step.",
                                        suggested_decision=None,
                                    )
                                    decision_source = "planner_auto_approved"
                                else:
                                    critic_invocations += 1
                                    review = self.critic.review(
                                        messages=self._context_messages(),
                                        decision=raw_decision,
                                        tool_manifests=self.registry.list_manifests(),
                                        observations=planner_observations,
                                    )
                                    decision_source = "planner_reviewed"
                            effective_planner_decision = self._resolve_reviewed_decision(raw_decision, review)
                            planner_repeated_previous_request = (
                                state_machine_repair is not None
                                and effective_planner_decision.decision == DecisionType.TOOL_CALL
                                and self.loop_controller.request_signature(effective_planner_decision) in request_signatures
                            )
                            if planner_repeated_previous_request or self._planner_response_should_use_state_machine_repair(
                                effective_planner_decision,
                                state_machine_repair,
                                completed_outputs,
                                has_prior_progress=has_prior_progress,
                            ):
                                raw_decision = state_machine_repair
                                review = DecisionReview(
                                    approved=True,
                                    issues=[],
                                    summary=state_machine_repair_summary,
                                    suggested_decision=None,
                                )
                                decision_source = "state_machine_repair"
                        except Exception:
                            if state_machine_repair is None:
                                raise
                            if self._should_use_state_machine_repair_after_planner_exception(
                                has_prior_progress=has_prior_progress,
                                knowledge_type=str(getattr(knowledge_request_intent, "knowledge_type", "") or ""),
                            ):
                                raw_decision = state_machine_repair
                                review = DecisionReview(
                                    approved=True,
                                    issues=[],
                                    summary=state_machine_repair_summary,
                                    suggested_decision=None,
                                )
                                decision_source = "state_machine_repair_exception"
                            else:
                                raw_decision = ToolDecision(
                                    decision=DecisionType.RESPOND,
                                    intent="respond_directly",
                                    reason="Planner failed before a grounded workflow could be safely recovered on the first turn.",
                                    response_hint="这句话还缺一点上下文，我先不乱猜。你是想让我总结哪一段内容，或者“他”具体指谁？",
                                    risk_level=RiskLevel.LOW,
                                )
                                review = DecisionReview(
                                    approved=True,
                                    issues=["planner_exception_safe_response"],
                                    summary="Returned a clarification response instead of falling back to an ungrounded first-turn workflow.",
                                    suggested_decision=None,
                                )
                                decision_source = "planner_exception_safe_response"
                    self.trace_store.append(
                        "decision_raw",
                        {
                            "trace_id": trace_id,
                            "step": step,
                            "decision": raw_decision.model_dump(mode="json"),
                            "source": decision_source,
                            "fallback": raw_decision is fallback_decision,
                            "workflow": raw_decision is workflow_decision,
                            "state_machine_repair": raw_decision is workflow_decision or raw_decision is fallback_decision,
                        },
                    )
                    self._emit_progress(
                        progress_callback,
                        "decision_made",
                        f"我初步打算走 {raw_decision.selected_tool or raw_decision.decision.value} 这条路。",
                        {
                            "step": step,
                            "decision": raw_decision.decision.value,
                            "selected_tool": raw_decision.selected_tool,
                        },
                    )

                    self.trace_store.append(
                        "decision_review",
                        {
                            "trace_id": trace_id,
                            "step": step,
                            "source": decision_source,
                            "review": review.model_dump(mode="json"),
                        },
                    )
                    self._emit_progress(
                        progress_callback,
                        "review",
                        "我又复查了一遍这一步，看看工具和参数是不是说得通。",
                        {"step": step, "approved": review.approved, "issues": review.issues},
                    )

                    if not review.approved and review.suggested_decision is None:
                        raise ValueError(f"Decision review rejected planner output: {review.summary or review.issues}")

                    decision = self._normalize_network_lookup_decision(
                        self._resolve_reviewed_decision(raw_decision, review),
                        user_text=user_text,
                    )
                    decision, upstream_constraint_trace = self._enforce_upstream_constraints_on_decision(
                        decision,
                        task_envelope=intent_bundle.task_envelope,
                        task_graph=intent_bundle.task_graph,
                        overall_task_goal=active_overall_goal,
                        completed_outputs=completed_outputs,
                        state_machine_repair=state_machine_repair,
                    )
                    if upstream_constraint_trace is not None:
                        self.trace_store.append(
                            "upstream_constraint_applied",
                            {
                                "trace_id": trace_id,
                                "step": step,
                                "phase": "pre_argument_planner",
                                "source": decision_source,
                                "details": upstream_constraint_trace,
                                "decision": decision.model_dump(mode="json"),
                            },
                        )
                    decision, workflow_argument_trace = self.workflow_argument_planner.plan(
                        decision=decision,
                        user_text=user_text,
                        workflow_family=bound_workflow_family,
                        decision_source=decision_source,
                        observations=planner_observations,
                        completed_outputs=completed_outputs,
                        overall_task_goal=active_overall_goal,
                        candidate_state=active_candidate_state,
                        workflow_state=observed_workflow_state,
                        task_envelope=intent_bundle.task_envelope,
                    )
                    decision, post_argument_constraint_trace = self._enforce_upstream_constraints_on_decision(
                        decision,
                        task_envelope=intent_bundle.task_envelope,
                        task_graph=intent_bundle.task_graph,
                        overall_task_goal=active_overall_goal,
                        completed_outputs=completed_outputs,
                        state_machine_repair=state_machine_repair,
                    )
                    if workflow_argument_trace is not None:
                        self.trace_store.append(
                            "workflow_argument_planner",
                            {
                                "trace_id": trace_id,
                                "step": step,
                                "decision_source": decision_source,
                                "trace": workflow_argument_trace,
                            },
                        )
                    if post_argument_constraint_trace is not None:
                        self.trace_store.append(
                            "upstream_constraint_applied",
                            {
                                "trace_id": trace_id,
                                "step": step,
                                "phase": "post_argument_planner",
                                "source": decision_source,
                                "details": post_argument_constraint_trace,
                                "decision": decision.model_dump(mode="json"),
                            },
                        )
                    self.validator.validate(decision)
                    self.validator.validate_against_task_state(
                        decision,
                        overall_task_goal=active_overall_goal,
                        completed_outputs=completed_outputs,
                    )
                    self.guardrails.validate(decision)
                except Exception as exc:  # noqa: BLE001
                        recovered_decision = self._validated_state_machine_repair(
                            state_machine_repair,
                            request_signatures=request_signatures,
                            request_signature=self.loop_controller.request_signature,
                            validate_decision=self.validator.validate,
                            validate_task_state=lambda d: self.validator.validate_against_task_state(
                                d,
                                overall_task_goal=active_overall_goal,
                                completed_outputs=completed_outputs,
                            ),
                            validate_guardrails=self.guardrails.validate,
                        )
                        if recovered_decision is not None:
                            decision = recovered_decision
                            decision, recovered_constraint_trace = self._enforce_upstream_constraints_on_decision(
                                decision,
                                task_envelope=intent_bundle.task_envelope,
                                task_graph=intent_bundle.task_graph,
                                overall_task_goal=active_overall_goal,
                                completed_outputs=completed_outputs,
                                state_machine_repair=state_machine_repair,
                            )
                            if recovered_constraint_trace is not None:
                                self.trace_store.append(
                                    "upstream_constraint_applied",
                                    {
                                        "trace_id": trace_id,
                                        "step": step,
                                        "phase": "decision_recovery",
                                        "source": "state_machine_repair_after_decision_error",
                                        "details": recovered_constraint_trace,
                                        "decision": decision.model_dump(mode="json"),
                                    },
                                )
                            self.validator.validate(decision)
                            self.validator.validate_against_task_state(
                                decision,
                                overall_task_goal=active_overall_goal,
                                completed_outputs=completed_outputs,
                            )
                            self.guardrails.validate(decision)
                            self.trace_store.append(
                                "decision_recovered",
                                {
                                    "trace_id": trace_id,
                                    "step": step,
                                    "reason": "state_machine_repair_after_decision_error",
                                    "details": str(exc),
                                    "decision": decision.model_dump(mode="json"),
                                },
                            )
                        elif self.loop_controller.should_finalize_after_decision_error(tool_results):
                            self.trace_store.append(
                                "loop_stop",
                                {
                                    "trace_id": trace_id,
                                    "step": step,
                                    "reason": "finalize_after_decision_error",
                                    "details": str(exc),
                                },
                            )
                            loop_stop_reason = "finalize_after_decision_error"
                            break
                        else:
                            raise

                artifacts.decision = decision
                if decision.overall_task_goal is not None and decision.overall_task_goal.required_outputs:
                    active_overall_goal = self._merge_goals(active_overall_goal, decision.overall_task_goal)
                elif active_overall_goal is None and decision.selected_tool:
                    manifest = self.registry.get_manifest(decision.selected_tool)
                    if manifest.produces:
                        active_overall_goal = TaskGoal(
                            summary=f"Complete intent: {decision.intent}",
                            required_outputs=list(manifest.produces),
                        )

                artifacts.overall_task_goal = active_overall_goal
                artifacts.candidate_state = active_candidate_state
                self.trace_store.append(
                    "decision_effective",
                    {
                        "trace_id": trace_id,
                        "step": step,
                        "source": decision_source,
                        "decision": decision.model_dump(mode="json"),
                        "active_overall_goal": None
                        if active_overall_goal is None
                        else active_overall_goal.model_dump(mode="json"),
                    },
                )
                decision_path.append(
                    {
                        "step": step,
                        "source": decision_source,
                        "decision": decision.decision.value,
                        "selected_tool": decision.selected_tool,
                        "intent": decision.intent,
                        "risk_level": decision.risk_level.value,
                    }
                )

                if decision.memory_write:
                    self.memory_store.remember(
                        MemoryRecord(
                            memory_type="episodic",
                            scope="session",
                            content=decision.memory_write,
                            importance=0.65,
                            tags=["llm"],
                        )
                    )

                if decision.decision == DecisionType.CLARIFY:
                    clarification_text = decision.response_hint or self._render_clarification_hint(
                        user_text=user_text,
                        intent=decision.intent,
                        missing_slots=[],
                        fallback="",
                        style_hint="Ask the user only for the missing information needed to continue.",
                    )
                    self._emit_progress(
                        progress_callback,
                        "clarify",
                        clarification_text,
                        {"step": step},
                    )
                    artifacts.pending_task = PendingTask(
                        task_id=f"pending_{uuid.uuid4().hex[:10]}",
                        intent=decision.intent,
                        summary=decision.reason or "Need additional information from the user.",
                        original_user_request=user_text,
                        clarification_prompt=clarification_text,
                        overall_task_goal=active_overall_goal or decision.overall_task_goal,
                        missing_slots=[],
                        collected_slots={},
                        resume_hint=clarification_text,
                    )
                    break

                if decision.decision in {DecisionType.RESPOND, DecisionType.CLARIFY, DecisionType.FINISH}:
                    break

                current_stop_reason = self.loop_controller.should_stop_on_duplicate_request(
                    decision=decision,
                    previous_signatures=request_signatures,
                    tool_results=tool_results,
                )
                if current_stop_reason is not None:
                    self.trace_store.append("loop_stop", {"trace_id": trace_id, "step": step, "reason": current_stop_reason})
                    loop_stop_reason = current_stop_reason
                    break

                request = self.registry.build_request(
                    trace_id=trace_id,
                    session_id=session_id,
                    tool_name=decision.selected_tool or "",
                    arguments=decision.arguments,
                )
                request_signatures.append(self.loop_controller.request_signature(decision))
                self.trace_store.append(
                    "tool_request",
                    {"trace_id": trace_id, "step": step, "request": request.model_dump(mode="json")},
                )
                self._emit_progress(
                    progress_callback,
                    "tool_start",
                    f"我现在开始调用 {request.tool_name}。",
                    {"step": step, "tool_name": request.tool_name, "arguments": request.arguments},
                )

                result = self.registry.execute(request)
                last_executed_decision = decision
                tool_results.append(result)
                artifacts.tool_results.append(result)
                if result.status == "success" and decision.selected_tool:
                    manifest = self.registry.get_manifest(decision.selected_tool)
                    effective_outputs = self.completion_judge.resolve_effective_outputs(
                        tool_name=decision.selected_tool,
                        produced_outputs=list(manifest.produces),
                        result=result,
                    )
                    for output_name in effective_outputs:
                        if output_name not in completed_outputs:
                            completed_outputs.append(output_name)
                    active_candidate_state = self._derive_candidate_state(decision, result, active_candidate_state)
                    if self._has_reliable_candidates(active_candidate_state, active_overall_goal) and OutputKind.OBJECT_CANDIDATES not in completed_outputs:
                        completed_outputs.append(OutputKind.OBJECT_CANDIDATES)
                artifacts.completed_outputs = list(completed_outputs)
                artifacts.candidate_state = active_candidate_state
                if self._should_offer_candidate_selection(
                    user_text=user_text,
                    candidate_state=active_candidate_state,
                    overall_task_goal=active_overall_goal,
                    completed_outputs=completed_outputs,
                ):
                    artifacts.pending_task = self._build_candidate_selection_pending(
                        user_text=user_text,
                        candidate_state=active_candidate_state,
                        overall_task_goal=active_overall_goal,
                    )
                    self._emit_progress(
                        progress_callback,
                        "waiting_for_selection",
                        "我先把最像的几个文件托出来给你选，选中之后我就继续往下做。",
                        {
                            "step": step,
                            "selection_candidates": [
                                candidate.model_dump(mode="json")
                                for candidate in artifacts.pending_task.selection_candidates
                            ],
                        },
                    )
                    self.trace_store.append(
                        "loop_stop",
                        {
                            "trace_id": trace_id,
                            "step": step,
                            "reason": "waiting_for_selection",
                        },
                    )
                    loop_stop_reason = "waiting_for_selection"
                    break
                self.history.append(
                    Message(role=Role.TOOL, content=f"{decision.selected_tool}: {result.model_dump(mode='json')}")
                )
                self.trace_store.append(
                    "tool_result",
                    {"trace_id": trace_id, "step": step, "result": result.model_dump(mode="json")},
                )
                if result.status == "success":
                    self._emit_progress(
                        progress_callback,
                        "tool_success",
                        f"{request.tool_name} 已经执行完了，我拿到了新结果。",
                        {"step": step, "tool_name": request.tool_name, "data": result.data},
                    )
                else:
                    self._emit_progress(
                        progress_callback,
                        "tool_error",
                        f"{request.tool_name} 这一步没跑通，我得调整一下。",
                        {
                            "step": step,
                            "tool_name": request.tool_name,
                            "error": None if result.error is None else result.error.message,
                        },
                    )

                completion = self.completion_judge.assess(
                    overall_task_goal=active_overall_goal,
                    completed_outputs=completed_outputs,
                    tool_results=tool_results,
                )
                completion, _, completion_review = self._review_execution_state(
                    user_text=user_text,
                    overall_task_goal=active_overall_goal,
                    completed_outputs=completed_outputs,
                    tool_results=tool_results,
                    candidate_state=active_candidate_state,
                    completion=completion,
                    stop_reason=loop_stop_reason,
                    document_delivery_intent=document_delivery_intent,
                    knowledge_request_intent=knowledge_request_intent,
                    site_search_intent=site_search_intent,
                    task_classification=intent_bundle.task_classification,
                )
                observation_summary = self._build_execution_summary(
                    user_text=user_text,
                    overall_task_goal=active_overall_goal,
                    completed_outputs=completed_outputs,
                    tool_results=tool_results,
                    candidate_state=active_candidate_state,
                    completion=completion,
                    stop_reason=loop_stop_reason,
                    document_delivery_intent=document_delivery_intent,
                    knowledge_request_intent=knowledge_request_intent,
                    site_search_intent=site_search_intent,
                    decision_path=decision_path,
                    planner_invocations=planner_invocations,
                    critic_invocations=critic_invocations,
                    planner_bypass_count=planner_bypass_count,
                    task_classification=None
                    if intent_bundle.task_classification is None
                    else intent_bundle.task_classification.model_dump(mode="json"),
                )
                observed_workflow_state = WorkflowState.model_validate(observation_summary.get("workflow_state") or {})
                artifacts.workflow_state = observed_workflow_state
                self.trace_store.append(
                    "completion_check",
                    {
                        "trace_id": trace_id,
                        "step": step,
                        "assessment": completion.model_dump(mode="json"),
                        "review": completion_review.model_dump(mode="json"),
                        "observed_workflow_state": observed_workflow_state.model_dump(mode="json"),
                        "active_overall_goal": None
                        if active_overall_goal is None
                        else active_overall_goal.model_dump(mode="json"),
                    },
                )
                if completion.done:
                    self._emit_progress(
                        progress_callback,
                        "completion",
                        "我判断该拿到的结果已经齐了，可以准备组织最终回复了。",
                        {
                            "step": step,
                            "completed_outputs": [item.value for item in completion.completed_outputs],
                        },
                    )
                    self.trace_store.append(
                        "loop_stop",
                        {
                            "trace_id": trace_id,
                            "step": step,
                            "reason": "completion_judge",
                            "details": completion.reason,
                        },
                    )
                    loop_stop_reason = "completion_judge"
                    break

                current_stop_reason = self.loop_controller.should_stop_after_result(tool_results)
                if current_stop_reason is not None:
                    self._emit_progress(
                        progress_callback,
                        "stop",
                        "我主动停下来了，避免继续重复或空转。",
                        {"step": step, "reason": current_stop_reason},
                    )
                    self.trace_store.append("loop_stop", {"trace_id": trace_id, "step": step, "reason": current_stop_reason})
                    loop_stop_reason = current_stop_reason
                    break
        except Exception as exc:  # noqa: BLE001
            error_text = self.llm_client.build_unavailable_response(exc)
            artifacts.final_response = error_text
            self.history.append(Message(role=Role.ASSISTANT, content=error_text))
            self.trace_store.append("error", {"trace_id": trace_id, "error": str(exc)})
            self._emit_progress(
                progress_callback,
                "error",
                f"中途出了点问题：{exc}",
                {"trace_id": trace_id, "error": str(exc)},
            )
            return artifacts

        final_observations = ContextBuilder.build_observations(tool_results)
        final_completion = self.completion_judge.assess(
            overall_task_goal=active_overall_goal,
            completed_outputs=completed_outputs,
            tool_results=tool_results,
        )
        final_completion, execution_summary, completion_review = self._review_execution_state(
            user_text=user_text,
            overall_task_goal=active_overall_goal,
            completed_outputs=completed_outputs,
            tool_results=tool_results,
            candidate_state=active_candidate_state,
            completion=final_completion,
            stop_reason=loop_stop_reason,
            document_delivery_intent=document_delivery_intent,
            knowledge_request_intent=knowledge_request_intent,
            site_search_intent=site_search_intent,
            decision_path=decision_path,
            planner_invocations=planner_invocations,
            critic_invocations=critic_invocations,
            planner_bypass_count=planner_bypass_count,
            task_classification=intent_bundle.task_classification,
        )
        artifacts.execution_summary = execution_summary
        if (
            artifacts.pending_task is None
            and loop_stop_reason == "finalize_after_decision_error"
            and active_candidate_state is not None
            and len(active_candidate_state.candidate_paths) >= 2
        ):
            artifacts.pending_task = self._build_candidate_selection_pending(
                user_text=user_text,
                candidate_state=active_candidate_state,
                overall_task_goal=active_overall_goal,
            )
        self._emit_progress(progress_callback, "responding", "结果已经整理好了，我现在把它翻译成人话。", {})
        speech_text = ""
        if artifacts.pending_task is not None:
            pending_bundle = self._render_pending_task_response_bundle(
                pending_task=artifacts.pending_task,
                execution_summary=execution_summary,
                observations=final_observations,
            )
            if pending_bundle is None:
                raise RuntimeError("pending_task_response_generation_failed")
            final_response = pending_bundle["display_text"]
            speech_text = pending_bundle["speech_text"]
        elif self._should_force_grounded_partial(execution_summary):
            partial_bundle = self._render_grounded_partial_response_bundle(
                execution_summary=execution_summary,
                observations=final_observations,
            )
            if partial_bundle is None:
                raise RuntimeError("partial_response_generation_failed")
            final_response = partial_bundle["display_text"]
            speech_text = partial_bundle["speech_text"]
        else:
            try:
                structured_bundle = self._build_structured_qq_history_response_bundle(execution_summary)
                if structured_bundle is None:
                    structured_bundle = self._build_structured_document_edit_response_bundle(execution_summary)
                if structured_bundle is not None:
                    final_response = structured_bundle["display_text"]
                    speech_text = structured_bundle["speech_text"]
                elif tool_results and self.config.tool_speech_enabled and hasattr(self.llm_client, "render_tool_response_bundle"):
                    try:
                        bundle = self.llm_client.render_tool_response_bundle(
                            system_name=self.config.system_name,
                            messages=self._context_messages(),
                            observations=final_observations,
                            response_hint=artifacts.decision.response_hint if artifacts.decision else None,
                            execution_summary=execution_summary,
                            persona_name=self.config.persona_name,
                            persona_profile=self.config.persona_profile,
                            display_style_prompt=self.config.display_style_prompt,
                            speech_style_prompt=self.config.speech_style_prompt,
                            speech_max_chars=self.config.speech_max_chars,
                        )
                    except TypeError:
                        bundle = self.llm_client.render_tool_response_bundle(
                            self.config.system_name,
                            self._context_messages(),
                            final_observations,
                            artifacts.decision.response_hint if artifacts.decision else None,
                            execution_summary,
                        )
                    final_response = bundle["display_text"]
                    speech_text = bundle["speech_text"]
                else:
                    try:
                        final_response = self.llm_client.render_response(
                            system_name=self.config.system_name,
                            messages=self._context_messages(),
                            observations=final_observations,
                            response_hint=artifacts.decision.response_hint if artifacts.decision else None,
                            execution_summary=execution_summary,
                            persona_name=self.config.persona_name,
                            persona_profile=self.config.persona_profile,
                            display_style_prompt=self.config.display_style_prompt,
                        )
                    except TypeError:
                        final_response = self.llm_client.render_response(
                            system_name=self.config.system_name,
                            messages=self._context_messages(),
                            observations=final_observations,
                            response_hint=artifacts.decision.response_hint if artifacts.decision else None,
                            execution_summary=execution_summary,
                        )
            except Exception as exc:  # noqa: BLE001
                final_response = self.llm_client.build_unavailable_response(exc)
        grounding_review = self.execution_critic.review_grounding(
            user_text=user_text,
            execution_summary=execution_summary,
            response_text=final_response,
        )
        execution_summary = self._merge_execution_review_into_summary(execution_summary, completion_review, "completion_review")
        execution_summary = self._merge_execution_review_into_summary(execution_summary, grounding_review, "grounding_review")
        artifacts.execution_summary = execution_summary
        workflow_payload = execution_summary.get("workflow_state")
        if isinstance(workflow_payload, dict):
            try:
                artifacts.workflow_state = WorkflowState.model_validate(workflow_payload)
            except Exception:
                artifacts.workflow_state = None
        if grounding_review.force_partial and execution_summary.get("stop_reason") != "duplicate_tool_request_after_success":
            partial_bundle = self._render_grounded_partial_response_bundle(
                execution_summary=execution_summary,
                observations=final_observations,
            )
            if partial_bundle is None:
                raise RuntimeError("partial_response_generation_failed")
            final_response = partial_bundle["display_text"]
            speech_text = partial_bundle["speech_text"]
        if not speech_text:
            speech_text = final_response
        artifacts.final_response = final_response
        artifacts.speech_text = speech_text
        self.history.append(Message(role=Role.ASSISTANT, content=final_response))
        self.trace_store.append("final_response", {"trace_id": trace_id, "text": final_response})
        artifacts.debug_summary = (
            f"我一共跑了 {len(tool_results)} 个工具步骤，"
            f"目前拿到的输出有：{', '.join(item.value for item in artifacts.completed_outputs) or '无'}。"
        )

        self.memory_store.remember(
            MemoryRecord(
                memory_type="episodic",
                scope="session",
                content=f"用户: {user_text}\n助手: {final_response}",
                importance=0.55,
                tags=["turn"],
            )
        )

        try:
            artifacts.tts_dispatched = self.voice_adapter.dispatch(speech_text)
        except Exception as exc:  # noqa: BLE001
            artifacts.tts_dispatched = False
            self.trace_store.append("tts_error", {"trace_id": trace_id, "error": str(exc)})
        self._emit_progress(
            progress_callback,
            "done",
            "这轮任务已经收尾了，你现在可以查看结果。",
            {
                "trace_id": trace_id,
                "completed_outputs": [item.value for item in artifacts.completed_outputs],
                "debug_summary": artifacts.debug_summary,
            },
        )
        return artifacts

    def _review_execution_state(
        self,
        *,
        user_text: str,
        overall_task_goal: TaskGoal | None,
        completed_outputs: list[OutputKind],
        tool_results: list[ToolCallResult],
        candidate_state: CandidateState | None,
        completion,
        stop_reason: str | None,
        document_delivery_intent=None,
        knowledge_request_intent=None,
        site_search_intent=None,
        decision_path: list[dict[str, object]] | None = None,
        planner_invocations: int = 0,
        critic_invocations: int = 0,
        planner_bypass_count: int = 0,
        task_classification=None,
    ) -> tuple[object, dict, ExecutionReview]:
        summary = self._build_execution_summary(
            user_text=user_text,
            overall_task_goal=overall_task_goal,
            completed_outputs=completed_outputs,
            tool_results=tool_results,
            candidate_state=candidate_state,
            completion=completion,
            stop_reason=stop_reason,
            document_delivery_intent=document_delivery_intent,
            knowledge_request_intent=knowledge_request_intent,
            site_search_intent=site_search_intent,
            decision_path=decision_path,
            planner_invocations=planner_invocations,
            critic_invocations=critic_invocations,
            planner_bypass_count=planner_bypass_count,
            task_classification=task_classification,
        )
        review = self.execution_critic.review_completion(
            user_text=user_text,
            execution_summary=summary,
        )
        reviewed_completion = self._apply_execution_review(completion, review)
        if reviewed_completion != completion:
            summary = self._build_execution_summary(
                user_text=user_text,
                overall_task_goal=overall_task_goal,
                completed_outputs=completed_outputs,
                tool_results=tool_results,
                candidate_state=candidate_state,
                completion=reviewed_completion,
                stop_reason=stop_reason,
                document_delivery_intent=document_delivery_intent,
                knowledge_request_intent=knowledge_request_intent,
                site_search_intent=site_search_intent,
                decision_path=decision_path,
                planner_invocations=planner_invocations,
                critic_invocations=critic_invocations,
                planner_bypass_count=planner_bypass_count,
                task_classification=task_classification,
            )
        summary["completion_review"] = review.model_dump(mode="json")
        return reviewed_completion, summary, review

    @staticmethod
    def _merge_execution_review_into_summary(summary: dict, review: ExecutionReview, key: str) -> dict:
        merged = dict(summary)
        merged[key] = review.model_dump(mode="json")
        if review.force_partial:
            merged["task_status"] = "partial"
            missing_outputs = [str(item) for item in merged.get("missing_outputs") or [] if str(item).strip()]
            for output_kind in review.missing_outputs:
                value = output_kind.value if isinstance(output_kind, OutputKind) else str(output_kind)
                if value not in missing_outputs:
                    missing_outputs.append(value)
            merged["missing_outputs"] = missing_outputs
        return merged

    @staticmethod
    def _derive_workflow_stage(
        *,
        overall_task_goal: TaskGoal | None,
        completed_outputs: list[OutputKind],
        candidate_state: CandidateState | None,
        candidate_paths: list[str],
        successful_actions: list[dict],
        completion,
    ) -> str:
        if completion.done and not completion.missing_outputs:
            return "completed"

        completed_values = {
            item.value if isinstance(item, OutputKind) else str(item).strip()
            for item in completed_outputs
            if str(item).strip()
        }
        required_values = {
            item.value if isinstance(item, OutputKind) else str(item).strip()
            for item in ((overall_task_goal.required_outputs if overall_task_goal is not None else []) or [])
            if str(item).strip()
        }
        tool_names = {
            str(item.get("tool_name", "")).strip()
            for item in successful_actions
            if isinstance(item, dict) and str(item.get("tool_name", "")).strip()
        }

        if "message_sent" in completed_values or "qq.send_file" in tool_names or "qq.send_text" in tool_names:
            return "delivered"
        if "file_written" in completed_values:
            return "artifact_ready"
        if (
            {"message_sent", "path_opened", "file_written"} & required_values
            and {"file_contents", "object_details", "search_results", "web_content"} & completed_values
        ):
            return "action_ready"
        if candidate_state is not None and candidate_state.workflow_stage:
            return candidate_state.workflow_stage
        if candidate_paths or "object_candidates" in completed_values:
            return "candidate_ready"
        if completed_values:
            return "progressing"
        return "searching"

    @staticmethod
    def _derive_primary_target_path(
        *,
        task_classification,
        candidate_state: CandidateState | None,
        candidate_paths: list[str],
        successful_actions: list[dict],
        written_files: list[str],
    ) -> tuple[str | None, str | None]:
        task_kind = ""
        if isinstance(task_classification, dict):
            task_kind = str(task_classification.get("task_kind", "")).strip().lower()

        delivered_path: str | None = None
        for action in successful_actions:
            if not isinstance(action, dict) or action.get("tool_name") != "qq.send_file":
                continue
            data = action.get("data") or {}
            candidate = str(data.get("path", "")).strip()
            if candidate:
                delivered_path = candidate
                break

        primary_path = delivered_path
        if primary_path is None:
            for item in written_files:
                candidate = str(item).strip()
                if candidate:
                    primary_path = candidate
                    break
        if primary_path is None and candidate_state is not None:
            for item in candidate_state.candidate_paths:
                candidate = str(item).strip()
                if candidate:
                    primary_path = candidate
                    break
        if primary_path is None:
            for item in candidate_paths:
                candidate = str(item).strip()
                if candidate:
                    primary_path = candidate
                    break
        if primary_path is None:
            for item in written_files:
                candidate = str(item).strip()
                if candidate:
                    primary_path = candidate
                    break

        delivery_target_path = primary_path if task_kind == "delivery" else None
        return primary_path, delivery_target_path

    @staticmethod
    def _next_actions_for_workflow_state(
        *,
        required_outputs: list[OutputKind],
        completed_outputs: list[OutputKind],
        candidate_paths: list[str],
        primary_target_path: str | None,
        successful_actions: list[dict],
    ) -> list[str]:
        completed_values = {
            item.value if isinstance(item, OutputKind) else str(item).strip()
            for item in completed_outputs
            if str(item).strip()
        }
        required_values = {
            item.value if isinstance(item, OutputKind) else str(item).strip()
            for item in required_outputs
            if str(item).strip()
        }
        tool_names = {
            str(item.get("tool_name", "")).strip()
            for item in successful_actions
            if isinstance(item, dict) and str(item.get("tool_name", "")).strip()
        }
        actions: list[str] = []
        if "object_candidates" in required_values and "object_candidates" not in completed_values:
            actions.append("search")
        if "file_contents" in required_values and "file_contents" not in completed_values and (candidate_paths or primary_target_path):
            actions.append("read")
        if "object_details" in required_values and "object_details" not in completed_values and (candidate_paths or primary_target_path):
            actions.append("inspect")
        if "search_results" in required_values and "search_results" not in completed_values:
            actions.append("search")
        if "web_content" in required_values and "web_content" not in completed_values:
            actions.append("fetch")
        if "file_written" in required_values and "file_written" not in completed_values and primary_target_path:
            actions.append("write")
        if "message_sent" in required_values and "message_sent" not in completed_values:
            if "qq.send_file" in tool_names:
                pass
            elif primary_target_path:
                actions.append("deliver")
        if "path_opened" in required_values and "path_opened" not in completed_values and primary_target_path:
            actions.append("open")
        return actions

    @staticmethod
    def _derive_workflow_family(
        *,
        task_classification,
        overall_task_goal: TaskGoal | None,
        successful_actions: list[dict],
    ) -> str:
        task_kind = ""
        if isinstance(task_classification, dict):
            task_kind = str(task_classification.get("task_kind", "")).strip().lower()
        if task_kind:
            if task_kind in {"history_lookup", "reply_lookup", "attachment_lookup"}:
                return "qq_history"
            if task_kind in {"delivery", "file_delivery"}:
                return "delivery"
            if task_kind in {"summarize", "document_summary"}:
                return "document_summary"
            if task_kind in {"document_edit", "edit", "rewrite", "transform"}:
                return "document_edit"
            if task_kind in {"lookup", "file_lookup", "local_lookup"}:
                return "local_lookup"
            if task_kind in {"inspect"}:
                return "inspect"
            return task_kind

        required_values = {
            item.value if isinstance(item, OutputKind) else str(item).strip()
            for item in ((overall_task_goal.required_outputs if overall_task_goal is not None else []) or [])
            if str(item).strip()
        }
        if "message_sent" in required_values:
            return "delivery"
        if "contact_candidates" in required_values:
            return "contact_resolution"
        if "web_content" in required_values or "search_results" in required_values:
            return "web_lookup"
        if "file_written" in required_values:
            if "object_details" in required_values:
                return "document_edit"
            return "artifact_generation"
        if "file_contents" in required_values:
            return "content_lookup"
        if "object_details" in required_values:
            return "inspection"

        tool_names = {
            str(item.get("tool_name", "")).strip()
            for item in successful_actions
            if isinstance(item, dict) and str(item.get("tool_name", "")).strip()
        }
        if any(name in {"qq.get_recent_messages", "qq.get_last_reply", "qq.search_history", "qq.get_recent_attachments"} for name in tool_names):
            return "qq_history"
        if any(name.startswith("web.") for name in tool_names):
            return "web_lookup"
        if any(name == "qq.search_contacts" for name in tool_names):
            return "contact_resolution"
        if any(name.startswith("qq.send") for name in tool_names):
            return "delivery"
        return "generic"

    @classmethod
    def _build_workflow_state(
        cls,
        *,
        overall_task_goal: TaskGoal | None,
        completed_outputs: list[OutputKind],
        completion,
        task_classification,
        candidate_state: CandidateState | None,
        candidate_paths: list[str],
        successful_actions: list[dict],
        primary_target_path: str | None,
    ) -> WorkflowState:
        workflow_family = cls._derive_workflow_family(
            task_classification=task_classification,
            overall_task_goal=overall_task_goal,
            successful_actions=successful_actions,
        )
        workflow_stage = cls._derive_workflow_stage(
            overall_task_goal=overall_task_goal,
            completed_outputs=completed_outputs,
            candidate_state=candidate_state,
            candidate_paths=candidate_paths,
            successful_actions=successful_actions,
            completion=completion,
        )
        candidates: list[WorkflowCandidate] = []
        candidate_names = list(candidate_state.candidate_names if candidate_state is not None else [])
        for index, path in enumerate(candidate_paths[:5], start=1):
            name = candidate_names[index - 1] if index - 1 < len(candidate_names) and candidate_names[index - 1] else Path(path).name
            subtitle = str(Path(path).parent)
            candidates.append(
                WorkflowCandidate(
                    candidate_id=str(index),
                    candidate_kind="file",
                    display_name=name,
                    path_or_ref=path,
                    subtitle=subtitle,
                    score=candidate_state.top_score if index == 1 and candidate_state is not None else 0.0,
                    evidence=[] if candidate_state is None else [candidate_state.source_tool, candidate_state.confidence_reason],
                )
            )
        return WorkflowState(
            workflow_family=workflow_family,
            workflow_stage=workflow_stage,
            required_outputs=[] if overall_task_goal is None else list(overall_task_goal.required_outputs),
            completed_outputs=list(completed_outputs),
            missing_outputs=list(completion.missing_outputs),
            primary_target_kind="file" if primary_target_path else "unknown",
            primary_target_ref=primary_target_path,
            candidates=candidates,
            next_allowed_actions=cls._next_actions_for_workflow_state(
                required_outputs=[] if overall_task_goal is None else list(overall_task_goal.required_outputs),
                completed_outputs=completed_outputs,
                candidate_paths=candidate_paths,
                primary_target_path=primary_target_path,
                successful_actions=successful_actions,
            ),
            metadata={},
        )

    @staticmethod
    def _build_execution_summary(
        user_text: str,
        overall_task_goal: TaskGoal | None,
        completed_outputs: list[OutputKind],
        tool_results: list[ToolCallResult],
        candidate_state: CandidateState | None,
        completion,
        stop_reason: str | None,
        document_delivery_intent=None,
        knowledge_request_intent=None,
        site_search_intent=None,
        decision_path: list[dict[str, object]] | None = None,
        planner_invocations: int = 0,
        critic_invocations: int = 0,
        planner_bypass_count: int = 0,
        task_classification=None,
    ) -> dict:
        successful_actions: list[dict] = []
        failed_actions: list[dict] = []
        candidate_paths: list[str] = []
        written_files: list[str] = []
        web_sources: list[dict] = []

        for result in tool_results:
            if result.status == "success":
                successful_actions.append({"tool_name": result.tool_name, "data": result.data})
                if result.tool_name == "retrieval.search_local_objects":
                    for candidate in result.data.get("candidates", [])[:5]:
                        path = candidate.get("path")
                        if path and path not in candidate_paths:
                            candidate_paths.append(path)
                if result.tool_name in {
                    "file.write",
                    "file.write_docx",
                    "file.write_xlsx",
                    "file.edit_docx",
                    "file.render_docx_from_template",
                    "document_agent.edit",
                }:
                    path = result.data.get("path")
                    if path:
                        written_files.append(path)
                if result.tool_name == "web.search":
                    for item in result.data.get("results", [])[:5]:
                        if isinstance(item, dict):
                            source = AgentKernel._normalize_web_source(item)
                            if source is not None and source not in web_sources:
                                web_sources.append(source)
                if result.tool_name == "web.fetch":
                    source = AgentKernel._normalize_web_source(result.data)
                    if source is not None and source not in web_sources:
                        web_sources.append(source)
                if result.tool_name == "web.open_page":
                    source = AgentKernel._normalize_web_source(result.data)
                    if source is not None and source not in web_sources:
                        web_sources.append(source)
                if result.tool_name == "web.research":
                    for item in result.data.get("sources", [])[:5]:
                        if isinstance(item, dict):
                            source = AgentKernel._normalize_web_source(item)
                            if source is not None and source not in web_sources:
                                web_sources.append(source)
            else:
                failed_actions.append(
                    {
                        "tool_name": result.tool_name,
                        "error": None if result.error is None else result.error.message,
                    }
                )

        requested_output_file = AgentKernel._extract_requested_output_file(user_text)
        if candidate_state is not None:
            for path in candidate_state.candidate_paths:
                if path and path not in candidate_paths:
                    candidate_paths.append(path)

        if any(
            isinstance(action, dict)
            and str(action.get("tool_name", "")).strip() in {"qq.send_file", "qq.send_text", "qq.send_voice"}
            for action in successful_actions
        ) and OutputKind.MESSAGE_SENT not in completed_outputs:
            completed_outputs.append(OutputKind.MESSAGE_SENT)

        workflow_stage = AgentKernel._derive_workflow_stage(
            overall_task_goal=overall_task_goal,
            completed_outputs=completed_outputs,
            candidate_state=candidate_state,
            candidate_paths=candidate_paths,
            successful_actions=successful_actions,
            completion=completion,
        )
        primary_target_path, delivery_target_path = AgentKernel._derive_primary_target_path(
            task_classification=task_classification,
            candidate_state=candidate_state,
            candidate_paths=candidate_paths,
            successful_actions=successful_actions,
            written_files=written_files,
        )
        workflow_state = AgentKernel._build_workflow_state(
            overall_task_goal=overall_task_goal,
            completed_outputs=completed_outputs,
            completion=completion,
            task_classification=task_classification,
            candidate_state=candidate_state,
            candidate_paths=candidate_paths,
            successful_actions=successful_actions,
            primary_target_path=primary_target_path,
        )

        return {
            "task_status": "completed" if completion.done else "partial",
            "workflow_stage": workflow_stage,
            "workflow_state": workflow_state.model_dump(mode="json"),
            "stop_reason": stop_reason,
            "decision_stats": {
                "planner_invocations": planner_invocations,
                "critic_invocations": critic_invocations,
                "planner_bypass_count": planner_bypass_count,
            },
            "decision_path": list(decision_path or []),
            "document_request": None
            if document_delivery_intent is None
            else document_delivery_intent.model_dump(mode="json"),
            "knowledge_request": None
            if knowledge_request_intent is None
            else knowledge_request_intent.model_dump(mode="json"),
            "site_search_request": None
            if site_search_intent is None
            else site_search_intent.model_dump(mode="json"),
            "task_classification": None
            if task_classification is None
            else (task_classification if isinstance(task_classification, dict) else task_classification.model_dump(mode="json")),
            "overall_task_goal": None
            if overall_task_goal is None
            else overall_task_goal.model_dump(mode="json"),
            "completed_outputs": [item.value for item in completed_outputs],
            "missing_outputs": [item.value for item in completion.missing_outputs],
            "candidate_paths": candidate_paths,
            "primary_target_path": primary_target_path,
            "delivery_target_path": delivery_target_path,
            "candidate_state": None if candidate_state is None else candidate_state.model_dump(mode="json"),
            "written_files": written_files,
            "requested_output_file": requested_output_file,
            "web_sources": web_sources,
            "successful_actions": successful_actions,
            "failed_actions": failed_actions,
        }

    @staticmethod
    def _normalize_web_source(payload: dict) -> dict | None:
        url = payload.get("final_url") or payload.get("url")
        if not isinstance(url, str) or not url.strip():
            return None
        source = {
            "title": str(payload.get("title", "")).strip(),
            "url": url.strip(),
            "source_domain": str(payload.get("source_domain", "")).strip(),
            "snippet": str(payload.get("snippet", "")).strip(),
            "excerpt": str(payload.get("excerpt", "")).strip(),
            "provider": str(payload.get("provider", "")).strip(),
            "extractor": str(payload.get("extractor", "")).strip(),
            "fetched_via": str(payload.get("fetched_via", "")).strip(),
            "published_at": payload.get("published_at"),
        }
        return source

    def _build_qq_history_workflow(
        self,
        *,
        user_text: str,
        overall_task_goal: TaskGoal | None,
        task_classification=None,
        recent_context: str = "",
    ) -> ToolDecision | None:
        task_kind = ""
        domain = ""
        if task_classification is not None:
            task_kind = str(getattr(task_classification, "task_kind", "") or "").strip().lower()
            domain = str(getattr(task_classification, "domain", "") or "").strip().lower()

        if domain != "qq_history" and task_kind not in {"history_lookup", "reply_lookup", "attachment_lookup"}:
            return None

        planned = self._plan_qq_history_arguments(
            user_text=user_text,
            task_kind=task_kind or "history_lookup",
            recent_context=recent_context,
        )
        selected_tool = str(planned.get("selected_tool") or "").strip()
        arguments = planned.get("arguments") if isinstance(planned.get("arguments"), dict) else {}

        if selected_tool not in {
            "qq.get_recent_messages",
            "qq.get_last_reply",
            "qq.search_history",
            "qq.get_recent_attachments",
        }:
            return ToolDecision(
                decision=DecisionType.CLARIFY,
                intent="clarify_qq_history_target",
                reason="The QQ history request could not be converted into a valid lookup step.",
                response_hint=self._render_clarification_hint(
                    user_text=user_text,
                    intent="clarify_qq_history_target",
                    missing_slots=["contact_query"],
                    fallback="",
                    style_hint="Ask which person, group, or chat thread should be checked, in one short natural sentence.",
                ),
                risk_level=RiskLevel.LOW,
                overall_task_goal=overall_task_goal,
            )

        if selected_tool == "qq.search_history" and not (
            str(arguments.get("query") or "").strip()
            or str(arguments.get("contact_query") or "").strip()
            or bool(arguments.get("reply_after_last_outbound"))
        ):
            return ToolDecision(
                decision=DecisionType.CLARIFY,
                intent="clarify_qq_history_target",
                reason="The QQ history lookup is missing the target contact or search clue.",
                response_hint=self._render_clarification_hint(
                    user_text=user_text,
                    intent="clarify_qq_history_target",
                    missing_slots=["contact_query"],
                    fallback="",
                    style_hint="Ask which person, group, or earlier conversation should be checked.",
                ),
                risk_level=RiskLevel.LOW,
                overall_task_goal=overall_task_goal,
            )

        intent = {
            "history_lookup": "lookup_qq_history",
            "reply_lookup": "lookup_qq_reply",
            "attachment_lookup": "lookup_qq_attachments",
        }.get(task_kind, "lookup_qq_history")
        summary = {
            "history_lookup": "Look up the requested QQ conversation history.",
            "reply_lookup": "Look up the requested QQ reply history.",
            "attachment_lookup": "Look up the requested QQ attachments.",
        }.get(task_kind, "Look up the requested QQ conversation history.")

        return ToolDecision(
            decision=DecisionType.TOOL_CALL,
            intent=intent,
            reason="Task classification identified a QQ history request, and the LLM produced a grounded QQ lookup step.",
            selected_tool=selected_tool,
            arguments=arguments,
            risk_level=RiskLevel.LOW,
            overall_task_goal=TaskGoal(
                summary=summary,
                required_outputs=[OutputKind.OBJECT_DETAILS],
                completion_mode="outputs",
            ),
            expected_step_outputs=[OutputKind.OBJECT_DETAILS],
        )

    def _build_system_utility_workflow(
            self,
            *,
            user_text: str,
            completed_outputs: list[OutputKind],
            overall_task_goal: TaskGoal | None,
            task_classification=None,
            recent_context: str = "",
    ) -> ToolDecision | None:
        text = str(user_text or "").strip()
        task_kind = ""
        if task_classification is not None:
            task_kind = str(getattr(task_classification, "task_kind", "") or "").strip().lower()

        # 1) 当前时间
        if task_kind == "get_current_time":
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="get_current_time",
                reason="Task classification identified a current-time system utility request.",
                selected_tool="system.get_time",
                arguments={"kind": "time"},
                risk_level=RiskLevel.LOW,
                overall_task_goal=TaskGoal(
                    summary="Answer the user's current time question.",
                    required_outputs=[],
                    completion_mode="success",
                ),
            )

        # 2) 创建提醒：由 LLM 直接产出最终执行参数
        if task_kind == "create_reminder":
            planned_args = self._plan_system_utility_arguments(
                user_text=text,
                task_kind="create_reminder",
                recent_context=recent_context,
            )

            when_iso = str(planned_args.get("when_iso", "") or "").strip()
            timezone = str(planned_args.get("timezone", "") or "Asia/Shanghai").strip()
            message = str(planned_args.get("message", "") or "").strip()

            if when_iso and message:
                return ToolDecision(
                    decision=DecisionType.TOOL_CALL,
                    intent="create_reminder",
                    reason="Task classification identified a reminder creation request, and the LLM produced structured execution arguments.",
                    selected_tool="system.create_reminder",
                    arguments={
                        "when_iso": when_iso,
                        "timezone": timezone,
                        "message": message,
                    },
                    risk_level=RiskLevel.LOW,
                    overall_task_goal=TaskGoal(
                        summary="Create the requested reminder.",
                        required_outputs=[],
                        completion_mode="success",
                    )
                )

            return ToolDecision(
                decision=DecisionType.CLARIFY,
                intent="clarify_reminder_details",
                reason="The reminder request could not be converted into complete structured execution arguments.",
                response_hint=self._render_clarification_hint(
                    user_text=text,
                    intent="clarify_reminder_details",
                    missing_slots=["when_iso", "message"],
                    fallback="",
                    style_hint="Ask for the exact reminder time and what should be said in one short natural sentence.",
                ),
                risk_level=RiskLevel.LOW,
                overall_task_goal=overall_task_goal,
            )

        # 3) 列提醒
        if task_kind == "list_reminders":
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="list_reminders",
                reason="Task classification identified a reminder listing request.",
                selected_tool="system.list_reminders",
                arguments={"status": "scheduled"},
                risk_level=RiskLevel.LOW,
                overall_task_goal=TaskGoal(
                    summary="List current reminders.",
                    required_outputs=[],
                    completion_mode="success",
                ),
            )

        # 4) 取消提醒
        if task_kind == "cancel_reminder":
            planned_args = self._plan_system_utility_arguments(
                user_text=text,
                task_kind="cancel_reminder",
                recent_context=recent_context,
            )
            reminder_id = str(planned_args.get("reminder_id", "") or "").strip()

            if reminder_id:
                return ToolDecision(
                    decision=DecisionType.TOOL_CALL,
                    intent="cancel_reminder",
                    reason="Task classification identified a reminder cancellation request, and the LLM produced the reminder identifier.",
                    selected_tool="system.cancel_reminder",
                    arguments={"reminder_id": reminder_id},
                    risk_level=RiskLevel.LOW,
                    overall_task_goal=TaskGoal(
                        summary="Cancel the specified reminder.",
                        required_outputs=[],
                        completion_mode="success",
                    ),
                )

            return ToolDecision(
                decision=DecisionType.CLARIFY,
                intent="clarify_reminder_to_cancel",
                reason="The user wants to cancel a reminder but the target reminder could not be determined.",
                response_hint=self._render_clarification_hint(
                    user_text=text,
                    intent="clarify_reminder_to_cancel",
                    missing_slots=["reminder_id"],
                    fallback="",
                    style_hint="Ask the user which reminder to cancel, using reminder id or reminder content.",
                ),
                risk_level=RiskLevel.LOW,
                overall_task_goal=overall_task_goal,
            )

        # 5) 极小兜底，只保留高确定性情况
        compact = "".join(text.lower().split())
        if compact in {"现在几点", "现在几点了", "几点了"}:
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="get_current_time",
                reason="Fallback matched a high-confidence current-time request.",
                selected_tool="system.get_time",
                arguments={"kind": "time"},
                risk_level=RiskLevel.LOW,
                overall_task_goal=TaskGoal(
                    summary="Answer the user's current time question.",
                    required_outputs=[],
                ),
            )

        return None

    @staticmethod
    def _extract_reminder_id_from_text(text: str) -> str | None:
        import re

        match = re.search(r"(rem_[A-Za-z0-9]+)", text)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _extract_reminder_request_from_text(text: str) -> dict[str, str]:
        normalized = str(text or "").strip()

        prefixes = ("提醒我", "设提醒", "定提醒", "设闹钟", "定闹钟", "叫我")
        body = normalized
        for prefix in prefixes:
            if body.startswith(prefix):
                body = body[len(prefix):].strip()
                break

        # 粗略拆分：前半是时间，后半是内容
        # 例如：明天早上九点开会
        # 第一版先尽量粗略，后续再增强
        if not body:
            return {"when_text": "", "message": ""}

        markers = ("后", "点", "分", "明天", "今天", "后天", "下午", "上午", "晚上", "中午")
        split_idx = -1
        for idx, ch in enumerate(body):
            if any(body[: idx + 1].endswith(marker) for marker in markers):
                split_idx = idx

        if split_idx == -1:
            return {"when_text": "", "message": body}

        when_text = body[: split_idx + 1].strip(" ，,。")
        message = body[split_idx + 1:].strip(" ，,。")
        return {
            "when_text": when_text,
            "message": message,
        }

    def _render_pending_task_response_bundle(
        self,
        *,
        pending_task: PendingTask,
        execution_summary: dict[str, Any],
        observations: list[str],
    ) -> dict[str, str] | None:
        response_hint = self._build_pending_task_response_hint(
            pending_task=pending_task,
            execution_summary=execution_summary,
        )
        return self._render_execution_response_bundle(
            observations=observations,
            response_hint=response_hint,
            execution_summary=execution_summary,
        )

    def _render_grounded_partial_response_bundle(
        self,
        *,
        execution_summary: dict[str, Any],
        observations: list[str],
    ) -> dict[str, str] | None:
        response_hint = self._build_grounded_partial_response_hint(execution_summary)
        return self._render_execution_response_bundle(
            observations=observations,
            response_hint=response_hint,
            execution_summary=execution_summary,
        )

    def _render_execution_response_bundle(
        self,
        *,
        observations: list[str],
        response_hint: str,
        execution_summary: dict[str, Any],
    ) -> dict[str, str] | None:
        try:
            system_name = getattr(self.config, "system_name", "local-agent")
            persona_name = getattr(self.config, "persona_name", None)
            persona_profile = getattr(self.config, "persona_profile", None)
            display_style_prompt = getattr(self.config, "display_style_prompt", None)
            if hasattr(self.llm_client, "render_tool_response_bundle"):
                return self.llm_client.render_tool_response_bundle(
                    system_name=system_name,
                    messages=self._context_messages(),
                    observations=observations,
                    response_hint=response_hint,
                    execution_summary=execution_summary,
                    persona_name=persona_name,
                    persona_profile=persona_profile,
                    display_style_prompt=display_style_prompt,
                    speech_style_prompt=getattr(self.config, "speech_style_prompt", None),
                    speech_max_chars=int(getattr(self.config, "speech_max_chars", 80) or 80),
                )
            final_response = self.llm_client.render_response(
                system_name=system_name,
                messages=self._context_messages(),
                observations=observations,
                response_hint=response_hint,
                execution_summary=execution_summary,
                persona_name=persona_name,
                persona_profile=persona_profile,
                display_style_prompt=display_style_prompt,
            )
        except Exception:
            return None
        return {"display_text": final_response, "speech_text": final_response}

    @staticmethod
    def _build_grounded_partial_response_hint(execution_summary: dict[str, Any]) -> str:
        candidate_paths = execution_summary.get("candidate_paths", [])
        missing_outputs = execution_summary.get("missing_outputs", [])
        stop_reason = str(execution_summary.get("stop_reason", "") or "").strip()
        requested_output_file = str(execution_summary.get("requested_output_file", "") or "").strip()

        hints = [
            "The task is only partially complete.",
            "Explain clearly what has already been confirmed and what is still missing.",
            "Do not claim the task is finished.",
        ]
        if candidate_paths:
            hints.append("Candidate paths are available. Mention the most relevant candidates and ask the user to choose one or narrow the scope.")
        if stop_reason == "waiting_for_selection":
            hints.append("The workflow is waiting for the user to select one candidate before the write step can continue.")
        if "file_written" in missing_outputs:
            if requested_output_file:
                hints.append(f"Do not say `{requested_output_file}` was written, because that write step has not succeeded yet.")
            else:
                hints.append("Do not say the result file was written, because the write step has not succeeded yet.")
        elif missing_outputs:
            hints.append(f"The still-missing outputs are: {', '.join(str(item) for item in missing_outputs)}.")
        return " ".join(hints)

    @staticmethod
    def _build_pending_task_response_hint(
        *,
        pending_task: PendingTask,
        execution_summary: dict[str, Any],
    ) -> str:
        missing_slots = ", ".join(slot for slot in pending_task.missing_slots if slot)
        stop_reason = str(execution_summary.get("stop_reason", "") or "").strip()
        hints = [
            "The task is currently paused and needs one more user clarification before execution can continue.",
            "Reply in natural Chinese and make the next required input feel concrete and contextual.",
            "Do not use generic canned wording.",
        ]
        if pending_task.selection_candidates:
            hints.append("There are candidate files available. Ask the user to choose one candidate or give a narrowing clue such as name, date, location, or file type.")
        if missing_slots:
            hints.append(f"The missing slots are: {missing_slots}.")
        if stop_reason == "waiting_for_selection":
            hints.append("The workflow is specifically waiting for candidate confirmation.")
        return " ".join(hints)

    @staticmethod
    def _get_primary_task_graph_subtask(task_graph: Any) -> Any | None:
        if task_graph is None:
            return None
        primary_task_id = str(getattr(task_graph, "primary_task_id", "") or "").strip()
        subtasks = list(getattr(task_graph, "subtasks", []) or [])
        if primary_task_id:
            for subtask in subtasks:
                if str(getattr(subtask, "task_id", "") or "").strip() == primary_task_id:
                    return subtask
        return subtasks[0] if subtasks else None

    def _build_task_graph_pending_task(
        self,
        *,
        user_text: str,
        task_graph: Any,
        overall_task_goal: TaskGoal | None,
    ) -> PendingTask | None:
        primary_subtask = self._get_primary_task_graph_subtask(task_graph)
        if primary_subtask is None:
            return None

        status = str(getattr(primary_subtask, "status", "") or "").strip().lower()
        missing_slots = [
            str(slot).strip()
            for slot in getattr(primary_subtask, "missing_slots", []) or []
            if str(slot).strip()
        ]
        needs_clarification = bool(getattr(task_graph, "needs_clarification", False))
        if status != "waiting_for_input" and not (needs_clarification and missing_slots):
            return None

        fallback = (
            str(getattr(task_graph, "followup_text", "") or "").strip()
            or "我先接住这件事了，但还差一个关键信息。你把要我查或处理的具体对象补一句，我就继续。"
        )
        clarification_text = self._render_clarification_hint(
            user_text=user_text,
            intent=str(getattr(primary_subtask, "kind", "") or "clarify").strip() or "clarify",
            missing_slots=missing_slots,
            fallback=fallback,
            style_hint=(
                "Try to continue the immediately preceding topic or claim when it is clear from context. "
                "Only ask the user for the missing target or subject if the context still does not identify one concrete thing to investigate."
            ),
        )
        summary = (
            str(getattr(primary_subtask, "summary", "") or "").strip()
            or str(getattr(primary_subtask, "task_text", "") or "").strip()
            or str(getattr(task_graph, "primary_task_text", "") or "").strip()
            or str(user_text or "").strip()
        )
        return PendingTask(
            task_id=str(getattr(primary_subtask, "task_id", "") or f"pending_{uuid.uuid4().hex[:10]}"),
            intent=str(getattr(primary_subtask, "kind", "") or "clarify").strip() or "clarify",
            summary=summary or "Need one more concrete detail before continuing.",
            original_user_request=user_text,
            state_kind="clarification",
            clarification_prompt=clarification_text,
            overall_task_goal=overall_task_goal,
            missing_slots=missing_slots,
            collected_slots=dict(getattr(primary_subtask, "slot_values", {}) or {}),
            resume_hint=clarification_text,
        )

    def render_follow_up_cancel_response(
        self,
        *,
        latest_user_text: str,
        pending_task: PendingTask,
    ) -> str:
        execution_summary = {
            "task_status": "cancelled",
            "stop_reason": "user_cancelled_pending_task",
            "pending_task": pending_task.model_dump(mode="json"),
        }
        response_hint = (
            "The user decided to cancel the unfinished task. "
            "Acknowledge the cancellation briefly in natural Chinese, and mention the cancelled task in a grounded way when helpful. "
            "Do not use generic canned wording."
        )
        bundle = self._render_execution_response_bundle(
            observations=[],
            response_hint=response_hint,
            execution_summary=execution_summary,
        )
        if bundle is not None:
            return bundle["display_text"]
        raise RuntimeError("follow_up_cancel_response_generation_failed")

    @staticmethod
    def _tool_action_label(action: str) -> str:
        normalized = str(action or "").strip().lower()
        mapping = {
            "replace": "替换了一段内容",
            "insert_after": "新增了一段内容",
            "delete": "删除了一段内容",
        }
        return mapping.get(normalized, "处理了一处内容")

    @classmethod
    def _build_structured_document_edit_response_bundle(cls, execution_summary: dict) -> dict[str, str] | None:
        task_classification = execution_summary.get("task_classification") or {}
        task_kind = str(task_classification.get("task_kind", "") or "").strip().lower()
        if task_kind != "document_edit":
            return None

        successful_actions = execution_summary.get("successful_actions") or []
        if not isinstance(successful_actions, list):
            return None

        for action in reversed(successful_actions):
            if not isinstance(action, dict):
                continue
            tool_name = str(action.get("tool_name", "") or "").strip()
            data = action.get("data") if isinstance(action.get("data"), dict) else {}
            if tool_name in {"file.edit_docx", "document_agent.edit"}:
                path = str(data.get("path") or data.get("output_path") or data.get("source_path") or "").strip()
                file_label = path or "目标文档"
                applied_edits = data.get("applied_edits") if isinstance(data.get("applied_edits"), list) else []
                try:
                    edit_count = int(data.get("edit_count") or len(applied_edits) or 0)
                except (TypeError, ValueError):
                    edit_count = len(applied_edits)
                summary = " ".join(str(data.get("summary", "") or "").split())
                lines = [f"已更新文件：{file_label}"]
                if summary:
                    lines.append(f"修改摘要：{summary}")
                if edit_count > 0:
                    lines.append(f"本次共应用 {edit_count} 处修改。")
                if applied_edits:
                    lines.append("实际写入摘要：")
                    for item in applied_edits[:2]:
                        if not isinstance(item, dict):
                            continue
                        label = cls._tool_action_label(item.get("action", ""))
                        block_id = str(item.get("block_id", "") or "").strip()
                        preview = " ".join(str(item.get("text_preview", "") or "").split())
                        if len(preview) > 160:
                            preview = preview[:157].rstrip() + "..."
                        detail = label
                        if block_id:
                            detail += f"（{block_id}）"
                        if preview:
                            detail += f"：{preview}"
                        lines.append(f"- {detail}")
                speech = "文档已经改好了。"
                if edit_count > 0:
                    speech = f"文档已经改好了，这次共处理 {edit_count} 处修改。"
                return {"display_text": "\n".join(lines), "speech_text": speech}

            if tool_name in {"file.render_docx_from_template", "file.write_docx", "file.write"}:
                path = str(data.get("path", "") or "").strip()
                if not path:
                    continue
                display = f"已生成文件：{path}"
                if tool_name == "file.write":
                    display = f"已写入文件：{path}"
                return {"display_text": display, "speech_text": "文件已经处理好了。"}

        return None

    @classmethod
    def _build_structured_qq_history_response_bundle(cls, execution_summary: dict) -> dict[str, str] | None:
        task_classification = execution_summary.get("task_classification") or {}
        task_kind = str(task_classification.get("task_kind", "") or "").strip().lower()
        domain = str(task_classification.get("domain", "") or "").strip().lower()
        if domain != "qq_history" and task_kind not in {"history_lookup", "reply_lookup"}:
            return None

        successful_actions = execution_summary.get("successful_actions") or []
        if not isinstance(successful_actions, list):
            return None

        for action in reversed(successful_actions):
            if not isinstance(action, dict):
                continue
            tool_name = str(action.get("tool_name", "") or "").strip()
            if tool_name != "qq.search_history":
                continue
            data = action.get("data") if isinstance(action.get("data"), dict) else {}
            messages = data.get("messages") if isinstance(data.get("messages"), list) else []
            contact_query = str(data.get("contact_query", "") or "").strip()
            contact_label = contact_query or "对方"
            if not messages:
                display = f"没有，我这边暂时没查到和{contact_label}相关的历史记录。"
                speech = f"没有查到和{contact_label}的历史记录。"
                return {"display_text": display, "speech_text": speech}

            inbound_messages = [item for item in messages if isinstance(item, dict) and str(item.get("direction", "") or "").strip().lower() == "inbound"]
            reference_messages = inbound_messages or [item for item in messages if isinstance(item, dict)]
            if not reference_messages:
                continue

            latest_message = reference_messages[0]
            latest_time = cls._format_qq_history_time(latest_message.get("created_at"))
            contact_name = str(latest_message.get("contact_name", "") or "").strip()
            if contact_name:
                contact_label = contact_name

            topic_text = cls._summarize_qq_history_topics(reference_messages)
            if inbound_messages:
                lines = [f"有，{contact_label}之前找过我。"]
                speech = f"有，{contact_label}之前找过我。"
            else:
                lines = [f"我这边查到和{contact_label}有过聊天记录，不过没看到对方主动来找我的消息。"]
                speech = f"我查到和{contact_label}有过聊天记录。"

            detail_parts: list[str] = []
            if latest_time:
                detail_parts.append(f"最近一次能对上的记录是{latest_time}")
            if topic_text:
                detail_parts.append(f"主要提到{topic_text}")
            if detail_parts:
                lines.append("，".join(detail_parts) + "。")
            return {"display_text": "\n".join(lines), "speech_text": speech}

        return None

    @staticmethod
    def _format_qq_history_time(value: object) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return ""
        return f"{dt.month}月{dt.day}日{dt.hour:02d}:{dt.minute:02d}"

    @staticmethod
    def _summarize_qq_history_topics(messages: list[dict[str, object]]) -> str:
        topics: list[str] = []
        seen: set[str] = set()
        for item in messages:
            if not isinstance(item, dict):
                continue
            for key in ("text", "summary"):
                raw = " ".join(str(item.get(key, "") or "").split())
                if not raw:
                    continue
                raw = raw.replace("对方在", "").replace("你在", "").replace("说了：", "").replace("发送了附件：", "")
                normalized = raw.strip("，。；： ")
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                topics.append(normalized)
                break
            if len(topics) >= 2:
                break
        if not topics:
            return ""
        quoted = [f"“{item[:18].rstrip()}”" for item in topics]
        return "、".join(quoted)

    @staticmethod
    def _allowed_actions_for_goal(
            *,
            workflow_family: str,
            overall_task_goal: TaskGoal | None,
            completed_outputs: list[OutputKind],
            candidate_state: CandidateState | None,
    ) -> set[str]:
        if overall_task_goal is None:
            return {"respond", "clarify", "tool_call"}

        required = set(overall_task_goal.required_outputs)
        completed = set(completed_outputs)
        missing = required - completed

        if workflow_family in {"document_summary", "local_lookup"}:
            if OutputKind.OBJECT_CANDIDATES in missing:
                return {"tool_call", "clarify"}
            if OutputKind.FILE_CONTENTS in missing:
                return {"tool_call", "clarify"}
            return {"respond", "clarify"}

        if workflow_family in {"document_edit"}:
            if OutputKind.OBJECT_CANDIDATES in missing:
                return {"tool_call", "clarify"}
            if OutputKind.OBJECT_DETAILS in missing:
                return {"tool_call", "clarify"}
            if OutputKind.FILE_WRITTEN in missing:
                return {"tool_call", "clarify"}
            return {"respond", "clarify"}

        if workflow_family in {"qq_history", "history_lookup", "reply_lookup", "attachment_lookup"}:
            if OutputKind.OBJECT_DETAILS in missing:
                return {"tool_call", "clarify"}
            return {"respond", "clarify"}

        if workflow_family in {"delivery"}:
            if OutputKind.OBJECT_CANDIDATES in missing:
                return {"tool_call", "clarify"}
            if OutputKind.MESSAGE_SENT in missing:
                return {"tool_call", "clarify"}
            return {"respond", "clarify"}

        if workflow_family in {"web_lookup", "web_target"}:
            if OutputKind.SEARCH_RESULTS in missing or OutputKind.WEB_CONTENT in missing:
                return {"tool_call", "clarify"}
            return {"respond", "clarify"}

        return {"respond", "tool_call", "clarify"}

    def _supports_message_delivery(self) -> bool:
        return self.registry.has_tool("qq.send_file")

    def _build_shared_workflow_proposals(
        self,
        *,
        user_text: str,
        completed_outputs: list[OutputKind],
        candidate_state: CandidateState | None,
        overall_task_goal: TaskGoal | None,
        task_classification,
        knowledge_intent,
        document_delivery_intent,
        site_search_intent,
        tool_results: list[ToolCallResult],
        recent_context: str = "",
        local_search_decision: ToolDecision | None = None,
        allow_programmatic_lookup_workflow: bool = True,
        local_search_source: str = "local_search_planner",
        local_search_priority: int = 98,
        local_search_reason: str = "LLM-planned initial local search workflow",
        document_summary_priority: int = 90,
        document_summary_reason: str = "Document summary workflow",
        document_operation_priority: int = 92,
        document_operation_reason: str = "Document operation workflow",
        file_delivery_priority: int = 88,
        file_delivery_reason: str = "File delivery workflow",
        local_lookup_priority: int = 84,
        local_lookup_reason: str = "Local lookup workflow",
        web_target_priority: int = 86,
        web_target_reason: str = "Web target workflow",
        web_lookup_priority: int = 74,
        web_lookup_reason: str = "General web lookup workflow",
        include_scheduled_task_fire: bool = False,
        scheduled_task_fire_priority: int = 97,
        scheduled_task_fire_reason: str = "Deliver a fired scheduled QQ notification.",
        runtime_channel: str | None = None,
        runtime_channel_context: dict[str, Any] | None = None,
        include_qq_history: bool = False,
        qq_history_priority: int = 94,
        qq_history_reason: str = "QQ history workflow",
        include_system_utility: bool = False,
        system_utility_priority: int = 93,
        system_utility_reason: str = "System utility workflow",
    ) -> list[WorkflowProposal]:
        supports_message_delivery = self._supports_message_delivery()
        proposals: list[WorkflowProposal] = [
            WorkflowProposal(
                source=local_search_source,
                family=self._local_search_planner_family(task_classification),
                priority=local_search_priority,
                reason=local_search_reason,
                decision=local_search_decision,
            )
        ]

        if include_scheduled_task_fire:
            proposals.append(
                WorkflowProposal(
                    source="scheduled_task_fire",
                    family="delivery",
                    priority=scheduled_task_fire_priority,
                    reason=scheduled_task_fire_reason,
                    decision=self._build_fired_scheduled_task_delivery_workflow(
                        runtime_channel=runtime_channel,
                        runtime_channel_context=runtime_channel_context,
                    ),
                )
            )

        if include_qq_history:
            proposals.append(
                WorkflowProposal(
                    source="qq_history",
                    family="qq_history",
                    priority=qq_history_priority,
                    reason=qq_history_reason,
                    decision=self._build_qq_history_workflow(
                        user_text=user_text,
                        overall_task_goal=overall_task_goal,
                        task_classification=task_classification,
                        recent_context=recent_context,
                    ),
                )
            )

        proposals.extend(
            [
                WorkflowProposal(
                    source="document_summary",
                    family="document_summary",
                    priority=document_summary_priority,
                    reason=document_summary_reason,
                    decision=None if not allow_programmatic_lookup_workflow else self._build_document_summary_workflow(
                        user_text=user_text,
                        completed_outputs=completed_outputs,
                        candidate_state=candidate_state,
                        overall_task_goal=overall_task_goal,
                        knowledge_intent=knowledge_intent,
                        task_classification=task_classification,
                        supports_message_delivery=supports_message_delivery,
                    ),
                ),
                WorkflowProposal(
                    source="document_operation",
                    family="document_operation",
                    priority=document_operation_priority,
                    reason=document_operation_reason,
                    decision=None if not allow_programmatic_lookup_workflow else self._build_document_operation_workflow(
                        user_text=user_text,
                        completed_outputs=completed_outputs,
                        candidate_state=candidate_state,
                        overall_task_goal=overall_task_goal,
                        knowledge_intent=knowledge_intent,
                        task_classification=task_classification,
                        supports_message_delivery=supports_message_delivery,
                    ),
                ),
                WorkflowProposal(
                    source="file_delivery",
                    family="file_delivery",
                    priority=file_delivery_priority,
                    reason=file_delivery_reason,
                    decision=None if not allow_programmatic_lookup_workflow else self._build_file_delivery_workflow(
                        user_text=user_text,
                        completed_outputs=completed_outputs,
                        candidate_state=candidate_state,
                        overall_task_goal=overall_task_goal,
                        knowledge_intent=knowledge_intent,
                        task_classification=task_classification,
                        supports_message_delivery=supports_message_delivery,
                    ),
                ),
                WorkflowProposal(
                    source="local_lookup",
                    family="local_lookup",
                    priority=local_lookup_priority,
                    reason=local_lookup_reason,
                    decision=None if not allow_programmatic_lookup_workflow else self._build_local_file_lookup_workflow(
                        user_text=user_text,
                        completed_outputs=completed_outputs,
                        candidate_state=candidate_state,
                        knowledge_intent=knowledge_intent,
                        overall_task_goal=overall_task_goal,
                        task_classification=task_classification,
                    ),
                ),
                WorkflowProposal(
                    source="web_target",
                    family="web_target",
                    priority=web_target_priority,
                    reason=web_target_reason,
                    decision=self._build_web_target_workflow(
                        user_text=user_text,
                        completed_outputs=completed_outputs,
                        tool_results=tool_results,
                    ),
                ),
                WorkflowProposal(
                    source="web_lookup",
                    family="web_lookup",
                    priority=web_lookup_priority,
                    reason=web_lookup_reason,
                    decision=self.web_retrieval_strategy.build_initial_lookup(
                        user_text=user_text,
                        completed_outputs=completed_outputs,
                        delivery_intent=document_delivery_intent,
                        knowledge_intent=knowledge_intent,
                        site_search_intent=site_search_intent,
                    ),
                ),
            ]
        )

        if include_system_utility:
            proposals.append(
                WorkflowProposal(
                    source="system_utility",
                    family="system_utility",
                    priority=system_utility_priority,
                    reason=system_utility_reason,
                    decision=self._build_system_utility_workflow(
                        user_text=user_text,
                        completed_outputs=completed_outputs,
                        overall_task_goal=overall_task_goal,
                        task_classification=task_classification,
                        recent_context=recent_context,
                    ),
                )
            )

        return proposals

    def _bind_workflow_plan(
            self,
            *,
            user_text: str,
            intent_bundle,
            overall_task_goal: TaskGoal | None,
            candidate_state: CandidateState | None,
            completed_outputs: list[OutputKind],
            tool_results: list[ToolCallResult],
            recent_context: str = "",
            runtime_channel: str | None = None,
            runtime_channel_context: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        proposals = self._build_shared_workflow_proposals(
            user_text=user_text,
            completed_outputs=completed_outputs,
            candidate_state=candidate_state,
            overall_task_goal=overall_task_goal,
            task_classification=intent_bundle.task_classification,
            knowledge_intent=intent_bundle.knowledge_request,
            document_delivery_intent=intent_bundle.document_delivery,
            site_search_intent=intent_bundle.site_search,
            tool_results=tool_results,
            recent_context=recent_context,
            local_search_decision=self._build_llm_local_search_workflow(
                user_text=user_text,
                completed_outputs=completed_outputs,
                candidate_state=candidate_state,
                overall_task_goal=overall_task_goal,
                knowledge_intent=intent_bundle.knowledge_request,
                task_classification=intent_bundle.task_classification,
                supports_message_delivery=self._supports_message_delivery(),
            ),
            include_scheduled_task_fire=True,
            runtime_channel=runtime_channel,
            runtime_channel_context=runtime_channel_context,
            include_qq_history=True,
            include_system_utility=True,
        )

        chosen = WorkflowSelector.choose(
            proposals=proposals,
            intent_bundle=intent_bundle,
            completed_outputs=completed_outputs,
            candidate_state=candidate_state,
        )

        if chosen is None:
            return {
                "workflow_family": "generic",
                "workflow_decision": None,
                "overall_task_goal": overall_task_goal,
            }

        chosen_family = "generic"
        for proposal in proposals:
            if proposal.decision is chosen:
                chosen_family = proposal.family
                break

        return {
            "workflow_family": chosen_family,
            "workflow_decision": chosen,
            "overall_task_goal": chosen.overall_task_goal or overall_task_goal,
        }

    @staticmethod
    def _should_force_grounded_partial(execution_summary: dict) -> bool:
        if execution_summary.get("task_status") == "completed":
            completion_review = execution_summary.get("completion_review") or {}
            grounding_review = execution_summary.get("grounding_review") or {}
            if completion_review.get("force_partial") or grounding_review.get("force_partial"):
                return True
            return False
        if execution_summary.get("stop_reason") == "duplicate_tool_request_after_success":
            return False
        if execution_summary.get("stop_reason") == "finalize_after_decision_error":
            return True
        if execution_summary.get("failed_actions"):
            return True
        completion_review = execution_summary.get("completion_review") or {}
        grounding_review = execution_summary.get("grounding_review") or {}
        if completion_review.get("force_partial") or grounding_review.get("force_partial"):
            return True
        return False

    def _plan_system_utility_arguments(
        self,
        *,
        user_text: str,
        task_kind: str,
        recent_context: str = "",
    ) -> dict[str, object]:
        if not hasattr(self.llm_client, "plan_system_utility_arguments"):
            return {}

        now_local = datetime.now().astimezone()
        try:
            payload = self.llm_client.plan_system_utility_arguments(
                user_text=user_text,
                task_kind=task_kind,
                current_time_iso=now_local.isoformat(),
                timezone=str(now_local.tzinfo or "Asia/Shanghai"),
                recent_context=recent_context,
                persona_name=getattr(self.config, "persona_name", "") or "",
                persona_profile=getattr(self.config, "persona_profile", "") or "",
            )
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _plan_qq_history_arguments(
        self,
        *,
        user_text: str,
        task_kind: str,
        recent_context: str = "",
    ) -> dict[str, object]:
        if not hasattr(self.llm_client, "plan_qq_history_arguments"):
            return {}

        now_local = datetime.now().astimezone()
        try:
            payload = self.llm_client.plan_qq_history_arguments(
                user_text=user_text,
                task_kind=task_kind,
                current_time_iso=now_local.isoformat(),
                timezone=str(now_local.tzinfo or "Asia/Shanghai"),
                recent_context=recent_context,
                persona_name=getattr(self.config, "persona_name", "") or "",
                persona_profile=getattr(self.config, "persona_profile", "") or "",
            )
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def handle_scheduled_task(
            self,
            *,
            task: dict,
    ) -> TurnArtifacts | None:
        task_type = str(task.get("task_type") or "").strip().lower()
        task_payload = task.get("task_payload") or {}
        if not isinstance(task_payload, dict):
            task_payload = {}

        session_id = str(task.get("session_id") or "").strip() or None
        channel = str(task.get("channel") or "").strip() or None
        channel_runtime = task_payload.get("channel_runtime")
        if not isinstance(channel_runtime, dict):
            channel_runtime = None

        if task_type == "notify":
            text = str(task.get("message") or task_payload.get("text") or "").strip()
            if not text:
                return None
            scheduled_runtime_context = dict(channel_runtime or {})
            scheduled_runtime_context["scheduled_task"] = {
                "phase": "fired",
                "task_type": "notify",
                "reminder_id": str(task.get("reminder_id") or "").strip() or None,
                "message": text,
            }
            return self.handle_user_input(
                f"把这句话发到当前QQ会话：{text}",
                runtime_session_id=session_id,
                runtime_channel=channel,
                runtime_channel_context=scheduled_runtime_context,
            )

        if task_type == "deferred_agent_task":
            instruction_text = str(task_payload.get("instruction_text") or "").strip()
            if not instruction_text:
                return None
            return self.handle_user_input(
                instruction_text,
                runtime_session_id=session_id,
                runtime_channel=channel,
                runtime_channel_context=channel_runtime,
            )

        return None

    def _render_clarification_hint(
            self,
            *,
            user_text: str,
            intent: str,
            missing_slots: list[str],
            fallback: str,
            style_hint: str = "",
    ) -> str:
        unavailable = self.llm_client.build_unavailable_response(
            RuntimeError("clarification_hint_generation_failed")
        )
        if not hasattr(self.llm_client, "render_clarification_hint"):
            return str(fallback).strip() or unavailable
        try:
            text = self.llm_client.render_clarification_hint(
                user_text=user_text,
                intent=intent,
                missing_slots=missing_slots,
                style_hint=style_hint,
            )
            return str(text).strip() or str(fallback).strip() or unavailable
        except Exception:
            return str(fallback).strip() or unavailable
