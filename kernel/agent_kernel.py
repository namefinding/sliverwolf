from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Any
from local_agent.app.scope_resolver import infer_scope_root
from local_agent.artifacts.output_planner import OutputArtifactPlanner
from local_agent.intent.models import AnswerabilityAssessment, IntentBundle, TaskEnvelope
from local_agent.intent.service import IntentService
from local_agent.kernel.completion_judge import CompletionJudge
from local_agent.kernel.context_builder import ContextBuilder
from local_agent.kernel.decision_critic import DecisionCritic
from local_agent.kernel.decision_validator import DecisionValidationError, DecisionValidator
from local_agent.kernel.execution_critic import ExecutionCritic
from local_agent.kernel.file_retrieval_strategy import FileRetrievalStrategy
from local_agent.kernel.guardrails import Guardrails
from local_agent.kernel.loop_controller import LoopController
from local_agent.kernel.request_intent_analyzer import RequestIntentAnalyzer
from local_agent.kernel.web_retrieval_strategy import WebRetrievalStrategy
from local_agent.llm.ollama_client import OllamaClient
from local_agent.memory.warm_memory import WarmMemoryService
from local_agent.modules.base import ToolRegistry
from local_agent.modules.system_utility.parser import parse_when_text
from local_agent.protocol.execution_contract import build_tool_execution_context
from local_agent.protocol.models import (
    AgentConfig,
    CandidateState,
    DecisionReview,
    DecisionType,
    ExecutionReview,
    InstructionIntent,
    MemoryCandidateIntent,
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
    ToolUseContext,
    TurnArtifacts,
    WorkflowCandidate,
    WorkflowNodeSpec,
    WorkflowSpec,
    WorkflowState,
)
from local_agent.storage.memory_store import SQLiteMemoryStore
from local_agent.storage.trace_audit import write_trace_audit_files
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
        "file.open_path",
        "file.reveal_in_explorer",
        "image.describe",
        "image.inspect",
        "image.read_text",
        "document_agent.summarize",
        "document_agent.read",
        "document_agent.inspect",
        "document_agent.compose",
        "document_agent.edit",
        "memory.recall",
        "qq.get_current_context",
        "qq.get_recent_messages",
        "qq.get_last_reply",
        "qq.search_history",
        "qq.get_recent_attachments",
        "qq.search_contacts",
        "qq.send_text",
        "qq.send_file",
        "qq.send_voice",
        "web.search",
        "web.research",
        "web.fetch",
        "system.get_time",
        "system.create_reminder",
        "system.create_scheduled_task",
        "system.list_reminders",
        "system.cancel_reminder",
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

    def _write_turn_trace_audit(self, trace_id: str) -> None:
        trace_path = getattr(getattr(self, "trace_store", None), "trace_path", None)
        if trace_path is None:
            return
        try:
            paths = write_trace_audit_files(trace_path=trace_path, trace_id=trace_id)
            self.trace_store.append(
                "trace_audit_written",
                {"trace_id": trace_id, **paths},
            )
        except Exception as exc:  # noqa: BLE001
            try:
                self.trace_store.append("trace_audit_error", {"trace_id": trace_id, "error": str(exc)})
            except Exception:
                return

    def _append_self_diagnosis(
        self,
        *,
        execution_summary: dict[str, Any],
        loop_stop_reason: str,
        trace_id: str,
        user_text: str,
    ) -> None:
        """任务异常终止时自动分析 trace，给出根因和修复建议。

        不自动改代码，只生成诊断报告写入 trace 和回复中。
        """
        trace_path = getattr(getattr(self, "trace_store", None), "trace_path", None)
        if not trace_path:
            return

        # 从 execution_summary 中提取关键错误信息
        tool_errors = []
        tool_successes = []
        for result in (execution_summary.get("tool_results") or []):
            if not isinstance(result, dict):
                continue
            tool_name = str(result.get("tool_name") or "")
            if result.get("status") == "error":
                err_msg = str(result.get("error", "") or "")
                if isinstance(result.get("error"), dict):
                    err_msg = str(result["error"].get("message") or result["error"])
                tool_errors.append((tool_name, err_msg[:200]))
            elif result.get("status") == "success":
                data_keys = list((result.get("data") or {}).keys())
                tool_successes.append((tool_name, data_keys))

        # 构建精简的诊断 prompt
        diag_prompt = (
            f"你是银狼 agent 的自诊断模块。请分析以下任务失败的原因并给出简明的修复建议。\n\n"
            f"停止原因: {loop_stop_reason}\n"
            f"用户请求: {user_text[:200]}\n"
            f"成功步骤: {json.dumps(tool_successes[-5:], ensure_ascii=False)}\n"
            f"失败步骤: {json.dumps(tool_errors[-5:], ensure_ascii=False)}\n"
            f"缺失输出: {execution_summary.get('missing_outputs', [])}\n\n"
            f"请用中文简洁回答（3-5句）：1) 根因是什么 2) 应该怎么修（改代码/改 prompt/改配置） 3) 用户现在该怎么办。"
        )

        diagnosis = ""
        try:
            chat_model = str(getattr(self.config, "chat_model", "") or getattr(self.config, "model", "") or "").strip()
            raw = self.llm_client._chat(
                [
                    {"role": "system", "content": "You are an honest, concise debug assistant. Diagnose agent failures in Chinese. Be specific about root cause and fix. Keep it under 150 characters."},
                    {"role": "user", "content": diag_prompt},
                ],
                model=chat_model or None,
            )
            diagnosis = str(raw).strip() if raw else ""
        except Exception:
            diagnosis = ""

        self.trace_store.append(
            "self_diagnosis",
            {
                "trace_id": trace_id,
                "loop_stop_reason": loop_stop_reason,
                "tool_errors": tool_errors,
                "diagnosis": diagnosis,
            },
        )

        # 如果诊断有效，追加到 history 中让最终回复能看到
        if diagnosis:
            self.history.append(
                Message(
                    role=Role.SYSTEM,
                    content=f"[自诊断] 任务因 {loop_stop_reason} 终止。诊断结果: {diagnosis[:300]}",
                )
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
        config = getattr(self, "config", None)
        workspace_root = str(getattr(config, "workspace_root", "") or "").strip()
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

    def _local_scope_hints_for_context(self, *texts: str) -> dict[str, str]:
        hints = self._local_scope_hints()
        evidence = "\n".join(str(text or "") for text in texts if str(text or "").strip()).lower()
        if not evidence:
            return {}

        alias_markers = {
            "workspace": ("workspace", "工作区", "项目目录", "本地项目"),
            "desktop": ("desktop", "桌面"),
            "downloads": ("downloads", "download", "下载"),
            "documents": ("documents", "document", "文档"),
            "pictures": ("pictures", "picture", "图片", "照片"),
        }
        scoped: dict[str, str] = {}
        for alias, path in hints.items():
            normalized_path = str(path or "").lower()
            markers = alias_markers.get(alias, (alias,))
            if any(marker and marker.lower() in evidence for marker in markers) or (
                normalized_path and normalized_path in evidence
            ):
                scoped[alias] = path
        return scoped

    def _recent_conversation_text(self, limit: int = 20) -> str:
        visible_messages = [
            f"{message.role.value}: {message.content}"
            for message in self.history
            if message.role in {Role.SYSTEM, Role.USER, Role.ASSISTANT}
        ]
        return "\n".join(visible_messages[-limit:])

    def _build_single_source_intent_bundle(
        self,
        *,
        user_text: str,
        recent_context: str,
        hot_context_summary: str,
        warm_memory_summary: str,
        learning_memory_summary: str,
        cold_memory_summary: str,
        active_task_summary: str,
        channel_context_summary: str,
    ) -> IntentBundle:
        context_layers_used = [
            name
            for name, value in (
                ("recent_context", recent_context),
                ("active_task_summary", active_task_summary),
                ("channel_context_summary", channel_context_summary),
                ("hot_context_summary", hot_context_summary),
                ("warm_memory_summary", warm_memory_summary),
                ("learning_memory_summary", learning_memory_summary),
                ("cold_memory_summary", cold_memory_summary),
            )
            if str(value or "").strip()
        ]
        memory_candidate = None
        try:
            memory_candidate = self.intent_service.analyze_memory_candidate(
                user_text,
                recent_context=recent_context,
                hot_context_summary=hot_context_summary,
                warm_memory_summary=warm_memory_summary,
                learning_memory_summary=learning_memory_summary,
                cold_memory_summary=cold_memory_summary,
                active_task_summary=active_task_summary,
                channel_context_summary=channel_context_summary,
            )
        except Exception:
            memory_candidate = None
        instruction_intent = self.intent_service._derive_instruction_intent(memory_candidate) if memory_candidate is not None else None
        contract: dict[str, Any] = {}
        contract_planner = getattr(self.llm_client, "plan_main_agent_contract", None)
        if callable(contract_planner):
            try:
                contract = contract_planner(
                    user_text=user_text,
                    recent_context=recent_context,
                    active_task_summary=active_task_summary,
                    channel_context_summary=channel_context_summary,
                    hot_context_summary=hot_context_summary,
                    warm_memory_summary=warm_memory_summary,
                    learning_memory_summary=learning_memory_summary,
                    cold_memory_summary=cold_memory_summary,
                    local_scope_hints=self._local_scope_hints_for_context(
                        user_text,
                        recent_context,
                        active_task_summary,
                        channel_context_summary,
                        hot_context_summary,
                        warm_memory_summary,
                        cold_memory_summary,
                    ),
                    tool_manifests=self.registry.list_manifests(),
                )
            except Exception as exc:  # noqa: BLE001
                contract = {"contract_error": str(exc)}
        contract = self._normalize_main_agent_contract(contract, user_text=user_text)
        recent_file_artifact = self._extract_recent_file_artifact_path(channel_context_summary)
        if recent_file_artifact and self._request_refers_to_recent_file(user_text):
            grounded = contract.get("grounded_inputs") if isinstance(contract.get("grounded_inputs"), dict) else {}
            grounded = dict(grounded)
            grounded.setdefault("target_path", recent_file_artifact)
            grounded.setdefault("file_path", recent_file_artifact)
            grounded.setdefault("resolved_subject", Path(recent_file_artifact).name)
            grounded.setdefault("source_context", str(Path(recent_file_artifact).parent))
            grounded.setdefault("reference_resolution", "recent_file_artifact")
            contract["grounded_inputs"] = grounded

        primary_objective = str(contract.get("primary_objective") or user_text or "").strip()
        required_outputs = [
            item for item in (contract.get("required_outputs") or [])
            if isinstance(item, OutputKind)
        ]
        preferred_tools = [
            tool
            for tool in (contract.get("preferred_tools") or [])
            if isinstance(tool, str) and self.registry.has_tool(tool)
        ]
        grounded_inputs = contract.get("grounded_inputs") if isinstance(contract.get("grounded_inputs"), dict) else {}
        workflow_spec = contract.get("workflow_spec") if isinstance(contract.get("workflow_spec"), WorkflowSpec) else None
        constraints = [
            str(item).strip()
            for item in (contract.get("constraints") or [])
            if str(item).strip()
        ]

        delegated_brief_parts = [
            "mode=main_agent_single_source",
            f"objective={primary_objective}",
            "rule=Use recent context, active task state, channel context, memories, and tool results directly. Do not reclassify or narrow the task through a secondary planner.",
            "rule=Downstream tools execute the main agent decision only; if required outputs are missing, continue or ask a concrete clarification.",
        ]
        if preferred_tools:
            delegated_brief_parts.append("preferred_tools=" + ", ".join(preferred_tools))
        if required_outputs:
            delegated_brief_parts.append("required_outputs=" + ", ".join(output.value for output in required_outputs))
        if workflow_spec is not None and workflow_spec.nodes:
            delegated_brief_parts.append(
                "workflow_nodes="
                + json.dumps(
                    [node.model_dump(mode="json") for node in workflow_spec.nodes],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        if grounded_inputs:
            delegated_brief_parts.append(
                "grounded_inputs=" + json.dumps(grounded_inputs, ensure_ascii=False, sort_keys=True)
            )
        if constraints:
            delegated_brief_parts.append("constraints=" + " | ".join(constraints))
        if active_task_summary.strip():
            delegated_brief_parts.append("active_task=" + active_task_summary.strip())
        if channel_context_summary.strip():
            delegated_brief_parts.append("channel_context=" + channel_context_summary.strip()[:900])
        execution_notes = [
            "The main agent contract is authoritative for task scope and completion.",
            "Downstream tools and recovery state machines must not replace the original required_outputs.",
            *constraints,
        ]
        if contract.get("contract_error"):
            execution_notes.append("main_agent_contract_planner_error=" + str(contract.get("contract_error")))
        return IntentBundle(
            memory_candidate_intent=memory_candidate or MemoryCandidateIntent(),
            instruction_intent=instruction_intent or InstructionIntent(),
            task_classification=None,
            answerability=AnswerabilityAssessment(
                answerability="main_agent_decides",
                preferred_family="main_agent",
                local_answer_kind="none",
                answer_basis=context_layers_used,
                confidence=1.0,
                rationale="single_source_main_agent",
            ),
            task_envelope=TaskEnvelope(
                mode="main_agent",
                conversation_mode=str(contract.get("conversation_mode") or "").strip()
                or ("continuation" if str(recent_context or "").strip() else "new_request"),
                primary_objective=primary_objective,
                needs_grounding=bool(contract.get("needs_grounding", bool(required_outputs))),
                context_layers_used=context_layers_used,
                allowed_families=[
                    str(item).strip()
                    for item in (contract.get("allowed_families") or [])
                    if str(item).strip()
                ],
                blocked_families=[
                    str(item).strip()
                    for item in (contract.get("blocked_families") or [])
                    if str(item).strip()
                ],
                required_outputs=required_outputs,
                preferred_tools=preferred_tools,
                planning_focus_text=json.dumps(grounded_inputs, ensure_ascii=False, sort_keys=True) if grounded_inputs else None,
                execution_notes=execution_notes,
                delegated_execution_brief="\n".join(delegated_brief_parts),
                workflow_spec=workflow_spec,
                rationale=str(contract.get("rationale") or "single_source_main_agent").strip(),
            ),
        )

    @staticmethod
    def _contract_objective_is_placeholder(value: str) -> bool:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return True
        return normalized in {
            "full user deliverable after resolving context",
            "full user deliverable",
            "primary objective",
            "complete the active task",
        }

    def _normalize_main_agent_contract(self, contract: dict[str, Any] | None, *, user_text: str) -> dict[str, Any]:
        normalized = dict(contract or {})
        if self._contract_objective_is_placeholder(str(normalized.get("primary_objective") or "")):
            normalized["primary_objective"] = str(user_text or "").strip()

        manifests = {
            manifest.tool_name: manifest
            for manifest in self.registry.list_manifests()
        }
        if self._contract_mentions_qq_recent_messages(normalized):
            if "qq.get_recent_messages" in manifests:
                normalized["workflow_nodes"] = [
                    {
                        "node_id": "step_1",
                        "tool": "qq.get_recent_messages",
                        "intent": "read recent QQ messages from the active channel runtime",
                        "reason": "The user asked for recent QQ messages; use the QQ runtime instead of local documents or web search.",
                        "produces": [output.value for output in manifests["qq.get_recent_messages"].produces],
                    }
                ]
                normalized["preferred_tools"] = ["qq.get_recent_messages"]
                normalized["allowed_families"] = ["qq_history"]
            else:
                blocked_families = [
                    str(item).strip()
                    for item in normalized.get("blocked_families") or []
                    if str(item).strip()
                ]
                for family in ("web_lookup", "document_summary", "document_operation", "local_lookup", "file_lookup"):
                    if family not in blocked_families:
                        blocked_families.append(family)
                normalized["primary_objective"] = (
                    "Tell the user the QQ runtime is unavailable, so recent QQ messages cannot be read in this turn."
                )
                normalized["workflow_nodes"] = [
                    {
                        "node_id": "step_1",
                        "tool": None,
                        "intent": "explain that the QQ runtime is unavailable for reading recent messages",
                        "reason": "Recent QQ messages require the active QQ runtime; web or local files are not valid substitutes.",
                        "produces": [OutputKind.MESSAGE_SENT.value],
                    }
                ]
                normalized["preferred_tools"] = []
                normalized["allowed_families"] = []
                normalized["blocked_families"] = blocked_families
        preferred_tools: list[str] = []
        for item in normalized.get("preferred_tools") or []:
            tool_name = str(item or "").strip()
            if tool_name and tool_name in manifests and tool_name not in preferred_tools:
                preferred_tools.append(tool_name)

        workflow_nodes: list[WorkflowNodeSpec] = []
        available_workflow_outputs: set[OutputKind] = set()
        output_rewrites: dict[OutputKind, list[OutputKind]] = {}
        for index, item in enumerate(normalized.get("workflow_nodes") or [], start=1):
            if not isinstance(item, dict):
                continue
            node_payload = dict(item)
            node_payload.setdefault("node_id", f"step_{index}")
            tool_name = str(node_payload.get("tool") or "").strip()
            if self._contract_node_should_use_qq_recent_messages(normalized, node_payload, manifests):
                tool_name = "qq.get_recent_messages"
                node_payload["produces"] = list(manifests["qq.get_recent_messages"].produces)
            if tool_name == "web.search" and "web.research" in manifests:
                tool_name = "web.research"
            node_payload["tool"] = tool_name if tool_name in manifests else None
            try:
                node = WorkflowNodeSpec.model_validate(node_payload)
            except Exception:
                continue

            original_produces = list(node.produces)
            if node.tool and node.tool in manifests:
                manifest_outputs = list(manifests[node.tool].produces)
                fixed_produces = [output_kind for output_kind in node.produces if output_kind in manifest_outputs]
                if node.tool == "web.research" and manifest_outputs:
                    fixed_produces = manifest_outputs
                if not fixed_produces and manifest_outputs:
                    fixed_produces = manifest_outputs
                if fixed_produces != node.produces:
                    node = node.model_copy(update={"produces": fixed_produces})
                if original_produces and fixed_produces:
                    for output_kind in original_produces:
                        if output_kind not in fixed_produces:
                            output_rewrites[output_kind] = fixed_produces

            if not node.produces:
                continue
            if node.requires:
                rewritten_requires: list[OutputKind] = []
                for output_kind in node.requires:
                    replacements = output_rewrites.get(output_kind)
                    if replacements:
                        for replacement in replacements:
                            if replacement not in rewritten_requires:
                                rewritten_requires.append(replacement)
                    elif output_kind not in rewritten_requires:
                        rewritten_requires.append(output_kind)
                executable_requires = [
                    output_kind
                    for output_kind in rewritten_requires
                    if output_kind in available_workflow_outputs and output_kind not in node.produces
                ]
                if executable_requires != node.requires:
                    node = node.model_copy(update={"requires": executable_requires})
            workflow_nodes.append(node)
            available_workflow_outputs.update(node.produces)
            if node.tool and node.tool not in preferred_tools:
                preferred_tools.append(node.tool)
        workflow_nodes = self._insert_document_compose_nodes(workflow_nodes, manifests)
        for node in workflow_nodes:
            if node.tool and node.tool not in preferred_tools:
                preferred_tools.append(node.tool)
        normalized["preferred_tools"] = preferred_tools

        required_outputs: list[OutputKind] = []
        for item in normalized.get("required_outputs") or []:
            if isinstance(item, OutputKind) and item not in required_outputs:
                required_outputs.append(item)
            elif isinstance(item, str):
                try:
                    output_kind = OutputKind(item.strip())
                except ValueError:
                    continue
                if output_kind not in required_outputs:
                    required_outputs.append(output_kind)

        if workflow_nodes:
            required_outputs = []
            for node in workflow_nodes:
                for output_kind in node.produces:
                    if output_kind not in required_outputs:
                        required_outputs.append(output_kind)
        else:
            inferred_outputs: list[OutputKind] = []
            for tool_name in preferred_tools:
                for output_kind in manifests[tool_name].produces:
                    if output_kind not in inferred_outputs:
                        inferred_outputs.append(output_kind)
            shallow_outputs = {OutputKind.OBJECT_CANDIDATES, OutputKind.CONTACT_CANDIDATES, OutputKind.DIRECTORY_ENTRIES}
            if inferred_outputs and (not required_outputs or set(required_outputs).issubset(shallow_outputs)):
                for output_kind in inferred_outputs:
                    if output_kind not in required_outputs:
                        required_outputs.append(output_kind)
        normalized["required_outputs"] = required_outputs
        normalized["workflow_spec"] = WorkflowSpec(
            workflow_name=str(normalized.get("workflow_name") or "main_agent_contract").strip() or "main_agent_contract",
            goal=TaskGoal(
                summary=str(normalized.get("primary_objective") or user_text or "").strip(),
                required_outputs=required_outputs,
                completion_mode="outputs",
            ) if required_outputs else None,
            nodes=workflow_nodes,
        ) if workflow_nodes else None

        allowed_families = [
            str(item).strip()
            for item in normalized.get("allowed_families") or []
            if str(item).strip()
        ]
        blocked_families = [
            str(item).strip()
            for item in normalized.get("blocked_families") or []
            if str(item).strip()
        ]
        for tool_name in preferred_tools:
            family = self._tool_family_for_selected_tool(tool_name)
            if family and family not in blocked_families and family not in allowed_families:
                allowed_families.append(family)
        normalized["allowed_families"] = allowed_families
        normalized["blocked_families"] = blocked_families

        if preferred_tools or required_outputs:
            normalized["needs_grounding"] = True
        return normalized

    @staticmethod
    def _insert_document_compose_nodes(
        workflow_nodes: list[WorkflowNodeSpec],
        manifests: dict[str, Any],
    ) -> list[WorkflowNodeSpec]:
        if "document_agent.compose" not in manifests or not workflow_nodes:
            return workflow_nodes
        if any(node.tool == "document_agent.compose" for node in workflow_nodes):
            return workflow_nodes

        inserted: list[WorkflowNodeSpec] = []
        has_prior_web = False
        for node in workflow_nodes:
            if node.tool in {"web.research", "web.fetch"} or OutputKind.WEB_CONTENT in node.produces:
                has_prior_web = True
            if has_prior_web and node.tool == "file.write_docx":
                compose_manifest = manifests["document_agent.compose"]
                inserted.append(
                    WorkflowNodeSpec(
                        node_id="compose_document",
                        tool="document_agent.compose",
                        intent="compose final document content from grounded web and tool materials",
                        reason="Document content should be synthesized before writing.",
                        requires=[OutputKind.WEB_CONTENT],
                        produces=list(compose_manifest.produces) or [OutputKind.FILE_CONTENTS, OutputKind.OBJECT_DETAILS],
                    )
                )
                requires = list(node.requires)
                if OutputKind.FILE_CONTENTS not in requires:
                    requires.append(OutputKind.FILE_CONTENTS)
                node = node.model_copy(update={"requires": requires})
                has_prior_web = False
            inserted.append(node)
        return inserted

    @staticmethod
    def _contract_node_should_use_qq_recent_messages(
        normalized_contract: dict[str, Any],
        node_payload: dict[str, Any],
        manifests: dict[str, ToolManifest],
    ) -> bool:
        if "qq.get_recent_messages" not in manifests:
            return False
        raw_tool = str(node_payload.get("tool") or "").strip()
        if raw_tool and raw_tool.startswith("qq."):
            return False
        haystack = " ".join(
            str(value or "")
            for value in (
                normalized_contract.get("primary_objective"),
                normalized_contract.get("grounded_inputs"),
                node_payload.get("intent"),
                node_payload.get("reason"),
            )
        ).lower()
        mentions_qq = "qq" in haystack
        mentions_messages = any(marker in haystack for marker in ("message", "messages", "消息", "聊天", "history", "历史"))
        mentions_recent = any(marker in haystack for marker in ("recent", "last", "latest", "最近", "最后", "前两", "两句"))
        return mentions_qq and mentions_messages and mentions_recent

    @classmethod
    def _contract_should_use_qq_recent_messages(
        cls,
        normalized_contract: dict[str, Any],
        manifests: dict[str, ToolManifest],
    ) -> bool:
        if "qq.get_recent_messages" not in manifests:
            return False
        haystack_parts = [
            normalized_contract.get("primary_objective"),
            normalized_contract.get("grounded_inputs"),
            normalized_contract.get("preferred_tools"),
            normalized_contract.get("allowed_families"),
        ]
        for node in normalized_contract.get("workflow_nodes") or []:
            if isinstance(node, dict):
                haystack_parts.extend([node.get("tool"), node.get("intent"), node.get("reason"), node.get("produces")])
        haystack = " ".join(str(value or "") for value in haystack_parts).lower()
        mentions_qq = "qq" in haystack
        mentions_messages = any(marker in haystack for marker in ("message", "messages", "消息", "聊天", "history", "历史"))
        mentions_recent = any(marker in haystack for marker in ("recent", "last", "latest", "最近", "最后", "前两", "两句"))
        return mentions_qq and mentions_messages and mentions_recent

    @staticmethod
    def _contract_mentions_qq_recent_messages(normalized_contract: dict[str, Any]) -> bool:
        haystack_parts = [
            normalized_contract.get("primary_objective"),
            normalized_contract.get("grounded_inputs"),
            normalized_contract.get("preferred_tools"),
            normalized_contract.get("allowed_families"),
        ]
        for node in normalized_contract.get("workflow_nodes") or []:
            if isinstance(node, dict):
                haystack_parts.extend([node.get("tool"), node.get("intent"), node.get("reason"), node.get("produces")])
        haystack = " ".join(str(value or "") for value in haystack_parts).lower()
        mentions_qq = "qq" in haystack
        mentions_messages = any(
            marker in haystack
            for marker in ("message", "messages", "消息", "聊天", "history", "历史", "娑堟伅", "鑱婂ぉ", "鍘嗗彶")
        )
        mentions_recent = any(
            marker in haystack
            for marker in ("recent", "last", "latest", "最近", "最后", "前两", "两句", "鏈€杩?", "鏈€鍚?", "鍓嶄袱", "涓ゅ彞")
        )
        return mentions_qq and mentions_messages and mentions_recent

    @staticmethod
    def _parse_task_envelope_grounded_inputs(task_envelope) -> dict[str, Any]:
        raw = str(getattr(task_envelope, "planning_focus_text", "") or "").strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except Exception:  # noqa: BLE001
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _first_grounded_string(grounded_inputs: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = grounded_inputs.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _latest_tool_time(tool_results: list[ToolCallResult] | None) -> datetime | None:
        for result in reversed(tool_results or []):
            if result.status != "success" or result.tool_name != "system.get_time":
                continue
            iso = str(result.data.get("iso") or "").strip()
            if not iso:
                continue
            try:
                return datetime.fromisoformat(iso)
            except ValueError:
                continue
        return None

    @staticmethod
    def _contract_recent_user_message_time(task_envelope) -> datetime | None:
        brief = str(getattr(task_envelope, "delegated_execution_brief", "") or "")
        if not brief:
            return None
        matches = re.findall(r"(?m)^\s*-\s*(20\d{2}-\d{2}-\d{2}T[0-9:.+-]+):\s*", brief)
        parsed: list[datetime] = []
        for raw in matches:
            try:
                dt = datetime.fromisoformat(raw)
            except ValueError:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            parsed.append(dt)
        if not parsed:
            return None
        return max(parsed).astimezone(timezone(timedelta(hours=8)))

    @staticmethod
    def _parse_explicit_local_datetime(text: str, *, default_tz) -> str:
        match = re.search(
            r"(?P<date>20\d{2}[-/年]\d{1,2}[-/月]\d{1,2})\s*(?:日|号)?\s*"
            r"(?P<hour>\d{1,2})\s*(?::|点)\s*(?P<minute>\d{1,2})?",
            str(text or ""),
        )
        if not match:
            return ""

    @staticmethod
    def _normalize_chinese_time_digits(text: str) -> str:
        digit_map = {
            "零": 0,
            "〇": 0,
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
        }

        def cn_number(value: str) -> int | None:
            value = value.strip()
            if not value:
                return None
            if value == "十":
                return 10
            if value.startswith("十"):
                tail = digit_map.get(value[1:], None)
                return 10 + int(tail or 0)
            if "十" in value:
                head, tail = value.split("十", 1)
                head_value = digit_map.get(head, None)
                tail_value = digit_map.get(tail, 0) if tail else 0
                if head_value is None or tail_value is None:
                    return None
                return int(head_value) * 10 + int(tail_value)
            if len(value) == 1 and value in digit_map:
                return digit_map[value]
            return None

        def replace_with_unit(match: re.Match[str]) -> str:
            number = cn_number(match.group("num"))
            if number is None:
                return match.group(0)
            return f"{number}{match.group('unit')}"

        return re.sub(
            r"(?P<num>[零〇一二两三四五六七八九十]{1,3})(?P<unit>点|分钟|小时)",
            replace_with_unit,
            str(text or ""),
        )
        date_text = match.group("date").replace("年", "-").replace("月", "-").replace("/", "-")
        parts = [int(part) for part in date_text.strip("-").split("-")]
        minute = int(match.group("minute") or 0)
        try:
            return datetime(
                parts[0],
                parts[1],
                parts[2],
                int(match.group("hour")),
                minute,
                0,
                tzinfo=default_tz,
            ).isoformat()
        except (ValueError, TypeError):
            return ""

    def _infer_when_iso_from_contract_context(
        self,
        *,
        grounded_inputs: dict[str, Any],
        task_envelope,
        user_text: str,
        tool_results: list[ToolCallResult] | None = None,
        timezone_name: str = "Asia/Shanghai",
    ) -> str:
        explicit = self._first_grounded_string(grounded_inputs, "when_iso", "scheduled_for", "time_iso", "datetime_iso")
        if explicit:
            return explicit
        now = self._contract_recent_user_message_time(task_envelope) or self._latest_tool_time(tool_results)
        if now is None:
            now = datetime.now(timezone(timedelta(hours=8)))
        tz = now.tzinfo or timezone(timedelta(hours=8))
        combined = "\n".join(
            part
            for part in (
                str(getattr(task_envelope, "primary_objective", "") or ""),
                str(getattr(task_envelope, "planning_focus_text", "") or ""),
                str(user_text or ""),
                self._first_grounded_string(grounded_inputs, "resolved_subject", "source_context", "instruction", "task"),
            )
            if part
        )
        normalized_combined = self._normalize_chinese_time_digits(combined)
        explicit_local = self._parse_explicit_local_datetime(normalized_combined, default_tz=tz)
        if explicit_local:
            return explicit_local
        parsed = parse_when_text(when_text=normalized_combined, now=now, timezone_name=timezone_name or "Asia/Shanghai")
        return "" if parsed is None else parsed.isoformat()

    @classmethod
    def _infer_reminder_message_from_contract_context(
        cls,
        *,
        grounded_inputs: dict[str, Any],
        task_envelope,
        user_text: str,
        fallback: str = "",
    ) -> str:
        explicit = cls._first_grounded_string(grounded_inputs, "message", "text", "reply_text", "speech_text")
        if explicit:
            return explicit
        combined = "\n".join(
            part
            for part in (
                str(user_text or ""),
                str(getattr(task_envelope, "planning_focus_text", "") or ""),
                str(getattr(task_envelope, "primary_objective", "") or ""),
            )
            if part
        )
        quote_match = re.search(
            r"(?:message|text|content|\u63d0\u9192\u6d88\u606f\u5185\u5bb9|\u901a\u77e5\u5185\u5bb9|\u63d0\u9192\u5185\u5bb9)\s*[:\uff1a]\s*['\"]([^'\"]{1,80})['\"]",
            combined,
            re.I,
        )
        if quote_match:
            return quote_match.group(1).strip()
        lines = [line.strip() for line in re.split(r"[\r\n]+", str(user_text or "")) if line.strip()]
        source = "\n".join(lines) if lines else str(user_text or "")
        cleaned = cls._normalize_chinese_time_digits(source)
        cleaned = re.sub(r"\d+\s*(?:\u5206\u949f|\u5c0f\u65f6)\u540e", " ", cleaned)
        cleaned = re.sub(
            r"(?:\u534a\u5c0f\u65f6\u540e|\u4eca\u5929|\u660e\u5929|\u540e\u5929|\u4e0a\u5348|\u4e2d\u5348|\u4e0b\u5348|\u665a\u4e0a|\u508d\u665a|\u51cc\u6668|\u65e9\u4e0a|\u65e9\u6668)?\s*\d{1,2}\s*(?:\u70b9|:)\s*\d{0,2}",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"(?:\u63d0\u9192\u6211|\u53eb\u6211|\u558a\u6211|\u544a\u8bc9\u6211|\u8ba9\u6211)", " ", cleaned)
        cleaned = re.sub(
            r"(?:\u8bf7|\u5e2e\u6211|\u7ed9\u6211|\u5230\u65f6|\u5230\u65f6\u5019|\u8bbe\u7f6e|\u8bbe\u4e2a|\u95f9\u949f|\u63d0\u9192|\u901a\u77e5|\u53d1QQ|QQ|\u5f53\u524d\u4f1a\u8bdd|\u4e00\u4e0b|\u4e00\u4e0b\u5427)",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" \uff0c,\u3002.!！?？\uff1a:")
        if cleaned:
            return cleaned[:80]
        fallback_text = str(fallback or getattr(task_envelope, "primary_objective", "") or "").strip()
        fallback_text = re.sub(r"^\s*(?:\u5728)?\d+\s*(?:\u5206\u949f|\u5c0f\u65f6)\u540e", "", fallback_text).strip(" \uff0c,\u3002")
        return fallback_text[:80]

    def _build_cached_reminder_fire_text(self, message: str) -> str:
        clean_message = str(message or "").strip()
        if not clean_message:
            return ""
        llm_client = getattr(self, "llm_client", None)
        if llm_client is not None and hasattr(llm_client, "render_reminder_fire_text"):
            try:
                rendered = llm_client.render_reminder_fire_text(
                    reminder_message=clean_message,
                    persona_name=str(getattr(self.config, "persona_name", "") or ""),
                    persona_profile=str(getattr(self.config, "persona_profile", "") or ""),
                    chat_style_prompt=str(getattr(self.config, "chat_style_prompt", "") or ""),
                )
                rendered = str(rendered or "").strip()
                if rendered:
                    return rendered[:120]
            except Exception:
                pass
        return clean_message[:120]

    def _ensure_cached_reminder_payload(self, task_payload: Any, message: str) -> dict[str, Any]:
        payload = dict(task_payload or {}) if isinstance(task_payload, dict) else {}
        if str(message or "").strip() and not str(payload.get("fire_text") or "").strip():
            payload["fire_text"] = self._build_cached_reminder_fire_text(str(message or "").strip())
            payload["fire_text_source"] = "creation_time_persona_render"
        return payload

    @staticmethod
    def _build_contract_file_path_from_subject(subject: str, source_context: str) -> str:
        clean_subject = str(subject or "").strip().strip("\"'")
        if not clean_subject:
            return ""
        subject_path = Path(clean_subject)
        if subject_path.is_absolute() or len(subject_path.parts) > 1:
            return clean_subject
        if "." not in subject_path.name:
            return ""
        clean_scope = str(source_context or "").strip().strip("\"'")
        if clean_scope and clean_scope.lower() not in {"web", "qq", "current chat", "local chat history"}:
            return str(Path(clean_scope) / clean_subject)
        return clean_subject

    def _build_contract_docx_path(self, *, subject: str, source_context: str, user_text: str) -> str:
        explicit_path = self._extract_requested_output_file(user_text)
        if explicit_path:
            path = Path(explicit_path)
            if path.suffix.lower() != ".docx":
                path = path.with_suffix(".docx")
            return str(path)

        clean_subject = str(subject or "").strip().strip("\"'") or "查询结果"
        safe_title = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff._-]+", "-", clean_subject).strip("-._") or "查询结果"
        if not safe_title.lower().endswith(".docx"):
            safe_title = f"{safe_title}.docx"

        clean_scope = str(source_context or "").strip().strip("\"'")
        if clean_scope and Path(clean_scope).is_absolute():
            base_dir = Path(clean_scope)
        else:
            base_dir = Path(getattr(getattr(self, "config", None), "workspace_root", "") or ".")
        return str(base_dir / safe_title)

    @staticmethod
    def _split_docx_paragraphs(content: str) -> list[str]:
        paragraphs = [part.strip(" -\t") for part in re.split(r"\n{1,}|\s{2,}", str(content or "")) if part.strip()]
        return paragraphs[:24]

    @staticmethod
    def _build_docx_content_from_tool_results(tool_results: list[ToolCallResult]) -> str:
        for result in reversed(tool_results):
            if result.status != "success":
                continue
            if result.tool_name not in {"web.research", "web.search", "web.fetch"}:
                continue
            chunks: list[str] = []
            observation = result.metrics.get("observation") if isinstance(result.metrics, dict) else None
            if observation:
                chunks.append(str(observation))
            for key in ("summary", "answer", "content", "text"):
                value = result.data.get(key)
                if isinstance(value, str) and value.strip():
                    chunks.append(value.strip())
            for key in ("results", "sources", "pages"):
                value = result.data.get(key)
                if isinstance(value, list):
                    for item in value[:6]:
                        if isinstance(item, dict):
                            title = str(item.get("title") or item.get("name") or item.get("url") or "").strip()
                            snippet = str(item.get("snippet") or item.get("summary") or item.get("content") or "").strip()
                            url = str(item.get("url") or item.get("source") or "").strip()
                            line = " - ".join(part for part in (title, snippet, url) if part)
                            if line:
                                chunks.append(line)
                        elif isinstance(item, str) and item.strip():
                            chunks.append(item.strip())
            if chunks:
                text = "\n".join(chunks)
                return text[:6000]
        return ""

    def _repair_decision_arguments_from_task_contract(
        self,
        decision: ToolDecision,
        *,
        task_envelope,
    ) -> tuple[ToolDecision, dict[str, Any] | None]:
        if decision.decision != DecisionType.TOOL_CALL or not decision.selected_tool:
            return decision, None
        grounded_inputs = self._parse_task_envelope_grounded_inputs(task_envelope)
        if not grounded_inputs:
            return decision, None

        selected_tool = str(decision.selected_tool or "").strip()
        arguments = dict(decision.arguments or {})
        repaired = False

        subject = self._first_grounded_string(
            grounded_inputs,
            "resolved_subject",
            "target_name",
            "target",
            "query",
            "topic",
            "search_query",
        )
        source_context = self._first_grounded_string(
            grounded_inputs,
            "source_context",
            "scope",
            "path",
            "path_scope",
            "directory",
        )
        target_path = self._first_grounded_string(
            grounded_inputs,
            "target_path",
            "source_path",
            "file_path",
            "path",
        )
        instruction = self._first_grounded_string(
            grounded_inputs,
            "instruction",
            "task",
            "edit_instruction",
            "operation",
        ) or str(getattr(task_envelope, "primary_objective", "") or "").strip()
        url = self._first_grounded_string(grounded_inputs, "url", "target_url", "source_url")
        contact_query = self._first_grounded_string(
            grounded_inputs,
            "contact_query",
            "contact",
            "contact_name",
            "person",
            "group",
        )
        message = self._first_grounded_string(
            grounded_inputs,
            "message",
            "text",
            "reply_text",
            "speech_text",
        )
        if selected_tool in {"system.create_reminder", "system.create_scheduled_task"}:
            message = self._infer_reminder_message_from_contract_context(
                grounded_inputs=grounded_inputs,
                task_envelope=task_envelope,
                user_text=str(getattr(task_envelope, "primary_objective", "") or ""),
                fallback=instruction,
            )
        when_iso = self._first_grounded_string(grounded_inputs, "when_iso", "scheduled_for", "time_iso", "datetime_iso")
        timezone_name = self._first_grounded_string(grounded_inputs, "timezone", "timezone_name", "tz")
        if not when_iso and selected_tool in {"system.create_reminder", "system.create_scheduled_task"}:
            when_iso = self._infer_when_iso_from_contract_context(
                grounded_inputs=grounded_inputs,
                task_envelope=task_envelope,
                user_text=str(getattr(task_envelope, "primary_objective", "") or ""),
                timezone_name=timezone_name or "Asia/Shanghai",
            )
        session_id = self._first_grounded_string(grounded_inputs, "session_id", "target_session_id")
        channel = self._first_grounded_string(grounded_inputs, "channel", "runtime_channel")
        task_type = self._first_grounded_string(grounded_inputs, "task_type", "schedule_type")
        reminder_id = self._first_grounded_string(grounded_inputs, "reminder_id", "id")

        if selected_tool == "memory.recall":
            query = subject or instruction or str(getattr(task_envelope, "primary_objective", "") or "").strip()
            if query and not str(arguments.get("query") or "").strip():
                arguments["query"] = query
                repaired = True
            arguments.setdefault("limit", 5)
        elif selected_tool == "file.search_by_name":
            if subject and not str(arguments.get("query") or "").strip():
                arguments["query"] = subject
                plan = self._build_local_retrieval_plan(subject, user_text=str(getattr(task_envelope, "primary_objective", "") or ""))
                if plan.get("query_terms") and not arguments.get("query_terms"):
                    arguments["query_terms"] = plan["query_terms"]
                if plan.get("alias_terms") and not arguments.get("alias_terms"):
                    arguments["alias_terms"] = plan["alias_terms"]
                if plan.get("extensions") and not arguments.get("extensions"):
                    arguments["extensions"] = plan["extensions"]
                repaired = True
            if source_context and not str(arguments.get("path") or "").strip():
                arguments["path"] = source_context
                repaired = True
            if "path" in arguments or "query" in arguments:
                arguments.setdefault("recursive", True)
                arguments.setdefault("include_dirs", True)
                arguments.setdefault("scope_mode", "subtree")
                arguments.setdefault("target_kind", "file")
                arguments.setdefault("top_k", 8)
        elif selected_tool == "retrieval.search_local_objects":
            if subject and not str(arguments.get("query") or "").strip():
                arguments["query"] = subject
                repaired = True
            if source_context and not str(arguments.get("path_scope") or "").strip():
                arguments["path_scope"] = source_context
                repaired = True
            if "path_scope" in arguments or "query" in arguments:
                arguments.setdefault("scope_mode", "subtree")
                arguments.setdefault("target_kind", "file")
                arguments.setdefault("top_k", 8)
                arguments.setdefault("rebuild_if_missing", True)
        elif selected_tool == "file.list":
            if source_context and not str(arguments.get("path") or "").strip():
                arguments["path"] = source_context
                repaired = True
            if "path" in arguments:
                arguments.setdefault("recursive", True)
                arguments.setdefault("include_dirs", True)
        elif selected_tool in {"web.search", "web.research"}:
            if subject and not str(arguments.get("query") or "").strip():
                arguments["query"] = subject
                repaired = True
            if selected_tool == "web.research" and "query" in arguments:
                arguments.setdefault("max_results", 5)
                arguments.setdefault("max_pages", 2)
                arguments.setdefault("prefer_browser", True)
            elif selected_tool == "web.search" and "query" in arguments:
                arguments.setdefault("max_results", 5)
        elif selected_tool in {"web.fetch", "web.open_page"}:
            if url and not str(arguments.get("url") or "").strip():
                arguments["url"] = url
                repaired = True
        elif selected_tool in {"qq.search_history", "qq.get_last_reply", "qq.get_recent_attachments"}:
            if selected_tool == "qq.search_history" and subject and not str(arguments.get("query") or "").strip():
                arguments["query"] = subject
                repaired = True
            if contact_query and not str(arguments.get("contact_query") or "").strip():
                arguments["contact_query"] = contact_query
                repaired = True
            if selected_tool == "qq.get_recent_attachments":
                arguments.setdefault("kind", "any")
                arguments.setdefault("limit", 5)
            elif selected_tool == "qq.search_history":
                arguments.setdefault("limit", 5)
        elif selected_tool == "qq.search_contacts":
            query = contact_query or subject
            if query and not str(arguments.get("query") or "").strip():
                arguments["query"] = query
                repaired = True
            arguments.setdefault("target_kind", "any")
            arguments.setdefault("limit", 5)
        elif selected_tool == "qq.send_text":
            if message and not str(arguments.get("message") or "").strip():
                arguments["message"] = message
                repaired = True
            arguments.setdefault("target_kind", "current")
        elif selected_tool == "qq.send_file":
            if target_path and not str(arguments.get("file_path") or "").strip():
                arguments["file_path"] = target_path
                repaired = True
            arguments.setdefault("target_kind", "current")
        elif selected_tool == "qq.send_voice":
            if message and not str(arguments.get("speech_text") or "").strip() and not str(arguments.get("audio_path") or "").strip():
                arguments["speech_text"] = message
                repaired = True
            arguments.setdefault("target_kind", "current")
        elif selected_tool == "document_agent.compose":
            if instruction and not str(arguments.get("instruction") or "").strip():
                arguments["instruction"] = instruction
                repaired = True
            if target_path and not str(arguments.get("output_path") or "").strip():
                arguments["output_path"] = target_path
                repaired = True
            if grounded_inputs and not isinstance(arguments.get("grounded_inputs"), dict):
                arguments["grounded_inputs"] = grounded_inputs
                repaired = True
            arguments.setdefault("recent_context", "")
            arguments.setdefault("resolved_facts", {})
            arguments.setdefault("source_materials", {})
            arguments.setdefault("constraints", {})
            arguments.setdefault("style_hints", {})
        elif selected_tool in {"document_agent.summarize", "document_agent.read", "document_agent.inspect", "document_agent.edit"}:
            if target_path and not str(arguments.get("source_path") or "").strip():
                arguments["source_path"] = target_path
                repaired = True
            if instruction and selected_tool != "document_agent.read" and not str(arguments.get("instruction") or "").strip():
                arguments["instruction"] = instruction
                repaired = True
            elif instruction and selected_tool == "document_agent.read" and "instruction" not in arguments:
                arguments["instruction"] = instruction
                repaired = True
            if grounded_inputs and not isinstance(arguments.get("grounded_inputs"), dict):
                arguments["grounded_inputs"] = grounded_inputs
                repaired = True
            arguments.setdefault("recent_context", "")
            arguments.setdefault("resolved_facts", {})
            arguments.setdefault("source_materials", {})
            arguments.setdefault("constraints", {})
            arguments.setdefault("style_hints", {})
            if selected_tool == "document_agent.edit":
                arguments.setdefault("allow_overwrite", True)
                arguments.setdefault("preserve_structure", True)
                arguments.setdefault("preserve_style", True)
        elif selected_tool == "system.get_time":
            if not str(arguments.get("kind") or "").strip():
                arguments["kind"] = self._first_grounded_string(grounded_inputs, "kind", "time_kind") or "datetime"
                repaired = True
            if timezone_name and not str(arguments.get("timezone_name") or "").strip():
                arguments["timezone_name"] = timezone_name
                repaired = True
        elif selected_tool in {"system.create_reminder", "system.create_scheduled_task"}:
            if selected_tool == "system.create_scheduled_task" and not str(arguments.get("task_type") or "").strip():
                arguments["task_type"] = task_type or "notify"
                repaired = True
            if when_iso and not str(arguments.get("when_iso") or "").strip():
                arguments["when_iso"] = when_iso
                repaired = True
            if not str(arguments.get("timezone") or "").strip():
                arguments["timezone"] = timezone_name or "Asia/Shanghai"
                repaired = True
            if (message or instruction) and not str(arguments.get("message") or "").strip():
                arguments["message"] = message or instruction
                repaired = True
            task_payload = grounded_inputs.get("task_payload")
            cached_payload = self._ensure_cached_reminder_payload(task_payload, str(arguments.get("message") or message or instruction or ""))
            if isinstance(task_payload, dict) and "task_payload" not in arguments:
                arguments["task_payload"] = cached_payload
                repaired = True
            elif "task_payload" not in arguments:
                arguments["task_payload"] = cached_payload
                repaired = True
            elif isinstance(arguments.get("task_payload"), dict):
                updated_payload = self._ensure_cached_reminder_payload(
                    arguments.get("task_payload"),
                    str(arguments.get("message") or message or instruction or ""),
                )
                if updated_payload != arguments.get("task_payload"):
                    arguments["task_payload"] = updated_payload
                    repaired = True
            if session_id and not str(arguments.get("session_id") or "").strip():
                arguments["session_id"] = session_id
                repaired = True
            if channel and not str(arguments.get("channel") or "").strip():
                arguments["channel"] = channel
                repaired = True
        elif selected_tool == "system.list_reminders":
            if not str(arguments.get("status") or "").strip():
                arguments["status"] = self._first_grounded_string(grounded_inputs, "status") or "scheduled"
                repaired = True
            if session_id and not str(arguments.get("session_id") or "").strip():
                arguments["session_id"] = session_id
                repaired = True
        elif selected_tool == "system.cancel_reminder":
            if reminder_id and not str(arguments.get("reminder_id") or "").strip():
                arguments["reminder_id"] = reminder_id
                repaired = True

        if not repaired:
            return decision, None
        repaired_decision = decision.model_copy(update={"arguments": arguments})
        return repaired_decision, {
            "reason": "repaired_tool_arguments_from_main_contract",
            "selected_tool": selected_tool,
            "filled_fields": sorted(set(arguments) - set(decision.arguments or {})),
        }

    def _try_repair_missing_arguments(
        self,
        *,
        decision: ToolDecision,
        error: str,
        user_text: str,
        recent_context: str,
        trace_id: str,
        step: int,
        tool_results: list[ToolCallResult] | None = None,
    ) -> ToolDecision | None:
        """当 decision 因缺少必填字段被 DecisionValidationError 拒绝时，调用 LLM 补全参数。

        不做复杂的 planner pipeline，极简 prompt 只要求 LLM 返回修复后的 JSON。
        """
        selected_tool = str(decision.selected_tool or "").strip()
        if not selected_tool:
            return None

        grounded_repair = self._repair_missing_arguments_from_tool_results(
            decision=decision,
            tool_results=tool_results or [],
        )
        if grounded_repair is not None:
            return grounded_repair

        # 从错误消息中提取缺失的字段名
        import re as _re
        missing_field_match = _re.search(r"requires non-empty (\w+)", error)
        if not missing_field_match:
            missing_field_match = _re.search(r"requires (?:a )?non-empty (\w+)", error)
        if not missing_field_match:
            missing_field_match = _re.search(r"(\w+) must be a ", error)

        missing_field = (
            missing_field_match.group(1).strip() if missing_field_match else ""
        )

        existing_args = dict(decision.arguments or {})
        # 去掉 metadata 类字段让 prompt 更干净
        for meta_key in ("execution_brief", "required_outputs", "grounded_inputs", "constraints"):
            existing_args.pop(meta_key, None)

        # 获取当前时间作为上下文
        try:
            from datetime import datetime as _dt
            from datetime import timezone as _tz, timedelta as _td
            now_local = _dt.now(_tz(_td(hours=8)))
            now_text = now_local.strftime("%Y-%m-%d %H:%M:%S") + " CST (UTC+8)"
        except Exception:
            now_text = "(unknown time)"

        prompt = (
            f"你是一个工具参数补全助手。以下工具调用缺少必填参数，请根据上下文补全后返回完整的 JSON 对象。\n\n"
            f"工具名称: {selected_tool}\n"
            f"已有参数: {json.dumps(existing_args, ensure_ascii=False)}\n"
            f"缺失的必填字段: {missing_field or '(从错误消息推断)'}\n"
            f"验证错误: {error}\n"
            f"用户原始请求: {user_text}\n"
            f"当前时间: {now_text}\n"
            f"最近对话上下文: {recent_context[-800:] if recent_context else '(无)'}\n\n"
            f"请返回一个完整有效的 JSON object，补全所有必填字段。只返回 JSON，不要任何解释文字。"
        )

        # 用 chat_model 做极简补全
        chat_model = str(getattr(self.config, "chat_model", "") or getattr(self.config, "model", "") or "").strip()
        try:
            raw = self.llm_client._chat(  # noqa: SLF001
                [
                    {"role": "system", "content": "You are a JSON argument repair assistant. Return ONLY valid JSON, no explanation."},
                    {"role": "user", "content": prompt},
                ],
                model=chat_model or None,
            )
        except Exception:
            return None

        if not raw or not raw.strip():
            return None

        # 尝试从 LLM 输出中提取 JSON
        def _extract_json_braces(text: str) -> str | None:
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                return None
            return text[start_idx:end_idx + 1]

        json_candidate = None
        for extraction in (raw.strip(), _extract_json_braces(raw)):
            if not extraction:
                continue
            try:
                json_candidate = json.loads(extraction)
                if isinstance(json_candidate, dict):
                    break
            except json.JSONDecodeError:
                continue

        if not isinstance(json_candidate, dict) or not json_candidate:
            return None

        # 合并：LLM 结果 + 原有参数 → 用 LLM 结果覆盖缺失字段
        merged = dict(existing_args)
        merged.update(json_candidate)

        return ToolDecision(
            decision=DecisionType.TOOL_CALL,
            intent=decision.intent,
            reason=f"{decision.reason} [arguments repaired by LLM]",
            selected_tool=selected_tool,
            arguments=merged,
            risk_level=decision.risk_level,
            overall_task_goal=decision.overall_task_goal,
            expected_step_outputs=decision.expected_step_outputs,
        )

    def _repair_missing_arguments_from_tool_results(
        self,
        *,
        decision: ToolDecision,
        tool_results: list[ToolCallResult],
    ) -> ToolDecision | None:
        selected_tool = str(decision.selected_tool or "").strip()
        arguments = dict(decision.arguments or {})
        source_context = ""
        grounded_inputs = arguments.get("grounded_inputs")
        if isinstance(grounded_inputs, dict):
            source_context = self._first_grounded_string(
                grounded_inputs,
                "source_context",
                "scope",
                "path",
                "path_scope",
                "directory",
            ) or ""
        if selected_tool == "file.metadata_many" and not arguments.get("paths"):
            paths = self._latest_listed_image_paths(tool_results, source_context=source_context)
            if paths:
                arguments["paths"] = paths
                arguments.setdefault("continue_on_error", True)
        elif selected_tool == "file.mkdir_many" and not arguments.get("paths"):
            paths = self._image_organization_folder_paths(tool_results, source_context=source_context)
            if paths:
                arguments["paths"] = paths
                arguments.setdefault("exist_ok", True)
                arguments.setdefault("parents", True)
                arguments.setdefault("continue_on_error", True)
        elif selected_tool in {"file.move_many", "file.copy_many"} and not arguments.get("items"):
            items = self._image_organization_move_items(tool_results, source_context=source_context)
            if items:
                arguments["items"] = items
                arguments.setdefault("continue_on_error", True)
        else:
            return None

        if arguments == (decision.arguments or {}):
            return None
        return decision.model_copy(
            update={
                "arguments": arguments,
                "reason": f"{decision.reason} [arguments filled from prior tool results]",
            }
        )

    def _route_unsafe_document_write_decision(
        self,
        decision: ToolDecision,
        *,
        task_envelope,
    ) -> tuple[ToolDecision, dict[str, Any] | None]:
        if decision.decision != DecisionType.TOOL_CALL:
            return decision, None
        if str(decision.selected_tool or "").strip() != "file.write":
            return decision, None
        if not self.registry.has_tool("document_agent.edit"):
            return decision, None

        arguments = dict(decision.arguments or {})
        target_path = str(arguments.get("path") or "").strip()
        if not target_path or not self._looks_like_document_agent_editable_path(target_path):
            return decision, None
        if not Path(target_path).exists():
            return decision, None

        grounded_inputs = self._parse_task_envelope_grounded_inputs(task_envelope)
        instruction = self._first_grounded_string(
            grounded_inputs,
            "instruction",
            "task",
            "edit_instruction",
            "operation",
        ) or str(getattr(task_envelope, "primary_objective", "") or "").strip()
        if not instruction:
            instruction = "Edit the document according to the main-agent task contract."

        routed_arguments = {
            "source_path": target_path,
            "instruction": instruction,
            "allow_overwrite": True,
            "preserve_structure": True,
            "preserve_style": True,
            "grounded_inputs": grounded_inputs,
            "recent_context": "",
            "resolved_facts": {},
            "source_materials": {},
            "constraints": {},
            "style_hints": {},
        }
        expected_outputs = list(decision.expected_step_outputs or [])
        if OutputKind.FILE_WRITTEN not in expected_outputs:
            expected_outputs.append(OutputKind.FILE_WRITTEN)
        routed = decision.model_copy(
            update={
                "selected_tool": "document_agent.edit",
                "arguments": routed_arguments,
                "expected_step_outputs": expected_outputs,
                "reason": (
                    str(decision.reason or "").strip()
                    + " Routed Office/document writes through document_agent.edit to preserve formatting and embedded media."
                ).strip(),
            }
        )
        return routed, {
            "reason": "routed_document_write_to_document_agent_edit",
            "original_tool": "file.write",
            "selected_tool": "document_agent.edit",
            "path": target_path,
            "preserve_structure": True,
            "preserve_style": True,
        }

    @staticmethod
    def _format_main_agent_context_observation(
        *,
        user_text: str,
        recent_context: str,
        active_task_summary: str,
        channel_context_summary: str,
        hot_context_summary: str,
        warm_memory_summary: str,
        learning_memory_summary: str,
        cold_memory_summary: str,
    ) -> str:
        sections: list[str] = [
            "Main agent is the single source of task understanding.",
            f"Latest user request: {str(user_text or '').strip()}",
        ]
        for label, value, limit in (
            ("Recent conversation", recent_context, 1600),
            ("Active task state", active_task_summary, 900),
            ("Channel context", channel_context_summary, 1200),
            ("Hot context", hot_context_summary, 700),
            ("Warm memory", warm_memory_summary, 700),
            ("Learning memory", learning_memory_summary, 700),
            ("Cold memory", cold_memory_summary, 500),
        ):
            text = str(value or "").strip()
            if text:
                sections.append(f"{label}:\n{text[-limit:]}")
        return "\n\n".join(sections)

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

            group_context_messages = runtime_channel_context.get("group_context_messages")
            if isinstance(group_context_messages, list):
                rendered_group_messages: list[str] = []
                for item in group_context_messages[-12:]:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("text") or "").strip()
                    if not text:
                        continue
                    role = str(item.get("role") or "").strip()
                    sender_name = str(item.get("sender_name") or "").strip()
                    sender_id = str(item.get("sender_id") or "").strip()
                    created_at = str(item.get("created_at") or "").strip()
                    label = sender_name or sender_id or role or "unknown"
                    prefix = f"{created_at} " if created_at else ""
                    rendered_group_messages.append(f"- {prefix}{label}: {text}")
                if rendered_group_messages:
                    parts.append("group_context_messages:\n" + "\n".join(rendered_group_messages))

            recent_artifacts = runtime_channel_context.get("recent_artifacts")
            if isinstance(recent_artifacts, list):
                rendered_artifacts: list[str] = []
                for item in recent_artifacts[-6:]:
                    if not isinstance(item, dict):
                        continue
                    kind = str(item.get("kind") or "artifact").strip()
                    role = str(item.get("role") or "").strip()
                    title = str(item.get("title") or "").strip()
                    path = str(item.get("path") or "").strip()
                    created_at = str(item.get("created_at") or "").strip()
                    if not path and not title:
                        continue
                    fields = [f"kind={kind}"]
                    if role:
                        fields.append(f"role={role}")
                    if title:
                        fields.append(f"title={title}")
                    if path:
                        fields.append(f"path={path}")
                    if created_at:
                        fields.append(f"created_at={created_at}")
                    rendered_artifacts.append("- " + "; ".join(fields))
                if rendered_artifacts:
                    parts.append("recent_artifacts:\n" + "\n".join(rendered_artifacts))

            last_file = runtime_channel_context.get("last_file_artifact")
            if isinstance(last_file, dict):
                path = str(last_file.get("path") or "").strip()
                title = str(last_file.get("title") or "").strip()
                role = str(last_file.get("role") or "").strip()
                if path or title:
                    parts.append(
                        "last_file_artifact="
                        + "; ".join(
                            part
                            for part in (
                                f"title={title}" if title else "",
                                f"path={path}" if path else "",
                                f"role={role}" if role else "",
                            )
                            if part
                        )
                    )

        return "\n".join(parts)

    @staticmethod
    def _request_refers_to_recent_file(user_text: str) -> bool:
        text = str(user_text or "")
        if not text.strip():
            return False
        file_markers = ("这个文件", "这份文件", "这个文档", "这份文档", "刚才那个", "刚刚那个", "上一个文件", "刚发的文件", "刚生成的文件")
        operation_markers = ("删", "删除", "打开", "发", "发送", "改", "修改", "重命名", "移动", "复制", "看看", "读", "读取")
        return any(marker in text for marker in file_markers) and any(marker in text for marker in operation_markers)

    @staticmethod
    def _extract_recent_file_artifact_path(channel_context_summary: str) -> str:
        text = str(channel_context_summary or "")
        if not text.strip():
            return ""
        patterns = [
            r"last_file_artifact=.*?path=([^;\n]+)",
            r"recent_artifacts:[\s\S]*?path=([^;\n]+)",
            r"last_written_file=([^\n]+)",
            r"last_sent_file=([^\n]+)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for raw in reversed(matches):
                candidate = str(raw or "").strip().strip("'\"")
                if candidate:
                    return candidate
        return ""

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
        if tool.startswith("computer."):
            return "computer_control"
        if tool.startswith("image."):
            return "image_understanding"
        if tool.startswith("memory."):
            return "memory"
        if tool.startswith("qq.send_"):
            return "delivery"
        if tool.startswith("qq."):
            return "qq_history"
        if tool.startswith("skill."):
            return "skill"
        if tool.startswith("app."):
            return "app_control"
        if tool.startswith("system.") or tool.startswith("time.") or tool.startswith("calendar."):
            return "system_utility"
        if tool in {"retrieval.search_local_objects", "file.search_by_name", "file.list", "file.read", "file.extract_text", "file.extract_structure"}:
            return "local_lookup"
        if tool in {"file.metadata", "file.preview", "file.open_path", "file.reveal_in_explorer"}:
            return "file_lookup"
        if tool in {"document_agent.summarize", "document_agent.read", "document_agent.inspect", "document_agent.compose"}:
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
        if decision.decision in {DecisionType.RESPOND, DecisionType.FINISH} and not decision.selected_tool and goal_missing_outputs:
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
            decision.decision in {DecisionType.RESPOND, DecisionType.FINISH}
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
        if selected_tool.startswith("skill."):
            return decision, None
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

    def _should_auto_approve_low_risk_tool_decision(self, decision: ToolDecision) -> bool:
        if decision.decision != DecisionType.TOOL_CALL:
            return False
        if not decision.selected_tool:
            return False
        if decision.memory_write:
            return False
        if decision.risk_level != RiskLevel.LOW:
            return False
        if decision.selected_tool not in self._TRUSTED_STATE_MACHINE_TOOLS:
            return False

        manifest = self.registry.get_manifest(decision.selected_tool)
        if not manifest.read_only:
            return False
        if manifest.destructive or manifest.requires_confirmation:
            return False
        return True

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
        if selected_tool == "file.search_by_name":
            return bool(str(arguments.get("path") or "").strip() and str(arguments.get("query") or "").strip())
        if selected_tool == "retrieval.search_local_objects":
            return bool(str(arguments.get("path_scope") or arguments.get("path") or "").strip() and str(arguments.get("query") or "").strip())
        if selected_tool == "file.list":
            return bool(str(arguments.get("path") or "").strip())
        if selected_tool in {"file.read", "file.extract_text", "image.read_text"}:
            paths = arguments.get("paths")
            return isinstance(paths, list) and any(str(path or "").strip() for path in paths)
        if selected_tool in {"file.metadata", "file.preview", "file.open_path", "file.reveal_in_explorer", "image.describe", "image.inspect"}:
            return bool(str(arguments.get("path") or "").strip())
        if selected_tool == "document_agent.compose":
            return bool(str(arguments.get("instruction") or "").strip())
        if selected_tool in {"document_agent.summarize", "document_agent.read", "document_agent.inspect", "document_agent.edit"}:
            return bool(str(arguments.get("source_path") or "").strip())
        if selected_tool == "memory.recall":
            return bool(str(arguments.get("query") or "").strip())
        if selected_tool == "system.get_time":
            return True
        if selected_tool in {"system.create_reminder", "system.create_scheduled_task"}:
            has_time = bool(str(arguments.get("when_iso") or "").strip())
            has_message = bool(str(arguments.get("message") or "").strip())
            if selected_tool == "system.create_scheduled_task":
                return has_time and has_message and bool(str(arguments.get("task_type") or "").strip())
            return has_time and has_message
        if selected_tool == "system.list_reminders":
            return True
        if selected_tool == "system.cancel_reminder":
            return bool(str(arguments.get("reminder_id") or "").strip())
        return False

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
            cleaned = re.sub(r"[,.!?:;，。！？：；、（）()《》【】\[\]{}\"'`]+", " ", cleaned)
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
        return Path(path).suffix.lower() in {".docx", ".md"}

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

        if action in {"inspect", "extract_document_structure", "search_document_blocks"}:
            if OutputKind.OBJECT_DETAILS in completed_outputs:
                return None
            if cls._looks_like_document_path(path):
                return cls._build_document_details_followup(
                    path=path,
                    user_text=user_text,
                    overall_task_goal=overall_task_goal,
                    reason="Delegate the grounded document inspection to the document sub-agent.",
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
            ):
                return cls._build_document_agent_read_followup(
                    user_text=user_text,
                    path=primary_path,
                    overall_task_goal=overall_task_goal,
                    reason="State transition: the grounded document should now be read by the document sub-agent.",
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
            if cls._looks_like_document_path(primary_path):
                return cls._build_document_details_followup(
                    path=primary_path,
                    user_text=user_text,
                    overall_task_goal=overall_task_goal,
                    reason="State transition: delegate the document inspection to the document sub-agent.",
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

    def _build_contract_workflow_followup(
        self,
        *,
        workflow_spec: WorkflowSpec | None,
        task_envelope: TaskEnvelope | None,
        user_text: str,
        candidate_state: CandidateState | None,
        overall_task_goal: TaskGoal | None,
        completed_outputs: list[OutputKind],
        tool_results: list[ToolCallResult] | None = None,
        supports_message_delivery: bool = False,
    ) -> ToolDecision | None:
        if workflow_spec is None or not workflow_spec.nodes:
            return None

        completed = set(completed_outputs)
        goal = workflow_spec.goal or overall_task_goal
        for node in workflow_spec.nodes:
            produces = [output for output in node.produces if isinstance(output, OutputKind)]
            if produces and set(produces).issubset(completed):
                continue
            requires = [output for output in node.requires if isinstance(output, OutputKind)]
            if any(output not in completed for output in requires):
                return None
            tool_name = str(node.tool or "").strip()
            if not tool_name:
                if OutputKind.MESSAGE_SENT in produces:
                    return ToolDecision(
                        decision=DecisionType.RESPOND,
                        intent=str(node.intent or f"workflow_{node.node_id}").strip() or f"workflow_{node.node_id}",
                        reason=str(
                            node.reason
                            or "The workflow node is a channel reply; render it as the final response."
                        ).strip(),
                        risk_level=RiskLevel.LOW,
                        overall_task_goal=goal,
                        expected_step_outputs=produces,
                    )
                return None
            if not self.registry.has_tool(tool_name):
                return None
            if tool_name == "qq.send_file" and not supports_message_delivery:
                return None

            arguments = self._build_contract_workflow_arguments(
                tool_name=tool_name,
                user_text=user_text,
                candidate_state=candidate_state,
                task_envelope=task_envelope,
                tool_results=tool_results or [],
            )
            if tool_name == "qq.send_text" and not str(arguments.get("message") or "").strip():
                return ToolDecision(
                    decision=DecisionType.RESPOND,
                    intent=str(node.intent or f"workflow_{node.node_id}").strip() or f"workflow_{node.node_id}",
                    reason=(
                        "The contract requested a current-channel text reply but did not provide an outbound "
                        "message body, so the kernel should render the final response instead of calling qq.send_text."
                    ),
                    risk_level=RiskLevel.LOW,
                    overall_task_goal=goal,
                    expected_step_outputs=produces,
                )
            tool_name, arguments = self._adapt_contract_workflow_tool_for_grounded_target(
                tool_name=tool_name,
                arguments=arguments,
                produces=produces,
            )
            decision = ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent=str(node.intent or f"workflow_{node.node_id}").strip() or f"workflow_{node.node_id}",
                reason=str(node.reason or "Run the next node from the main agent workflow contract.").strip(),
                selected_tool=tool_name,
                arguments=arguments,
                risk_level=RiskLevel.LOW,
                overall_task_goal=goal,
                expected_step_outputs=produces,
            )
            if task_envelope is not None:
                decision, _trace = self._repair_decision_arguments_from_task_contract(
                    decision,
                    task_envelope=task_envelope,
                )
                decision, _trace = self._route_unsafe_document_write_decision(
                    decision,
                    task_envelope=task_envelope,
                )
            return decision
        return None

    def _scoped_manifests(self, task_envelope) -> list:
        """按需返回相关工具 manifest，避免每步传全部 70+ 个工具."""
        preferred = list(getattr(task_envelope, "preferred_tools", []) or [])
        allowed_families = set(getattr(task_envelope, "allowed_families", []) or [])
        if not preferred and not allowed_families:
            return self.registry.list_manifests()

        all_tools = self.registry.list_manifests()
        scoped = []
        for m in all_tools:
            family = self._tool_family_for_selected_tool(m.tool_name)
            if family in allowed_families or m.tool_name in preferred:
                scoped.append(m)
        # 至少保留 skill + system_utility + memory 模块（skills永远可见）
        for m in all_tools:
            family = self._tool_family_for_selected_tool(m.tool_name)
            if family in {"skill", "system_utility", "memory"} and m not in scoped:
                scoped.append(m)
        return scoped if scoped else all_tools

    @staticmethod
    def _has_authoritative_contract_workflow(task_envelope: TaskEnvelope | None) -> bool:
        # ReAct 模式
        return False

    @staticmethod
    def _build_contract_blocked_decision(
        *,
        workflow_spec: WorkflowSpec | None,
        overall_task_goal: TaskGoal | None,
        completed_outputs: list[OutputKind],
    ) -> ToolDecision | None:
        goal = workflow_spec.goal if workflow_spec is not None and workflow_spec.goal is not None else overall_task_goal
        missing_outputs = [
            output
            for output in ((goal.required_outputs if goal is not None else []) or [])
            if output not in set(completed_outputs or [])
        ]
        if not missing_outputs:
            return None
        return ToolDecision(
            decision=DecisionType.CLARIFY,
            intent="contract_workflow_blocked",
            reason=(
                "The authoritative main-agent workflow has pending required outputs, "
                "but no executable next node is currently available."
            ),
            response_hint="我已经按主任务流程推进到这里，但还缺少继续执行所需的中间结果，需要补齐后才能完成。",
            risk_level=RiskLevel.LOW,
            overall_task_goal=goal,
        )

    def _adapt_contract_workflow_tool_for_grounded_target(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        produces: list[OutputKind],
    ) -> tuple[str, dict[str, Any]]:
        if tool_name != "file.read" or OutputKind.FILE_CONTENTS not in produces:
            return tool_name, arguments
        if not self.registry.has_tool("file.extract_text"):
            return tool_name, arguments
        paths = arguments.get("paths")
        if not isinstance(paths, list) or not paths:
            return tool_name, arguments
        first_path = next((str(path).strip() for path in paths if isinstance(path, str) and path.strip()), "")
        if not first_path or not self._looks_like_document_path(first_path):
            return tool_name, arguments
        return (
            "file.extract_text",
            {
                "paths": paths,
                "encoding": str(arguments.get("encoding") or "utf-8"),
                "max_chars": int(arguments.get("max_chars") or arguments.get("max_bytes") or 12000),
                "max_rows_per_sheet": 10,
            },
        )

    def _build_contract_workflow_arguments(
        self,
        *,
        tool_name: str,
        user_text: str,
        candidate_state: CandidateState | None,
        task_envelope: TaskEnvelope | None,
        tool_results: list[ToolCallResult] | None = None,
    ) -> dict[str, Any]:
        grounded_inputs = self._parse_task_envelope_grounded_inputs(task_envelope)
        subject = self._first_grounded_string(
            grounded_inputs,
            "resolved_subject",
            "target_name",
            "target",
            "query",
            "topic",
            "search_query",
        )
        source_context = self._first_grounded_string(
            grounded_inputs,
            "source_context",
            "scope",
            "path",
            "path_scope",
            "directory",
        )
        target_path = self._first_grounded_string(
            grounded_inputs,
            "target_path",
            "source_path",
            "file_path",
            "path",
        )
        if not target_path and tool_name in {"file.write", "file.read"}:
            target_path = self._build_contract_file_path_from_subject(subject, source_context)
        if not target_path and tool_name == "file.write_docx":
            target_path = self._build_contract_docx_path(
                subject=subject,
                source_context=source_context,
                user_text=user_text,
            )
        if not target_path and candidate_state is not None:
            target_path = self._select_candidate_path_for_action(candidate_state, "read") or ""
        instruction = self._first_grounded_string(
            grounded_inputs,
            "instruction",
            "task",
            "edit_instruction",
            "operation",
        ) or str(getattr(task_envelope, "primary_objective", "") or user_text or "").strip()
        url = self._first_grounded_string(grounded_inputs, "url", "target_url", "source_url")
        contact_query = self._first_grounded_string(
            grounded_inputs,
            "contact_query",
            "contact",
            "contact_name",
            "person",
            "group",
        )
        message = self._first_grounded_string(
            grounded_inputs,
            "message",
            "text",
            "reply_text",
            "speech_text",
        )
        if tool_name in {"system.create_reminder", "system.create_scheduled_task"}:
            message = self._infer_reminder_message_from_contract_context(
                grounded_inputs=grounded_inputs,
                task_envelope=task_envelope,
                user_text=user_text,
                fallback=instruction,
            )
        when_iso = self._first_grounded_string(grounded_inputs, "when_iso", "scheduled_for", "time_iso", "datetime_iso")
        timezone_name = self._first_grounded_string(grounded_inputs, "timezone", "timezone_name", "tz")
        if not when_iso and tool_name in {"system.create_reminder", "system.create_scheduled_task"}:
            when_iso = self._infer_when_iso_from_contract_context(
                grounded_inputs=grounded_inputs,
                task_envelope=task_envelope,
                user_text=user_text,
                tool_results=tool_results,
                timezone_name=timezone_name or "Asia/Shanghai",
            )
        session_id = self._first_grounded_string(grounded_inputs, "session_id", "target_session_id")
        channel = self._first_grounded_string(grounded_inputs, "channel", "runtime_channel")
        task_type = self._first_grounded_string(grounded_inputs, "task_type", "schedule_type")
        reminder_id = self._first_grounded_string(grounded_inputs, "reminder_id", "id")

        if tool_name == "memory.recall":
            query = subject or instruction or str(getattr(task_envelope, "primary_objective", "") or user_text or "").strip()
            return {"query": query, "limit": 5} if query else {}
        if tool_name == "file.search_by_name":
            args: dict[str, Any] = {}
            if target_path:
                args["path"] = str(Path(target_path).parent)
                args["query"] = Path(target_path).name
                args["query_terms"] = [Path(target_path).stem, Path(target_path).name]
                args["extensions"] = [Path(target_path).suffix] if Path(target_path).suffix else []
                args.setdefault("recursive", True)
                args.setdefault("include_dirs", True)
                args.setdefault("scope_mode", "subtree")
                args.setdefault("target_kind", "file")
                args.setdefault("top_k", 8)
                return args
            if source_context:
                args["path"] = source_context
            if subject:
                args["query"] = subject
                plan = self._build_local_retrieval_plan(subject, user_text=str(getattr(task_envelope, "primary_objective", "") or user_text))
                if plan.get("query_terms"):
                    args["query_terms"] = plan["query_terms"]
                if plan.get("alias_terms"):
                    args["alias_terms"] = plan["alias_terms"]
                if plan.get("extensions"):
                    args["extensions"] = plan["extensions"]
            args.setdefault("recursive", True)
            args.setdefault("include_dirs", True)
            args.setdefault("scope_mode", "subtree")
            args.setdefault("target_kind", "file")
            args.setdefault("top_k", 8)
            return args
        if tool_name == "retrieval.search_local_objects":
            args = {}
            if source_context:
                args["path_scope"] = source_context
            if subject:
                args["query"] = subject
            args.setdefault("scope_mode", "subtree")
            args.setdefault("target_kind", "file")
            args.setdefault("top_k", 8)
            args.setdefault("rebuild_if_missing", True)
            return args
        if tool_name == "file.list":
            args = {}
            if source_context:
                args["path"] = source_context
            args.setdefault("recursive", True)
            args.setdefault("include_dirs", True)
            return args
        if tool_name == "file.metadata_many":
            paths = self._latest_listed_image_paths(tool_results or [], source_context=source_context)
            return {"paths": paths, "continue_on_error": True} if paths else {}
        if tool_name == "file.mkdir_many":
            folders = self._image_organization_folder_paths(tool_results or [], source_context=source_context)
            return {"paths": folders, "exist_ok": True, "parents": True, "continue_on_error": True} if folders else {}
        if tool_name in {"file.move_many", "file.copy_many"}:
            items = self._image_organization_move_items(tool_results or [], source_context=source_context)
            return {"items": items, "continue_on_error": True} if items else {}
        if tool_name == "file.extract_text":
            return {"paths": [target_path], "max_chars": 12000, "max_rows_per_sheet": 10} if target_path else {}
        if tool_name == "file.read":
            return {"paths": [target_path], "encoding": "utf-8", "max_bytes": 12000} if target_path else {}
        if tool_name == "file.write":
            content = self._first_grounded_string(
                grounded_inputs,
                "content",
                "file_content",
                "body",
                "text",
            ) or instruction
            args = {"path": target_path} if target_path else {}
            if content:
                args["content"] = content
            return args
        if tool_name == "document_agent.compose":
            if not target_path:
                target_path = self._build_contract_docx_path(
                    subject=subject,
                    source_context=source_context,
                    user_text=user_text,
                )
            args = {
                "instruction": instruction,
                "output_path": target_path,
                "title": subject or str(getattr(task_envelope, "primary_objective", "") or "").strip(),
                "recent_context": "",
                "resolved_facts": {},
                "source_materials": self._collect_document_source_materials(tool_results or []),
                "style_hints": {},
                "max_chars": 12000,
            }
            if grounded_inputs:
                args["grounded_inputs"] = grounded_inputs
            return args
        if tool_name == "file.write_docx":
            composed_content = ""
            composed_title = None
            composed_doc = self._latest_composed_document(tool_results or [])
            if composed_doc:
                composed_title = str(composed_doc.get("title") or "").strip() or None
                composed_content = self._clean_web_document_text(str(composed_doc.get("content") or "").strip())
                if not composed_content:
                    files = composed_doc.get("files")
                    if isinstance(files, list):
                        for file_entry in files:
                            if isinstance(file_entry, dict):
                                composed_content = self._clean_web_document_text(str(file_entry.get("content") or "").strip())
                                if composed_content:
                                    break
            elif any(
                result.status == "success" and result.tool_name in {"web.research", "web.fetch"}
                for result in (tool_results or [])
            ):
                composed_content, composed_title = self._compose_web_write_content(
                    user_text=user_text,
                    tool_results=tool_results or [],
                    delivery_intent=SimpleNamespace(title=subject or None),
                    recent_context="",
                )
            content = composed_content or self._first_grounded_string(
                grounded_inputs,
                "content",
                "file_content",
                "body",
                "text",
            ) or self._build_docx_content_from_tool_results(tool_results or []) or instruction
            content = self._clean_web_document_text(content) or content
            title = composed_title or subject or str(getattr(task_envelope, "primary_objective", "") or "").strip() or Path(target_path).stem
            args = {"path": target_path} if target_path else {}
            if title:
                args["title"] = title
            if content:
                args["content"] = content
                args["paragraphs"] = self._split_docx_paragraphs(content)
            args["overwrite"] = True
            return args
        if tool_name in {"file.metadata", "file.preview", "file.open_path", "file.reveal_in_explorer", "file.delete", "image.describe", "image.inspect"}:
            return {"path": target_path} if target_path else {}
        if tool_name == "image.read_text":
            return {"paths": [target_path], "max_chars": 8000} if target_path else {}
        if tool_name in {"document_agent.summarize", "document_agent.read", "document_agent.inspect", "document_agent.edit"}:
            args = {"source_path": target_path, "instruction": instruction} if target_path else {"instruction": instruction}
            if grounded_inputs:
                args["grounded_inputs"] = grounded_inputs
            args.setdefault("recent_context", "")
            args.setdefault("resolved_facts", {})
            args.setdefault("source_materials", {})
            args.setdefault("constraints", {})
            args.setdefault("style_hints", {})
            if tool_name == "document_agent.edit":
                args.setdefault("allow_overwrite", True)
                args.setdefault("preserve_structure", True)
                args.setdefault("preserve_style", True)
            return args
        if tool_name in {"web.search", "web.research"}:
            args = {"query": subject} if subject else {}
            if tool_name == "web.research" and subject:
                args.update({"max_results": 5, "max_pages": 2, "prefer_browser": True})
            elif subject:
                args.setdefault("max_results", 5)
            return args
        if tool_name == "web.fetch":
            return {"url": url} if url else {}
        if tool_name == "qq.search_history":
            args = {"limit": 5}
            if subject:
                args["query"] = subject
            if contact_query:
                args["contact_query"] = contact_query
            return args
        if tool_name == "qq.get_recent_messages":
            args = {"limit": 2, "include_assistant": True}
            if session_id:
                args["session_id"] = session_id
            return args
        if tool_name == "qq.search_contacts":
            return {"query": contact_query or subject, "target_kind": "any", "limit": 5} if (contact_query or subject) else {}
        if tool_name in {"qq.get_current_context", "qq.get_last_reply"}:
            return {}
        if tool_name == "qq.get_recent_attachments":
            args = {"kind": "any", "limit": 5}
            if contact_query:
                args["contact_query"] = contact_query
            return args
        if tool_name == "qq.send_text":
            return {"message": message, "target_kind": "current"} if message else {"target_kind": "current"}
        if tool_name == "qq.send_file":
            return {"file_path": target_path, "target_kind": "current"} if target_path else {"target_kind": "current"}
        if tool_name == "qq.send_voice":
            return {"speech_text": message, "target_kind": "current"} if message else {"target_kind": "current"}
        if tool_name == "system.get_time":
            kind = self._first_grounded_string(grounded_inputs, "kind", "time_kind") or "datetime"
            args = {"kind": kind}
            if timezone_name:
                args["timezone_name"] = timezone_name
            return args
        if tool_name == "system.create_reminder":
            task_payload = grounded_inputs.get("task_payload")
            args = {
                "when_iso": when_iso,
                "timezone": timezone_name or "Asia/Shanghai",
                "message": message or instruction,
                "task_payload": self._ensure_cached_reminder_payload(task_payload, message or instruction),
            }
            if session_id:
                args["session_id"] = session_id
            if channel:
                args["channel"] = channel
            return {key: value for key, value in args.items() if value not in (None, "")}
        if tool_name == "system.create_scheduled_task":
            task_payload = grounded_inputs.get("task_payload")
            args = {
                "task_type": task_type or "notify",
                "when_iso": when_iso,
                "timezone": timezone_name or "Asia/Shanghai",
                "message": message or instruction,
                "task_payload": self._ensure_cached_reminder_payload(task_payload, message or instruction)
                if (task_type or "notify") == "notify"
                else (task_payload if isinstance(task_payload, dict) else {}),
            }
            if session_id:
                args["session_id"] = session_id
            if channel:
                args["channel"] = channel
            return {key: value for key, value in args.items() if value not in (None, "")}
        if tool_name == "system.list_reminders":
            args = {"status": self._first_grounded_string(grounded_inputs, "status") or "scheduled"}
            if session_id:
                args["session_id"] = session_id
            return args
        if tool_name == "system.cancel_reminder":
            return {"reminder_id": reminder_id} if reminder_id else {}
        return {}

    _IMAGE_ORGANIZATION_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".webp",
        ".heic",
        ".tif",
        ".tiff",
    }

    @classmethod
    def _is_image_file_path(cls, path: str) -> bool:
        try:
            return Path(path).suffix.lower() in cls._IMAGE_ORGANIZATION_EXTENSIONS
        except Exception:
            return False

    @classmethod
    def _latest_listed_image_paths(
        cls,
        tool_results: list[ToolCallResult],
        *,
        source_context: str | None = None,
        limit: int = 300,
    ) -> list[str]:
        source_root = Path(source_context).resolve() if source_context else None
        for result in reversed(tool_results):
            if result.status != "success" or result.tool_name != "file.list":
                continue
            entries = result.data.get("entries", [])
            paths: list[str] = []
            for entry in entries:
                if not isinstance(entry, dict) or bool(entry.get("is_dir")):
                    continue
                raw_path = str(entry.get("path") or "").strip()
                if not raw_path or not cls._is_image_file_path(raw_path):
                    continue
                path = Path(raw_path)
                if source_root is not None:
                    try:
                        path.resolve().relative_to(source_root)
                    except Exception:
                        continue
                paths.append(str(path))
                if len(paths) >= limit:
                    break
            if paths:
                return paths
        return []

    @classmethod
    def _latest_image_metadata_items(cls, tool_results: list[ToolCallResult]) -> list[dict[str, Any]]:
        for result in reversed(tool_results):
            if result.status != "success" or result.tool_name != "file.metadata_many":
                continue
            raw_items = result.data.get("items") or result.data.get("results") or []
            items: list[dict[str, Any]] = []
            for item in raw_items:
                if not isinstance(item, dict) or not bool(item.get("ok", True)):
                    continue
                path = str(item.get("path") or "").strip()
                if path and cls._is_image_file_path(path):
                    items.append(item)
            if items:
                return items
        return []

    @staticmethod
    def _image_group_from_metadata(item: dict[str, Any]) -> str:
        raw_modified = item.get("modified_at")
        try:
            modified = datetime.fromtimestamp(float(raw_modified))
            return modified.strftime("%Y-%m")
        except Exception:
            return "未分类"

    @classmethod
    def _image_organization_destination_root(
        cls,
        metadata_items: list[dict[str, Any]],
        *,
        source_context: str | None = None,
    ) -> Path | None:
        if source_context:
            return Path(source_context) / "图片整理"
        for item in metadata_items:
            path = str(item.get("path") or "").strip()
            if path:
                return Path(path).parent / "图片整理"
        return None

    @classmethod
    def _image_organization_folder_paths(
        cls,
        tool_results: list[ToolCallResult],
        *,
        source_context: str | None = None,
    ) -> list[str]:
        metadata_items = cls._latest_image_metadata_items(tool_results)
        if not metadata_items:
            # 没有 metadata 时，用 file.list 结果推断
            paths = cls._latest_listed_image_paths(tool_results, source_context=source_context)
            if not paths:
                return []
            from datetime import datetime as _dt
            group = _dt.now().strftime("%Y-%m")
            if source_context:
                dest_root = Path(source_context) / "图片整理"
            else:
                dest_root = Path(paths[0]).parent / "图片整理"
            return [str(dest_root / group)]
        destination_root = cls._image_organization_destination_root(metadata_items, source_context=source_context)
        if destination_root is None:
            return []
        groups = sorted({cls._image_group_from_metadata(item) for item in metadata_items})
        return [str(destination_root / group) for group in groups]

    @classmethod
    def _image_organization_move_items(
        cls,
        tool_results: list[ToolCallResult],
        *,
        source_context: str | None = None,
    ) -> list[dict[str, Any]]:
        metadata_items = cls._latest_image_metadata_items(tool_results)
        if not metadata_items:
            # 没有 metadata 结果时，直接用 file.list 的图片路径
            paths = cls._latest_listed_image_paths(tool_results, source_context=source_context)
            if not paths:
                return []
            # 用当前时间做简单分组
            from datetime import datetime as _dt
            group = _dt.now().strftime("%Y-%m")
            if source_context:
                dest_root = Path(source_context) / "图片整理"
            else:
                dest_root = Path(paths[0]).parent / "图片整理"
            return [
                {
                    "src_path": p,
                    "dest_path": str(dest_root / group / Path(p).name),
                    "overwrite": False,
                }
                for p in paths
            ]
        destination_root = cls._image_organization_destination_root(metadata_items, source_context=source_context)
        if destination_root is None:
            return []
        items: list[dict[str, Any]] = []
        for item in metadata_items:
            src = Path(str(item.get("path") or "").strip())
            if not src.name:
                continue
            try:
                src.resolve().relative_to(destination_root.resolve())
                continue
            except Exception:
                pass
            group = cls._image_group_from_metadata(item)
            items.append(
                {
                    "src_path": str(src),
                    "dest_path": str(destination_root / group / src.name),
                    "overwrite": False,
                }
            )
        return items

    @staticmethod
    def _build_contract_workflow_debug_payload(
        *,
        workflow_spec: WorkflowSpec | None,
        completed_outputs: list[OutputKind],
        candidate_state: CandidateState | None = None,
        observed_workflow_state: WorkflowState | None = None,
        active_overall_goal: TaskGoal | None = None,
        selected_decision: ToolDecision | None = None,
    ) -> dict[str, Any]:
        completed_values = [
            output.value if isinstance(output, OutputKind) else str(output)
            for output in completed_outputs
        ]
        completed_set = set(completed_outputs)
        nodes: list[dict[str, Any]] = []
        next_node_id: str | None = None
        blocked_by: list[str] = []
        selected_node_id: str | None = None
        selected_outputs = set(getattr(selected_decision, "expected_step_outputs", []) or [])
        selected_tool = str(getattr(selected_decision, "selected_tool", "") or "").strip()

        if workflow_spec is not None:
            for node in workflow_spec.nodes:
                produces = [output for output in node.produces if isinstance(output, OutputKind)]
                requires = [output for output in node.requires if isinstance(output, OutputKind)]
                missing_requires = [output for output in requires if output not in completed_set]
                missing_produces = [output for output in produces if output not in completed_set]
                if produces and not missing_produces:
                    status = "completed"
                elif missing_requires:
                    status = "blocked"
                else:
                    status = "ready"
                    if next_node_id is None:
                        next_node_id = node.node_id
                if selected_node_id is None and selected_tool and node.tool == selected_tool:
                    if not selected_outputs or selected_outputs == set(produces):
                        selected_node_id = node.node_id
                if status == "blocked" and next_node_id is None:
                    blocked_by = [output.value for output in missing_requires]

                nodes.append(
                    {
                        "node_id": node.node_id,
                        "tool": node.tool,
                        "intent": node.intent,
                        "status": status,
                        "requires": [output.value for output in requires],
                        "produces": [output.value for output in produces],
                        "missing_requires": [output.value for output in missing_requires],
                        "missing_outputs": [output.value for output in missing_produces],
                    }
                )

        required_outputs = []
        if active_overall_goal is not None:
            required_outputs = [
                output.value if isinstance(output, OutputKind) else str(output)
                for output in active_overall_goal.required_outputs
            ]
        missing_outputs = [output for output in required_outputs if output not in completed_values]
        return {
            "workflow_name": None if workflow_spec is None else workflow_spec.workflow_name,
            "goal": None if workflow_spec is None or workflow_spec.goal is None else workflow_spec.goal.model_dump(mode="json"),
            "nodes": nodes,
            "node_count": len(nodes),
            "next_node_id": next_node_id,
            "selected_node_id": selected_node_id,
            "selected_tool": selected_tool or None,
            "selected_decision": None if selected_decision is None else selected_decision.model_dump(mode="json"),
            "blocked_by": blocked_by,
            "completed_outputs": completed_values,
            "required_outputs": required_outputs,
            "missing_outputs": missing_outputs,
            "candidate_state": None if candidate_state is None else candidate_state.model_dump(mode="json"),
            "observed_workflow_state": None if observed_workflow_state is None else observed_workflow_state.model_dump(mode="json"),
        }

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
                return "inspect"
            return None
        if OutputKind.FILE_CONTENTS in missing_outputs:
            if AgentKernel._looks_like_image_text_request(user_text):
                return "read_image_text"
            return "read"
        if OutputKind.OBJECT_DETAILS in missing_outputs:
            if AgentKernel._looks_like_image_request(user_text):
                return "describe_image"
            if AgentKernel._looks_like_preview_request(user_text):
                return "preview"
            return "inspect"
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
        cleaned = re.sub(r"[,.!?:;，。！？：；、（）()《》【】\[\]{}\"'`]+", " ", cleaned)
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
        explicit_exts = [
            ext
            for ext in (".docx", ".pptx", ".xlsx", ".md", ".txt", ".pdf", ".log", ".csv", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
            if ext in lowered
        ]
        if explicit_exts:
            return cls._dedupe_preserve_order(explicit_exts)
        if re.search(r"(?<![a-z0-9])(?:ppt|powerpoint|presentation)(?![a-z0-9])", lowered):
            return [".pptx"]
        if re.search(r"(?<![a-z0-9])(?:excel|sheet|spreadsheet)(?![a-z0-9])", lowered):
            return [".xlsx", ".csv"]
        if re.search(r"(?<![a-z0-9])(?:word|docx?)(?![a-z0-9])", lowered):
            return [".docx"]
        if re.search(r"(?<![a-z0-9])(?:markdown|md)(?![a-z0-9])", lowered):
            return [".md"]
        if re.search(r"(?<![a-z0-9])pdf(?![a-z0-9])", lowered):
            return [".pdf"]
        if re.search(r"(?<![a-z0-9])txt(?![a-z0-9])", lowered):
            return [".txt"]
        if re.search(r"(?<![a-z0-9])log file(?![a-z0-9])", lowered):
            return [".log"]
        return []

    @classmethod
    def _build_local_retrieval_plan(cls, query: str, *, user_text: str = "") -> dict[str, object]:
        extensions = cls._local_file_extensions_for_query(query, user_text=user_text)
        planned_query = FileQueryNormalizer.strip_file_type_terms(query, extensions) if extensions else query
        planned_query = planned_query or query
        query_terms = cls._dedupe_preserve_order(cls._tokenize_query(planned_query))
        return {
            "query": planned_query,
            "query_terms": query_terms or cls._tokenize_query(planned_query),
            "alias_terms": [],
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
            return None

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
            return None

        selected_tool = str(plan.get("selected_tool", "") or "").strip()
        arguments = plan.get("arguments", {})
        if selected_tool not in set(allowed_tools) or not isinstance(arguments, dict):
            return None

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
            return None

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
    def _looks_like_tool_observation_text(text: str) -> bool:
        compact = str(text or "").strip()
        if not compact:
            return False
        return bool(re.match(r"^req_[0-9a-f]+\s+\w+\.", compact)) or " results_sample=" in compact

    @staticmethod
    def _looks_like_mojibake_text(text: str) -> bool:
        sample = str(text or "")[:400]
        if not sample:
            return False
        markers = ("å", "æ", "ç", "ä", "è", "é", "Â", "¤", "", "", "", "")
        marker_count = sum(sample.count(marker) for marker in markers)
        return marker_count >= 8

    @classmethod
    def _clean_web_document_text(cls, text: str) -> str:
        lines: list[str] = []
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if cls._looks_like_tool_observation_text(line):
                continue
            if cls._looks_like_mojibake_text(line):
                continue
            lines.append(line)
        return "\n\n".join(lines).strip()

    @staticmethod
    def _latest_web_research_bundle(tool_results: list[ToolCallResult]) -> dict[str, Any] | None:
        for result in reversed(tool_results):
            if result.status == "success" and result.tool_name in {"web.research", "web.fetch"}:
                return result.data
        return None

    @staticmethod
    def _latest_composed_document(tool_results: list[ToolCallResult]) -> dict[str, Any] | None:
        for result in reversed(tool_results):
            if result.status == "success" and result.tool_name == "document_agent.compose":
                return result.data
        return None

    @staticmethod
    def _collect_document_source_materials(tool_results: list[ToolCallResult]) -> dict[str, Any]:
        materials: dict[str, Any] = {}
        tool_result_entries: list[dict[str, Any]] = []
        for result in tool_results:
            if result.status != "success":
                continue
            entry = {
                "tool_name": result.tool_name,
                "data": result.data,
                "produced_outputs": [output.value for output in result.produced_outputs],
            }
            tool_result_entries.append(entry)
            if result.tool_name == "web.research":
                materials["research_bundle"] = result.data
            elif result.tool_name == "web.fetch" and "research_bundle" not in materials:
                materials["web_result"] = result.data
            elif result.tool_name and result.tool_name.startswith("file."):
                materials.setdefault("file_results", []).append(result.data)
        if tool_result_entries:
            materials["tool_results"] = tool_result_entries
        return materials

    @staticmethod
    def _latest_web_write_content(tool_results: list[ToolCallResult]) -> str:
        for result in reversed(tool_results):
            if result.status != "success":
                continue
            if result.tool_name == "web.research":
                summary = result.data.get("summary")
                if isinstance(summary, str) and summary.strip():
                    return AgentKernel._clean_web_document_text(summary)
                content = result.data.get("content")
                if isinstance(content, str) and content.strip():
                    cleaned_content = AgentKernel._clean_web_document_text(content)
                    if cleaned_content:
                        return cleaned_content
                snippets: list[str] = []
                for item in result.data.get("sources", [])[:3]:
                    if not isinstance(item, dict):
                        continue
                    title = str(item.get("title", "")).strip()
                    url = str(item.get("url", "")).strip()
                    excerpt = str(item.get("excerpt", "") or item.get("content", "")).strip()
                    line = "\n".join(part for part in (title, url, excerpt) if part)
                    line = AgentKernel._clean_web_document_text(line)
                    if line:
                        snippets.append(line)
                if snippets:
                    return "\n\n".join(snippets)
            if result.tool_name == "web.fetch":
                content = result.data.get("content")
                if isinstance(content, str) and content.strip():
                    return AgentKernel._clean_web_document_text(content)
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
        llm_client = getattr(self, "llm_client", None)
        if bundle is None or not hasattr(llm_client, "compose_web_research_document"):
            return fallback_content, None
        try:
            composed = llm_client.compose_web_research_document(
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
        content = self._clean_web_document_text(str(composed.get("content") or "").strip())
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

    def _build_legacy_recovery_followup(
        self,
        *,
        user_text: str,
        planner_user_text: str,
        overall_task_goal: TaskGoal | None,
        completed_outputs: list[OutputKind],
        candidate_state: CandidateState | None,
        tool_results: list[ToolCallResult],
        last_executed_decision: ToolDecision | None,
        document_delivery_intent,
        knowledge_request_intent,
        site_search_intent,
        recent_context: str,
    ) -> ToolDecision | None:
        if not tool_results or last_executed_decision is None:
            return None
        for builder in (
            lambda: self._build_candidate_write_followup(
                user_text=planner_user_text,
                overall_task_goal=overall_task_goal,
                completed_outputs=completed_outputs,
                candidate_state=candidate_state,
            ),
            lambda: self._build_docx_edit_followup(
                user_text=planner_user_text,
                overall_task_goal=overall_task_goal,
                completed_outputs=completed_outputs,
                candidate_state=candidate_state,
            ),
            lambda: self._build_web_write_followup(
                user_text=user_text,
                overall_task_goal=overall_task_goal,
                completed_outputs=completed_outputs,
                tool_results=tool_results,
                delivery_intent=document_delivery_intent,
                recent_context=recent_context,
            ),
            lambda: self.web_retrieval_strategy.build_empty_result_fallback(
                user_text=user_text,
                last_decision=last_executed_decision,
                last_result=tool_results[-1],
                delivery_intent=document_delivery_intent,
                knowledge_intent=knowledge_request_intent,
                site_search_intent=site_search_intent,
            ),
            lambda: self.file_retrieval_strategy.build_empty_result_fallback(
                user_text=user_text,
                last_decision=last_executed_decision,
                last_result=tool_results[-1],
                candidate_state=candidate_state,
                reliable_candidates=self._has_reliable_candidates(candidate_state, overall_task_goal),
            ),
        ):
            decision = builder()
            if decision is not None:
                return decision
        return None

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
        llm_metrics_start = len(getattr(self.llm_client, "last_call_metrics", []) or [])
        session_id = str(runtime_session_id or "").strip() or f"session_{uuid.uuid4().hex[:10]}"
        artifacts = TurnArtifacts(trace_id=trace_id)
        tool_results: list[ToolCallResult] = []
        request_signatures: list[str] = []
        active_overall_goal: TaskGoal | None = seed_overall_task_goal
        active_candidate_state: CandidateState | None = seed_candidate_state
        observed_workflow_state: WorkflowState | None = seed_workflow_state
        completed_outputs: list[OutputKind] = []
        loop_stop_reason: str | None = None
        last_executed_decision: ToolDecision | None = None
        planner_invocations = 0
        critic_invocations = 0
        planner_bypass_count = 0
        decision_path: list[dict[str, object]] = []
        self.history.append(Message(role=Role.USER, content=user_text))
        recent_context = self._recent_conversation_text()
        channel_context_summary = self._build_channel_context_summary(runtime_channel, runtime_channel_context)
        intent_bundle = self._build_single_source_intent_bundle(
            user_text=user_text,
            recent_context=recent_context,
            hot_context_summary=self.hot_context_summary,
            warm_memory_summary=self.warm_memory_summary,
            learning_memory_summary=self.learning_memory_summary,
            cold_memory_summary=self.cold_memory_summary,
            active_task_summary=self.active_task_summary,
            channel_context_summary=channel_context_summary,
        )
        task_graph = getattr(intent_bundle, "task_graph", None)
        planner_user_text = user_text
        document_delivery_intent = intent_bundle.document_delivery
        knowledge_request_intent = intent_bundle.knowledge_request
        site_search_intent = intent_bundle.site_search
        authoritative_required_outputs = list(
            (active_overall_goal.required_outputs if active_overall_goal is not None else [])
            or getattr(intent_bundle.task_envelope, "required_outputs", [])
            or []
        )
        main_goal_locked = bool(authoritative_required_outputs)
        authoritative_contract_workflow = self._has_authoritative_contract_workflow(intent_bundle.task_envelope)

        scripted_planner_decision_count = self._scripted_planner_decision_count()
        scripted_planner_decisions_used = 0

        llm_runtime = {
            "provider": str(getattr(self.config, "llm_provider", "ollama") or "ollama"),
            "model": str(getattr(self.config, "model", "") or ""),
            "chat_model": str(getattr(self.config, "chat_model", "") or ""),
            "critic_model": str(getattr(self.config, "critic_model", "") or ""),
            "response_model": str(getattr(self.config, "response_model", "") or ""),
        }

        self.trace_store.append(
            "user_input",
            {
                "trace_id": trace_id,
                "text": user_text,
                "planner_text": planner_user_text,
                "llm_runtime": llm_runtime,
            },
        )
        self.trace_store.append(
            "intent_context",
            {
                "trace_id": trace_id,
                "llm_runtime": llm_runtime,
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
            "main_agent_context",
            {
                "trace_id": trace_id,
                "llm_runtime": llm_runtime,
                "planner_text": planner_user_text,
                "mode": "single_source_main_agent",
                "context_layers_used": intent_bundle.task_envelope.context_layers_used,
                "task_envelope": intent_bundle.task_envelope.model_dump(mode="json"),
            },
        )
        self.trace_store.append(
            "workflow_contract_debug",
            {
                "trace_id": trace_id,
                "source": "main_agent_contract",
                "primary_objective": intent_bundle.task_envelope.primary_objective,
                "workflow": self._build_contract_workflow_debug_payload(
                    workflow_spec=getattr(intent_bundle.task_envelope, "workflow_spec", None),
                    completed_outputs=[],
                    candidate_state=active_candidate_state,
                    observed_workflow_state=observed_workflow_state,
                    active_overall_goal=TaskGoal(
                        summary=str(getattr(intent_bundle.task_envelope, "primary_objective", "") or "").strip()
                        or "Complete the active task.",
                        required_outputs=list(getattr(intent_bundle.task_envelope, "required_outputs", []) or []),
                        completion_mode="outputs",
                    )
                    if getattr(intent_bundle.task_envelope, "required_outputs", None)
                    else active_overall_goal,
                ),
            },
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
        self._emit_progress(progress_callback, "received", "我先记下你的需求，准备开始分析。", {"trace_id": trace_id})

        try:
            for step in range(self.config.max_steps):
                observations = ContextBuilder.build_observations(tool_results)

                self._emit_progress(
                    progress_callback,
                    "planning",
                    f"第 {step + 1} 步：我在判断下一步该直接回答，还是去调用工具。",
                    {"step": step},
                )

                fallback_decision = None
                scripted_planner_has_pending_decision = scripted_planner_decisions_used < scripted_planner_decision_count
                bound_workflow_family = "generic"
                bound_workflow_decision = None
                bound_goal = None

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
                contract_workflow_decision = None if scripted_planner_has_pending_decision else self._build_contract_workflow_followup(
                    workflow_spec=getattr(intent_bundle.task_envelope, "workflow_spec", None),
                    task_envelope=intent_bundle.task_envelope,
                    user_text=planner_user_text,
                    candidate_state=active_candidate_state,
                    overall_task_goal=active_overall_goal,
                    completed_outputs=completed_outputs,
                    tool_results=tool_results,
                    supports_message_delivery=self.registry.has_tool("qq.send_file"),
                )
                workflow_decision = contract_workflow_decision
                if (
                    workflow_decision is None
                    and not authoritative_contract_workflow
                    and not scripted_planner_has_pending_decision
                ):
                    workflow_decision = self._build_state_transition_followup(
                        user_text=planner_user_text,
                        workflow_state=observed_workflow_state,
                        candidate_state=active_candidate_state,
                        overall_task_goal=active_overall_goal,
                        completed_outputs=completed_outputs,
                        supports_message_delivery=self.registry.has_tool("qq.send_file"),
                    )
                if (
                    workflow_decision is None
                    and not authoritative_contract_workflow
                    and not scripted_planner_has_pending_decision
                ):
                    workflow_decision = self._build_candidate_action_followup(
                        user_text=planner_user_text,
                        overall_task_goal=active_overall_goal,
                        completed_outputs=completed_outputs,
                        candidate_state=active_candidate_state,
                        supports_message_delivery=self._supports_message_delivery(),
                    )
                contract_blocked_decision = None
                if (
                    workflow_decision is None
                    and authoritative_contract_workflow
                    and not scripted_planner_has_pending_decision
                ):
                    contract_blocked_decision = self._build_contract_blocked_decision(
                        workflow_spec=getattr(intent_bundle.task_envelope, "workflow_spec", None),
                        overall_task_goal=active_overall_goal,
                        completed_outputs=completed_outputs,
                    )
                    workflow_decision = contract_blocked_decision
                if contract_workflow_decision is not None:
                    state_machine_source = "contract_workflow"
                elif contract_blocked_decision is not None:
                    state_machine_source = "contract_blocked"
                elif workflow_decision is not None:
                    state_machine_source = "observed_workflow"
                else:
                    state_machine_source = "no_workflow_decision"
                self.trace_store.append(
                    "state_machine_debug",
                    {
                        "trace_id": trace_id,
                        "step": step,
                        "phase": "before_decision",
                        "source": state_machine_source,
                        "workflow": self._build_contract_workflow_debug_payload(
                            workflow_spec=getattr(intent_bundle.task_envelope, "workflow_spec", None),
                            completed_outputs=completed_outputs,
                            candidate_state=active_candidate_state,
                            observed_workflow_state=observed_workflow_state,
                            active_overall_goal=active_overall_goal,
                            selected_decision=workflow_decision,
                        ),
                    },
                )
                if (
                    tool_results
                    and last_executed_decision is not None
                    and not authoritative_contract_workflow
                ):
                    fallback_decision = self._build_legacy_recovery_followup(
                        user_text=user_text,
                        planner_user_text=planner_user_text,
                        overall_task_goal=active_overall_goal,
                        completed_outputs=completed_outputs,
                        candidate_state=active_candidate_state,
                        tool_results=tool_results,
                        last_executed_decision=last_executed_decision,
                        document_delivery_intent=document_delivery_intent,
                        knowledge_request_intent=knowledge_request_intent,
                        site_search_intent=site_search_intent,
                        recent_context=recent_context,
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
                        self._format_main_agent_context_observation(
                            user_text=user_text,
                            recent_context=recent_context,
                            active_task_summary=self.active_task_summary,
                            channel_context_summary=channel_context_summary,
                            hot_context_summary=self.hot_context_summary,
                            warm_memory_summary=self.warm_memory_summary,
                            learning_memory_summary=self.learning_memory_summary,
                            cold_memory_summary=self.cold_memory_summary,
                        ),
                    ]
                    planner_observations.append(
                        "Main agent execution contract. This is authoritative task scope, not tool arguments. "
                        "When calling a tool, convert grounded_inputs into that tool's declared input schema and do not copy unrelated keys directly:\n"
                        + intent_bundle.task_envelope.model_dump_json(exclude_none=True)
                    )

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
                    if contract_blocked_decision is not None and state_machine_repair is contract_blocked_decision:
                        raw_decision = contract_blocked_decision
                        review = DecisionReview(
                            approved=True,
                            issues=["authoritative_contract_blocked"],
                            summary="The main-agent workflow is authoritative and has no executable next node, so the kernel did not re-plan the task.",
                            suggested_decision=None,
                        )
                        decision_source = "contract_blocked"
                    elif (
                        (not main_goal_locked) or state_machine_repair is contract_workflow_decision
                    ) and self._should_short_circuit_state_machine_repair(
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
                                tool_manifests=self._scoped_manifests(intent_bundle.task_envelope),
                                observations=planner_observations,
                                allowed_decisions=sorted(allowed_actions),
                                bound_workflow_family=bound_workflow_family,
                            )
                            raw_decision = self._normalize_network_lookup_decision(raw_decision, user_text=user_text)
                            scripted_planner_decisions_used += 1
                            if raw_decision.selected_tool and raw_decision.selected_tool.startswith("skill."):
                                # skill 调用跳过所有 contract/action 约束检查
                                pass
                            elif not self._planner_decision_within_allowed_actions(
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
                                if raw_decision.selected_tool and raw_decision.selected_tool.startswith("skill."):
                                    # skill calls skip critic — they're self-contained and pre-validated
                                    review = DecisionReview(
                                        approved=True, issues=[], summary="Skill call — auto-approved.", suggested_decision=None,
                                    )
                                    decision_source = "skill_auto_approved"
                                elif self._should_auto_approve_low_risk_tool_decision(raw_decision):
                                    review = DecisionReview(
                                        approved=True,
                                        issues=[],
                                        summary="Auto-approved a low-risk read-only tool call; validator and guardrails still run before execution.",
                                        suggested_decision=None,
                                    )
                                    decision_source = "planner_low_risk_auto_approved"
                                elif self._should_auto_approve_write_followup(raw_decision, tool_results) or self._should_auto_approve_candidate_followup(raw_decision, tool_results):
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
                                        tool_manifests=self._scoped_manifests(intent_bundle.task_envelope),
                                        observations=planner_observations,
                                    )
                                    decision_source = "planner_reviewed"
                            effective_planner_decision = self._resolve_reviewed_decision(raw_decision, review)
                            planner_repeated_previous_request = (
                                state_machine_repair is not None
                                and effective_planner_decision.decision == DecisionType.TOOL_CALL
                                and self.loop_controller.request_signature(effective_planner_decision) in request_signatures
                            )
                            if raw_decision.selected_tool and raw_decision.selected_tool.startswith("skill."):
                                pass  # skill 跳过 state machine repair
                            elif planner_repeated_previous_request or self._planner_response_should_use_state_machine_repair(
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
                    decision, argument_repair_trace = self._repair_decision_arguments_from_task_contract(
                        decision,
                        task_envelope=intent_bundle.task_envelope,
                    )
                    if argument_repair_trace is not None:
                        self.trace_store.append(
                            "decision_argument_repair",
                            {
                                "trace_id": trace_id,
                                "step": step,
                                "source": decision_source,
                                "details": argument_repair_trace,
                                "decision": decision.model_dump(mode="json"),
                            },
                        )
                    decision, document_write_reroute_trace = self._route_unsafe_document_write_decision(
                        decision,
                        task_envelope=intent_bundle.task_envelope,
                    )
                    if document_write_reroute_trace is not None:
                        self.trace_store.append(
                            "decision_tool_reroute",
                            {
                                "trace_id": trace_id,
                                "step": step,
                                "source": decision_source,
                                "details": document_write_reroute_trace,
                                "decision": decision.model_dump(mode="json"),
                            },
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
                    decision, post_argument_constraint_trace = self._enforce_upstream_constraints_on_decision(
                        decision,
                        task_envelope=intent_bundle.task_envelope,
                        task_graph=intent_bundle.task_graph,
                        overall_task_goal=active_overall_goal,
                        completed_outputs=completed_outputs,
                        state_machine_repair=state_machine_repair,
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
                        elif (
                            isinstance(exc, DecisionValidationError)
                            and decision is not None
                            and decision.decision == DecisionType.TOOL_CALL
                            and decision.selected_tool
                        ):
                            repaired_decision = self._try_repair_missing_arguments(
                                decision=decision,
                                error=str(exc),
                                user_text=user_text,
                                recent_context=recent_context,
                                trace_id=trace_id,
                                step=step,
                                tool_results=tool_results,
                            )
                            if repaired_decision is not None:
                                decision = repaired_decision
                                decision, post_repair_constraint_trace = self._enforce_upstream_constraints_on_decision(
                                    decision,
                                    task_envelope=intent_bundle.task_envelope,
                                    task_graph=intent_bundle.task_graph,
                                    overall_task_goal=active_overall_goal,
                                    completed_outputs=completed_outputs,
                                    state_machine_repair=state_machine_repair,
                                )
                                self.trace_store.append(
                                    "decision_argument_repaired_by_llm",
                                    {
                                        "trace_id": trace_id,
                                        "step": step,
                                        "original_error": str(exc),
                                        "decision": decision.model_dump(mode="json"),
                                    },
                                )
                            else:
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
                if main_goal_locked:
                    if active_overall_goal is None:
                        active_overall_goal = TaskGoal(
                            summary=str(getattr(intent_bundle.task_envelope, "primary_objective", "") or "").strip()
                            or "Complete the active task.",
                            required_outputs=list(authoritative_required_outputs),
                            completion_mode="outputs",
                        )
                elif decision.overall_task_goal is not None and decision.overall_task_goal.required_outputs:
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
                    execution_context=build_tool_execution_context(
                        decision=decision,
                        user_text=user_text,
                        task_envelope=intent_bundle.task_envelope,
                        overall_task_goal=active_overall_goal,
                        recent_context=recent_context,
                        workflow_family=bound_workflow_family,
                    ),
                )
                tool_use_context = ToolUseContext.from_execution_context(
                    trace_id=trace_id,
                    session_id=session_id,
                    workspace_root=self.config.workspace_root,
                    execution_context=request.execution_context,
                    channel=runtime_channel,
                    access_policy=(
                        runtime_channel_context.get("access_policy")
                        if isinstance(runtime_channel_context, dict)
                        and isinstance(runtime_channel_context.get("access_policy"), dict)
                        else None
                    ),
                    runtime_settings=runtime_channel_context if isinstance(runtime_channel_context, dict) else None,
                    completed_outputs=completed_outputs,
                    metadata={
                        "step": step,
                        "decision_source": decision_source,
                        "risk_level": decision.risk_level.value,
                        "selected_tool": decision.selected_tool,
                    },
                )
                request_signatures.append(self.loop_controller.request_signature(decision))
                self.trace_store.append(
                    "tool_request",
                    {
                        "trace_id": trace_id,
                        "step": step,
                        "request": request.model_dump(mode="json"),
                        "tool_use_context": tool_use_context.model_dump(mode="json"),
                    },
                )
                self._emit_progress(
                    progress_callback,
                    "tool_start",
                    f"我现在开始调用 {request.tool_name}。",
                    {"step": step, "tool_name": request.tool_name, "arguments": request.arguments},
                )

                result = self.registry.execute(request, context=tool_use_context)
                last_executed_decision = decision
                tool_results.append(result)
                artifacts.tool_results.append(result)
                if result.status == "success":
                    for output_name in result.produced_outputs:
                        if output_name not in completed_outputs:
                            completed_outputs.append(output_name)
                    active_candidate_state = self._derive_candidate_state(decision, result, active_candidate_state)
                    if self._has_reliable_candidates(active_candidate_state, active_overall_goal) and OutputKind.OBJECT_CANDIDATES not in completed_outputs:
                        completed_outputs.append(OutputKind.OBJECT_CANDIDATES)
                artifacts.completed_outputs = list(completed_outputs)
                artifacts.candidate_state = active_candidate_state

                # skill 执行完后立刻判断是否所有 required_outputs 已满足，是则跳过下一轮 decide
                is_skill = decision.selected_tool and decision.selected_tool.startswith("skill.")
                if is_skill and active_overall_goal is not None and active_overall_goal.required_outputs:
                    missing = [o for o in active_overall_goal.required_outputs if o not in completed_outputs]
                    if not missing:
                        self.trace_store.append(
                            "loop_stop",
                            {"trace_id": trace_id, "step": step, "reason": "skill_completed_all_outputs"},
                        )
                        loop_stop_reason = "skill_completed_all_outputs"
                        break

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
                    task_classification=None if authoritative_contract_workflow else intent_bundle.task_classification,
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
                    if authoritative_contract_workflow or intent_bundle.task_classification is None
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
                self.trace_store.append(
                    "state_machine_progress",
                    {
                        "trace_id": trace_id,
                        "step": step,
                        "phase": "after_tool_result",
                        "executed_tool": result.tool_name,
                        "tool_status": result.status,
                        "produced_outputs": [
                            output.value if isinstance(output, OutputKind) else str(output)
                            for output in result.produced_outputs
                        ],
                        "workflow": self._build_contract_workflow_debug_payload(
                            workflow_spec=getattr(intent_bundle.task_envelope, "workflow_spec", None),
                            completed_outputs=completed_outputs,
                            candidate_state=active_candidate_state,
                            observed_workflow_state=observed_workflow_state,
                            active_overall_goal=active_overall_goal,
                        ),
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
            # 不走模板化回复。让上层的自诊断 + 正常回复逻辑处理
            error_text = ""
            artifacts.final_response = error_text
            self.trace_store.append("error", {"trace_id": trace_id, "error": str(exc)})
            self._emit_progress(
                progress_callback,
                "error",
                f"中途出了点问题：{exc}",
                {"trace_id": trace_id, "error": str(exc)},
            )
            self._append_llm_call_metrics_trace(trace_id, llm_metrics_start)
            self._write_turn_trace_audit(trace_id)
            return artifacts

        final_observations = [
            obs[:300] for obs in ContextBuilder.build_observations(tool_results)[:8]
        ]
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
            task_classification=None if authoritative_contract_workflow else intent_bundle.task_classification,
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
        # 自诊断：当任务异常终止时，尝试分析 trace 给出根因和修复建议
        if loop_stop_reason and loop_stop_reason not in ("waiting_for_selection", "completion_judge"):
            self._append_self_diagnosis(
                execution_summary=execution_summary,
                loop_stop_reason=loop_stop_reason,
                trace_id=trace_id,
                user_text=user_text,
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
                    structured_bundle = self._build_structured_document_inspect_response_bundle(execution_summary)
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
                final_response = f"这一步出了点问题：{exc}。看看 trace 里的 self_diagnosis 能定位到根因。"
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
        self._append_llm_call_metrics_trace(trace_id, llm_metrics_start)
        self._write_turn_trace_audit(trace_id)
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

    def _append_llm_call_metrics_trace(self, trace_id: str, metrics_start: int) -> None:
        metrics = list(getattr(self.llm_client, "last_call_metrics", []) or [])
        if metrics_start < 0:
            metrics_start = 0
        turn_metrics = metrics[metrics_start:]
        if not turn_metrics:
            return
        total_tokens = 0
        total_elapsed_ms = 0.0
        compact_calls: list[dict[str, Any]] = []
        for item in turn_metrics:
            if not isinstance(item, dict):
                continue
            usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
            try:
                total_tokens += int(usage.get("total_tokens") or 0)
            except (TypeError, ValueError):
                pass
            try:
                total_elapsed_ms += float(item.get("elapsed_ms") or 0.0)
            except (TypeError, ValueError):
                pass
            compact_calls.append(
                {
                    "provider": item.get("provider"),
                    "model": item.get("model"),
                    "elapsed_ms": item.get("elapsed_ms"),
                    "usage": usage,
                }
            )
        if not compact_calls:
            return
        self.trace_store.append(
            "llm_call_metrics",
            {
                "trace_id": trace_id,
                "call_count": len(compact_calls),
                "total_tokens": total_tokens,
                "total_elapsed_ms": round(total_elapsed_ms, 2),
                "calls": compact_calls,
            },
        )

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
        tool_evidence: list[dict] = []

        for result in tool_results:
            if result.status == "success":
                successful_actions.append(
                    {
                        "tool_name": result.tool_name,
                        "data": result.data,
                        "produced_outputs": [item.value for item in result.produced_outputs],
                        "evidence": list(result.evidence),
                    }
                )
                for item in result.evidence:
                    if isinstance(item, dict) and item not in tool_evidence:
                        tool_evidence.append(item)
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
            "tool_evidence": tool_evidence,
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
            response_hint = (
                str(response_hint or "").strip()
                + "\nAlways reply in natural Chinese. Do not switch to English for partial success, errors, or tool summaries."
            ).strip()
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
            "Reply in natural Chinese, even if tool observations or errors are in English.",
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
    def _build_structured_document_inspect_response_bundle(cls, execution_summary: dict) -> dict[str, str] | None:
        successful_actions = execution_summary.get("successful_actions") or []
        if not isinstance(successful_actions, list):
            return None

        for action in reversed(successful_actions):
            if not isinstance(action, dict):
                continue
            tool_name = str(action.get("tool_name", "") or "").strip()
            if tool_name != "document_agent.inspect":
                continue
            data = action.get("data") if isinstance(action.get("data"), dict) else {}
            answer = " ".join(str(data.get("answer", "") or "").split())
            summary = " ".join(str(data.get("summary", "") or "").split())
            speech = " ".join(str(data.get("speech_text", "") or "").split())
            path = str(data.get("path", "") or "").strip()
            if not answer and not summary and not speech:
                continue

            lines: list[str] = []
            if answer:
                lines.append(answer)
            elif summary:
                lines.append(summary)

            if summary and summary != answer:
                lines.append(f"依据：{summary}")

            findings = data.get("findings") if isinstance(data.get("findings"), list) else []
            evidence_items = data.get("evidence") if isinstance(data.get("evidence"), list) else []
            detail_lines: list[str] = []
            for item in findings[:2]:
                if not isinstance(item, dict):
                    continue
                claim = " ".join(str(item.get("claim", "") or "").split())
                evidence = " ".join(str(item.get("evidence", "") or "").split())
                if claim and evidence:
                    detail_lines.append(f"{claim}：{evidence}")
                elif claim:
                    detail_lines.append(claim)
            if not detail_lines:
                for item in evidence_items[:2]:
                    if not isinstance(item, dict):
                        continue
                    excerpt = " ".join(str(item.get("excerpt", "") or "").split())
                    reason = " ".join(str(item.get("reason", "") or "").split())
                    if excerpt and reason:
                        detail_lines.append(f"{reason}：{excerpt}")
                    elif excerpt:
                        detail_lines.append(excerpt)

            if detail_lines:
                lines.append("证据：" + "；".join(detail_lines))
            if path:
                lines.append(f"文件：{path}")

            display = "\n".join(line for line in lines if line).strip()
            speech_text = speech or answer or summary or display
            return {"display_text": display, "speech_text": speech_text}

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
            text = str(task_payload.get("fire_text") or task.get("message") or task_payload.get("text") or "").strip()
            if not text:
                return None
            scheduled_runtime_context = dict(channel_runtime or {})
            scheduled_runtime_context["scheduled_task"] = {
                "phase": "fired",
                "task_type": "notify",
                "reminder_id": str(task.get("reminder_id") or "").strip() or None,
                "message": text,
            }
            trace_id = f"trace_{uuid.uuid4().hex[:12]}"
            access_policy = (
                scheduled_runtime_context.get("access_policy")
                if isinstance(scheduled_runtime_context.get("access_policy"), dict)
                else None
            )
            request = ToolRegistry.build_request(
                trace_id=trace_id,
                session_id=session_id or "",
                tool_name="qq.send_text",
                arguments={"message": text, "target_kind": "current"},
                execution_context={
                    "execution_brief": "Fire cached reminder notification directly without re-planning.",
                    "required_outputs": [OutputKind.MESSAGE_SENT.value],
                    "grounded_inputs": {
                        "scheduled_reminder_id": str(task.get("reminder_id") or "").strip(),
                        "cached_fire_text": text,
                        "original_message": str(task.get("message") or "").strip(),
                    },
                },
            )
            tool_use_context = ToolUseContext.from_execution_context(
                trace_id=trace_id,
                session_id=session_id or "",
                workspace_root=self.config.workspace_root,
                execution_context=request.execution_context,
                channel=channel,
                access_policy=access_policy,
                runtime_settings=scheduled_runtime_context,
                metadata={
                    "scheduled_task": True,
                    "reminder_id": str(task.get("reminder_id") or "").strip() or None,
                    "dispatch_mode": "cached_direct_fire",
                },
            )
            if getattr(self, "trace_store", None) is not None:
                self.trace_store.append(
                    "scheduled_task_cached_fire",
                    {
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "reminder_id": str(task.get("reminder_id") or "").strip() or None,
                        "message": text,
                    },
                )
            result = self.registry.execute(request, context=tool_use_context)
            if getattr(self, "trace_store", None) is not None:
                self.trace_store.append(
                    "scheduled_task_cached_fire_result",
                    {
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "reminder_id": str(task.get("reminder_id") or "").strip() or None,
                        "result": result.model_dump(mode="json"),
                    },
                )
            return TurnArtifacts(
                tool_results=[result],
                final_response=text,
                speech_text=text,
                completed_outputs=[OutputKind.MESSAGE_SENT] if result.status == "success" else [],
                trace_id=trace_id,
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
