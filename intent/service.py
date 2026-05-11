from __future__ import annotations

from local_agent.intent.models import AnswerabilityAssessment, IntentBundle, TaskClassification, TaskEnvelope
from local_agent.kernel.request_intent_analyzer import RequestIntentAnalyzer
from local_agent.protocol.models import InstructionIntent, MemoryCandidateIntent, OutputKind, SiteSearchIntent, TaskGraphIntent


class IntentService:
    def __init__(self, analyzer: RequestIntentAnalyzer) -> None:
        self.analyzer = analyzer

    def analyze(
        self,
        user_text: str,
        *,
        recent_context: str = "",
        hot_context_summary: str = "",
        warm_memory_summary: str = "",
        learning_memory_summary: str = "",
        cold_memory_summary: str = "",
        active_task_summary: str = "",
        channel_context_summary: str = "",
    ) -> IntentBundle:
        layered_context_summary = self._build_layered_context_summary(
            recent_context=recent_context,
            hot_context_summary=hot_context_summary,
            warm_memory_summary=warm_memory_summary,
            learning_memory_summary=learning_memory_summary,
            cold_memory_summary=cold_memory_summary,
            active_task_summary=active_task_summary,
            channel_context_summary=channel_context_summary,
        )
        context_kwargs = {
            "recent_context": recent_context,
            "hot_context_summary": hot_context_summary,
            "warm_memory_summary": warm_memory_summary,
            "learning_memory_summary": learning_memory_summary,
            "cold_memory_summary": cold_memory_summary,
            "active_task_summary": active_task_summary,
            "channel_context_summary": channel_context_summary,
            "layered_context_summary": layered_context_summary,
        }

        document_delivery = self._invoke("analyze_document_delivery", user_text, **context_kwargs)
        knowledge_request = self._invoke("analyze_knowledge_request", user_text, **context_kwargs)
        site_search = self._invoke("analyze_site_search", user_text, **context_kwargs)
        memory_candidate_intent = self._invoke_optional(
            "analyze_memory_candidate_intent",
            MemoryCandidateIntent,
            user_text,
            **context_kwargs,
        )
        instruction_intent = self._derive_instruction_intent(memory_candidate_intent)
        task_graph = self._invoke_optional(
            "analyze_task_graph",
            TaskGraphIntent,
            user_text,
            **context_kwargs,
        )
        task_graph = self._normalize_task_graph(
            task_graph=task_graph,
            site_search=site_search,
        )

        if knowledge_request.knowledge_type in {"local_workspace", "qq_history", "casual_chat", "system_utility"}:
            site_search = SiteSearchIntent()

        task_classification = self._derive_task_classification(
            user_text=user_text,
            knowledge_request=knowledge_request,
            document_delivery=document_delivery,
            site_search=site_search,
            task_graph=task_graph,
        )
        answerability = self._derive_answerability(
            user_text=user_text,
            knowledge_request=knowledge_request,
            document_delivery=document_delivery,
            memory_candidate_intent=memory_candidate_intent,
            task_graph=task_graph,
            warm_memory_summary=warm_memory_summary,
            recent_context=recent_context,
        )
        task_envelope = self._build_task_envelope(
            user_text=user_text,
            recent_context=recent_context,
            hot_context_summary=hot_context_summary,
            warm_memory_summary=warm_memory_summary,
            learning_memory_summary=learning_memory_summary,
            cold_memory_summary=cold_memory_summary,
            active_task_summary=active_task_summary,
            channel_context_summary=channel_context_summary,
            knowledge_request=knowledge_request,
            document_delivery=document_delivery,
            instruction_intent=instruction_intent,
            task_graph=task_graph,
            task_classification=task_classification,
        )

        return IntentBundle(
            document_delivery=document_delivery,
            knowledge_request=knowledge_request,
            site_search=site_search,
            memory_candidate_intent=memory_candidate_intent,
            instruction_intent=instruction_intent,
            task_graph=task_graph,
            answerability=answerability,
            task_classification=task_classification,
            task_envelope=task_envelope,
        )

    @staticmethod
    def _normalize_task_graph(
        *,
        task_graph: TaskGraphIntent,
        site_search,
    ) -> TaskGraphIntent:
        subtasks = list(getattr(task_graph, "subtasks", []) or [])
        if not subtasks:
            return task_graph

        site_query = str(getattr(site_search, "query", "") or "").strip()
        any_waiting = False
        normalized_subtasks = []

        for subtask in subtasks:
            slot_values = dict(getattr(subtask, "slot_values", {}) or {})
            missing_slots = [
                str(slot).strip()
                for slot in getattr(subtask, "missing_slots", []) or []
                if str(slot).strip()
            ]
            status = str(getattr(subtask, "status", "") or "ready").strip().lower() or "ready"
            kind = str(getattr(subtask, "kind", "") or "").strip().lower()

            content_value = str(slot_values.get("content", "") or "").strip()
            if not content_value and kind == "web_lookup" and site_query:
                slot_values["content"] = site_query
                content_value = site_query

            if content_value and "content" in missing_slots:
                missing_slots = [slot for slot in missing_slots if slot != "content"]

            if status == "waiting_for_input" and not missing_slots:
                status = "ready"

            if status == "waiting_for_input":
                any_waiting = True

            normalized_subtasks.append(
                subtask.model_copy(
                    update={
                        "slot_values": slot_values,
                        "missing_slots": missing_slots,
                        "status": status,
                    }
                )
            )

        primary_task_id = str(getattr(task_graph, "primary_task_id", "") or "").strip()
        primary_subtask = None
        for subtask in normalized_subtasks:
            subtask_id = str(getattr(subtask, "task_id", "") or "").strip()
            if primary_task_id and subtask_id == primary_task_id:
                primary_subtask = subtask
                break
        if primary_subtask is None and normalized_subtasks:
            primary_subtask = normalized_subtasks[0]

        followup_text = str(getattr(task_graph, "followup_text", "") or "").strip() or None
        if primary_subtask is not None and str(getattr(primary_subtask, "status", "") or "").strip().lower() != "waiting_for_input":
            followup_text = None

        return task_graph.model_copy(
            update={
                "subtasks": normalized_subtasks,
                "needs_clarification": any_waiting,
                "followup_text": followup_text,
            }
        )

    def analyze_memory_candidate(
        self,
        user_text: str,
        *,
        recent_context: str = "",
        hot_context_summary: str = "",
        warm_memory_summary: str = "",
        learning_memory_summary: str = "",
        cold_memory_summary: str = "",
        active_task_summary: str = "",
        channel_context_summary: str = "",
    ) -> InstructionIntent:
        layered_context_summary = self._build_layered_context_summary(
            recent_context=recent_context,
            hot_context_summary=hot_context_summary,
            warm_memory_summary=warm_memory_summary,
            learning_memory_summary=learning_memory_summary,
            cold_memory_summary=cold_memory_summary,
            active_task_summary=active_task_summary,
            channel_context_summary=channel_context_summary,
        )
        return self._invoke_optional(
            "analyze_memory_candidate_intent",
            MemoryCandidateIntent,
            user_text,
            recent_context=recent_context,
            hot_context_summary=hot_context_summary,
            warm_memory_summary=warm_memory_summary,
            learning_memory_summary=learning_memory_summary,
            cold_memory_summary=cold_memory_summary,
            active_task_summary=active_task_summary,
            channel_context_summary=channel_context_summary,
            layered_context_summary=layered_context_summary,
        )

    def analyze_instruction(
        self,
        user_text: str,
        *,
        recent_context: str = "",
        hot_context_summary: str = "",
        warm_memory_summary: str = "",
        learning_memory_summary: str = "",
        cold_memory_summary: str = "",
        active_task_summary: str = "",
        channel_context_summary: str = "",
    ) -> InstructionIntent:
        candidate = self.analyze_memory_candidate(
            user_text,
            recent_context=recent_context,
            hot_context_summary=hot_context_summary,
            warm_memory_summary=warm_memory_summary,
            learning_memory_summary=learning_memory_summary,
            cold_memory_summary=cold_memory_summary,
            active_task_summary=active_task_summary,
            channel_context_summary=channel_context_summary,
        )
        return self._derive_instruction_intent(candidate)

    def _invoke(self, method_name: str, user_text: str, **context_kwargs):
        method = getattr(self.analyzer, method_name)
        try:
            return method(user_text, **context_kwargs)
        except TypeError:
            return method(user_text)

    def _invoke_optional(self, method_name: str, default_factory, user_text: str, **context_kwargs):
        method = getattr(self.analyzer, method_name, None)
        if method is None:
            return default_factory()
        try:
            return method(user_text, **context_kwargs)
        except TypeError:
            return method(user_text)
        except Exception:
            return default_factory()

    def _derive_task_classification(
        self,
        *,
        user_text: str,
        knowledge_request,
        document_delivery,
        site_search,
        task_graph,
    ) -> TaskClassification:
        graph_classification = self._classification_from_task_graph(
            task_graph=task_graph,
            knowledge_type=str(getattr(knowledge_request, "knowledge_type", "") or ""),
        )
        if graph_classification is not None:
            return graph_classification

        classification = self._classification_from_structured_intents(
            knowledge_request=knowledge_request,
            document_delivery=document_delivery,
            site_search=site_search,
        )
        if not str(classification.rationale or "").strip():
            classification.rationale = "short_circuit_from_existing_intents"
        else:
            classification.rationale = f"short_circuit:{classification.rationale}"
        return classification

    @staticmethod
    def _classification_from_structured_intents(
        *,
        knowledge_request,
        document_delivery,
        site_search,
    ) -> TaskClassification:
        knowledge_type = str(getattr(knowledge_request, "knowledge_type", "") or "").strip().lower()
        if knowledge_type == "qq_history":
            return TaskClassification(
                domain="qq_history",
                task_kind="history_lookup",
                preferred_families=["qq_history"],
                confidence=0.68,
                rationale="structured_intent_projection:qq_history",
            )
        if knowledge_type == "system_utility":
            return TaskClassification(
                domain="system_utility",
                task_kind="system_utility",
                preferred_families=["system_utility"],
                confidence=0.56,
                rationale="structured_intent_projection:system_utility",
            )
        if getattr(site_search, "site", None) or knowledge_type in {"general_external_topic", "time_sensitive_external_topic"}:
            task_kind = "research_document" if bool(getattr(document_delivery, "save_output", False)) else "research"
            return TaskClassification(
                domain="web",
                task_kind=task_kind,
                preferred_families=["web_lookup", "web_target"],
                confidence=0.64,
                rationale="structured_intent_projection:web",
            )
        if knowledge_type == "local_workspace":
            if bool(getattr(document_delivery, "save_output", False)):
                return TaskClassification(
                    domain="local_workspace",
                    task_kind="document_edit",
                    preferred_families=["document_operation", "local_lookup", "file_lookup"],
                    confidence=0.62,
                    rationale="structured_intent_projection:local_saved_document",
                )
            return TaskClassification(
                domain="local_workspace",
                task_kind="lookup",
                preferred_families=["local_lookup", "file_lookup"],
                confidence=0.52,
                rationale="structured_intent_projection:local_lookup",
            )
        return TaskClassification(
            domain="unknown",
            task_kind="unknown",
            preferred_families=[],
            confidence=0.0,
            rationale="structured_intent_projection:unknown",
        )

    @staticmethod
    def _classification_from_task_graph(*, task_graph, knowledge_type: str) -> TaskClassification | None:
        primary_subtask = IntentService._primary_task_graph_subtask(task_graph)
        if primary_subtask is None:
            return None
        confidence = float(getattr(task_graph, "confidence", 0.0) or 0.0)
        if confidence < 0.7:
            return None

        kind = str(getattr(primary_subtask, "kind", "") or "").strip().lower()
        if not kind:
            return None
        knowledge = str(knowledge_type or "").strip().lower()

        if kind in {"document_edit", "edit", "rewrite", "transform"}:
            return TaskClassification(
                domain="local_workspace",
                task_kind="document_edit",
                preferred_families=["document_operation", "local_lookup", "file_lookup"],
                confidence=max(confidence, 0.82),
                rationale="task_graph_primary_subtask:document_edit",
            )
        if kind in {"document_summary", "summarize"}:
            return TaskClassification(
                domain="local_workspace",
                task_kind="document_summary",
                preferred_families=["document_summary", "local_lookup", "file_lookup"],
                confidence=max(confidence, 0.78),
                rationale="task_graph_primary_subtask:document_summary",
            )
        if kind in {"qq_history", "history_lookup"}:
            return TaskClassification(
                domain="qq_history",
                task_kind="history_lookup",
                preferred_families=["qq_history"],
                confidence=max(confidence, 0.78),
                rationale="task_graph_primary_subtask:qq_history",
            )
        if kind in {"web_lookup", "web_research", "research"}:
            return TaskClassification(
                domain="web",
                task_kind="research",
                preferred_families=["web_lookup", "web_target"],
                confidence=max(confidence, 0.78),
                rationale="task_graph_primary_subtask:web_lookup",
            )
        if kind in {"system_utility", "reminder", "time_lookup"}:
            return TaskClassification(
                domain="system_utility",
                task_kind="create_reminder" if kind == "reminder" else "system_utility",
                preferred_families=["system_utility"],
                confidence=max(confidence, 0.74),
                rationale="task_graph_primary_subtask:system_utility",
            )
        if knowledge == "local_workspace" and kind in {"local_lookup", "file_lookup", "lookup"}:
            return TaskClassification(
                domain="local_workspace",
                task_kind="lookup",
                preferred_families=["local_lookup", "file_lookup"],
                confidence=max(confidence, 0.72),
                rationale="task_graph_primary_subtask:local_lookup",
            )
        return None

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
    def _derive_instruction_intent(memory_candidate: MemoryCandidateIntent) -> InstructionIntent:
        kind = str(getattr(memory_candidate, "kind", "") or "").strip().lower()
        instruction_kinds = {
            "naming",
            "preference",
            "workflow_method",
            "tool_policy",
            "correction",
            "style",
            "boundary",
        }
        if not bool(getattr(memory_candidate, "is_memory_candidate", False)) or kind not in instruction_kinds:
            return InstructionIntent()
        return InstructionIntent(
            is_instruction=True,
            scope=str(getattr(memory_candidate, "scope", "none") or "none"),
            kind=kind,
            apply_this_turn=bool(getattr(memory_candidate, "apply_this_turn", False)),
            persist_memory=bool(getattr(memory_candidate, "persist_memory", False)),
            normalized_instruction=str(getattr(memory_candidate, "normalized_text", None) or "").strip() or None,
            memory_text=str(getattr(memory_candidate, "memory_text", None) or "").strip() or None,
            preferred_families=list(getattr(memory_candidate, "preferred_families", []) or []),
            blocked_families=list(getattr(memory_candidate, "blocked_families", []) or []),
            preferred_tools=list(getattr(memory_candidate, "preferred_tools", []) or []),
            response_style=str(getattr(memory_candidate, "response_style", None) or "").strip() or None,
            confidence=float(getattr(memory_candidate, "confidence", 0.0) or 0.0),
            rationale=str(getattr(memory_candidate, "rationale", "") or ""),
        )

    @staticmethod
    def _derive_answerability(
        *,
        user_text: str,
        knowledge_request,
        document_delivery,
        memory_candidate_intent,
        task_graph,
        warm_memory_summary: str,
        recent_context: str,
    ) -> AnswerabilityAssessment:
        knowledge_type = str(getattr(knowledge_request, "knowledge_type", "") or "").strip().lower()
        needs_grounding = bool(getattr(knowledge_request, "needs_grounding", False))
        wants_document = bool(getattr(document_delivery, "wants_document", False))
        save_output = bool(getattr(document_delivery, "save_output", False))
        subtask_count = len(getattr(task_graph, "subtasks", []) or [])
        is_memory_candidate = bool(getattr(memory_candidate_intent, "is_memory_candidate", False))
        memory_kind = str(getattr(memory_candidate_intent, "kind", "") or "").strip().lower()
        lowered = str(user_text or "").lower()

        if wants_document or save_output:
            return AnswerabilityAssessment(
                answerability="verification_required",
                preferred_family="document_operation",
                local_answer_kind="none",
                answer_basis=["document_request"],
                confidence=0.91,
                rationale="document_output_requested",
            )

        if knowledge_type in {"time_sensitive_external_topic", "general_external_topic"} and needs_grounding:
            return AnswerabilityAssessment(
                answerability="verification_required",
                preferred_family="web_lookup",
                local_answer_kind="none",
                answer_basis=["external_grounding"],
                confidence=0.9,
                rationale="external_verification_needed",
            )

        if knowledge_type in {"local_workspace", "qq_history", "system_utility"}:
            preferred_family = {
                "local_workspace": "local_lookup",
                "qq_history": "qq_history",
                "system_utility": "system_utility",
            }.get(knowledge_type, "local_lookup")
            return AnswerabilityAssessment(
                answerability="local_tool_needed",
                preferred_family=preferred_family,
                local_answer_kind="none",
                answer_basis=[knowledge_type],
                confidence=0.86,
                rationale="local_capability_sufficient",
            )

        if is_memory_candidate and memory_kind in {"user_fact", "naming", "preference", "correction"}:
            return AnswerabilityAssessment(
                answerability="memory_or_local_answerable",
                preferred_family="chat",
                local_answer_kind="memory_fact" if memory_kind == "user_fact" else "direct_chat",
                answer_basis=["memory_candidate"],
                confidence=0.84,
                rationale="memory_candidate_can_be_handled_locally",
            )

        if knowledge_type == "casual_chat" and subtask_count <= 1 and not needs_grounding:
            basis: list[str] = ["casual_chat"]
            if str(warm_memory_summary or "").strip():
                basis.append("warm_memory")
            if str(recent_context or "").strip():
                basis.append("recent_context")
            local_answer_kind = "date_time" if any(token in lowered for token in ("今天", "明天", "后天", "星期", "周几", "几点", "日期")) else "direct_chat"
            return AnswerabilityAssessment(
                answerability="memory_or_local_answerable",
                preferred_family="chat",
                local_answer_kind=local_answer_kind,
                answer_basis=basis,
                confidence=0.73,
                rationale="casual_chat_answerable_without_verification",
            )

        return AnswerabilityAssessment(
            answerability="verification_required" if needs_grounding else "local_tool_needed",
            preferred_family="web_lookup" if needs_grounding else "chat",
            local_answer_kind="none",
            answer_basis=["fallback"],
            confidence=0.52,
            rationale="default_answerability_fallback",
        )

    @staticmethod
    def _build_task_envelope(
        *,
        user_text: str,
        recent_context: str,
        hot_context_summary: str,
        warm_memory_summary: str,
        learning_memory_summary: str,
        cold_memory_summary: str,
        active_task_summary: str,
        channel_context_summary: str,
        knowledge_request,
        document_delivery,
        instruction_intent,
        task_graph,
        task_classification,
    ) -> TaskEnvelope:
        knowledge_type = str(getattr(knowledge_request, "knowledge_type", "") or "").strip().lower()
        needs_grounding = bool(getattr(knowledge_request, "needs_grounding", False))
        wants_document = bool(getattr(document_delivery, "wants_document", False))
        save_output = bool(getattr(document_delivery, "save_output", False))
        task_kind = "" if task_classification is None else str(getattr(task_classification, "task_kind", "") or "").strip().lower()

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

        conversation_mode = "continuation" if str(recent_context or "").strip() else "new_request"
        primary_objective = str(user_text or "").strip()
        blocked: list[str] = []
        allowed: list[str] = []
        required_outputs: list[OutputKind] = []
        mode = "generic"
        style_intent: str | None = None
        rationale = ""
        preferred_tools: list[str] = []
        instruction_scope: str | None = None
        instruction_kind: str | None = None
        instruction_summary: str | None = None
        planning_focus_text = str(getattr(task_graph, "primary_task_text", "") or "").strip() or None
        primary_subtask = None
        primary_task_id = str(getattr(task_graph, "primary_task_id", "") or "").strip()
        for subtask in getattr(task_graph, "subtasks", []) or []:
            subtask_id = str(getattr(subtask, "task_id", "") or "").strip()
            if primary_task_id and subtask_id == primary_task_id:
                primary_subtask = subtask
                break
        if primary_subtask is None:
            subtasks = list(getattr(task_graph, "subtasks", []) or [])
            primary_subtask = subtasks[0] if subtasks else None
        primary_waiting_for_input = (
            primary_subtask is not None
            and str(getattr(primary_subtask, "status", "") or "").strip().lower() == "waiting_for_input"
        )
        primary_missing_slots = [
            str(slot).strip()
            for slot in getattr(primary_subtask, "missing_slots", []) or []
            if str(slot).strip()
        ]
        subtask_count = len(getattr(task_graph, "subtasks", []) or [])
        execution_notes: list[str] = []
        known_failure_avoidance = IntentService._extract_memory_guidance(learning_memory_summary, max_items=3)
        tool_order_constraints: list[str] = []
        response_strategy: str | None = None

        if save_output:
            required_outputs.append(OutputKind.FILE_WRITTEN)
        if task_kind in {"document_edit", "edit", "rewrite", "transform"} and OutputKind.FILE_WRITTEN not in required_outputs:
            required_outputs.append(OutputKind.FILE_WRITTEN)

        if knowledge_type == "local_workspace":
            mode = "local_task"
            blocked = ["web_lookup", "web_target", "qq_history", "system_utility"]
            rationale = "local_workspace_boundaries"
        elif knowledge_type == "qq_history":
            mode = "qq_history_task"
            allowed = ["qq_history"]
            blocked = [
                "local_collection",
                "local_lookup",
                "file_lookup",
                "file_delivery",
                "document_summary",
                "document_operation",
                "web_lookup",
                "web_target",
                "system_utility",
            ]
            rationale = "qq_history_boundaries"
        elif knowledge_type == "system_utility":
            mode = "system_utility_task"
            allowed = ["system_utility"]
            blocked = [
                "local_collection",
                "local_lookup",
                "file_lookup",
                "file_delivery",
                "document_summary",
                "document_operation",
                "web_lookup",
                "web_target",
                "qq_history",
            ]
            rationale = "system_utility_boundaries"
        elif knowledge_type == "casual_chat" and not needs_grounding:
            mode = "creative_write" if (wants_document or save_output) else "chat"
            style_intent = "document_reply" if wants_document else "conversational_reply"
            blocked = [
                "local_collection",
                "local_lookup",
                "file_lookup",
                "file_delivery",
                "document_summary",
                "document_operation",
                "web_lookup",
                "web_target",
                "qq_history",
                "system_utility",
                "delivery",
            ]
            rationale = "casual_chat_should_stay_in_main_agent"
        elif needs_grounding:
            mode = "grounded_lookup"
            blocked = ["qq_history", "system_utility"]
            for output_kind in (OutputKind.SEARCH_RESULTS, OutputKind.WEB_CONTENT):
                if output_kind not in required_outputs:
                    required_outputs.append(output_kind)
            rationale = "grounded_lookup_prefers_research_or_official_sources"
        elif task_kind in {"delivery", "proxy_send"}:
            mode = "delivery"
            allowed = ["delivery", "file_delivery"]
            rationale = "delivery_task_boundaries"

        if planning_focus_text:
            execution_notes.append(f"当前主焦点：{planning_focus_text}")
        if subtask_count > 1:
            execution_notes.append("这是一个已拆解的多步骤任务；优先推进当前主子任务，不要擅自改题。")
        if primary_waiting_for_input and primary_missing_slots:
            execution_notes.append(
                f"当前主子任务仍缺少必要信息（{', '.join(primary_missing_slots)}）；先结合上下文补全，补不出来再澄清，不要仅凭方法指令直接执行。"
            )
        if mode == "qq_history_task":
            tool_order_constraints.append("先查 QQ 历史，再组织结论；先给结论，再补充证据。")
            response_strategy = "qq_history_conclusion_then_evidence"
        elif mode == "system_utility_task":
            tool_order_constraints.append("优先使用本地系统能力完成，不要先跳去联网搜索。")
        elif mode == "grounded_lookup":
            tool_order_constraints.append("先做轻量检索与核验，只有必要时再做更深的网页研究。")
        elif mode == "local_task":
            tool_order_constraints.append("优先在本地工作区或明确路径范围内完成，不要把本地任务改成外部搜索。")
        if save_output:
            known_failure_avoidance.append("在确认 FILE_WRITTEN 之前，不要声称文件已经写入成功。")
        if wants_document:
            response_strategy = response_strategy or "document_status_before_completion_claim"

        instruction_enabled = bool(getattr(instruction_intent, "is_instruction", False))
        if instruction_enabled:
            instruction_scope = str(getattr(instruction_intent, "scope", "") or "").strip().lower() or None
            instruction_kind = str(getattr(instruction_intent, "kind", "") or "").strip().lower() or None
            instruction_summary = str(
                getattr(instruction_intent, "normalized_instruction", None)
                or getattr(instruction_intent, "memory_text", None)
                or ""
            ).strip() or None
            if getattr(instruction_intent, "response_style", None):
                style_intent = str(getattr(instruction_intent, "response_style") or "").strip() or style_intent
                response_strategy = response_strategy or style_intent
            if bool(getattr(instruction_intent, "apply_this_turn", False)):
                preferred_families = [
                    str(item).strip()
                    for item in getattr(instruction_intent, "preferred_families", []) or []
                    if str(item).strip()
                ]
                blocked_families = [
                    str(item).strip()
                    for item in getattr(instruction_intent, "blocked_families", []) or []
                    if str(item).strip()
                ]
                preferred_tools = [
                    str(item).strip()
                    for item in getattr(instruction_intent, "preferred_tools", []) or []
                    if str(item).strip()
                ]
                if preferred_families and not primary_waiting_for_input:
                    allowed = preferred_families
                    blocked = [family for family in blocked if family not in set(preferred_families)]
                for family in blocked_families:
                    if family not in blocked:
                        blocked.append(family)
                if primary_waiting_for_input and preferred_families:
                    execution_notes.append(
                        "已识别到本轮的方法偏好，但主任务对象还没完全补全；先由主 Agent 解决目标内容，再把执行方式下发给子模块。"
                    )
                if preferred_families == ["system_utility"] and not primary_waiting_for_input:
                    mode = "system_utility_task"
                elif preferred_families == ["qq_history"] and not primary_waiting_for_input:
                    mode = "qq_history_task"
                elif preferred_families == ["delivery"] and not primary_waiting_for_input:
                    mode = "delivery"
                rationale = (
                    f"{rationale}|instruction_override".strip("|")
                    if rationale
                    else "instruction_override"
                )

        delegated_execution_brief = IntentService._build_delegated_execution_brief(
            mode=mode,
            primary_objective=primary_objective,
            planning_focus_text=planning_focus_text,
            allowed_families=allowed,
            blocked_families=blocked,
            preferred_tools=preferred_tools,
            execution_notes=execution_notes,
            known_failure_avoidance=known_failure_avoidance,
            tool_order_constraints=tool_order_constraints,
            response_strategy=response_strategy,
        )

        return TaskEnvelope(
            mode=mode,
            conversation_mode=conversation_mode,
            primary_objective=primary_objective,
            needs_grounding=needs_grounding,
            context_layers_used=context_layers_used,
            allowed_families=allowed,
            blocked_families=blocked,
            required_outputs=required_outputs,
            style_intent=style_intent,
            instruction_scope=instruction_scope,
            instruction_kind=instruction_kind,
            instruction_summary=instruction_summary,
            preferred_tools=preferred_tools,
            planning_focus_text=planning_focus_text,
            execution_notes=execution_notes,
            known_failure_avoidance=known_failure_avoidance,
            tool_order_constraints=tool_order_constraints,
            response_strategy=response_strategy,
            delegated_execution_brief=delegated_execution_brief,
            subtask_count=subtask_count,
            rationale=rationale,
        )

    @staticmethod
    def _extract_memory_guidance(summary: str, *, max_items: int = 3) -> list[str]:
        lines: list[str] = []
        for raw_line in str(summary or "").splitlines():
            line = raw_line.strip()
            if not line or line.endswith(":"):
                continue
            if ". [" in line:
                line = line.split("] ", 1)[-1].strip()
            line = line.lstrip("- ").strip()
            if line and line not in lines:
                lines.append(line)
            if len(lines) >= max_items:
                break
        return lines

    @staticmethod
    def _build_delegated_execution_brief(
        *,
        mode: str,
        primary_objective: str,
        planning_focus_text: str | None,
        allowed_families: list[str],
        blocked_families: list[str],
        preferred_tools: list[str],
        execution_notes: list[str],
        known_failure_avoidance: list[str],
        tool_order_constraints: list[str],
        response_strategy: str | None,
    ) -> str:
        parts = [f"mode={mode}", f"objective={primary_objective}"]
        if planning_focus_text:
            parts.append(f"focus={planning_focus_text}")
        if allowed_families:
            parts.append(f"allowed_families={', '.join(allowed_families)}")
        if blocked_families:
            parts.append(f"blocked_families={', '.join(blocked_families)}")
        if preferred_tools:
            parts.append(f"preferred_tools={', '.join(preferred_tools)}")
        if tool_order_constraints:
            parts.append("tool_order_constraints=" + " | ".join(tool_order_constraints))
        if execution_notes:
            parts.append("execution_notes=" + " | ".join(execution_notes))
        if known_failure_avoidance:
            parts.append("avoid=" + " | ".join(known_failure_avoidance))
        if response_strategy:
            parts.append(f"response_strategy={response_strategy}")
        return "\n".join(parts)

    @staticmethod
    def _build_layered_context_summary(
        *,
        recent_context: str = "",
        hot_context_summary: str = "",
        warm_memory_summary: str = "",
        learning_memory_summary: str = "",
        cold_memory_summary: str = "",
        active_task_summary: str = "",
        channel_context_summary: str = "",
    ) -> str:
        sections: list[tuple[str, str, int]] = [
            ("Recent Context", recent_context, 900),
            ("Active Task", active_task_summary, 500),
            ("Channel Context", channel_context_summary, 350),
            ("Hot Context", hot_context_summary, 500),
            ("Warm Memory", warm_memory_summary, 350),
            ("Learning Memory", learning_memory_summary, 300),
            ("Cold Memory", cold_memory_summary, 250),
        ]
        rendered: list[str] = []
        for label, raw_text, limit in sections:
            compact = IntentService._compact_context_block(raw_text, max_chars=limit)
            if compact:
                rendered.append(f"[{label}]\n{compact}")
        return "\n\n".join(rendered)

    @staticmethod
    def _compact_context_block(text: str, *, max_chars: int) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 1].rstrip() + "…"
