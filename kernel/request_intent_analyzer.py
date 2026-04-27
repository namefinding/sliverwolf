from __future__ import annotations

import re
from typing import Any

from local_agent.intent.models import TaskClassification
from local_agent.protocol.models import (
    DocumentDeliveryIntent,
    InstructionIntent,
    KnowledgeRequestIntent,
    MemoryCandidateIntent,
    SiteSearchIntent,
    TaskGraphIntent,
)


class RequestIntentAnalyzer:
    _LOCAL_PATH_PATTERN = re.compile(
        r"([A-Za-z]:\\|\\\\|/|[A-Za-z0-9_.\\/-]+\.(txt|md|json|csv|yaml|yml|docx|doc|pdf|xlsx|xls|pptx|ppt|png|jpe?g|webp|gif|bmp))",
        flags=re.IGNORECASE,
    )
    _LOCAL_ANCHOR_PATTERN = re.compile(
        r"(桌面|工作区|目录|路径|文件夹|文件|文档|日志|word|docx|pdf|markdown|md|txt)",
        flags=re.IGNORECASE,
    )
    _QQ_HISTORY_PATTERN = re.compile(
        r"(聊天记录|历史记录|会话记录|回复了什么|回了什么|聊过什么|发过什么文件|聊天历史|附件记录|之前说过什么)",
        flags=re.IGNORECASE,
    )
    _WEB_PATTERN = re.compile(
        r"(网页|网站|链接|github|知乎|b站|bilibili|最新|最近|今天|新闻|资料|来源|搜索|搜一下|查一下|look up|search|find)",
        flags=re.IGNORECASE,
    )
    _TIME_SENSITIVE_PATTERN = re.compile(
        r"(今天|今日|最新|最近|刚刚|现在|目前|news|latest|today|current)",
        flags=re.IGNORECASE,
    )
    _SYSTEM_UTILITY_PATTERN = re.compile(
        r"(提醒|定时|闹钟|倒计时|几点|几号|日期|星期|周几|时间|reminder|alarm|timer)",
        flags=re.IGNORECASE,
    )
    _DOCUMENT_OUTPUT_PATTERN = re.compile(
        r"(写成|整理成|导出成|输出成|保存成|生成一份|生成一个|导出到|写入|保存到).{0,24}(docx|word|markdown|md|txt|xlsx|excel|pptx|ppt|文档|报告|提纲)",
        flags=re.IGNORECASE,
    )
    _EXPLICIT_OUTPUT_FILE_PATTERN = re.compile(
        r"([A-Za-z]:\\[^\s]+|\\\\[^\\\s]+\\[^\s]+|[A-Za-z0-9_.\\/-]+\.(txt|md|json|csv|yaml|yml|docx|doc|pdf|xlsx|xls|pptx|ppt|html|log))",
        flags=re.IGNORECASE,
    )

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    def analyze_document_delivery(
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
        layered_context_summary: str = "",
    ) -> DocumentDeliveryIntent:
        payload = self._call_llm_json(
            "analyze_document_delivery",
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
        if payload is not None:
            try:
                return DocumentDeliveryIntent.model_validate(payload)
            except Exception:
                pass
        return self._minimal_document_delivery_fallback(user_text)

    def analyze_knowledge_request(
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
        layered_context_summary: str = "",
    ) -> KnowledgeRequestIntent:
        payload = self._call_llm_json(
            "analyze_knowledge_request",
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
        if payload is not None:
            try:
                llm_intent = KnowledgeRequestIntent.model_validate(payload)
                return self._apply_minimal_knowledge_constraints(llm_intent, user_text)
            except Exception:
                pass
        return self._minimal_knowledge_fallback(user_text)

    def analyze_site_search(
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
        layered_context_summary: str = "",
    ) -> SiteSearchIntent:
        payload = self._call_llm_json(
            "analyze_site_search",
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
        if payload is not None:
            try:
                return SiteSearchIntent.model_validate(payload)
            except Exception:
                pass
        return SiteSearchIntent()

    def analyze_instruction_intent(
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
        layered_context_summary: str = "",
    ) -> InstructionIntent:
        payload = self._call_llm_json(
            "analyze_instruction_intent",
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
        if payload is not None:
            try:
                return InstructionIntent.model_validate(payload)
            except Exception:
                pass
        return InstructionIntent()

    def analyze_memory_candidate_intent(
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
        layered_context_summary: str = "",
    ) -> MemoryCandidateIntent:
        payload = self._call_llm_json(
            "analyze_memory_candidate_intent",
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
        if payload is not None:
            try:
                return MemoryCandidateIntent.model_validate(payload)
            except Exception:
                pass
        return MemoryCandidateIntent()

    def analyze_task_graph(
        self,
        user_text: str,
        *,
        recent_context: str = "",
        hot_context_summary: str = "",
        warm_memory_summary: str = "",
        cold_memory_summary: str = "",
        active_task_summary: str = "",
        channel_context_summary: str = "",
        layered_context_summary: str = "",
    ) -> TaskGraphIntent:
        payload = self._call_llm_json(
            "analyze_task_graph",
            user_text,
            recent_context=recent_context,
            hot_context_summary=hot_context_summary,
            warm_memory_summary=warm_memory_summary,
            cold_memory_summary=cold_memory_summary,
            active_task_summary=active_task_summary,
            channel_context_summary=channel_context_summary,
            layered_context_summary=layered_context_summary,
        )
        if payload is not None:
            try:
                return TaskGraphIntent.model_validate(payload)
            except Exception:
                pass
        return TaskGraphIntent(
            is_multi_task=False,
            primary_task_text=str(user_text or "").strip() or None,
            confidence=0.0,
            rationale="default_single_task_fallback",
        )

    def analyze_task_classification(
        self,
        *,
        user_text: str,
        knowledge_type: str,
        document_delivery: DocumentDeliveryIntent,
        site_search: SiteSearchIntent,
        recent_context: str = "",
        hot_context_summary: str = "",
        warm_memory_summary: str = "",
        cold_memory_summary: str = "",
        active_task_summary: str = "",
        channel_context_summary: str = "",
        layered_context_summary: str = "",
    ) -> TaskClassification:
        payload = self._call_llm_json(
            "analyze_task_classification",
            user_text=user_text,
            knowledge_type=knowledge_type,
            document_delivery=document_delivery.model_dump(mode="json"),
            site_search=site_search.model_dump(mode="json"),
            recent_context=recent_context,
            hot_context_summary=hot_context_summary,
            warm_memory_summary=warm_memory_summary,
            cold_memory_summary=cold_memory_summary,
            active_task_summary=active_task_summary,
            channel_context_summary=channel_context_summary,
            layered_context_summary=layered_context_summary,
        )
        if payload is not None:
            try:
                return self._normalize_task_classification(payload)
            except Exception:
                pass
        return self._minimal_task_classification(
            user_text=user_text,
            knowledge_type=knowledge_type,
            document_delivery=document_delivery,
            site_search=site_search,
        )

    def _call_llm_json(self, method_name: str, user_text: str | None = None, **kwargs) -> dict[str, Any] | None:
        method = getattr(self.llm_client, method_name, None)
        if method is None:
            return None
        try:
            if user_text is None:
                payload = method(**kwargs)
            else:
                payload = method(user_text, **kwargs)
        except TypeError:
            try:
                if user_text is None:
                    payload = method(**kwargs)
                else:
                    payload = method(user_text)
            except Exception:
                return None
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @classmethod
    def _apply_minimal_knowledge_constraints(
        cls,
        llm_intent: KnowledgeRequestIntent,
        user_text: str,
    ) -> KnowledgeRequestIntent:
        text = str(user_text or "").strip()
        if cls._LOCAL_PATH_PATTERN.search(text) or (
            cls._LOCAL_ANCHOR_PATTERN.search(text) and not cls._QQ_HISTORY_PATTERN.search(text)
        ):
            if any(token in text for token in ("写入", "加一条", "修改", "新增", "添加", "后面加")):
                return llm_intent.model_copy(
                    update={
                        "needs_grounding": False,
                        "time_sensitive": False,
                        "lookup_requested": False,
                        "knowledge_type": "local_workspace",
                        "confidence": max(llm_intent.confidence, 0.8),
                        "rationale": llm_intent.rationale or "explicit_local_task",
                    }
                )
        if cls._LOCAL_PATH_PATTERN.search(text) and llm_intent.knowledge_type not in {"local_workspace", "system_utility"}:
            return llm_intent.model_copy(
                update={
                    "needs_grounding": False,
                    "time_sensitive": False,
                    "lookup_requested": False,
                    "knowledge_type": "local_workspace",
                    "confidence": max(llm_intent.confidence, 0.82),
                    "rationale": "explicit_local_path",
                }
            )
        if cls._QQ_HISTORY_PATTERN.search(text) and llm_intent.knowledge_type == "unknown":
            return llm_intent.model_copy(
                update={
                    "needs_grounding": False,
                    "time_sensitive": False,
                    "lookup_requested": False,
                    "knowledge_type": "qq_history",
                    "confidence": 0.76,
                    "rationale": "explicit_history_anchor",
                }
            )
        if cls._SYSTEM_UTILITY_PATTERN.search(text) and llm_intent.knowledge_type == "unknown":
            return llm_intent.model_copy(
                update={
                    "needs_grounding": False,
                    "time_sensitive": False,
                    "lookup_requested": False,
                    "knowledge_type": "system_utility",
                    "confidence": 0.72,
                    "rationale": "explicit_system_utility_anchor",
                }
            )
        return llm_intent

    @classmethod
    def _minimal_document_delivery_fallback(cls, user_text: str) -> DocumentDeliveryIntent:
        text = str(user_text or "").strip()
        if cls._DOCUMENT_OUTPUT_PATTERN.search(text):
            output_file = None
            match = cls._EXPLICIT_OUTPUT_FILE_PATTERN.search(text)
            if match:
                output_file = match.group(1)
            return DocumentDeliveryIntent(
                wants_document=True,
                save_output=bool(output_file),
                artifact_type="document",
                output_format=None,
                output_file=output_file,
                confidence=0.62,
                rationale="minimal_document_output_fallback",
            )
        return DocumentDeliveryIntent()

    @classmethod
    def _minimal_knowledge_fallback(cls, user_text: str) -> KnowledgeRequestIntent:
        text = str(user_text or "").strip()
        if not text:
            return KnowledgeRequestIntent()
        if cls._LOCAL_PATH_PATTERN.search(text):
            return KnowledgeRequestIntent(
                needs_grounding=False,
                time_sensitive=False,
                lookup_requested=False,
                knowledge_type="local_workspace",
                confidence=0.82,
                rationale="minimal_local_path_fallback",
            )
        if cls._QQ_HISTORY_PATTERN.search(text):
            return KnowledgeRequestIntent(
                needs_grounding=False,
                time_sensitive=False,
                lookup_requested=False,
                knowledge_type="qq_history",
                confidence=0.78,
                rationale="minimal_history_fallback",
            )
        if cls._SYSTEM_UTILITY_PATTERN.search(text):
            return KnowledgeRequestIntent(
                needs_grounding=False,
                time_sensitive=False,
                lookup_requested=False,
                knowledge_type="system_utility",
                confidence=0.72,
                rationale="minimal_system_fallback",
            )
        if cls._WEB_PATTERN.search(text):
            return KnowledgeRequestIntent(
                needs_grounding=True,
                time_sensitive=bool(cls._TIME_SENSITIVE_PATTERN.search(text)),
                lookup_requested=True,
                knowledge_type=(
                    "time_sensitive_external_topic"
                    if cls._TIME_SENSITIVE_PATTERN.search(text)
                    else "general_external_topic"
                ),
                confidence=0.68,
                rationale="minimal_web_fallback",
            )
        if cls._LOCAL_ANCHOR_PATTERN.search(text):
            return KnowledgeRequestIntent(
                needs_grounding=False,
                time_sensitive=False,
                lookup_requested=False,
                knowledge_type="local_workspace",
                confidence=0.66,
                rationale="minimal_local_anchor_fallback",
            )
        return KnowledgeRequestIntent()

    @classmethod
    def _minimal_task_classification(
        cls,
        *,
        user_text: str,
        knowledge_type: str,
        document_delivery: DocumentDeliveryIntent,
        site_search: SiteSearchIntent,
    ) -> TaskClassification:
        text = str(user_text or "").strip()
        lowered = text.lower()

        if knowledge_type == "qq_history":
            return TaskClassification(
                domain="qq_history",
                task_kind="history_lookup",
                preferred_families=["qq_history"],
                confidence=0.68,
                rationale="minimal_history_classification",
            )

        if knowledge_type == "system_utility":
            if any(token in lowered for token in ("提醒", "定时", "闹钟", "倒计时", "reminder", "alarm", "timer")):
                return TaskClassification(
                    domain="system_utility",
                    task_kind="create_reminder",
                    preferred_families=["system_utility"],
                    confidence=0.68,
                    rationale="minimal_reminder_classification",
                )
            return TaskClassification(
                domain="system_utility",
                task_kind="get_current_time",
                preferred_families=["system_utility"],
                confidence=0.56,
                rationale="minimal_system_utility_classification",
            )

        if site_search.site or knowledge_type in {"general_external_topic", "time_sensitive_external_topic"}:
            task_kind = "research_document" if (document_delivery.wants_document or document_delivery.save_output) else "research"
            return TaskClassification(
                domain="web",
                task_kind=task_kind,
                preferred_families=["web_lookup", "web_target"],
                confidence=0.64,
                rationale="minimal_web_classification",
            )

        if knowledge_type == "local_workspace":
            if any(token in lowered for token in ("写入", "加一条", "修改", "新增", "添加", "后面加", "append", "insert", "edit", "update", "rewrite")):
                return TaskClassification(
                    domain="local_workspace",
                    task_kind="document_edit",
                    preferred_families=["document_operation", "local_lookup", "file_lookup"],
                    confidence=0.7,
                    rationale="minimal_document_edit_classification",
                )
            if document_delivery.wants_document or document_delivery.save_output:
                return TaskClassification(
                    domain="local_workspace",
                    task_kind="summarize",
                    preferred_families=["document_summary", "local_lookup", "file_lookup"],
                    confidence=0.58,
                    rationale="minimal_document_summary_classification",
                )
            return TaskClassification(
                domain="local_workspace",
                task_kind="lookup",
                preferred_families=["local_lookup", "file_lookup"],
                confidence=0.52,
                rationale="minimal_local_lookup_classification",
            )

        return TaskClassification(
            domain="unknown",
            task_kind="unknown",
            preferred_families=[],
            confidence=0.0,
            rationale="minimal_unknown_classification",
        )

    @staticmethod
    def _normalize_task_classification(payload: dict[str, Any]) -> TaskClassification:
        normalized = dict(payload or {})
        normalized["domain"] = str(normalized.get("domain", "unknown") or "unknown").strip().lower()
        normalized["task_kind"] = str(normalized.get("task_kind", "unknown") or "unknown").strip().lower()

        preferred = normalized.get("preferred_families", [])
        if not isinstance(preferred, list):
            preferred = []
        allowed_families = {
            "document_operation",
            "document_summary",
            "file_delivery",
            "local_lookup",
            "file_lookup",
            "local_collection",
            "qq_history",
            "web_target",
            "web_lookup",
            "system_utility",
        }
        normalized["preferred_families"] = [
            str(item).strip()
            for item in preferred
            if str(item).strip() in allowed_families
        ]

        run_mode = str(normalized.get("run_mode", "immediate") or "immediate").strip().lower()
        normalized["run_mode"] = run_mode if run_mode in {"immediate", "scheduled"} else "immediate"

        scheduled_task_type = str(normalized.get("scheduled_task_type", "") or "").strip().lower()
        normalized["scheduled_task_type"] = (
            scheduled_task_type if scheduled_task_type in {"notify", "deferred_agent_task"} else None
        )

        payload_hint = normalized.get("scheduled_task_payload_hint", {})
        normalized["scheduled_task_payload_hint"] = payload_hint if isinstance(payload_hint, dict) else {}

        try:
            normalized["confidence"] = max(0.0, min(1.0, float(normalized.get("confidence", 0.0))))
        except (TypeError, ValueError):
            normalized["confidence"] = 0.0

        normalized["rationale"] = str(normalized.get("rationale", "") or "").strip()
        return TaskClassification.model_validate(normalized)
