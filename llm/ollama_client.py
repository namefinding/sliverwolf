from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from local_agent.intent.models import TaskClassification
from local_agent.protocol.models import (
    DocumentDeliveryIntent,
    DecisionReview,
    FollowUpAssessment,
    InstructionIntent,
    KnowledgeRequestIntent,
    MemoryCandidateIntent,
    Message,
    PendingTask,
    OutputKind,
    ProxySendIntent,
    RiskLevel,
    SiteSearchIntent,
    TaskGraphIntent,
    TurnCompletionDecision,
    ToolDecision,
    ToolManifest,
    WorkflowSpec,
)


OUTPUT_KIND_ALIASES = {
    "directory_entries": OutputKind.DIRECTORY_ENTRIES.value,
    "dir_entries": OutputKind.DIRECTORY_ENTRIES.value,
    "directory_list": OutputKind.DIRECTORY_ENTRIES.value,
    "folder_list": OutputKind.DIRECTORY_ENTRIES.value,
    "folders": OutputKind.DIRECTORY_ENTRIES.value,
    "file_contents": OutputKind.FILE_CONTENTS.value,
    "file_content": OutputKind.FILE_CONTENTS.value,
    "read_file": OutputKind.FILE_CONTENTS.value,
    "extracted_text": OutputKind.FILE_CONTENTS.value,
    "search_matches": OutputKind.SEARCH_MATCHES.value,
    "matches": OutputKind.SEARCH_MATCHES.value,
    "object_candidates": OutputKind.OBJECT_CANDIDATES.value,
    "candidates": OutputKind.OBJECT_CANDIDATES.value,
    "contact_candidates": OutputKind.CONTACT_CANDIDATES.value,
    "contacts": OutputKind.CONTACT_CANDIDATES.value,
    "object_details": OutputKind.OBJECT_DETAILS.value,
    "candidate_details": OutputKind.OBJECT_DETAILS.value,
    "file_written": OutputKind.FILE_WRITTEN.value,
    "written_file": OutputKind.FILE_WRITTEN.value,
    "saved_file": OutputKind.FILE_WRITTEN.value,
    "path_opened": OutputKind.PATH_OPENED.value,
    "path_created": OutputKind.PATH_CREATED.value,
    "path_updated": OutputKind.PATH_UPDATED.value,
    "path_deleted": OutputKind.PATH_DELETED.value,
    "memory_items": OutputKind.MEMORY_ITEMS.value,
    "memory_results": OutputKind.MEMORY_ITEMS.value,
    "memory_saved": OutputKind.MEMORY_SAVED.value,
    "search_results": OutputKind.SEARCH_RESULTS.value,
    "web_results": OutputKind.SEARCH_RESULTS.value,
    "web_content": OutputKind.WEB_CONTENT.value,
    "page_content": OutputKind.WEB_CONTENT.value,
    "message_sent": OutputKind.MESSAGE_SENT.value,
    "messages_sent": OutputKind.MESSAGE_SENT.value,
}

ALLOWED_OUTPUT_VALUES = {member.value for member in OutputKind}

ACTION_TOOL_ALIASES = {
    "file.write": "file.write",
    "file.write_many": "file.write_many",
    "file.read": "file.read",
    "file.read_contents": "file.read",
    "file.read_content": "file.read",
    "file.get_contents": "file.read",
    "file.get_content": "file.read",
    "file.search_by_name": "file.search_by_name",
    "file.extract_text": "file.extract_text",
    "file.extract_structure": "file.extract_structure",
    "file.search_blocks": "file.search_blocks",
    "file.list": "file.list",
    "file.append": "file.append",
    "file.append_many": "file.append_many",
    "file.edit_docx": "file.edit_docx",
    "file.render_docx_from_template": "file.render_docx_from_template",
    "file.metadata": "file.metadata",
    "file.metadata_many": "file.metadata_many",
    "file.preview": "file.preview",
    "file.preview_many": "file.preview_many",
    "file.mkdir": "file.mkdir",
    "file.mkdir_many": "file.mkdir_many",
    "file.copy": "file.copy",
    "file.copy_many": "file.copy_many",
    "file.move": "file.move",
    "file.move_many": "file.move_many",
    "file.rename": "file.rename",
    "file.rename_many": "file.rename_many",
    "file.delete": "file.delete",
    "file.delete_many": "file.delete_many",
    "file.open_path": "file.open_path",
    "file.open_many": "file.open_many",
    "file.reveal_in_explorer": "file.reveal_in_explorer",
    "file.reveal_many": "file.reveal_many",
    "document_agent.summarize": "document_agent.summarize",
    "document_agent.read": "document_agent.read",
    "document_agent.inspect": "document_agent.inspect",
    "document_agent.edit": "document_agent.edit",
    "image.inspect": "image.inspect",
    "image.describe": "image.describe",
    "image.read_text": "image.read_text",
    "image.capture_screen": "image.capture_screen",
    "image.capture_region": "image.capture_region",
    "retrieval.search_local_objects": "retrieval.search_local_objects",
    "retrieval.inspect_local_candidate": "retrieval.inspect_local_candidate",
    "write_file": "file.write",
    "write_files": "file.write_many",
    "append_file": "file.append",
    "append_files": "file.append_many",
    "save_file": "file.write",
    "read_file": "file.read",
    "search_by_name": "file.search_by_name",
    "extract_text": "file.extract_text",
    "extract_structure": "file.extract_structure",
    "search_blocks": "file.search_blocks",
    "read_contents": "file.read",
    "read_content": "file.read",
    "get_contents": "file.read",
    "get_content": "file.read",
    "open_file": "file.read",
    "preview_file": "file.preview",
    "inspect_docx_structure": "file.extract_structure",
    "search_docx_blocks": "file.search_blocks",
    "edit_docx": "file.edit_docx",
    "edit_word_document": "file.edit_docx",
    "document_summary": "document_agent.summarize",
    "summarize_document": "document_agent.summarize",
    "read_document": "document_agent.read",
    "inspect_document": "document_agent.inspect",
    "document_inspect": "document_agent.inspect",
    "search_document_blocks": "document_agent.inspect",
    "document_edit": "document_agent.edit",
    "edit_document": "document_agent.edit",
    "render_docx_from_template": "file.render_docx_from_template",
    "apply_template_to_docx": "file.render_docx_from_template",
    "get_metadata": "file.metadata",
    "create_folder": "file.mkdir",
    "create_directory": "file.mkdir",
    "create_folders": "file.mkdir_many",
    "copy_file": "file.copy",
    "copy_files": "file.copy_many",
    "move_file": "file.move",
    "move_files": "file.move_many",
    "rename_file": "file.rename",
    "rename_files": "file.rename_many",
    "delete_file": "file.delete",
    "delete_files": "file.delete_many",
    "open_path": "file.open_path",
    "open_folder": "file.open_path",
    "open_paths": "file.open_many",
    "reveal_in_explorer": "file.reveal_in_explorer",
    "reveal_paths": "file.reveal_many",
    "inspect_image": "image.inspect",
    "describe_image": "image.describe",
    "image_describe": "image.describe",
    "analyze_image": "image.describe",
    "image_inspect": "image.inspect",
    "ocr_image": "image.read_text",
    "read_image_text": "image.read_text",
    "capture_screen": "image.capture_screen",
    "screenshot": "image.capture_screen",
    "capture_region": "image.capture_region",
    "list_files": "file.list",
    "list_folders": "file.list",
    "list_directories": "file.list",
    "search_files": "retrieval.search_local_objects",
    "web.search": "web.search",
    "web.fetch": "web.fetch",
    "web.open_page": "web.open_page",
    "web.research": "web.research",
    "qq.get_current_context": "qq.get_current_context",
    "qq.get_recent_messages": "qq.get_recent_messages",
    "qq.get_last_reply": "qq.get_last_reply",
    "qq.search_history": "qq.search_history",
    "qq.get_recent_attachments": "qq.get_recent_attachments",
    "qq.search_contacts": "qq.search_contacts",
    "qq.send_text": "qq.send_text",
    "qq.send_file": "qq.send_file",
    "qq.send_voice": "qq.send_voice",
    "search_web": "web.search",
    "web_search": "web.search",
    "fetch_web": "web.fetch",
    "fetch_url": "web.fetch",
    "browse_web": "web.fetch",
    "read_webpage": "web.fetch",
    "open_webpage": "web.open_page",
    "open_website": "web.open_page",
    "open_url": "web.open_page",
    "visit_website": "web.open_page",
    "research_web": "web.research",
    "web_research": "web.research",
    "search_online": "web.research",
    "search_local_objects": "retrieval.search_local_objects",
    "inspect_candidate": "retrieval.inspect_local_candidate",
    "inspect": "retrieval.inspect_local_candidate",
    "qq_context": "qq.get_current_context",
    "current_chat_context": "qq.get_current_context",
    "recent_messages": "qq.get_recent_messages",
    "get_recent_messages": "qq.get_recent_messages",
    "conversation_context": "qq.get_recent_messages",
    "last_reply": "qq.get_last_reply",
    "get_last_reply": "qq.get_last_reply",
    "search_history": "qq.search_history",
    "history_search": "qq.search_history",
    "recent_attachments": "qq.get_recent_attachments",
    "get_recent_attachments": "qq.get_recent_attachments",
    "search_contacts": "qq.search_contacts",
    "find_contact": "qq.search_contacts",
    "send_message": "qq.send_text",
    "send_text": "qq.send_text",
    "send_file": "qq.send_file",
    "file_transfer": "qq.send_file",
    "transfer_file": "qq.send_file",
    "send_voice": "qq.send_voice",
}

DECISION_ALIASES = {
    "tool_call": "tool_call",
    "use_tool": "tool_call",
    "call_tool": "tool_call",
    "execute": "tool_call",
    "respond": "respond",
    "reply": "respond",
    "answer": "respond",
    "response": "respond",
    "clarify": "clarify",
    "ask_user": "clarify",
    "finish": "finish",
    "done": "finish",
}

class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int = 120,
        chat_model: str | None = None,
        critic_model: str | None = None,
        response_model: str | None = None,
        vision_model: str | None = None,
        keep_alive: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.chat_model = chat_model or response_model or model
        self.critic_model = critic_model or model
        self.response_model = response_model or model
        self.vision_model = vision_model or model
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive

    def _chat(self, messages: list[dict[str, str]], stream: bool = False, model: str | None = None) -> str:
        request_payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": 0.2},
        }
        if self.keep_alive:
            request_payload["keep_alive"] = self.keep_alive
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=request_payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["message"]["content"]

    def describe_image(
        self,
        image_path: str | Path,
        *,
        prompt: str,
        model: str | None = None,
    ) -> str:
        payload_bytes = Path(image_path).read_bytes()
        encoded_image = base64.b64encode(payload_bytes).decode("ascii")
        request_payload = {
            "model": model or self.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [encoded_image],
                }
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        if self.keep_alive:
            request_payload["keep_alive"] = self.keep_alive
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=request_payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["message"]["content"]).strip()

    @staticmethod
    def _compact_text(value: str, *, max_chars: int, prefer_tail: bool = False) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        if max_chars <= 16:
            return text[:max_chars]
        if prefer_tail:
            return "...\n" + text[-(max_chars - 4) :]
        return text[: max_chars - 4] + "\n..."

    def _compact_lightweight_context(
        self,
        *,
        user_text: str,
        recent_context: str = "",
        hot_context_summary: str = "",
        warm_memory_summary: str = "",
        cold_memory_summary: str = "",
        active_task_summary: str = "",
        recent_context_chars: int = 1200,
        summary_chars: int = 480,
        user_text_chars: int = 400,
    ) -> dict[str, str]:
        return {
            "user_text": self._compact_text(user_text, max_chars=user_text_chars),
            "recent_context": self._compact_text(recent_context, max_chars=recent_context_chars, prefer_tail=True),
            "hot_context_summary": self._compact_text(hot_context_summary, max_chars=summary_chars),
            "warm_memory_summary": self._compact_text(warm_memory_summary, max_chars=summary_chars),
            "cold_memory_summary": self._compact_text(cold_memory_summary, max_chars=summary_chars),
            "active_task_summary": self._compact_text(active_task_summary, max_chars=summary_chars),
        }

    @staticmethod
    def _extract_json_block(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            parts = stripped.split("```")
            for part in parts:
                candidate = part.strip()
                if not candidate or candidate == "json":
                    continue
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                return json.loads(candidate)

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start >= 0 and end > start:
                return json.loads(stripped[start : end + 1])
            raise

    @staticmethod
    def _normalize_nullable_string(value: Any, *, lowercase: bool = False) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if normalized.lower() in {"null", "none", "nil", "undefined", "n/a"}:
            return None
        return normalized.lower() if lowercase else normalized

    @staticmethod
    def _normalize_decision_value(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        if not normalized:
            return None
        return DECISION_ALIASES.get(normalized, normalized)

    @staticmethod
    def _normalize_selected_tool_value(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        if not normalized:
            return None
        mapped = ACTION_TOOL_ALIASES.get(normalized, normalized if "." in normalized else None)
        return mapped

    @staticmethod
    def _coerce_tool_decision(payload: dict[str, Any], fallback: ToolDecision | None = None) -> ToolDecision:
        normalized = OllamaClient._normalize_tool_decision_payload(payload, fallback=fallback)
        return ToolDecision.model_validate(normalized)

    @staticmethod
    def _normalize_document_delivery_payload(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        wants_document = normalized.get("wants_document")
        save_output = normalized.get("save_output")
        artifact_type = OllamaClient._normalize_nullable_string(normalized.get("artifact_type"), lowercase=True)
        confidence = normalized.get("confidence", 0.0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        output_format = OllamaClient._normalize_nullable_string(normalized.get("output_format"), lowercase=True)
        output_file = OllamaClient._normalize_nullable_string(normalized.get("output_file"))
        title = OllamaClient._normalize_nullable_string(normalized.get("title"))
        rationale = normalized.get("rationale")
        if not isinstance(rationale, str):
            rationale = ""
        return {
            "wants_document": bool(wants_document),
            "save_output": bool(save_output),
            "artifact_type": artifact_type,
            "output_format": output_format,
            "output_file": output_file,
            "title": title,
            "confidence": max(0.0, min(1.0, confidence_value)),
            "rationale": rationale,
        }

    @staticmethod
    def _normalize_knowledge_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        needs_grounding = normalized.get("needs_grounding")
        time_sensitive = normalized.get("time_sensitive")
        lookup_requested = normalized.get("lookup_requested")
        knowledge_type = OllamaClient._normalize_nullable_string(normalized.get("knowledge_type"), lowercase=True) or "unknown"
        confidence = normalized.get("confidence", 0.0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        rationale = normalized.get("rationale")
        if not isinstance(rationale, str):
            rationale = ""
        return {
            "needs_grounding": bool(needs_grounding),
            "time_sensitive": bool(time_sensitive),
            "lookup_requested": bool(lookup_requested),
            "knowledge_type": knowledge_type,
            "confidence": max(0.0, min(1.0, confidence_value)),
            "rationale": rationale,
        }

    @staticmethod
    def _normalize_site_search_payload(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        site = OllamaClient._normalize_nullable_string(normalized.get("site"), lowercase=True)
        query = OllamaClient._normalize_nullable_string(normalized.get("query"))
        content_type = OllamaClient._normalize_nullable_string(normalized.get("content_type"), lowercase=True) or "generic"
        action = OllamaClient._normalize_nullable_string(normalized.get("action"), lowercase=True) or "search"
        site_scope = OllamaClient._normalize_nullable_string(normalized.get("site_scope"), lowercase=True) or "none"
        if site_scope not in {"none", "preferred", "required"}:
            site_scope = "preferred" if site else "none"
        open_first = bool(normalized.get("open_first"))
        confidence = normalized.get("confidence", 0.0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        rationale = normalized.get("rationale")
        if not isinstance(rationale, str):
            rationale = ""
        return {
            "site": site,
            "query": query,
            "content_type": content_type,
            "action": action,
            "site_scope": site_scope if site else "none",
            "open_first": open_first,
            "confidence": max(0.0, min(1.0, confidence_value)),
            "rationale": rationale,
        }

    @staticmethod
    def _normalize_instruction_intent_payload(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload or {})
        scope = OllamaClient._normalize_nullable_string(normalized.get("scope"), lowercase=True) or "none"
        if scope not in {"none", "turn", "session", "persistent"}:
            scope = "none"
        kind = OllamaClient._normalize_nullable_string(normalized.get("kind"), lowercase=True) or "none"
        if kind not in {"none", "naming", "preference", "workflow_method", "tool_policy", "correction", "style", "boundary"}:
            kind = "none"

        def _normalize_family_list(value: Any) -> list[str]:
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
                "delivery",
            }
            if not isinstance(value, list):
                return []
            return [
                item
                for item in (
                    str(entry).strip()
                    for entry in value
                )
                if item in allowed_families
            ]

        def _normalize_tool_list(value: Any) -> list[str]:
            if not isinstance(value, list):
                return []
            tools: list[str] = []
            for entry in value:
                normalized_tool = OllamaClient._normalize_selected_tool_value(entry)
                if normalized_tool:
                    tools.append(normalized_tool)
            return list(dict.fromkeys(tools))

        response_style = OllamaClient._normalize_nullable_string(normalized.get("response_style"), lowercase=True)
        if response_style not in {None, "default", "concise", "direct_answer_first", "conversational", "formal"}:
            response_style = None

        confidence = normalized.get("confidence", 0.0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        rationale = normalized.get("rationale")
        if not isinstance(rationale, str):
            rationale = ""
        is_instruction = bool(normalized.get("is_instruction"))
        apply_this_turn = bool(normalized.get("apply_this_turn"))
        persist_memory = bool(normalized.get("persist_memory"))
        normalized_instruction = OllamaClient._normalize_nullable_string(normalized.get("normalized_instruction"))
        memory_text = OllamaClient._normalize_nullable_string(normalized.get("memory_text"))
        if not is_instruction:
            scope = "none"
            kind = "none"
            apply_this_turn = False
            persist_memory = False
            normalized_instruction = None
            memory_text = None
        return {
            "is_instruction": is_instruction,
            "scope": scope,
            "kind": kind,
            "apply_this_turn": apply_this_turn,
            "persist_memory": persist_memory,
            "normalized_instruction": normalized_instruction,
            "memory_text": memory_text,
            "preferred_families": _normalize_family_list(normalized.get("preferred_families")),
            "blocked_families": _normalize_family_list(normalized.get("blocked_families")),
            "preferred_tools": _normalize_tool_list(normalized.get("preferred_tools")),
            "response_style": response_style,
            "confidence": max(0.0, min(1.0, confidence_value)),
            "rationale": rationale,
        }

    @staticmethod
    def _normalize_memory_candidate_intent_payload(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload or {})
        scope = OllamaClient._normalize_nullable_string(normalized.get("scope"), lowercase=True) or "none"
        if scope not in {"none", "turn", "session", "persistent"}:
            scope = "none"
        kind = OllamaClient._normalize_nullable_string(normalized.get("kind"), lowercase=True) or "none"
        if kind not in {
            "none",
            "user_fact",
            "naming",
            "preference",
            "workflow_method",
            "tool_policy",
            "correction",
            "style",
            "boundary",
        }:
            kind = "none"

        def _normalize_family_list(value: Any) -> list[str]:
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
                "delivery",
            }
            if not isinstance(value, list):
                return []
            return [
                item
                for item in (str(entry).strip() for entry in value)
                if item in allowed_families
            ]

        def _normalize_tool_list(value: Any) -> list[str]:
            if not isinstance(value, list):
                return []
            tools: list[str] = []
            for entry in value:
                normalized_tool = OllamaClient._normalize_selected_tool_value(entry)
                if normalized_tool:
                    tools.append(normalized_tool)
            return list(dict.fromkeys(tools))

        response_style = OllamaClient._normalize_nullable_string(normalized.get("response_style"), lowercase=True)
        if response_style not in {None, "default", "concise", "direct_answer_first", "conversational", "formal"}:
            response_style = None

        confidence = normalized.get("confidence", 0.0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        rationale = normalized.get("rationale")
        if not isinstance(rationale, str):
            rationale = ""

        is_memory_candidate = bool(normalized.get("is_memory_candidate"))
        apply_this_turn = bool(normalized.get("apply_this_turn"))
        persist_memory = bool(normalized.get("persist_memory"))
        should_write_memory = bool(normalized.get("should_write_memory"))
        overwrite_existing = bool(normalized.get("overwrite_existing"))
        normalized_text = OllamaClient._normalize_nullable_string(normalized.get("normalized_text"))
        memory_text = OllamaClient._normalize_nullable_string(normalized.get("memory_text"))
        memory_key = OllamaClient._normalize_nullable_string(normalized.get("memory_key"))
        canonical_value = normalized.get("canonical_value")
        if not isinstance(canonical_value, dict):
            canonical_value = {}

        if not is_memory_candidate:
            scope = "none"
            kind = "none"
            apply_this_turn = False
            persist_memory = False
            should_write_memory = False
            overwrite_existing = False
            normalized_text = None
            memory_text = None
            memory_key = None
            canonical_value = {}

        if scope in {"session", "persistent"} and persist_memory and not should_write_memory:
            should_write_memory = True

        return {
            "is_memory_candidate": is_memory_candidate,
            "scope": scope,
            "kind": kind,
            "apply_this_turn": apply_this_turn,
            "persist_memory": persist_memory,
            "should_write_memory": should_write_memory,
            "overwrite_existing": overwrite_existing,
            "normalized_text": normalized_text,
            "memory_text": memory_text,
            "memory_key": memory_key,
            "canonical_value": canonical_value,
            "preferred_families": _normalize_family_list(normalized.get("preferred_families")),
            "blocked_families": _normalize_family_list(normalized.get("blocked_families")),
            "preferred_tools": _normalize_tool_list(normalized.get("preferred_tools")),
            "response_style": response_style,
            "confidence": max(0.0, min(1.0, confidence_value)),
            "rationale": rationale,
        }

    @staticmethod
    def _normalize_task_graph_payload(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload or {})
        subtasks_raw = normalized.get("subtasks")
        subtasks: list[dict[str, Any]] = []
        if isinstance(subtasks_raw, list):
            for index, entry in enumerate(subtasks_raw, start=1):
                if not isinstance(entry, dict):
                    continue
                task_id = OllamaClient._normalize_nullable_string(entry.get("task_id")) or f"task_{index}"
                task_text = OllamaClient._normalize_nullable_string(entry.get("task_text")) or ""
                summary = OllamaClient._normalize_nullable_string(entry.get("summary")) or task_text
                kind = OllamaClient._normalize_nullable_string(entry.get("kind"), lowercase=True) or "generic"
                if kind not in {
                    "generic",
                    "web_lookup",
                    "document_edit",
                    "local_lookup",
                    "direct_answer",
                    "qq_history",
                    "system_utility",
                    "delivery",
                }:
                    kind = "generic"
                status = OllamaClient._normalize_nullable_string(entry.get("status"), lowercase=True) or "ready"
                if status not in {"ready", "waiting_for_input", "blocked", "completed"}:
                    status = "ready"
                missing_slots = entry.get("missing_slots")
                if not isinstance(missing_slots, list):
                    missing_slots = []
                slot_values_raw = entry.get("slot_values")
                slot_values = {
                    str(key).strip(): str(value).strip()
                    for key, value in (slot_values_raw or {}).items()
                    if str(key).strip() and str(value).strip()
                } if isinstance(slot_values_raw, dict) else {}
                rationale = OllamaClient._normalize_nullable_string(entry.get("rationale")) or ""
                subtasks.append(
                    {
                        "task_id": task_id,
                        "order": index,
                        "summary": summary,
                        "task_text": task_text,
                        "kind": kind,
                        "status": status,
                        "missing_slots": [str(item).strip() for item in missing_slots if str(item).strip()],
                        "slot_values": slot_values,
                        "rationale": rationale,
                    }
                )
        primary_task_text = OllamaClient._normalize_nullable_string(normalized.get("primary_task_text"))
        primary_task_id = OllamaClient._normalize_nullable_string(normalized.get("primary_task_id"))
        followup_text = OllamaClient._normalize_nullable_string(normalized.get("followup_text"))
        rationale = OllamaClient._normalize_nullable_string(normalized.get("rationale")) or ""
        confidence = normalized.get("confidence", 0.0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        return {
            "is_multi_task": bool(normalized.get("is_multi_task")) or len(subtasks) > 1,
            "primary_task_text": primary_task_text,
            "primary_task_id": primary_task_id,
            "needs_clarification": bool(normalized.get("needs_clarification")),
            "followup_text": followup_text,
            "subtasks": subtasks,
            "confidence": max(0.0, min(1.0, confidence_value)),
            "rationale": rationale,
        }

    @staticmethod
    def _shorten_speech_text(text: str, max_chars: int) -> str:
        compact = OllamaClient._normalize_conversational_text(text, for_speech=True)
        compact = " ".join(str(compact).replace("\r", "\n").split())
        compact = compact.replace("：", "，")
        limit = max(20, max_chars)
        if len(compact) <= limit:
            return compact
        cutoff = max(limit - 1, 1)
        trimmed = compact[:cutoff].rstrip("，,；;：:。.!? ")
        if not trimmed:
            trimmed = compact[:cutoff]
        return trimmed + "。"

    @staticmethod
    def _strip_stage_directions(text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return cleaned
        pattern = re.compile(r"^\s*[（(【\[]([^（）()\[\]【】\n]{1,24})[）)】\]]\s*")
        cue_pattern = re.compile(
            r"(轻|笑|眨|挑眉|耸肩|敲|点|抬|压低|放轻|沉默|顿了顿|停顿|语气|声音|指尖|桌面|屏幕|尾音|嗓音|口吻|低声|轻声|淡淡|懒懒|慢悠悠|若有所思|歪头|眯眼|哼|咳|清了清嗓子|roleplay|narrat)",
            flags=re.IGNORECASE,
        )
        while True:
            match = pattern.match(cleaned)
            if match is None:
                break
            cue = match.group(1).strip()
            if not cue_pattern.search(cue):
                break
            cleaned = cleaned[match.end():].lstrip()
        return cleaned

    @classmethod
    def _normalize_conversational_text(cls, text: str, *, for_speech: bool = False) -> str:
        cleaned = cls._sanitize_response_text(text)
        cleaned = cls._strip_stage_directions(cleaned)
        cleaned = re.sub(r"^[,，。.!！？:：;；\-\s]+", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        if for_speech:
            cleaned = cleaned.replace("嗯……", "嗯，").replace("呃……", "呃，")
        return cleaned

    def _fallback_tool_response_bundle(
        self,
        *,
        raw: str,
        system_name: str,
        messages: list[Message],
        observations: list[str],
        response_hint: str | None,
        execution_summary: dict[str, Any] | None,
        persona_name: str | None,
        persona_profile: str | None,
        display_style_prompt: str | None,
        speech_max_chars: int,
    ) -> dict[str, str]:
        display_text = self.render_response(
            system_name=system_name,
            messages=messages,
            observations=observations,
            response_hint=response_hint,
            execution_summary=execution_summary,
            persona_name=persona_name,
            persona_profile=persona_profile,
            display_style_prompt=display_style_prompt,
        )
        fallback_speech = self._shorten_speech_text(display_text, speech_max_chars)
        if not fallback_speech:
            fallback_speech = self._shorten_speech_text(raw, speech_max_chars)
        return {
            "display_text": self._normalize_conversational_text(display_text or raw),
            "speech_text": fallback_speech or self._shorten_speech_text(display_text or raw, speech_max_chars),
        }

    @staticmethod
    def _normalize_output_kinds(outputs: list[Any]) -> list[str]:
        normalized: list[str] = []
        for item in outputs:
            if isinstance(item, OutputKind):
                value = item.value
            elif isinstance(item, str):
                key = item.strip().lower()
                value = OUTPUT_KIND_ALIASES.get(key, key)
            else:
                continue
            if value in ALLOWED_OUTPUT_VALUES and value not in normalized:
                normalized.append(value)
        return normalized

    @staticmethod
    def _normalize_task_goal(raw_goal: Any) -> dict[str, Any] | None:
        if not isinstance(raw_goal, dict):
            return None
        outputs = raw_goal.get("required_outputs", [])
        if not isinstance(outputs, list):
            outputs = []
        return {
            "summary": raw_goal.get("summary", ""),
            "required_outputs": OllamaClient._normalize_output_kinds(outputs),
        }

    @staticmethod
    def _normalize_tool_decision_payload(
            payload: dict[str, Any],
            fallback: ToolDecision | None = None,
    ) -> dict[str, Any]:
        """
        Normalize arbitrary LLM JSON into the one and only supported ToolDecision shape.

        Canonical top-level fields:
          - decision
          - intent
          - reason
          - selected_tool
          - arguments
          - risk_level
          - response_hint
          - memory_write
          - overall_task_goal
          - expected_step_outputs

        Important rule:
          - `decision` is the only top-level decision field.
          - `selected_tool` is the only top-level tool field.
          - legacy `action` is NOT treated as a first-class schema field anymore.
            It is only used as a last-resort compatibility hint for tool aliases.
        """
        normalized = dict(payload)

        # ----------------------------
        # 1) Common response text aliases
        # ----------------------------
        for alias_key in (
                "conclusion",
                "answer",
                "reply",
                "response",
                "content",
                "message",
                "text",
                "final_response",
        ):
            if alias_key in normalized and "response_hint" not in normalized:
                normalized["response_hint"] = normalized.get(alias_key)

        if "conclusion" in normalized and "reason" not in normalized:
            normalized["reason"] = str(
                normalized.get("conclusion", "")).strip() or "Model returned a direct conclusion."

        # ----------------------------
        # 2) Canonical decision field
        # ----------------------------
        decision = OllamaClient._normalize_decision_value(normalized.get("decision"))

        # Legacy compatibility:
        # only map legacy action -> decision for obvious conversational actions,
        # NOT for tool-like actions such as edit_docx / write_file / search_web.
        legacy_action = normalized.get("action")
        legacy_action_normalized = None
        if isinstance(legacy_action, str):
            legacy_action_normalized = legacy_action.strip().lower()

        if decision is None and legacy_action_normalized in {"respond", "reply", "answer", "response", "clarify",
                                                             "ask_user", "finish", "done"}:
            decision = DECISION_ALIASES.get(legacy_action_normalized)

        # ----------------------------
        # 3) Canonical tool field aliases
        # ----------------------------
        if "tool" in normalized and "selected_tool" not in normalized:
            normalized["selected_tool"] = normalized.pop("tool")

        if "tool_name" in normalized and "selected_tool" not in normalized:
            normalized["selected_tool"] = normalized.pop("tool_name")

        selected_tool = OllamaClient._normalize_selected_tool_value(normalized.get("selected_tool"))

        # Legacy compatibility:
        # If decision is already tool_call, and selected_tool is still missing,
        # allow a legacy tool-like `action` to become selected_tool.
        if selected_tool is None and decision == "tool_call" and legacy_action_normalized:
            selected_tool = OllamaClient._normalize_selected_tool_value(legacy_action_normalized)

        # ----------------------------
        # 4) Canonical arguments field aliases
        # ----------------------------
        if "parameters" in normalized and "arguments" not in normalized:
            normalized["arguments"] = normalized.pop("parameters")

        if "tool_input" in normalized and "arguments" not in normalized:
            normalized["arguments"] = normalized.pop("tool_input")

        arguments = normalized.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}

        # ----------------------------
        # 5) Infer decision from selected_tool only when safe
        # ----------------------------
        if decision is None:
            if selected_tool is not None:
                decision = "tool_call"
            else:
                decision = "respond"

        # ----------------------------
        # 6) Normalize risk level
        # ----------------------------
        risk_level = normalized.get("risk_level", RiskLevel.LOW.value)
        if isinstance(risk_level, RiskLevel):
            normalized_risk = risk_level.value
        elif isinstance(risk_level, str):
            lowered = risk_level.strip().lower()
            if lowered in {RiskLevel.LOW.value, RiskLevel.MEDIUM.value, RiskLevel.HIGH.value}:
                normalized_risk = lowered
            else:
                normalized_risk = RiskLevel.LOW.value
        else:
            normalized_risk = RiskLevel.LOW.value

        # ----------------------------
        # 7) Normalize goal aliases
        # ----------------------------
        if "task_goal" in normalized and "overall_task_goal" not in normalized:
            normalized["overall_task_goal"] = normalized.pop("task_goal")

        overall_task_goal = OllamaClient._normalize_task_goal(normalized.get("overall_task_goal"))

        expected_step_outputs = OllamaClient._normalize_output_kinds(
            normalized.get("expected_step_outputs", [])
            if isinstance(normalized.get("expected_step_outputs", []), list)
            else []
        )

        # ----------------------------
        # 8) Canonical intent / reason / response_hint
        # ----------------------------
        intent = normalized.get("intent")
        if not isinstance(intent, str) or not intent.strip():
            if selected_tool:
                intent = selected_tool.replace(".", "_")
            else:
                intent = "respond"

        reason = normalized.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            if decision == "tool_call" and selected_tool:
                reason = f"Use {selected_tool} to continue the task."
            elif decision == "clarify":
                reason = "The request still needs clarification."
            elif decision == "finish":
                reason = "The task is complete."
            else:
                reason = "Respond directly to the user."

        response_hint = normalized.get("response_hint")
        if not isinstance(response_hint, str):
            response_hint = None
        elif not response_hint.strip():
            response_hint = None
        else:
            response_hint = response_hint.strip()

        memory_write = normalized.get("memory_write")
        if not isinstance(memory_write, str) or not memory_write.strip():
            memory_write = None
        else:
            memory_write = memory_write.strip()

        # ----------------------------
        # 9) Strict top-level cleanup
        # ----------------------------
        canonical = {
            "decision": decision,
            "intent": intent.strip(),
            "reason": reason.strip(),
            "selected_tool": selected_tool,
            "arguments": arguments,
            "risk_level": normalized_risk,
            "response_hint": response_hint,
            "memory_write": memory_write,
            "overall_task_goal": overall_task_goal,
            "expected_step_outputs": expected_step_outputs,
        }

        # ----------------------------
        # 10) Tool-specific argument lifting
        # ----------------------------
        selected_tool = canonical["selected_tool"]
        arguments = canonical["arguments"]

        if selected_tool == "file.write":
            for field in ("path", "content", "encoding", "overwrite"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "file.write_many":
            if "items" not in arguments and "item" in normalized:
                item_value = normalized["item"]
                arguments["items"] = item_value if isinstance(item_value, list) else [item_value]
            if "continue_on_error" in normalized and "continue_on_error" not in arguments:
                arguments["continue_on_error"] = normalized["continue_on_error"]

        elif selected_tool == "file.append":
            for field in ("path", "content", "encoding", "create"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "file.append_many":
            if "items" not in arguments and "item" in normalized:
                item_value = normalized["item"]
                arguments["items"] = item_value if isinstance(item_value, list) else [item_value]
            if "continue_on_error" in normalized and "continue_on_error" not in arguments:
                arguments["continue_on_error"] = normalized["continue_on_error"]

        elif selected_tool == "file.read":
            if "paths" not in arguments and "path" in normalized:
                path_value = normalized["path"]
                arguments["paths"] = [path_value] if isinstance(path_value, str) else path_value
            for field in ("encoding", "max_bytes"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "file.extract_text":
            if "paths" not in arguments and "path" in normalized:
                path_value = normalized["path"]
                arguments["paths"] = [path_value] if isinstance(path_value, str) else path_value
            for field in ("encoding", "max_chars", "max_rows_per_sheet"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "file.extract_structure":
            for field in ("path", "encoding", "include_text", "max_blocks", "max_chars_per_block",
                          "max_rows_per_sheet"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "file.search_blocks":
            for field in ("path", "query", "terms", "max_matches", "max_blocks", "max_chars_per_block"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "file.search_by_name":
            for field in ("path", "query", "recursive", "target_kind", "extensions", "include_dirs", "top_k",
                          "query_terms", "alias_terms", "scope_mode"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "file.list":
            for field in ("path", "recursive", "include_dirs", "patterns"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool in {"file.metadata", "file.preview", "file.open_path", "file.reveal_in_explorer"}:
            for field in ("path", "encoding", "max_chars", "max_children"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool in {"file.metadata_many", "file.open_many", "file.reveal_many"}:
            if "paths" not in arguments and "path" in normalized:
                path_value = normalized["path"]
                arguments["paths"] = [path_value] if isinstance(path_value, str) else path_value
            if "continue_on_error" in normalized and "continue_on_error" not in arguments:
                arguments["continue_on_error"] = normalized["continue_on_error"]

        elif selected_tool == "file.preview_many":
            if "paths" not in arguments and "path" in normalized:
                path_value = normalized["path"]
                arguments["paths"] = [path_value] if isinstance(path_value, str) else path_value
            for field in ("encoding", "max_chars", "max_children", "continue_on_error"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "file.mkdir":
            for field in ("path", "exist_ok", "parents"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "file.mkdir_many":
            if "paths" not in arguments and "path" in normalized:
                path_value = normalized["path"]
                arguments["paths"] = [path_value] if isinstance(path_value, str) else path_value
            for field in ("exist_ok", "parents", "continue_on_error"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool in {"file.copy", "file.move"}:
            for field in ("src_path", "dest_path", "overwrite"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool in {"file.copy_many", "file.move_many", "file.rename_many"}:
            if "items" not in arguments and "item" in normalized:
                item_value = normalized["item"]
                arguments["items"] = item_value if isinstance(item_value, list) else [item_value]
            if "continue_on_error" in normalized and "continue_on_error" not in arguments:
                arguments["continue_on_error"] = normalized["continue_on_error"]

        elif selected_tool == "file.rename":
            for field in ("path", "new_name", "overwrite"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "file.delete":
            for field in ("path", "recursive", "missing_ok"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "file.delete_many":
            if "paths" not in arguments and "path" in normalized:
                path_value = normalized["path"]
                arguments["paths"] = [path_value] if isinstance(path_value, str) else path_value
            for field in ("recursive", "missing_ok", "continue_on_error"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "file.edit_docx":
            for field in ("source_path", "output_path", "edits", "overwrite"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "file.render_docx_from_template":
            for field in ("template_path", "output_path", "source_path", "content", "paragraphs", "title", "overwrite"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "document_agent.summarize":
            for field in ("source_path", "instruction", "recent_context", "grounded_inputs", "resolved_facts", "source_materials", "constraints", "style_hints", "max_chars"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "document_agent.read":
            for field in ("source_path", "instruction", "recent_context", "grounded_inputs", "resolved_facts", "source_materials", "constraints", "style_hints", "max_chars"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "document_agent.inspect":
            for field in ("source_path", "instruction", "recent_context", "grounded_inputs", "resolved_facts", "source_materials", "constraints", "style_hints", "max_blocks", "max_chars_per_block", "max_matches"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "document_agent.edit":
            for field in (
                "source_path",
                "output_path",
                "instruction",
                "recent_context",
                "grounded_inputs",
                "resolved_facts",
                "source_materials",
                "constraints",
                "style_hints",
                "allow_overwrite",
                "preserve_structure",
                "preserve_style",
                "max_chars",
                "max_blocks",
                "max_chars_per_block",
            ):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "retrieval.search_local_objects":
            for field in ("query", "target_kind", "top_k", "path_scope", "scope_mode", "extensions",
                          "rebuild_if_missing", "query_terms", "alias_terms"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "retrieval.inspect_local_candidate":
            for field in ("path", "max_chars", "max_children"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "image.inspect":
            for field in ("path", "include_ocr", "ocr_max_chars"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "image.describe":
            for field in ("path", "prompt", "focus", "include_ocr", "ocr_max_chars", "max_description_chars"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "image.read_text":
            if "paths" not in arguments and "path" in normalized:
                path_value = normalized["path"]
                arguments["paths"] = [path_value] if isinstance(path_value, str) else path_value
            if "max_chars" in normalized and "max_chars" not in arguments:
                arguments["max_chars"] = normalized["max_chars"]

        elif selected_tool == "image.capture_screen":
            for field in ("output_path", "delay_ms", "all_screens"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "image.capture_region":
            for field in ("output_path", "x", "y", "width", "height", "delay_ms"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "web.search":
            for field in ("query", "max_results", "domains", "preferred_domains", "recency_days", "language", "query_terms", "alias_terms"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "web.fetch":
            for field in ("url", "max_chars", "allow_insecure", "prefer_browser"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "web.open_page":
            if "url" in normalized and "url" not in arguments:
                arguments["url"] = normalized["url"]

        elif selected_tool == "web.research":
            for field in (
                    "query",
                    "max_results",
                    "max_pages",
                    "max_chars",
                    "domains",
                    "preferred_domains",
                    "recency_days",
                    "language",
                    "allow_insecure",
                    "prefer_browser",
                    "query_terms",
                    "alias_terms",
            ):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "qq.get_recent_messages":
            for field in ("limit", "include_assistant"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "qq.get_last_reply":
            if "contact_query" in normalized and "contact_query" not in arguments:
                arguments["contact_query"] = normalized["contact_query"]

        elif selected_tool == "qq.search_history":
            for field in ("query", "contact_query", "limit", "reply_after_last_outbound"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "qq.get_recent_attachments":
            for field in ("contact_query", "kind", "limit"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "qq.search_contacts":
            for field in ("query", "limit"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "qq.send_text":
            for field in ("contact_query", "message", "text", "target_kind"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]
            if "message" not in arguments and isinstance(arguments.get("text"), str):
                arguments["message"] = arguments["text"]

        elif selected_tool == "qq.send_file":
            for field in ("contact_query", "file_path", "target_kind"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        elif selected_tool == "qq.send_voice":
            for field in ("contact_query", "speech_text", "audio_path", "target_kind"):
                if field in normalized and field not in arguments:
                    arguments[field] = normalized[field]

        canonical["arguments"] = arguments

        # ----------------------------
        # 11) Enforce non-tool decisions to be schema-clean
        # ----------------------------
        if canonical["decision"] != "tool_call":
            canonical["selected_tool"] = None
            canonical["arguments"] = {}

        return canonical

    @staticmethod
    def _compact_grounded_inputs(grounded_inputs: dict[str, Any] | None, *, max_chars: int = 1200) -> dict[str, Any]:
        if not isinstance(grounded_inputs, dict) or not grounded_inputs:
            return {}
        compact: dict[str, Any] = {}
        for key, value in grounded_inputs.items():
            normalized_key = str(key or "").strip()
            if not normalized_key:
                continue
            if isinstance(value, str):
                text = value.strip()
                if text:
                    compact[normalized_key] = text[:max_chars]
                continue
            if isinstance(value, (int, float, bool)) or value is None:
                compact[normalized_key] = value
                continue
            if isinstance(value, list):
                items: list[Any] = []
                for item in value[:8]:
                    if isinstance(item, str):
                        stripped = item.strip()
                        if stripped:
                            items.append(stripped[:240])
                    elif isinstance(item, (int, float, bool)) or item is None:
                        items.append(item)
                if items:
                    compact[normalized_key] = items
                continue
            if isinstance(value, dict):
                nested: dict[str, Any] = {}
                for nested_key, nested_value in list(value.items())[:8]:
                    nested_name = str(nested_key or "").strip()
                    if not nested_name:
                        continue
                    if isinstance(nested_value, str):
                        stripped = nested_value.strip()
                        if stripped:
                            nested[nested_name] = stripped[:240]
                    elif isinstance(nested_value, (int, float, bool)) or nested_value is None:
                        nested[nested_name] = nested_value
                if nested:
                    compact[normalized_key] = nested
        return compact

    @staticmethod
    def _infer_goal_fields(payload: dict[str, Any], tool_manifests: list[ToolManifest]) -> dict[str, Any]:
        selected_tool = payload.get("selected_tool")
        if not selected_tool:
            return payload

        manifest = next((tool for tool in tool_manifests if tool.tool_name == selected_tool), None)
        if manifest is None or not manifest.produces:
            return payload

        updated = dict(payload)
        if not updated.get("overall_task_goal"):
            updated["overall_task_goal"] = {
                "summary": f"Complete the current {updated.get('intent', 'task')}.",
                "required_outputs": [output_kind.value for output_kind in manifest.produces],
            }
        if not updated.get("expected_step_outputs"):
            updated["expected_step_outputs"] = [output_kind.value for output_kind in manifest.produces]
        return updated

    def decide(
            self,
            messages: list[Message],
            tool_manifests: list[ToolManifest],
            observations: list[str],
            allowed_decisions=None,
            bound_workflow_family: str | None = None,
    ) -> ToolDecision:
        tools_json = json.dumps(
            [tool.model_dump(mode="json") for tool in tool_manifests],
            ensure_ascii=False,
            indent=2,
        )
        conversation = "\n".join(f"{message.role.value}: {message.content}" for message in messages[-8:])
        observation_text = "\n".join(observations[-6:]) if observations else "No tool results yet."

        system_prompt = (
            "You are the decision kernel for a local agent.\n"
            "Return exactly one JSON object for the next step.\n"
            "No markdown. No explanations. No code fences.\n"
            "Keep overall_task_goal for the full user deliverable and expected_step_outputs for this step only.\n"
            "All listed tools are callable. Choose the narrowest valid tool that directly serves the request.\n"
            "Prefer grounded lookup/read tools before write tools.\n"
            "For QQ history questions, prefer specific QQ history tools over generic recent-message tools.\n"
            "For local document editing, prefer structure/edit tools over treating office files as plain text.\n"
            "If observations already contain a grounded state-machine next step, follow it unless you have a clearly better valid action.\n"
            "decision must be one of respond, tool_call, clarify, finish.\n"
            "risk_level must be one of low, medium, high.\n"
            "arguments must always be a JSON object.\n"
            "If decision is tool_call, selected_tool must be valid and arguments must be executable.\n"
            "If decision is respond, clarify, or finish, selected_tool must be null and arguments must be {}.\n"
            "Required schema:\n"
            "{\n"
            '  "decision": "respond|tool_call|clarify|finish",\n'
            '  "intent": "short_intent_name",\n'
            '  "reason": "short reason",\n'
            '  "selected_tool": "tool.name.or.null",\n'
            '  "arguments": {},\n'
            '  "risk_level": "low|medium|high",\n'
            '  "response_hint": "short hint or null",\n'
            '  "memory_write": "string or null",\n'
            '  "overall_task_goal": {"summary": "full deliverable", "required_outputs": ["output_a", "output_b"]} or null,\n'
            '  "expected_step_outputs": ["output_x", "output_y"]\n'
            "}\n"
            "Return only the JSON object."
        )
        user_prompt = (
            f"Available tools:\n{tools_json}\n\n"
            f"Recent conversation:\n{conversation}\n\n"
            f"Recent observations:\n{observation_text}\n\n"
            "Return exactly one JSON object for the next step."
        )
        raw = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        payload = self._normalize_tool_decision_payload(self._extract_json_block(raw))
        payload = self._infer_goal_fields(payload, tool_manifests)
        return ToolDecision.model_validate(payload)

    def plan_workflow_spec(
        self,
        *,
        messages: list[Message],
        tool_manifests: list[ToolManifest],
        intent_bundle: dict[str, Any] | None = None,
        recent_context: str = "",
        max_nodes: int = 6,
    ) -> WorkflowSpec:
        tool_summaries = [
            {
                "tool_name": tool.tool_name,
                "description": tool.description,
                "side_effect": tool.side_effect,
                "produces": [item.value for item in tool.produces],
            }
            for tool in tool_manifests
        ]
        conversation = "\n".join(f"{message.role.value}: {message.content}" for message in messages[-8:])
        payload = self._chat_json_object(
            task_label="workflow spec planner",
            schema=(
                '{"workflow_name":"...","goal":{"summary":"...","required_outputs":["..."],"completion_mode":"outputs"},'
                '"nodes":[{"node_id":"step_1","tool":"tool.name|null","intent":"...","reason":"...",'
                '"requires":["..."],"produces":["..."]}]}'
            ),
            rules=[
                "Create the whole workflow before execution starts.",
                "Use only tools from tool_manifests, or null only for the final response node.",
                "Nodes must be in execution order.",
                "Do not include tool arguments; arguments are planned later for each locked node.",
                "requires and produces must contain only values from allowed_output_values; never use raw slots such as query, location, user_request, date, or city there.",
                "Prefer specialized sub-agent tools for complex domains, such as document_agent.edit for document editing.",
                "For tasks that combine research and document editing, include research/time/file-location nodes before document_agent.edit.",
                "For weather or forecast requests, do not add the current year or words like latest to the query; if no place is known from the user request or recent context, prefer a final response node that asks for the city/location.",
                "Keep the workflow minimal and no longer than max_nodes.",
                "The final non-response node should produce all outputs needed by goal.required_outputs.",
            ],
            payload={
                "conversation": conversation,
                "recent_context": recent_context,
                "intent_bundle": intent_bundle or {},
                "tool_manifests": tool_summaries,
                "allowed_output_values": sorted(ALLOWED_OUTPUT_VALUES),
                "max_nodes": max_nodes,
            },
            model=self.model,
        )
        nodes = payload.get("nodes")
        if isinstance(nodes, list):
            payload["nodes"] = nodes[:max(1, max_nodes)]
        goal = payload.get("goal")
        if isinstance(goal, dict):
            goal["required_outputs"] = self._normalize_output_kinds(goal.get("required_outputs", []))
        return WorkflowSpec.model_validate(payload)

    def render_response(
        self,
        system_name: str,
        messages: list[Message],
        observations: list[str],
        response_hint: str | None,
        execution_summary: dict[str, Any] | None = None,
        persona_name: str | None = None,
        persona_profile: str | None = None,
        display_style_prompt: str | None = None,
    ) -> str:
        conversation = "\n".join(f"{message.role.value}: {message.content}" for message in messages[-10:])
        observation_text = "\n".join(observations[-8:]) if observations else "No extra observations."
        normalized_execution_summary = self._inject_local_time_hints(execution_summary)
        execution_text = json.dumps(normalized_execution_summary, ensure_ascii=False, indent=2)
        persona_block = self._build_persona_block(persona_name, persona_profile)

        system_prompt = (
            f"You are the outward response layer for {system_name}. "
            "Based on the conversation and tool results, reply in natural, fluent Chinese that is informative and easy to scan. "
            "Cover the main answer and the most relevant supporting details, but do not ramble or over-explain. "
            "If execution_summary.document_request.wants_document is true, format the answer like a compact document with a title and 3 to 5 short sections or grouped bullets. "
            "Do not expose JSON, schemas, protocol details, or internal reasoning. "
            "You must stay grounded in the execution summary. "
            "Never claim that a file was written, saved, deleted, or modified unless execution_summary.successful_actions explicitly includes that successful tool action. "
            "If execution_summary.task_status is partial, say clearly that part of the task is still unfinished. "
            "If execution_summary.web_sources contains sources, briefly cite one to three of the most relevant sources using their title or domain in the answer."
            " For direct QQ history questions, answer the conclusion first in plain Chinese, then add one short evidence sentence if available."
            " Avoid report-like openings such as '根据记录' unless the user explicitly asked for a summary or report."
            " For scheduled-task or reminder results, if display_when_local is present, use that local display time instead of raw UTC when_iso when telling the user when it will trigger.Never describe a UTC timestamp as local time."
            f"{persona_block}"
            f"\n{display_style_prompt or ''}".rstrip()
        )
        user_prompt = (
            f"Recent conversation:\n{conversation}\n\n"
            f"Tool observations:\n{observation_text}\n\n"
            f"Execution summary:\n{execution_text}\n\n"
            f"Hint:\n{response_hint or ''}"
        )
        raw = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.response_model,
        )
        return self._normalize_conversational_text(raw)

    def render_clarification_hint(
        self,
        *,
        user_text: str,
        intent: str,
        missing_slots: list[str],
        style_hint: str = "",
    ) -> str:
        payload = self._chat_json_object(
            task_label="clarification hint writer",
            schema='{"text":"string"}',
            rules=[
                "Write exactly one natural Chinese clarification reply.",
                "Ask only for the missing information needed to continue.",
                "Keep it concise, concrete, and conversational.",
                "Do not mention schemas, JSON, slot names, or internal workflow terms directly.",
            ],
            payload={
                "user_text": user_text,
                "intent": intent,
                "missing_slots": missing_slots,
                "style_hint": style_hint,
            },
            model=self.response_model,
        )
        text = self._normalize_conversational_text(str((payload or {}).get("text", "") or ""))
        if not text:
            raise ValueError("clarification hint text is empty")
        return text

    @staticmethod
    def _sanitize_response_text(text: str) -> str:
        cleaned = str(text or "")
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"^.*?</think>\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"^\s*Rendering the response\.\.\.\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*(#{1,6}\s*)思考过程\s*$", r"\1总结", cleaned, flags=re.MULTILINE)
        return cleaned.strip()

    @staticmethod
    def _format_user_datetime(when_iso: str, timezone_name: str = "Asia/Shanghai") -> str:
        dt = datetime.fromisoformat(str(when_iso).strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(timezone_name))
        local_dt = dt.astimezone(ZoneInfo(timezone_name))
        return local_dt.strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def _inject_local_time_hints(cls, execution_summary: dict[str, Any] | None) -> dict[str, Any]:
        summary = dict(execution_summary or {})
        successful_actions = summary.get("successful_actions")
        if not isinstance(successful_actions, list):
            return summary

        rewritten_actions: list[dict[str, Any]] = []
        reminder_render_hints: list[dict[str, str]] = []

        for action in successful_actions:
            if not isinstance(action, dict):
                rewritten_actions.append(action)
                continue

            tool_name = str(action.get("tool_name", "") or "").strip()
            data = action.get("data")
            if not isinstance(data, dict):
                rewritten_actions.append(action)
                continue

            rewritten_action = dict(action)
            rewritten_data = dict(data)

            if tool_name in {"system.create_reminder", "system.create_scheduled_task"}:
                task = rewritten_data.get("task")
                if isinstance(task, dict):
                    rewritten_task = dict(task)
                    when_iso = str(rewritten_task.get("when_iso", "") or "").strip()
                    timezone_name = str(
                        rewritten_task.get("timezone", "") or "Asia/Shanghai").strip() or "Asia/Shanghai"
                    message = str(rewritten_task.get("message", "") or "").strip()

                    if when_iso:
                        try:
                            display_when_local = cls._format_user_datetime(when_iso, timezone_name)
                            rewritten_task["display_when_local"] = display_when_local
                            rewritten_task["display_timezone"] = timezone_name
                            reminder_render_hints.append(
                                {
                                    "message": message,
                                    "display_when_local": display_when_local,
                                    "display_timezone": timezone_name,
                                }
                            )
                        except Exception:
                            pass

                    rewritten_data["task"] = rewritten_task

            rewritten_action["data"] = rewritten_data
            rewritten_actions.append(rewritten_action)

        summary["successful_actions"] = rewritten_actions
        if reminder_render_hints:
            summary["reminder_render_hints"] = reminder_render_hints
        return summary

    @staticmethod
    def _build_intent_context_payload(
        *,
        user_text: str,
        recent_context: str = "",
        hot_context_summary: str = "",
        warm_memory_summary: str = "",
        cold_memory_summary: str = "",
        active_task_summary: str = "",
        channel_context_summary: str = "",
        layered_context_summary: str = "",
        extra: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {"user_text": user_text}
        if layered_context_summary.strip():
            payload["layered_context_summary"] = layered_context_summary.strip()
        if channel_context_summary.strip():
            payload["channel_context_summary"] = channel_context_summary.strip()
        if recent_context.strip():
            payload["recent_context"] = recent_context.strip()
        if hot_context_summary.strip():
            payload["hot_context_summary"] = hot_context_summary.strip()
        if warm_memory_summary.strip():
            payload["warm_memory_summary"] = warm_memory_summary.strip()
        if cold_memory_summary.strip():
            payload["cold_memory_summary"] = cold_memory_summary.strip()
        if active_task_summary.strip():
            payload["active_task_summary"] = active_task_summary.strip()
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def analyze_document_delivery(
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
    ) -> dict[str, Any]:
        compact = self._compact_lightweight_context(
            user_text=user_text,
            recent_context=recent_context,
            hot_context_summary=hot_context_summary,
            warm_memory_summary=warm_memory_summary,
            cold_memory_summary=cold_memory_summary,
            active_task_summary=active_task_summary,
        )
        system_prompt = (
            "You extract structured delivery intent for a local agent. "
            "Return exactly one JSON object. "
            "Focus on whether the user wants the final answer organized like a document, "
            "whether they want it saved to a file, the preferred output format, an explicit output file if given, "
            "what kind of artifact they want (document, spreadsheet, or slides), "
            "and a concise suitable document title when appropriate. "
            "Use recent conversation and memory to resolve references like 'that file' or 'the previous one', but do not invent missing paths. "
            "Do not guess a file path unless the user explicitly provided one. "
            "If the user mentions a local file path as the thing to open, read, summarize, or inspect, do not treat that path as output_file. "
            "Only set output_file when the request explicitly says to save, export, write, or store the result to that file. "
            "If the user wants a document-style answer shown in chat but did not explicitly ask to save it, set wants_document=true and save_output=false. "
            "Questions about past QQ chat content, such as asking whether someone contacted the agent before or what they said, are not document requests by themselves unless the user explicitly asks for a report, summary file, or saved artifact. "
            "Prefer layered_context_summary as the main cross-layer view when it is present, and use the raw per-layer fields to verify details. "
            'Return schema: {"wants_document": bool, "save_output": bool, "artifact_type": "document|spreadsheet|slides|null", "output_format": "docx|xlsx|pptx|md|txt|null", "output_file": "string|null", "title": "string|null", "confidence": 0..1, "rationale": "short reason"}'
        )
        user_prompt = (
            "Intent context:\n"
            + self._build_intent_context_payload(
                user_text=compact["user_text"],
                recent_context=compact["recent_context"],
                hot_context_summary=compact["hot_context_summary"],
                warm_memory_summary=compact["warm_memory_summary"],
                cold_memory_summary=compact["cold_memory_summary"],
                active_task_summary=compact["active_task_summary"],
                channel_context_summary=channel_context_summary,
                layered_context_summary=layered_context_summary,
            )
            + "\n\nReturn JSON only."
        )
        raw = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.response_model,
        )
        payload = self._normalize_document_delivery_payload(self._extract_json_block(raw))
        return DocumentDeliveryIntent.model_validate(payload).model_dump(mode="json")

    def analyze_knowledge_request(
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
    ) -> dict[str, Any]:
        compact = self._compact_lightweight_context(
            user_text=user_text,
            recent_context=recent_context,
            hot_context_summary=hot_context_summary,
            warm_memory_summary=warm_memory_summary,
            cold_memory_summary=cold_memory_summary,
            active_task_summary=active_task_summary,
        )
        system_prompt = (
            "You classify whether a local agent should ground a user request with web research. "
            "Return exactly one JSON object. "
            "Set needs_grounding=true when the request asks about an external real-world topic and the answer should be grounded in web facts or would benefit from sourced lookup. "
            "Set time_sensitive=true for requests involving latest, current, today, recent, market, price, news, or other changing information. "
            "Set lookup_requested=true when the user explicitly asks to search, look up, check, find, collect info, or investigate. "
            "Use knowledge_type from: time_sensitive_external_topic, general_external_topic, local_workspace, qq_history, casual_chat, unknown. "
            "Requests about QQ chat history, past replies, relayed-message follow-ups, or previously sent attachments should use knowledge_type=qq_history and should not be grounded with web research. "
            "This includes natural phrasings like asking whether a person contacted the agent before, whether they ever messaged the agent, what they said earlier, what was discussed before, or what someone replied with in QQ. "
            "Requests about local files, workspace paths, or editing code should use local_workspace and should not be grounded with web research. "
            "If the user mentions a local file or path and asks to add, write, modify, update, append, or follow a previous style, prefer local_workspace even when the utterance also contains words like previous, earlier, or style reference. "
            "Do not classify a request as qq_history merely because recent context contains QQ-related messages; use qq_history only when the user is actually asking about past chat content, replies, contacts, or attachments. "
            "If the utterance mentions a person name or nickname together with questions like whether they contacted you before or what they said, prefer qq_history over web even if the phrasing is short and informal. "
            "Use recent conversation, hot context, and recalled memory before deciding. "
            "Use knowledge_type=system_utility when the user is asking you to use a local system capability such as:telling the current time/date/weekday;creating, listing, or cancelling reminders, alarms, or timers. Do NOT use knowledge_type=system_utility when the user is asking for conceptual knowledge about those things, such as: how alarms work, how to implement reminders; definitions, principles, tutorials, or comparisons"
            "Personal fact updates or memory notes such as birthdays, names, preferences, or '记一下/记住这个事实' should usually be knowledge_type=casual_chat unless the user explicitly asks to create a timed reminder, save to a file, or edit a local document. "
            "If the latest utterance contains pronouns or vague references, resolve them from context instead of assuming it is a web request. "
            "Prefer layered_context_summary as the main cross-layer view when it is present, and use the raw per-layer fields to verify details. "
            'Return schema: {"needs_grounding": bool, "time_sensitive": bool, "lookup_requested": bool, ' '"knowledge_type": "time_sensitive_external_topic|general_external_topic|local_workspace|qq_history|casual_chat|system_utility|unknown", ''"confidence": 0..1, "rationale": "short reason"}'
        )
        user_prompt = (
            "Intent context:\n"
            + self._build_intent_context_payload(
                user_text=compact["user_text"],
                recent_context=compact["recent_context"],
                hot_context_summary=compact["hot_context_summary"],
                warm_memory_summary=compact["warm_memory_summary"],
                cold_memory_summary=compact["cold_memory_summary"],
                active_task_summary=compact["active_task_summary"],
                channel_context_summary=channel_context_summary,
                layered_context_summary=layered_context_summary,
            )
            + "\n\nReturn JSON only."
        )
        raw = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.response_model,
        )
        payload = self._normalize_knowledge_request_payload(self._extract_json_block(raw))
        return KnowledgeRequestIntent.model_validate(payload).model_dump(mode="json")

    def analyze_site_search(
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
    ) -> dict[str, Any]:
        compact = self._compact_lightweight_context(
            user_text=user_text,
            recent_context=recent_context,
            hot_context_summary=hot_context_summary,
            warm_memory_summary=warm_memory_summary,
            cold_memory_summary=cold_memory_summary,
            active_task_summary=active_task_summary,
        )
        system_prompt = (
            "You extract site-specific search intent for a local agent. "
            "Return exactly one JSON object. "
            "Detect requests like searching on Zhihu, Bilibili, or GitHub. "
            "Use site from: zhihu, bilibili, github, null. "
            "Use content_type from: generic, article, question, video, repo. "
            "Use site_scope=required only when the user explicitly asks to search within, only use, or open a named site. "
            "Use site_scope=preferred when a site is merely a likely useful source but the user did not restrict the search. "
            "Use site_scope=none when no site should influence search. "
            "Set action=search unless the user explicitly wants to open the first result. "
            "Use conversation context only to resolve a previously mentioned site or query target; otherwise prefer null. "
            "Prefer layered_context_summary as the main cross-layer view when it is present, and use the raw per-layer fields to verify details. "
            'Return schema: {"site": "zhihu|bilibili|github|null", "query": "string|null", "content_type": "generic|article|question|video|repo", "action": "search|open_first", "site_scope": "none|preferred|required", "open_first": bool, "confidence": 0..1, "rationale": "short reason"}'
        )
        user_prompt = (
            "Intent context:\n"
            + self._build_intent_context_payload(
                user_text=compact["user_text"],
                recent_context=compact["recent_context"],
                hot_context_summary=compact["hot_context_summary"],
                warm_memory_summary=compact["warm_memory_summary"],
                cold_memory_summary=compact["cold_memory_summary"],
                active_task_summary=compact["active_task_summary"],
                channel_context_summary=channel_context_summary,
                layered_context_summary=layered_context_summary,
            )
            + "\n\nReturn JSON only."
        )
        raw = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.response_model,
        )
        payload = self._normalize_site_search_payload(self._extract_json_block(raw))
        return SiteSearchIntent.model_validate(payload).model_dump(mode="json")

    def analyze_memory_candidate_intent(
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
    ) -> dict[str, Any]:
        compact = self._compact_lightweight_context(
            user_text=user_text,
            recent_context=recent_context,
            hot_context_summary=hot_context_summary,
            warm_memory_summary=warm_memory_summary,
            cold_memory_summary=cold_memory_summary,
            active_task_summary=active_task_summary,
        )
        payload = self._chat_json_object(
            task_label="memory candidate intent analyzer",
            schema=(
                '{"is_memory_candidate":false,'
                '"scope":"none|turn|session|persistent",'
                '"kind":"none|user_fact|naming|preference|workflow_method|tool_policy|correction|style|boundary",'
                '"apply_this_turn":false,'
                '"persist_memory":false,'
                '"should_write_memory":false,'
                '"overwrite_existing":false,'
                '"normalized_text":"string|null",'
                '"memory_text":"string|null",'
                '"memory_key":"string|null",'
                '"canonical_value":{"value":"..."},'
                '"preferred_families":["document_operation","document_summary","file_delivery","local_lookup","file_lookup","local_collection","qq_history","web_target","web_lookup","system_utility","delivery"],'
                '"blocked_families":["document_operation","document_summary","file_delivery","local_lookup","file_lookup","local_collection","qq_history","web_target","web_lookup","system_utility","delivery"],'
                '"preferred_tools":["tool.name"],'
                '"response_style":"default|concise|direct_answer_first|conversational|formal|null",'
                '"confidence":0.0,'
                '"rationale":"short reason"}'
            ),
            rules=[
                "Decide whether the latest user message contains a reusable memory candidate for the main agent, including user facts, naming preferences, style preferences, workflow methods, tool policies, corrections, or boundaries.",
                "Use kind=user_fact for profile-like facts the user is telling the agent to remember, such as birthdays, names, relationships, identities, or stable background facts.",
                "Use naming/preference/workflow_method/tool_policy/correction/style/boundary when the content is an instruction about how the agent should behave.",
                "scope=turn for one-turn guidance that should shape this execution but not be kept long-term.",
                "scope=session for instructions or facts that should matter in this conversation but are not clearly durable.",
                "scope=persistent for long-term user facts or durable behavioral preferences the agent should remember later.",
                "Set apply_this_turn=true whenever the memory candidate should influence the current turn.",
                "Set persist_memory=true and should_write_memory=true only when the content should really be remembered beyond this turn.",
                "Set overwrite_existing=true when the user is clearly correcting an earlier remembered fact or preference.",
                "Provide memory_key when you can express the stable key, such as user.birthday, user.preferred_name, user.response_style, or user.workflow_preference.local_first.",
                "Provide canonical_value as a small structured object when it helps later reuse, for example {'relative_day':'tomorrow'} or {'preferred_name':'阿榆'}.",
                "For workflow_method or tool_policy candidates, populate preferred_families, blocked_families, or preferred_tools when that helps execution this turn.",
                "Examples: '先调用时间模块再判断季节' should usually be is_memory_candidate=true, scope=turn, kind=workflow_method, apply_this_turn=true, persist_memory=false.",
                "Examples: '以后你称呼我阿榆' should usually be is_memory_candidate=true, scope=persistent, kind=naming, apply_this_turn=true, persist_memory=true, should_write_memory=true.",
                "Examples: '我的生日是明天，你也记一下' should usually be is_memory_candidate=true, scope=persistent or session, kind=user_fact, apply_this_turn=true, persist_memory=true, should_write_memory=true.",
                "Do not force ordinary questions into memory mode. If the latest message is just a normal question or chat turn, return is_memory_candidate=false and scope=none.",
                "Prefer layered_context_summary as the main cross-layer view when it is present, and use the raw per-layer fields to verify details.",
            ],
            payload=(
                "Intent context:\n"
                + self._build_intent_context_payload(
                    user_text=compact["user_text"],
                    recent_context=compact["recent_context"],
                    hot_context_summary=compact["hot_context_summary"],
                    warm_memory_summary=compact["warm_memory_summary"],
                    cold_memory_summary=compact["cold_memory_summary"],
                    active_task_summary=compact["active_task_summary"],
                    channel_context_summary=channel_context_summary,
                    layered_context_summary=layered_context_summary,
                )
                + "\n\nReturn JSON only."
            ),
            model=self.response_model,
        )
        normalized = self._normalize_memory_candidate_intent_payload(payload)
        return MemoryCandidateIntent.model_validate(normalized).model_dump(mode="json")

    def analyze_instruction_intent(
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
    ) -> dict[str, Any]:
        payload = self.analyze_memory_candidate_intent(
            user_text,
            recent_context=recent_context,
            hot_context_summary=hot_context_summary,
            warm_memory_summary=warm_memory_summary,
            cold_memory_summary=cold_memory_summary,
            active_task_summary=active_task_summary,
            channel_context_summary=channel_context_summary,
            layered_context_summary=layered_context_summary,
        )
        kind = str(payload.get("kind", "") or "").strip().lower()
        is_instruction = bool(payload.get("is_memory_candidate")) and kind in {
            "naming",
            "preference",
            "workflow_method",
            "tool_policy",
            "correction",
            "style",
            "boundary",
        }
        projected = {
            "is_instruction": is_instruction,
            "scope": payload.get("scope", "none"),
            "kind": kind if is_instruction else "none",
            "apply_this_turn": bool(payload.get("apply_this_turn")) if is_instruction else False,
            "persist_memory": bool(payload.get("persist_memory")) if is_instruction else False,
            "normalized_instruction": payload.get("normalized_text") if is_instruction else None,
            "memory_text": payload.get("memory_text") if is_instruction else None,
            "preferred_families": payload.get("preferred_families", []) if is_instruction else [],
            "blocked_families": payload.get("blocked_families", []) if is_instruction else [],
            "preferred_tools": payload.get("preferred_tools", []) if is_instruction else [],
            "response_style": payload.get("response_style") if is_instruction else None,
            "confidence": payload.get("confidence", 0.0) if is_instruction else 0.0,
            "rationale": payload.get("rationale", ""),
        }
        normalized = self._normalize_instruction_intent_payload(projected)
        return InstructionIntent.model_validate(normalized).model_dump(mode="json")

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
    ) -> dict[str, Any]:
        compact = self._compact_lightweight_context(
            user_text=user_text,
            recent_context=recent_context,
            hot_context_summary=hot_context_summary,
            warm_memory_summary=warm_memory_summary,
            cold_memory_summary=cold_memory_summary,
            active_task_summary=active_task_summary,
        )
        payload = self._chat_json_object(
            task_label="task graph analyzer",
            schema=(
                '{"is_multi_task":false,'
                '"primary_task_text":"string|null",'
                '"primary_task_id":"string|null",'
                '"needs_clarification":false,'
                '"followup_text":"string|null",'
                '"subtasks":['
                '{"task_id":"task_1","summary":"short summary","task_text":"user-facing task text",'
                '"kind":"generic|web_lookup|document_edit|local_lookup|direct_answer|qq_history|system_utility|delivery",'
                '"status":"ready|waiting_for_input|blocked|completed",'
                '"missing_slots":["content"],'
                '"slot_values":{"content":"...","date_rule":"today"},'
                '"rationale":"short reason"}],'
                '"confidence":0.0,'
                '"rationale":"short reason"}'
            ),
            rules=[
                "Decompose the latest user request into ordered subtasks only when the utterance truly contains multiple actionable asks or when recent same-turn message fragments clearly extend earlier asks.",
                "Use channel_context_summary and layered_context_summary to recover fragmented same-turn messages, especially QQ-style sequential messages that were sent seconds apart.",
                "When a short fragment is clearly a supplement to the immediately previous subtask, merge it into that subtask's task_text or slot_values instead of creating a brand new task.",
                "When the latest message is mainly a method instruction such as asking the agent to search, check online, or look into something, first resolve what the user wants investigated from the immediately preceding conversation context before deciding that content is missing.",
                "For short follow-ups like '你上网查查看', '你去搜一下', or '帮我查一下', inherit the concrete topic or claim from recent_context or recent_user_messages when there is one clear recent referent, and rewrite primary_task_text/task_text to that resolved objective.",
                "Only keep a web_lookup subtask in waiting_for_input for missing content when neither recent_context nor channel recent_user_messages provides a single clear topic to investigate.",
                "Choose primary_task_text as the first executable subtask text in order. If no subtask is executable yet, use null.",
                "Use status=waiting_for_input only for the specific subtask that still lacks required content. Do not block other ready subtasks.",
                "If a later fragment fills the missing content of an earlier file-edit task, reflect the filled content in slot_values and task_text.",
                "For direct questions like arithmetic or short direct answers, use kind=direct_answer.",
                "For local document/file editing, use kind=document_edit or local_lookup depending on whether the edit instruction is already actionable.",
                "If active_task_summary contains an older clarification but the latest user message clearly resolves or corrects that ambiguity, do not repeat the old clarification. Mark the corrected subtask ready when the new message is sufficient.",
                "Statements like '我的生日是明天，你也记一下' or '记住X的生日是...' are memory or fact updates, not reminder creation, local file lookup, or document editing unless the user explicitly asks to save them to a file or create a timed reminder.",
                "If the whole request is really a single task, return one ready subtask or return is_multi_task=false with primary_task_text equal to the user request.",
                "Prefer layered_context_summary as the main cross-layer view when it is present, and use the raw per-layer fields to verify details.",
            ],
            payload=(
                "Intent context:\n"
                + self._build_intent_context_payload(
                    user_text=compact["user_text"],
                    recent_context=compact["recent_context"],
                    hot_context_summary=compact["hot_context_summary"],
                    warm_memory_summary=compact["warm_memory_summary"],
                    cold_memory_summary=compact["cold_memory_summary"],
                    active_task_summary=compact["active_task_summary"],
                    channel_context_summary=channel_context_summary,
                    layered_context_summary=layered_context_summary,
                )
                + "\n\nReturn JSON only."
            ),
            model=self.response_model,
        )
        normalized = self._normalize_task_graph_payload(payload)
        return TaskGraphIntent.model_validate(normalized).model_dump(mode="json")

    def analyze_task_classification(
            self,
            *,
            user_text: str,
            knowledge_type: str,
            document_delivery: dict[str, Any] | None = None,
            site_search: dict[str, Any] | None = None,
            recent_context: str = "",
            hot_context_summary: str = "",
            warm_memory_summary: str = "",
            cold_memory_summary: str = "",
            active_task_summary: str = "",
            channel_context_summary: str = "",
            layered_context_summary: str = "",
    ) -> dict[str, Any]:
        compact = self._compact_lightweight_context(
            user_text=user_text,
            recent_context=recent_context,
            hot_context_summary=hot_context_summary,
            warm_memory_summary=warm_memory_summary,
            cold_memory_summary=cold_memory_summary,
            active_task_summary=active_task_summary,
        )
        payload = self._chat_json_object(
            task_label="task classifier",
            schema=(
                '{"domain":"local_workspace|qq_history|web|system_utility|unknown",'
                '"task_kind":"document_edit|delivery|summarize|lookup|inspect|collection_action|attachment_lookup|reply_lookup|history_lookup|site_search|research|research_document|create_reminder|get_current_time|list_reminders|cancel_reminder|unknown",'
                '"preferred_families":["document_operation","document_summary","file_delivery","local_lookup","file_lookup","local_collection","qq_history","web_target","web_lookup","system_utility"],'
                '"run_mode":"immediate|scheduled",'
                '"scheduled_task_type":"notify|deferred_agent_task|null",'
                '"scheduled_task_payload_hint":{"instruction_text":"string","text":"string","origin_user_text":"string"},'
                '"confidence":0.0,"rationale":"short reason"}'
            ),
            rules=[
                "Classify the real task first, then decide whether execution is immediate or scheduled.",
                "Use scheduled only when the user clearly wants the task done later, by timer, after a delay, or at a future time.",
                "Use notify for pure reminders and deferred_agent_task for delayed actions that should be executed later.",
                "Keep scheduled_task_payload_hint lightweight; do not invent grounded paths or tool arguments.",
                "Use recent context and memory to resolve omitted references.",
                "Prefer layered_context_summary as the main cross-layer view when it is present, and use the raw per-layer fields to verify details.",
                "If the request is about a local file or document and asks to add, write, modify, or update content, classify it as domain=local_workspace and task_kind=document_edit even when the text also mentions previous style, earlier content, or reference words.",
                "Use qq_history only when the user is explicitly asking about past chat records, replies, attachments, or prior conversation content.",
                "Questions about whether a named person contacted the agent before, whether they ever messaged the agent, what they said earlier, or what was discussed before in QQ should be domain=qq_history, not web.",
                "Do not turn an informal QQ history question into web research or research_document just because the user asks '说了什么' or mentions a name like ssy.",
            ],
            payload=(
                "Intent context:\n"
                + self._build_intent_context_payload(
                    user_text=compact["user_text"],
                    recent_context=compact["recent_context"],
                    hot_context_summary=compact["hot_context_summary"],
                    warm_memory_summary=compact["warm_memory_summary"],
                    cold_memory_summary=compact["cold_memory_summary"],
                    active_task_summary=compact["active_task_summary"],
                    channel_context_summary=channel_context_summary,
                    layered_context_summary=layered_context_summary,
                    extra={
                        "knowledge_type": knowledge_type,
                        "document_delivery": document_delivery or {},
                        "site_search": site_search or {},
                    },
                )
            ),
            model=self.response_model,
        )
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
        if run_mode not in {"immediate", "scheduled"}:
            run_mode = "immediate"
        normalized["run_mode"] = run_mode

        scheduled_task_type = str(normalized.get("scheduled_task_type", "") or "").strip().lower()
        if scheduled_task_type not in {"", "notify", "deferred_agent_task"}:
            scheduled_task_type = ""
        normalized["scheduled_task_type"] = scheduled_task_type or None

        payload_hint = normalized.get("scheduled_task_payload_hint", {})
        if not isinstance(payload_hint, dict):
            payload_hint = {}
        normalized["scheduled_task_payload_hint"] = payload_hint

        try:
            normalized["confidence"] = max(0.0, min(1.0, float(normalized.get("confidence", 0.0))))
        except (TypeError, ValueError):
            normalized["confidence"] = 0.0

        normalized["rationale"] = str(normalized.get("rationale", "") or "").strip()
        return normalized

    def plan_scheduled_task_arguments(
            self,
            *,
            user_text: str,
            task_classification: dict[str, Any] | None = None,
            current_time_iso: str,
            timezone: str,
            recent_context: str = "",
            hot_context_summary: str = "",
            warm_memory_summary: str = "",
            cold_memory_summary: str = "",
            active_task_summary: str = "",
    ) -> dict[str, Any]:
        system_prompt = (
            "You plan structured arguments for a local scheduled-task tool.\n"
            "Return exactly one JSON object with this schema:\n"
            '{'
            '"run_mode":"immediate|scheduled",'
            '"task_type":"notify|deferred_agent_task",'
            '"when_iso":"string",'
            '"timezone":"string",'
            '"message":"string",'
            '"task_payload":{"instruction_text":"string","origin_user_text":"string","text":"string"}'
            '}\n'
            "Rules:\n"
            "1. If the request should be done now rather than later, return run_mode=immediate.\n"
            "2. If the request clearly refers to a future execution time, countdown, timer, reminder, tonight, tomorrow, or after X minutes, return run_mode=scheduled.\n"
            "3. You MUST resolve future time relative to the provided current_time_iso and timezone.\n"
            "4. when_iso MUST be a full ISO-8601 datetime string with timezone offset, for example 2026-04-16T23:15:00+08:00.\n"
            "5. Never output a naive datetime without timezone.\n"
            "6. Never invent a past year or unrelated calendar date when the user asked for a relative delay like '三分钟后'.\n"
            "7. Use task_type=notify when the scheduled task is only to notify/remind the user.\n"
            "8. Use task_type=deferred_agent_task when the scheduled task is to perform a real future action.\n"
            "9. For notify tasks, put the reminder text in message, and optionally also in task_payload.text.\n"
            "10. For deferred_agent_task, put the executable future instruction into task_payload.instruction_text.\n"
            "11. Always preserve the original delayed request in task_payload.origin_user_text.\n"
            "12. Do not invent concrete file paths, URLs, or grounded tool parameters that should only be resolved when the task actually executes.\n"
            "13. Return JSON only."
        )
        user_prompt = (
                "Intent context:\n"
                + self._build_intent_context_payload(
            user_text=user_text,
            recent_context=recent_context,
            hot_context_summary=hot_context_summary,
            warm_memory_summary=warm_memory_summary,
            cold_memory_summary=cold_memory_summary,
            active_task_summary=active_task_summary,
            extra={
                "task_classification": task_classification or {},
                "current_time_iso": current_time_iso,
                "timezone": timezone,
            },
        )
                + "\n\nReturn JSON only."
        )
        raw = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.response_model,
        )
        payload = self._extract_json_block(raw)
        normalized = dict(payload or {})

        run_mode = str(normalized.get("run_mode", "immediate") or "immediate").strip().lower()
        if run_mode not in {"immediate", "scheduled"}:
            run_mode = "immediate"
        normalized["run_mode"] = run_mode

        task_type = str(normalized.get("task_type", "notify") or "notify").strip().lower()
        if task_type not in {"notify", "deferred_agent_task"}:
            task_type = "notify"
        normalized["task_type"] = task_type

        normalized["when_iso"] = str(normalized.get("when_iso", "") or "").strip()
        normalized["timezone"] = str(normalized.get("timezone", timezone) or timezone).strip()
        normalized["message"] = str(normalized.get("message", "") or "").strip()

        task_payload = normalized.get("task_payload", {})
        if not isinstance(task_payload, dict):
            task_payload = {}
        task_payload["origin_user_text"] = str(
            task_payload.get("origin_user_text", user_text) or user_text
        ).strip()

        if "instruction_text" in task_payload:
            task_payload["instruction_text"] = str(task_payload.get("instruction_text", "") or "").strip()
        if "text" in task_payload:
            task_payload["text"] = str(task_payload.get("text", "") or "").strip()

        normalized["task_payload"] = task_payload
        return normalized

    def plan_local_search_step(
        self,
        *,
        user_text: str,
        task_classification: dict[str, Any] | None = None,
        current_time_iso: str,
        timezone: str,
        scope_hints: dict[str, str] | None = None,
        allowed_tools: list[str] | None = None,
        tool_schemas: dict[str, Any] | None = None,
        default_target_kind: str = "file",
        preferred_extensions: list[str] | None = None,
        recent_context: str = "",
        hot_context_summary: str = "",
        warm_memory_summary: str = "",
        cold_memory_summary: str = "",
        active_task_summary: str = "",
    ) -> dict[str, Any]:
        allowed = [
            tool
            for tool in (allowed_tools or ["file.list", "file.search_by_name", "retrieval.search_local_objects"])
            if isinstance(tool, str) and tool.strip()
        ]
        payload = self._chat_json_object(
            task_label="local search step planner",
            schema=(
                '{"selected_tool":"file.list|file.search_by_name|retrieval.search_local_objects|null",'
                '"reason":"string",'
                '"arguments":{"path":"string","path_scope":"string","query":"string","query_terms":["string"],'
                '"alias_terms":["string"],"recursive":true,"include_dirs":true,"patterns":["string"],'
                '"scope_mode":"subtree|shallow_first","target_kind":"any|file|folder","extensions":["string"],'
                '"top_k":8,"rebuild_if_missing":true}}'
            ),
            rules=[
                "This is only for the first local search step before any file is grounded.",
                "Choose exactly one tool from allowed_tools.",
                "Use file.search_by_name when the user likely knows the title, filename, or a short explicit name.",
                "Use retrieval.search_local_objects when the user describes the file semantically, partially, or fuzzily.",
                "Use file.list when the user mainly anchors the request to a concrete directory or wants to enumerate within a scope first.",
                "Resolve scope aliases such as desktop or downloads by using the exact path from scope_hints when available.",
                "Do not keep pure scope words like desktop or 桌面 inside query unless they are truly part of the filename.",
                "For file.search_by_name and retrieval.search_local_objects, arguments must contain a useful non-empty query.",
                "For file.list, arguments must contain a concrete path and may include patterns when file type is known.",
                "Never invent a path outside scope_hints unless it already appears clearly in the user text or recent context.",
                "Use preferred_extensions when they fit the task, but do not force unrelated file types.",
                "Return JSON only.",
            ],
            payload={
                "user_text": user_text,
                "task_classification": task_classification or {},
                "current_time_iso": current_time_iso,
                "timezone": timezone,
                "recent_context": recent_context,
                "hot_context_summary": hot_context_summary,
                "warm_memory_summary": warm_memory_summary,
                "cold_memory_summary": cold_memory_summary,
                "active_task_summary": active_task_summary,
                "scope_hints": scope_hints or {},
                "allowed_tools": allowed,
                "tool_schemas": tool_schemas or {},
                "default_target_kind": default_target_kind,
                "preferred_extensions": preferred_extensions or [],
            },
            model=self.response_model,
        )
        normalized = dict(payload or {})
        selected_tool = self._normalize_selected_tool_value(normalized.get("selected_tool"))
        if selected_tool not in set(allowed):
            selected_tool = None
        arguments = normalized.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        normalized["selected_tool"] = selected_tool
        normalized["reason"] = str(normalized.get("reason", "") or "").strip()
        normalized["arguments"] = arguments
        return normalized

    def plan_local_collection_intent(
        self,
        *,
        user_text: str,
        task_kind: str = "collection_action",
        scope_hints: dict[str, str] | None = None,
        recent_context: str = "",
    ) -> dict[str, Any]:
        payload = self._chat_json_object(
            task_label="local collection intent planner",
            schema=(
                '{"should_handle":true,"action":"move|copy|null","destination":"string|null","source_scope":"string|null",'
                '"selection_query":"string|null","category":"image|document|spreadsheet|slides|video|audio|archive|null",'
                '"patterns":["string"],"extensions":["string"],"use_directory_listing":false,"rationale":"short reason"}'
            ),
            rules=[
                "Set should_handle=true only when the user is asking to batch move, batch copy, organize, or gather local files.",
                "Return the final structured intent only; do not explain the workflow.",
                "Resolve scope aliases such as desktop or downloads by using the exact path from scope_hints when available.",
                "Never invent a path outside scope_hints unless it already appears clearly in the user request or recent context.",
                "destination must be the target folder for the batch action.",
                "source_scope should be the search scope for candidate files; use workspace when no stronger scope is grounded.",
                "selection_query should keep only the matching condition, without command words or destination words.",
                "Use use_directory_listing=true when the request is mainly type-based enumeration such as all images, all zips, or all screenshots in a scope.",
                "patterns and extensions should align with category when the file type is clear.",
                "Return JSON only.",
            ],
            payload={
                "user_text": user_text,
                "task_kind": task_kind,
                "scope_hints": scope_hints or {},
                "recent_context": recent_context,
            },
            model=self.response_model,
        )
        normalized = dict(payload or {})
        normalized["should_handle"] = bool(normalized.get("should_handle"))
        action = self._normalize_nullable_string(normalized.get("action"), lowercase=True)
        normalized["action"] = action if action in {"move", "copy"} else None
        normalized["destination"] = self._normalize_nullable_string(normalized.get("destination"))
        normalized["source_scope"] = self._normalize_nullable_string(normalized.get("source_scope"))
        normalized["selection_query"] = self._normalize_nullable_string(normalized.get("selection_query"))
        category = self._normalize_nullable_string(normalized.get("category"), lowercase=True)
        normalized["category"] = category if category in {
            "image",
            "document",
            "spreadsheet",
            "slides",
            "video",
            "audio",
            "archive",
        } else None
        patterns = normalized.get("patterns")
        if not isinstance(patterns, list):
            patterns = []
        normalized["patterns"] = [
            item.strip()
            for item in patterns
            if isinstance(item, str) and item.strip()
        ]
        extensions = normalized.get("extensions")
        if not isinstance(extensions, list):
            extensions = []
        normalized["extensions"] = [
            item.strip()
            for item in extensions
            if isinstance(item, str) and item.strip()
        ]
        normalized["use_directory_listing"] = bool(normalized.get("use_directory_listing"))
        normalized["rationale"] = self._normalize_nullable_string(normalized.get("rationale")) or ""
        return normalized

    def plan_workflow_tool_arguments(
        self,
        *,
        selected_tool: str,
        tool_input_schema: dict[str, Any],
        user_text: str,
        workflow_family: str,
        step_intent: str = "",
        step_reason: str = "",
        current_time_iso: str,
        timezone: str,
        recent_context: str = "",
        hot_context_summary: str = "",
        warm_memory_summary: str = "",
        cold_memory_summary: str = "",
        active_task_summary: str = "",
        overall_task_goal: dict[str, Any] | None = None,
        expected_step_outputs: list[str] | None = None,
        completed_outputs: list[str] | None = None,
        candidate_state: dict[str, Any] | None = None,
        workflow_state: dict[str, Any] | None = None,
        current_arguments: dict[str, Any] | None = None,
        observations: list[str] | None = None,
        task_classification: dict[str, Any] | None = None,
        task_envelope: dict[str, Any] | None = None,
        execution_brief: str | None = None,
    ) -> dict[str, Any]:
        schema_text = json.dumps(tool_input_schema or {}, ensure_ascii=False)
        payload = self._chat_json_object(
            task_label="workflow step argument planner",
            schema=schema_text,
            rules=[
                "Return only the final structured arguments object for the already-selected tool.",
                "Do not change the selected tool, workflow family, or execution goal.",
                "Use only fields that belong to the provided tool input schema.",
                "Preserve grounded hints from current_arguments unless the user request or stronger context clearly requires a correction.",
                "Never invent a file path, URL, contact, reminder id, or other grounded identifier unless it already appears in current_arguments, candidate_state, workflow_state, or observations.",
                "Resolve relative time expressions into exact ISO-8601 datetimes with timezone when the schema requires a time field.",
                "If a required field is already grounded in current_arguments, keep it unless there is a clear contradiction.",
                "Treat execution_brief and task_envelope as the main orchestration result from the primary agent. Follow them instead of reinterpreting the whole task from scratch.",
                "Return JSON only.",
            ],
            payload={
                "selected_tool": selected_tool,
                "workflow_family": workflow_family,
                "step_intent": step_intent,
                "step_reason": step_reason,
                "user_request": user_text,
                "current_local_datetime": current_time_iso,
                "timezone": timezone,
                "recent_context": recent_context,
                "hot_context_summary": hot_context_summary,
                "warm_memory_summary": warm_memory_summary,
                "cold_memory_summary": cold_memory_summary,
                "active_task_summary": active_task_summary,
                "overall_task_goal": overall_task_goal or {},
                "expected_step_outputs": expected_step_outputs or [],
                "completed_outputs": completed_outputs or [],
                "candidate_state": candidate_state or {},
                "workflow_state": workflow_state or {},
                "current_arguments": current_arguments or {},
                "observations": observations or [],
                "task_classification": task_classification or {},
                "task_envelope": task_envelope or {},
                "execution_brief": execution_brief or "",
                "tool_input_schema": tool_input_schema or {},
            },
            model=self.response_model,
        )
        return dict(payload or {})

    def plan_web_tool_arguments(
        self,
        *,
        selected_tool: str,
        tool_input_schema: dict[str, Any],
        user_text: str,
        workflow_family: str,
        step_intent: str = "",
        step_reason: str = "",
        current_time_iso: str,
        timezone: str,
        recent_context: str = "",
        hot_context_summary: str = "",
        warm_memory_summary: str = "",
        cold_memory_summary: str = "",
        active_task_summary: str = "",
        overall_task_goal: dict[str, Any] | None = None,
        expected_step_outputs: list[str] | None = None,
        completed_outputs: list[str] | None = None,
        candidate_state: dict[str, Any] | None = None,
        workflow_state: dict[str, Any] | None = None,
        current_arguments: dict[str, Any] | None = None,
        observations: list[str] | None = None,
        task_classification: dict[str, Any] | None = None,
        task_envelope: dict[str, Any] | None = None,
        execution_brief: str | None = None,
    ) -> dict[str, Any]:
        schema_text = json.dumps(tool_input_schema or {}, ensure_ascii=False)
        payload = self._chat_json_object(
            task_label="web agent argument planner",
            schema=schema_text,
            rules=[
                "Return only the final structured arguments object for the already-selected web tool.",
                "Use only fields that belong to the provided tool input schema.",
                "Act as a web-search sub-agent: understand the user's real information need, then produce concise search keywords.",
                "Do not blindly preserve current_arguments.query; rewrite it when it contains scaffolding, bad additions, duplicated words, or mismatched dates.",
                "For weather or forecast requests, include the user-specified city or region and the requested day/date in query; do not add words like latest, recent, or current year just because the request is time-related.",
                "For news or latest-development requests, use the key topic plus freshness words such as latest/recent/today only when they help the search.",
                "For conceptual questions, prefer clean topic keywords without adding latest or a year.",
                "Never invent a specific location, person, domain, or date that is not in the user request, context, current time, or grounded observations.",
                "Use domains only as a hard filter when the user explicitly requires a site/source, such as 'search on Bilibili', 'only official site', or 'just GitHub'.",
                "Use preferred_domains for soft source preferences; preferred_domains should never make a broad current-events search fail if other sources are better.",
                "For ordinary factual/current questions, keep domains empty unless the user restricted the search.",
                "Set recency_days only for genuinely time-sensitive/news/current requests.",
                "Keep max_results/max_pages close to current_arguments unless the schema/default strongly suggests otherwise.",
                "Treat execution_brief and task_envelope as the main orchestration result from the primary agent. Follow them instead of reinterpreting the whole task from scratch.",
                "Return JSON only.",
            ],
            payload={
                "selected_tool": selected_tool,
                "workflow_family": workflow_family,
                "step_intent": step_intent,
                "step_reason": step_reason,
                "user_request": user_text,
                "current_local_datetime": current_time_iso,
                "timezone": timezone,
                "recent_context": recent_context,
                "hot_context_summary": hot_context_summary,
                "warm_memory_summary": warm_memory_summary,
                "cold_memory_summary": cold_memory_summary,
                "active_task_summary": active_task_summary,
                "overall_task_goal": overall_task_goal or {},
                "expected_step_outputs": expected_step_outputs or [],
                "completed_outputs": completed_outputs or [],
                "candidate_state": candidate_state or {},
                "workflow_state": workflow_state or {},
                "current_arguments": current_arguments or {},
                "observations": observations or [],
                "task_classification": task_classification or {},
                "task_envelope": task_envelope or {},
                "execution_brief": execution_brief or "",
                "tool_input_schema": tool_input_schema or {},
            },
            model=self.response_model,
        )
        return self._normalize_web_tool_arguments(
            payload,
            user_text=user_text,
            current_time_iso=current_time_iso,
            current_arguments=current_arguments,
        )

    @classmethod
    def _normalize_web_tool_arguments(
        cls,
        payload: dict[str, Any] | None,
        *,
        user_text: str,
        current_time_iso: str,
        current_arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        args = dict(payload or {})
        current_args = dict(current_arguments or {})
        if not args.get("query") and current_args.get("query"):
            args["query"] = current_args["query"]

        for field in ("domains", "preferred_domains"):
            raw_domains = args.get(field, [])
            if raw_domains is None:
                args[field] = []
            elif isinstance(raw_domains, str):
                args[field] = [raw_domains.strip().lower()] if raw_domains.strip() else []
            elif isinstance(raw_domains, list):
                args[field] = [
                    str(item).strip().lower()
                    for item in raw_domains
                    if str(item).strip()
                ]
            else:
                args[field] = []
        if args.get("domains"):
            hard = set(args["domains"])
            args["preferred_domains"] = [
                domain for domain in args.get("preferred_domains", []) if domain not in hard
            ]

        query = " ".join(str(args.get("query", "") or "").split())
        user_lower = str(user_text or "").lower()
        is_weather = cls._web_query_is_weather(query) or cls._web_query_is_weather(user_text)
        is_time_sensitive = cls._web_request_is_time_sensitive(user_text)
        current_year = cls._extract_year_from_iso(current_time_iso)

        if query and (is_weather or not is_time_sensitive):
            if "\u6700\u65b0" not in user_text:
                query = re.sub(r"(\s|^)(\u6700\u65b0|\u6700\u8fd1)(\s|$)", " ", query).strip()
            if not any(term in user_lower for term in ("latest", "recent", "news")):
                query = re.sub(r"\b(latest|recent|news)\b", " ", query, flags=re.IGNORECASE).strip()
            if current_year and current_year not in str(user_text):
                query = re.sub(rf"(?<![\d.]){re.escape(current_year)}(?![\d.])", " ", query).strip()
            query = " ".join(query.split())
            args["query"] = query

        recency = args.get("recency_days")
        try:
            recency_value = int(recency) if recency is not None else None
        except (TypeError, ValueError):
            recency_value = None
        if recency_value is not None and recency_value <= 0:
            recency_value = None
        if recency_value is None and is_time_sensitive and not is_weather:
            recency_value = 14
        args["recency_days"] = recency_value
        if is_weather and "max_pages" in args:
            try:
                args["max_pages"] = max(1, min(1, int(args.get("max_pages", 1) or 1)))
            except (TypeError, ValueError):
                args["max_pages"] = 1
            args["prefer_browser"] = False
        return args

    @staticmethod
    def _extract_year_from_iso(current_time_iso: str) -> str:
        match = re.match(r"^(\d{4})-", str(current_time_iso or ""))
        return match.group(1) if match else str(datetime.now().year)

    @staticmethod
    def _web_query_is_weather(text: str) -> bool:
        lowered = str(text or "").lower()
        return any(
            term in lowered
            for term in (
                "weather",
                "forecast",
                "temperature",
                "\u5929\u6c14",
                "\u6c14\u6e29",
                "\u6e29\u5ea6",
                "\u964d\u96e8",
                "\u4e0b\u96e8",
                "\u9884\u62a5",
            )
        )

    @staticmethod
    def _web_request_is_time_sensitive(text: str) -> bool:
        lowered = str(text or "").lower()
        return any(
            term in lowered
            for term in (
                "\u6700\u8fd1",
                "\u6700\u65b0",
                "\u4eca\u5929",
                "\u4eca\u65e5",
                "\u5f53\u524d",
                "\u73b0\u5728",
                "\u65b0\u95fb",
                "latest",
                "recent",
                "today",
                "current",
                "news",
            )
        )

    @staticmethod
    def _normalize_proxy_send_payload(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload or {})
        normalized["should_handle"] = bool(normalized.get("should_handle"))
        recipient_query = normalized.get("recipient_query")
        message_body = normalized.get("message_body")
        intent_label = str(normalized.get("intent_label", "send_message") or "send_message").strip().lower()
        rationale = normalized.get("rationale")
        try:
            confidence_value = float(normalized.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence_value = 0.0
        if intent_label not in {"send_message", "notify", "remind", "relay"}:
            intent_label = "send_message"
        return {
            "should_handle": bool(normalized.get("should_handle")),
            "recipient_query": str(recipient_query).strip() if isinstance(recipient_query, str) and recipient_query.strip() else None,
            "message_body": str(message_body).strip() if isinstance(message_body, str) and message_body.strip() else None,
            "intent_label": intent_label,
            "confidence": max(0.0, min(1.0, confidence_value)),
            "rationale": str(rationale).strip() if isinstance(rationale, str) else "",
        }

    def classify_proxy_send_intent(
        self,
        *,
        user_text: str,
        recent_messages: list[dict[str, Any]] | None = None,
    ) -> ProxySendIntent:
        compact_user_text = self._compact_text(user_text, max_chars=320)
        compact_messages = []
        for item in (recent_messages or [])[-6:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip() or "unknown"
            text = str(item.get("text", "")).strip()
            if text:
                compact_messages.append({"role": role, "text": self._compact_text(text, max_chars=220)})
        payload = self._chat_json_object(
            task_label="QQ proxy-send classifier",
            schema=(
                '{"should_handle":true,"recipient_query":"string|null","message_body":"string|null",'
                '"intent_label":"send_message|notify|remind|relay","confidence":0.0,"rationale":"short reason"}'
            ),
            rules=[
                "Set should_handle=true only when the user wants the assistant to contact another QQ person or group on their behalf.",
                "Do not treat ordinary chat with the assistant as proxy send.",
                "Judge the user's communicative goal semantically rather than matching fixed phrases.",
                "recipient_query should keep the target wording used by the user.",
                "message_body should keep only the content to send, without wrapper phrases.",
            ],
            payload={
                "recent_messages": compact_messages,
                "latest_user_message": compact_user_text,
            },
            model=self.chat_model,
        )
        normalized = self._normalize_proxy_send_payload(payload)
        return ProxySendIntent.model_validate(normalized)

    def classify_proxy_selection_intent(
        self,
        *,
        user_text: str,
        pending_request: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        compact_candidates = []
        for item in candidates[:5]:
            if not isinstance(item, dict):
                continue
            compact_candidates.append(
                {
                    "candidate_id": str(item.get("candidate_id", "") or "").strip(),
                    "name": str(item.get("name", "") or "").strip(),
                    "target_id": str(item.get("target_id", "") or "").strip(),
                    "kind": str(item.get("kind", "") or "").strip(),
                }
            )
        payload = self._chat_json_object(
            task_label="QQ proxy-send selection agent",
            schema='{"action":"select|cancel|wait","candidate_id":"string|null","confidence":0.0,"rationale":"short reason"}',
            rules=[
                "Decide whether the latest user message resolves a pending QQ proxy-send contact selection.",
                "Use select only when exactly one candidate is clearly identified.",
                "Use cancel only when the user clearly wants to stop this pending send.",
                "Use wait when the message is ambiguous, unrelated, or needs the normal agent to handle it.",
                "candidate_id must be one of the provided candidate ids when action is select, otherwise null.",
            ],
            payload={
                "latest_user_message": self._compact_text(user_text, max_chars=240),
                "pending_request": {
                    "recipient_query": str((pending_request or {}).get("recipient_query", "") or "").strip(),
                    "message_body": self._compact_text(str((pending_request or {}).get("message_body", "") or ""), max_chars=240),
                    "intent_label": str((pending_request or {}).get("intent_label", "") or "").strip(),
                },
                "candidates": compact_candidates,
            },
            model=self.chat_model,
        )
        action = str(payload.get("action", "wait") or "wait").strip().lower()
        if action not in {"select", "cancel", "wait"}:
            action = "wait"
        candidate_ids = {item["candidate_id"] for item in compact_candidates if item.get("candidate_id")}
        candidate_id = str(payload.get("candidate_id", "") or "").strip()
        if action != "select" or candidate_id not in candidate_ids:
            candidate_id = None
            if action == "select":
                action = "wait"
        try:
            confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "action": action,
            "candidate_id": candidate_id,
            "confidence": confidence,
            "rationale": str(payload.get("rationale", "") or "").strip(),
        }

    def chat_reply(
        self,
        system_name: str,
        messages: list[Message],
        persona_name: str | None = None,
        persona_profile: str | None = None,
        chat_style_prompt: str | None = None,
    ) -> str:
        conversation = "\n".join(f"{message.role.value}: {message.content}" for message in messages[-12:])
        persona_block = self._build_persona_block(persona_name, persona_profile)
        system_prompt = (
            f"You are the conversational layer for {system_name}. "
            "Reply in natural, warm Chinese with enough detail to be genuinely useful. "
            "Cover the main point and the most relevant context, but avoid rambling or repeating yourself. "
            "Sound like a real person in chat, not like narrated roleplay. "
            "Avoid bracketed stage directions, action descriptions, and dramatic asides. "
            "Do not expose internal tool logic, schemas, or chain-of-thought. "
            "If the user is just chatting, answer directly without inventing tool usage."
            f"{persona_block}"
            f"\n{chat_style_prompt or ''}".rstrip()
        )
        user_prompt = f"Recent conversation:\n{conversation}\n\nReply naturally to the latest user message."
        raw = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.chat_model,
        )
        return self._normalize_conversational_text(raw)

    def reflect_runtime_learning(
        self,
        *,
        user_text: str,
        turn_result: dict[str, Any],
        execution_summary: dict[str, Any],
        existing_memories: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        system_prompt = (
            "You extract compact runtime learning lessons for a local agent after a real user turn. "
            "Return exactly one JSON object in the form "
            '{"memories":[{"memory_type":"failure_pattern|workflow_lesson|eval_lesson|success_pattern","content":"..."}]}. '
            "Prefer failure_pattern, workflow_lesson, or eval_lesson when the task is partial, failed, selected the wrong scope, "
            "stopped at candidate listing, or otherwise drifted from the user request. "
            "Only emit success_pattern when the turn clearly completed the required outputs without obvious drift. "
            "Keep each content short, concrete, and reusable. "
            "Do not mention trace ids or internal field names in the content."
        )
        user_prompt = (
            f"User request:\n{user_text}\n\n"
            f"Turn result:\n{json.dumps(turn_result, ensure_ascii=False, indent=2)}\n\n"
            f"Execution summary:\n{json.dumps(execution_summary, ensure_ascii=False, indent=2)}\n\n"
            f"Existing drafted memories:\n{json.dumps(existing_memories or [], ensure_ascii=False, indent=2)}\n\n"
            "Return JSON only."
        )
        raw = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.critic_model,
        )
        payload = self._extract_json_block(raw)
        memories = payload.get("memories", [])
        if not isinstance(memories, list):
            return []
        normalized: list[dict[str, str]] = []
        allowed = {"failure_pattern", "workflow_lesson", "eval_lesson", "success_pattern"}
        for item in memories[:3]:
            if not isinstance(item, dict):
                continue
            memory_type = str(item.get("memory_type", "")).strip()
            content = str(item.get("content", "")).strip()
            if memory_type not in allowed or not content:
                continue
            normalized.append({"memory_type": memory_type, "content": content})
        return normalized

    def render_tool_response_bundle(
        self,
        system_name: str,
        messages: list[Message],
        observations: list[str],
        response_hint: str | None,
        execution_summary: dict[str, Any] | None = None,
        persona_name: str | None = None,
        persona_profile: str | None = None,
        display_style_prompt: str | None = None,
        speech_style_prompt: str | None = None,
        speech_max_chars: int = 80,
    ) -> dict[str, str]:
        conversation = "\n".join(f"{message.role.value}: {message.content}" for message in messages[-10:])
        observation_text = "\n".join(observations[-8:]) if observations else "No extra observations."
        normalized_execution_summary = self._inject_local_time_hints(execution_summary)
        execution_text = json.dumps(normalized_execution_summary, ensure_ascii=False, indent=2)
        persona_block = self._build_persona_block(persona_name, persona_profile)
        system_prompt = (
            f"You are the outward response layer for {system_name}. "
            "Produce exactly one JSON object with two Chinese strings: display_text and speech_text. "
            "display_text is for the screen: it should be fuller than a one-line answer, cover the main result plus the most relevant details, and stay easy to scan. "
            "It should feel informative, not wordy. "
            "If execution_summary.document_request.wants_document is true, format display_text like a compact document with a title and 3 to 5 short sections or grouped bullets. "
            "speech_text is for TTS only: it should be shorter, more conversational, more human, avoid repeating the whole screen text, avoid long paths and long lists, and focus on the conclusion or next step. "
            "Avoid bracketed stage directions, action descriptions, and theatrical roleplay in both fields. "
            "Do not expose JSON, schemas, protocol details, or internal reasoning inside either field. "
            "Both fields must stay grounded in execution_summary. "
            "Never claim that a file was written, saved, deleted, or modified unless execution_summary.successful_actions explicitly includes that successful tool action. "
            "If execution_summary.successful_actions includes file.edit_docx with applied_edits.text_preview, only describe those exact edit previews or a simpler abstraction of them; never invent richer sample content. "
            "If execution_summary.task_status is partial, say clearly that part of the task is unfinished. "
            "If execution_summary.web_sources contains sources, display_text should briefly mention one to three of the most relevant source titles or domains. "
            "For reminder or scheduled-task results, if successful_actions.data.task.display_when_local is present, use that local time in display_text and speech_text instead of the raw UTC when_iso. "
            "For direct QQ history questions, display_text should answer the conclusion first and then one short evidence sentence; speech_text should keep that same order even more briefly. "
            "Avoid report-like openings such as '根据记录' unless the user explicitly asked for a summary or report. "
            f"Keep speech_text under about {max(20, speech_max_chars)} Chinese characters when possible. "
            f"{persona_block}"
            f"\nDisplay style: {display_style_prompt or ''}\nSpeech style: {speech_style_prompt or ''}".rstrip()
        )
        user_prompt = (
            f"Recent conversation:\n{conversation}\n\n"
            f"Tool observations:\n{observation_text}\n\n"
            f"Execution summary:\n{execution_text}\n\n"
            f"Hint:\n{response_hint or ''}\n\n"
            'Return JSON only, for example: {"display_text":"...","speech_text":"..."}'
        )
        raw = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.response_model,
        )
        try:
            payload = self._extract_json_block(raw)
            display_text = self._normalize_conversational_text(payload.get("display_text", ""))
            speech_text = self._normalize_conversational_text(payload.get("speech_text", ""), for_speech=True)
            if not display_text:
                raise ValueError("display_text is required in tool response bundle")
            if not speech_text:
                speech_text = self._shorten_speech_text(display_text, speech_max_chars)
        except Exception:
            return self._fallback_tool_response_bundle(
                raw=raw,
                system_name=system_name,
                messages=messages,
                observations=observations,
                response_hint=response_hint,
                execution_summary=execution_summary,
                persona_name=persona_name,
                persona_profile=persona_profile,
                display_style_prompt=display_style_prompt,
                speech_max_chars=speech_max_chars,
            )
        return {"display_text": display_text, "speech_text": speech_text}

    @staticmethod
    def _build_persona_block(persona_name: str | None, persona_profile: str | None) -> str:
        parts: list[str] = []
        if persona_name:
            parts.append(f"Persona name: {persona_name}.")
        if persona_profile:
            parts.append(f"Persona profile: {persona_profile}")
        if not parts:
            return ""
        return "\n" + " ".join(parts)

    def _chat_json_object(
        self,
        *,
        task_label: str,
        schema: str,
        payload: Any,
        rules: list[str] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        system_lines = [
            f"You are the local-agent {task_label}.",
            "Return exactly one JSON object.",
            "No markdown.",
            "No explanation.",
            f"Schema: {schema}",
        ]
        if rules:
            system_lines.append("Rules:")
            system_lines.extend(f"{index}. {rule}" for index, rule in enumerate(rules, start=1))
        user_prompt = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)
        raw = self._chat(
            [
                {"role": "system", "content": "\n".join(system_lines)},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
        )
        return dict(self._extract_json_block(raw) or {})

    def critique_decision(
        self,
        messages: list[Message],
        decision: ToolDecision,
        tool_manifests: list[ToolManifest],
        observations: list[str],
    ) -> DecisionReview:
        tools_json = json.dumps(
            [tool.model_dump(mode="json") for tool in tool_manifests],
            ensure_ascii=False,
            indent=2,
        )
        decision_json = decision.model_dump_json(indent=2)
        conversation = "\n".join(f"{message.role.value}: {message.content}" for message in messages[-8:])
        latest_user_message = next((message.content for message in reversed(messages) if message.role.value == "user"), "")
        observation_text = "\n".join(observations[-6:]) if observations else "No tool results yet."

        system_prompt = (
            "You are a decision critic for a local agent.\n"
            "Review whether the planner decision is safe, consistent, and executable.\n"
            "Be strict about tool choice, argument completeness, and field consistency.\n"
            "Return exactly one JSON object. No markdown.\n"
            "Approve only when the decision is internally consistent.\n"
            "expected_step_outputs must match the current tool step.\n"
            "overall_task_goal must stay aligned with the full user deliverable, not just the current step.\n"
            "If you reject and can fix safely, provide a full suggested_decision; otherwise use null.\n"
            "Prefer fixing selected_tool, arguments, or expected_step_outputs before changing overall_task_goal.\n"
            "Never invent a file path unless it already appears in the decision or observations.\n"
            "If search returned zero candidates, prefer keeping the planner step, asking for clarification, or listing a directory.\n"
            "decision values must stay within respond, tool_call, clarify, finish.\n"
            "Schema:\n"
            "{\n"
            '  "approved": true,\n'
            '  "issues": ["issue1", "issue2"],\n'
            '  "summary": "short review summary",\n'
            '  "suggested_decision": null\n'
            "}\n"
            "If suggested_decision is not null, it must be a full ToolDecision-compatible JSON object.\n"
        )
        user_prompt = (
            f"Latest user request:\n{latest_user_message}\n\n"
            f"Available tools:\n{tools_json}\n\n"
            f"Recent conversation:\n{conversation}\n\n"
            f"Recent observations:\n{observation_text}\n\n"
            f"Planner decision:\n{decision_json}\n\n"
            "Return exactly one JSON object."
        )
        raw = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.critic_model,
        )
        payload = self._extract_json_block(raw)
        payload.setdefault("approved", False)
        payload.setdefault("issues", [])
        payload.setdefault("summary", "")
        payload.setdefault("suggested_decision", None)
        if isinstance(payload.get("suggested_decision"), dict):
            payload["suggested_decision"] = self._normalize_tool_decision_payload(
                payload["suggested_decision"],
                fallback=decision,
            )
            payload["suggested_decision"] = self._infer_goal_fields(payload["suggested_decision"], tool_manifests)
        return DecisionReview.model_validate(payload)

    def classify_follow_up(
        self,
        pending_task: PendingTask,
        user_text: str,
    ) -> FollowUpAssessment:
        system_prompt = (
            "You classify whether the latest user message continues a pending task.\n"
            "Return exactly one JSON object with this schema:\n"
            "{\n"
            '  "action": "resume|resume_with_correction|new_request|cancel",\n'
            '  "rationale": "short reason",\n'
            '  "slot_updates": {"slot_name": "value"},\n'
            '  "merged_user_request": "full resumed request or null",\n'
            '  "assistant_response": "only for cancel, otherwise null"\n'
            "}\n"
            "Rules:\n"
            "1. Use resume when the user is clarifying the pending task.\n"
            "2. Use resume_with_correction when the user is still replying to the pending task but is correcting a wrong assumption, subject, recipient, object, or intended action from the previous turn.\n"
            "3. Replies like '不是他，是我', '对，我是在告诉你...', or '我的生日是明天，你也记一下' after a mistaken clarification usually mean resume_with_correction, not new_request.\n"
            "4. Subject switches matter: changing from another person to the speaker, or from the speaker to another named person, should be treated as a correction when it is clearly answering the pending clarification.\n"
            "2. Use cancel when the user explicitly gives up, stops, or says never mind.\n"
            "5. Use new_request only when the user clearly changed topic and the pending task should no longer be continued.\n"
            "6. merged_user_request should combine the original request with the new clarification or correction in natural language.\n"
            "7. slot_updates should contain any filled details, such as file_name, folder_name, location, path clue, corrected subject, or corrected content.\n"
            "8. Treat additive follow-ups such as supplementary content, date details, style constraints, or 'add this too' as resume, not new_request.\n"
            "Return JSON only."
        )
        user_prompt = (
            f"Pending task:\n{pending_task.model_dump_json(indent=2)}\n\n"
            f"Latest user message:\n{user_text}\n\n"
            "Return exactly one JSON object."
        )
        raw = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.chat_model,
        )
        payload = self._extract_json_block(raw)
        payload.setdefault("action", "new_request")
        payload.setdefault("rationale", "")
        payload.setdefault("slot_updates", {})
        payload.setdefault("merged_user_request", None)
        payload.setdefault("assistant_response", None)
        if not isinstance(payload.get("slot_updates"), dict):
            payload["slot_updates"] = {}
        return FollowUpAssessment.model_validate(payload)

    def classify_task_attribution(
        self,
        task_candidates: list[PendingTask],
        user_text: str,
    ) -> FollowUpAssessment:
        system_prompt = (
            "You decide whether the latest user message continues one of several recent unfinished tasks.\n"
            "Return exactly one JSON object with this schema:\n"
            "{\n"
            '  "action": "resume|resume_with_correction|new_request|cancel",\n'
            '  "target_task_id": "task id or null",\n'
            '  "rationale": "short reason",\n'
            '  "slot_updates": {"slot_name": "value"},\n'
            '  "merged_user_request": "full resumed request or null",\n'
            '  "assistant_response": "only for cancel, otherwise null"\n'
            "}\n"
            "Rules:\n"
            "1. Choose resume only when the latest user message is best understood as continuing exactly one candidate task.\n"
            "2. Choose resume_with_correction when the latest user message is still about one candidate task but corrects a mistaken subject, target, or intended action from the prior task framing.\n"
            "3. Choose cancel only when the user clearly wants to stop a candidate task.\n"
            "4. Choose new_request when the message starts a different topic or no candidate is a good fit.\n"
            "5. Prefer the most recent and semantically closest candidate when multiple are plausible, but never guess if the fit is weak.\n"
            "6. merged_user_request should naturally combine the original request with the new clarification or correction when action is resume or resume_with_correction.\n"
            "7. If the latest message clearly answers a pending clarification while correcting the earlier misunderstanding, do not classify it as new_request.\n"
            "Return JSON only."
        )
        compact_candidates = []
        for task in task_candidates[:6]:
            compact_candidates.append(
                {
                    "task_id": task.task_id,
                    "intent": task.intent,
                    "summary": task.summary,
                    "original_user_request": task.original_user_request,
                    "state_kind": task.state_kind,
                    "missing_slots": task.missing_slots,
                    "collected_slots": task.collected_slots,
                    "selection_candidates": [
                        {
                            "candidate_id": candidate.candidate_id,
                            "name": candidate.name,
                            "path": candidate.path,
                        }
                        for candidate in task.selection_candidates[:3]
                    ],
                }
            )
        user_prompt = (
            f"Candidate tasks:\n{json.dumps(compact_candidates, ensure_ascii=False, indent=2)}\n\n"
            f"Latest user message:\n{user_text}\n\n"
            "Return exactly one JSON object."
        )
        raw = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.chat_model,
        )
        payload = self._extract_json_block(raw)
        payload.setdefault("action", "new_request")
        payload.setdefault("target_task_id", None)
        payload.setdefault("rationale", "")
        payload.setdefault("slot_updates", {})
        payload.setdefault("merged_user_request", None)
        payload.setdefault("assistant_response", None)
        if not isinstance(payload.get("slot_updates"), dict):
            payload["slot_updates"] = {}
        if payload.get("target_task_id") is not None:
            payload["target_task_id"] = str(payload["target_task_id"]).strip() or None
        return FollowUpAssessment.model_validate(payload)

    def classify_turn_completion(
        self,
        *,
        raw_user_turn_text: str,
        recent_context: str = "",
        hot_context_summary: str = "",
        pending_task_summary: str = "",
        event_summaries: list[str] | None = None,
        attachment_refs: list[str] | None = None,
        typing_active: bool = False,
        event_count: int = 0,
        turn_age_ms: int = 0,
        silence_ms: int = 0,
        quiet_window_ms: int = 0,
        idle_timeout_ms: int = 0,
        persona_name: str = "",
    ) -> TurnCompletionDecision:
        compact = self._compact_lightweight_context(
            user_text=raw_user_turn_text,
            recent_context=recent_context,
            hot_context_summary=hot_context_summary,
            warm_memory_summary=pending_task_summary,
            user_text_chars=360,
            summary_chars=320,
        )
        payload = self._chat_json_object(
            task_label="live-turn completion judge",
            schema='{"finalize":true,"confidence":0.0,"wait_ms":0,"reason":"short reason","source":"llm","turn_kind":"uncertain|execute_task|direct_reply|memory_update|instruction_update|chat","ask_followup":false,"followup_text":"","understood_task":"one concise Chinese sentence summarizing the actionable task","should_ack_task":false,"task_ack_text":"natural Chinese acknowledgement for task turns only"}',
            rules=[
                "Finalize only when the buffered content is already actionable or the user has very likely finished this turn.",
                "Wait when the user looks mid-thought, is sending a multipart request, or may still append details.",
                "Be conservative for vague starters such as greetings, lead-ins, or 'help me'.",
                "Prefer wait_ms around 800 to 2500 when more input is likely.",
                "Treat recent context and pending task summary as soft hints only.",
                "Use turn_kind=execute_task only when the agent should immediately run tools, edit files, search, send, schedule, or perform another concrete execution step.",
                "Use turn_kind=memory_update when the user is telling the agent a fact to remember or correcting a remembered fact, such as birthdays, names, relationships, or profile details.",
                "Use turn_kind=instruction_update when the user is setting a naming rule, style preference, workflow method, or other behavioral instruction.",
                "Use turn_kind=direct_reply or chat for ordinary conversational turns that should simply receive one normal reply.",
                "If finalize is false and a brief nudge would help, you may set ask_followup=true.",
                "When ask_followup is true, followup_text must be one short natural Chinese sentence asking whether there are any additional details before proceeding.",
                "Keep followup_text concise and specific; do not use bullets or long explanations.",
                "When finalize is true, understood_task must summarize the task you believe should be executed, including target object, action, key content, date/style constraints, and relevant context when available.",
                "When finalize is false, understood_task should briefly describe what is still incomplete or likely pending.",
                "Set should_ack_task=true only for task turns that require execution, tools, file changes, reminders, lookups, sending messages, or other concrete actions.",
                "Set should_ack_task=false for ordinary chat, greetings, opinions, casual discussion, or questions that should simply be answered normally.",
                "Personal fact updates or memory notes like '我的生日是明天，你也记一下' are memory_update by default, not reminder creation, unless the user explicitly asks to remind them later or at a specific time.",
                "When turn_kind is memory_update or instruction_update, should_ack_task must be false because these turns should normally get one direct reply rather than a pre-execution acknowledgement.",
                "When should_ack_task=true, task_ack_text must be a short natural Chinese message telling the user what task you understood and are about to handle.",
                "task_ack_text is a pre-execution acknowledgement only: never claim that the task is already completed, never state lookup results, and never say you already checked, found, or confirmed anything.",
                "For search, history, file, reminder, or delivery tasks, task_ack_text should describe the next action you will take, not the outcome.",
                "When should_ack_task=false, task_ack_text must be empty.",
            ],
            payload={
                "raw_user_turn_text": compact["user_text"],
                "recent_context": compact["recent_context"],
                "hot_context_summary": compact["hot_context_summary"],
                "pending_task_summary": compact["warm_memory_summary"],
                "event_summaries": event_summaries or [],
                "attachment_refs": attachment_refs or [],
                "typing_active": typing_active,
                "event_count": event_count,
                "turn_age_ms": turn_age_ms,
                "silence_ms": silence_ms,
                "quiet_window_ms": quiet_window_ms,
                "idle_timeout_ms": idle_timeout_ms,
                "persona_name": str(persona_name or "").strip(),
            },
            model=self.chat_model,
        )
        payload["finalize"] = bool(payload.get("finalize"))
        payload["ask_followup"] = bool(payload.get("ask_followup"))
        try:
            payload["confidence"] = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
        except (TypeError, ValueError):
            payload["confidence"] = 0.0
        try:
            payload["wait_ms"] = max(0, int(payload.get("wait_ms", 0) or 0))
        except (TypeError, ValueError):
            payload["wait_ms"] = 0
        payload["reason"] = str(payload.get("reason", "") or "").strip()
        turn_kind = str(payload.get("turn_kind", "uncertain") or "uncertain").strip().lower()
        if turn_kind not in {"uncertain", "execute_task", "direct_reply", "memory_update", "instruction_update", "chat"}:
            turn_kind = "uncertain"
        payload["turn_kind"] = turn_kind
        payload["followup_text"] = str(payload.get("followup_text", "") or "").strip()
        payload["understood_task"] = str(payload.get("understood_task", "") or "").strip()
        payload["should_ack_task"] = bool(payload.get("should_ack_task"))
        payload["task_ack_text"] = str(payload.get("task_ack_text", "") or "").strip()
        source = str(payload.get("source", "llm") or "llm").strip().lower()
        payload["source"] = source if source in {"llm", "rule", "fallback"} else "llm"
        if payload["finalize"]:
            payload["ask_followup"] = False
            payload["followup_text"] = ""
        elif not payload["ask_followup"]:
            payload["followup_text"] = ""
        if payload["turn_kind"] in {"memory_update", "instruction_update", "direct_reply", "chat"}:
            payload["should_ack_task"] = False
        if not payload["should_ack_task"]:
            payload["task_ack_text"] = ""
        elif not payload["task_ack_text"] and payload["understood_task"]:
            payload["task_ack_text"] = f"我理解的任务是：{payload['understood_task']}。我先按这个处理。"
        return TurnCompletionDecision.model_validate(payload)

    def build_unavailable_response(self, error: Exception) -> str:
        return f"LLM 决策或自然语言生成失败。错误信息：{error}"

    def plan_system_utility_arguments(
        self,
        *,
        user_text: str,
        task_kind: str,
        current_time_iso: str,
        timezone: str,
        recent_context: str = "",
        persona_name: str = "",
        persona_profile: str = "",
    ) -> dict[str, Any]:
        compact = self._compact_lightweight_context(
            user_text=user_text,
            recent_context=recent_context,
            hot_context_summary=persona_name,
            warm_memory_summary=persona_profile,
            summary_chars=360,
        )
        schema = "{}"
        rules = [
            "Return final execution arguments only.",
            "Convert natural time expressions into exact ISO-8601 with timezone when time is required.",
            "Keep user intent intact; message text may be natural and persona-aware but must stay faithful.",
        ]
        if task_kind == "create_reminder":
            schema = '{"when_iso":"...","timezone":"...","message":"..."}'
        elif task_kind == "cancel_reminder":
            schema = '{"reminder_id":"..."}'

        payload = self._chat_json_object(
            task_label="system utility argument planner",
            schema=schema,
            rules=rules,
            payload={
                "user_request": compact["user_text"],
                "task_kind": task_kind,
                "current_local_datetime": current_time_iso,
                "timezone": timezone,
                "recent_context": compact["recent_context"],
                "persona_name": compact["hot_context_summary"],
                "persona_profile": compact["warm_memory_summary"],
            },
            model=self.response_model,
        )
        return dict(payload or {})

    def plan_qq_history_arguments(
        self,
        *,
        user_text: str,
        task_kind: str,
        current_time_iso: str,
        timezone: str,
        recent_context: str = "",
        persona_name: str = "",
        persona_profile: str = "",
    ) -> dict[str, Any]:
        compact = self._compact_lightweight_context(
            user_text=user_text,
            recent_context=recent_context,
            hot_context_summary=persona_name,
            warm_memory_summary=persona_profile,
            summary_chars=360,
        )
        payload = self._chat_json_object(
            task_label="QQ history argument planner",
            schema=(
                '{"selected_tool":"qq.get_recent_messages|qq.get_last_reply|qq.search_history|qq.get_recent_attachments",'
                '"arguments":{}}'
            ),
            rules=[
                "Use only QQ history tools.",
                "Prefer qq.search_history when the user asks what was discussed earlier, wants a summary of prior chat content, or mentions a specific contact outside the current chat window.",
                "Keep explicit person or group names in contact_query exactly as the user expressed them.",
                "Use qq.get_recent_messages only for immediate current-session context when no specific contact or history thread needs to be resolved.",
                "Use qq.get_last_reply only for a narrow latest-reply lookup.",
                "Use qq.get_recent_attachments when the user is asking about files, pictures, audio, or other attachments.",
                "Return final execution arguments only, without explanations.",
            ],
            payload={
                "user_request": user_text,
                "task_kind": task_kind,
                "current_local_datetime": current_time_iso,
                "timezone": timezone,
                "recent_context": compact["recent_context"],
                "persona_name": compact["hot_context_summary"],
                "persona_profile": compact["warm_memory_summary"],
            },
            model=self.response_model,
        )
        selected_tool = self._normalize_selected_tool_value(payload.get("selected_tool"))
        if selected_tool not in {
            "qq.get_recent_messages",
            "qq.get_last_reply",
            "qq.search_history",
            "qq.get_recent_attachments",
        }:
            return {}

        raw_arguments = payload.get("arguments")
        arguments = raw_arguments if isinstance(raw_arguments, dict) else {}

        def _limit(default: int, minimum: int = 1, maximum: int = 50) -> int:
            try:
                value = int(arguments.get("limit", default))
            except (TypeError, ValueError):
                value = default
            return max(minimum, min(maximum, value))

        normalized_arguments: dict[str, Any] = {}
        if selected_tool == "qq.get_recent_messages":
            normalized_arguments["limit"] = _limit(8, maximum=30)
            normalized_arguments["include_assistant"] = bool(arguments.get("include_assistant", True))
        elif selected_tool == "qq.get_last_reply":
            normalized_arguments["contact_query"] = self._normalize_nullable_string(arguments.get("contact_query"))
        elif selected_tool == "qq.search_history":
            normalized_arguments["query"] = self._normalize_nullable_string(arguments.get("query"))
            normalized_arguments["contact_query"] = self._normalize_nullable_string(arguments.get("contact_query"))
            normalized_arguments["limit"] = _limit(12, maximum=50)
            normalized_arguments["reply_after_last_outbound"] = bool(arguments.get("reply_after_last_outbound", False))
        elif selected_tool == "qq.get_recent_attachments":
            kind = self._normalize_nullable_string(arguments.get("kind"), lowercase=True) or "any"
            if kind in {"images", "image"}:
                kind = "image"
            elif kind in {"files", "file"}:
                kind = "file"
            elif kind in {"voice", "audio", "audios"}:
                kind = "audio"
            else:
                kind = "any"
            normalized_arguments["contact_query"] = self._normalize_nullable_string(arguments.get("contact_query"))
            normalized_arguments["kind"] = kind
            normalized_arguments["limit"] = _limit(5, maximum=20)

        return {
            "selected_tool": selected_tool,
            "arguments": normalized_arguments,
        }

    def compose_web_research_document(
        self,
        *,
        user_text: str,
        title: str | None,
        research_bundle: dict[str, Any],
        recent_context: str = "",
        hot_context_summary: str = "",
        warm_memory_summary: str = "",
        cold_memory_summary: str = "",
        active_task_summary: str = "",
    ) -> dict[str, str]:
        compact = self._compact_context_payload(
            user_text=user_text,
            recent_context=recent_context,
            hot_context_summary=hot_context_summary,
            warm_memory_summary=warm_memory_summary,
            cold_memory_summary=cold_memory_summary,
            active_task_summary=active_task_summary,
            recent_context_chars=800,
            summary_chars=240,
            user_text_chars=360,
        )
        sources: list[dict[str, str]] = []
        for item in research_bundle.get("sources", [])[:4]:
            if not isinstance(item, dict):
                continue
            sources.append(
                {
                    "title": self._compact_text(str(item.get("title", "")), max_chars=160),
                    "url": self._compact_text(str(item.get("url", "")), max_chars=260),
                    "snippet": self._compact_text(str(item.get("snippet", "")), max_chars=320),
                    "excerpt": self._compact_text(str(item.get("excerpt") or item.get("content") or ""), max_chars=900),
                }
            )
        if not sources:
            for item in research_bundle.get("results", [])[:5]:
                if not isinstance(item, dict):
                    continue
                sources.append(
                    {
                        "title": self._compact_text(str(item.get("title", "")), max_chars=160),
                        "url": self._compact_text(str(item.get("url", "")), max_chars=260),
                        "snippet": self._compact_text(str(item.get("snippet", "")), max_chars=420),
                    }
                )
        payload = self._chat_json_object(
            task_label="web research document composer",
            schema='{"title":"...","content":"..."}',
            rules=[
                "Write the final document body for the user's requested saved artifact.",
                "Use the provided web research sources as grounding material; do not paste raw page chrome, navigation text, CSS, or HTML fragments.",
                "Synthesize concise natural Chinese with a clear title, short sections, and practical bullets where helpful.",
                "Mention source names or URLs briefly at the end when useful.",
                "Return only JSON fields title and content.",
            ],
            payload={
                "user_request": compact["user_text"],
                "requested_title": title,
                "recent_context": compact["recent_context"],
                "memory_context": {
                    "hot": compact["hot_context_summary"],
                    "warm": compact["warm_memory_summary"],
                    "cold": compact["cold_memory_summary"],
                    "active": compact["active_task_summary"],
                },
                "research_query": research_bundle.get("query"),
                "sources": sources,
                "content_excerpt": self._compact_text(str(research_bundle.get("content", "")), max_chars=2200),
            },
            model=self.response_model,
        )
        content = self._normalize_nullable_string(payload.get("content"))
        if not content:
            content = self._compact_text(str(research_bundle.get("content", "")), max_chars=1200)
        composed_title = self._normalize_nullable_string(payload.get("title")) or title or ""
        return {"title": composed_title, "content": content}

    def summarize_document_for_agent(
        self,
        *,
        instruction: str,
        source_path: str,
        extracted_text: str,
        recent_context: str = "",
        grounded_inputs: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        compact = self._compact_lightweight_context(
            user_text=instruction,
            recent_context=recent_context,
            summary_chars=360,
            user_text_chars=360,
        )
        compact_grounded_inputs = self._compact_grounded_inputs(grounded_inputs)
        payload = self._chat_json_object(
            task_label="document agent summary",
            schema='{"summary":"...","speech_text":"..."}',
            rules=[
                "Return only the final grounded summary for the document request.",
                "Use only the provided document text excerpt and user instruction.",
                "Treat grounded_inputs as already-resolved facts from the main agent.",
                "Do not invent facts that are absent from the document text.",
                "Write concise natural Chinese.",
            ],
            payload={
                "instruction": compact["user_text"],
                "source_path": source_path,
                "recent_context": compact["recent_context"],
                "grounded_inputs": compact_grounded_inputs,
                "document_text_excerpt": self._compact_text(extracted_text, max_chars=8000),
            },
            model=self.response_model,
        )
        summary = self._normalize_nullable_string(payload.get("summary")) or self._compact_text(extracted_text, max_chars=320)
        speech_text = self._normalize_nullable_string(payload.get("speech_text")) or self._shorten_speech_text(summary, 80)
        return {
            "summary": summary,
            "speech_text": speech_text,
        }

    def plan_document_docx_edits(
        self,
        *,
        instruction: str,
        source_path: str,
        extracted_text: str,
        structure_blocks: list[dict[str, str]],
        recent_context: str = "",
        grounded_inputs: dict[str, Any] | None = None,
        preserve_structure: bool = True,
        preserve_style: bool = True,
        suggested_append_anchor_block_id: str | None = None,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        compact = self._compact_lightweight_context(
            user_text=instruction,
            recent_context=recent_context,
            summary_chars=360,
            user_text_chars=360,
        )
        compact_grounded_inputs = self._compact_grounded_inputs(grounded_inputs)
        payload = self._chat_json_object(
            task_label="document agent docx edit planner",
            schema='{"edits":[{"block_id":"p-1","action":"replace|insert_after|delete","text":"..."}],"summary":"..."}',
            rules=[
                "Return only final edit operations for the existing document.",
                "Use only block_id values that appear in the provided structure_blocks.",
                "Allowed actions are replace, insert_after, and delete.",
                "When the user asks to append or add a line, prefer insert_after and use the suggested append anchor when it fits.",
                "Treat grounded_inputs as authoritative facts already resolved by the main agent.",
                "You are the document sub-agent: decide the exact block_id, action, and final written text yourself from the document structure and task package.",
                "The main agent provides the target file and materials only; do not assume it has chosen edit locations for you.",
                "Use grounded_inputs.resolved_facts for dates and fixed facts, grounded_inputs.source_materials for external research or QQ context, and grounded_inputs.constraints for write boundaries.",
                "If required source_materials or resolved_facts are missing for the requested edit, prefer returning no edits with a summary that states the missing context instead of fabricating content.",
                "If the instruction says today/今天/日期写今天, use grounded_inputs.resolved_facts.current_date and current_date_mmdd; never infer the date from old document entries.",
                "If grounded_inputs.date_entry_style_examples or recent_document_paragraphs are present, follow that nearby style for new text instead of copying the raw instruction wording.",
                "For dated log append requests, write one clean standalone entry such as 'MMDD：content' when that matches the document style; do not concatenate command fragments.",
                "Preserve the document structure unless the user clearly requests a broader rewrite.",
                "Keep the written text faithful to the user's requested change.",
            ],
            payload={
                "instruction": compact["user_text"],
                "source_path": source_path,
                "recent_context": compact["recent_context"],
                "grounded_inputs": compact_grounded_inputs,
                "preserve_structure": preserve_structure,
                "preserve_style": preserve_style,
                "suggested_append_anchor_block_id": suggested_append_anchor_block_id,
                "document_text_excerpt": self._compact_text(extracted_text, max_chars=7000),
                "structure_blocks": structure_blocks,
            },
            model=model_override or self.response_model,
        )
        raw_edits = payload.get("edits")
        if isinstance(raw_edits, dict):
            edits = [raw_edits]
        elif isinstance(raw_edits, list):
            edits = raw_edits
        else:
            single_edit = {
                "block_id": payload.get("block_id"),
                "action": payload.get("action"),
                "text": payload.get("text"),
            }
            edits = [single_edit] if any(value is not None for value in single_edit.values()) else []
        summary = self._normalize_nullable_string(payload.get("summary"))
        return {"edits": edits, "summary": summary}

    def plan_document_inspection(
        self,
        *,
        instruction: str,
        source_path: str,
        extracted_text: str,
        recent_context: str = "",
        grounded_inputs: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        compact = self._compact_lightweight_context(
            user_text=instruction,
            recent_context=recent_context,
            summary_chars=320,
            user_text_chars=320,
        )
        compact_grounded_inputs = self._compact_grounded_inputs(grounded_inputs)
        payload = self._chat_json_object(
            task_label="document agent inspection planner",
            schema='{"mode":"structure|search_blocks","query":"..."}',
            rules=[
                "Return only the inspection mode and optional query.",
                "Use search_blocks when the user is asking where a specific topic, sentence, or keyword appears in the document.",
                "Use structure when the user wants the outline, sections, layout, or a general structural inspection.",
                "Treat grounded_inputs as authoritative facts from the main agent.",
                "When mode is structure, query may be empty.",
            ],
            payload={
                "instruction": compact["user_text"],
                "source_path": source_path,
                "recent_context": compact["recent_context"],
                "grounded_inputs": compact_grounded_inputs,
                "document_text_excerpt": self._compact_text(extracted_text, max_chars=5000),
            },
            model=self.response_model,
        )
        mode = self._normalize_nullable_string(payload.get("mode"), lowercase=True) or "structure"
        if mode not in {"structure", "search_blocks"}:
            mode = "structure"
        query = self._normalize_nullable_string(payload.get("query")) or ""
        return {"mode": mode, "query": query}

    def rewrite_text_document_for_agent(
        self,
        *,
        instruction: str,
        source_path: str,
        original_text: str,
        recent_context: str = "",
        grounded_inputs: dict[str, Any] | None = None,
        preserve_structure: bool = True,
    ) -> dict[str, str]:
        compact = self._compact_lightweight_context(
            user_text=instruction,
            recent_context=recent_context,
            summary_chars=360,
            user_text_chars=360,
        )
        compact_grounded_inputs = self._compact_grounded_inputs(grounded_inputs)
        payload = self._chat_json_object(
            task_label="document agent text rewrite",
            schema='{"content":"...","summary":"..."}',
            rules=[
                "Return only the fully rewritten document content.",
                "Use only the provided original text and user instruction.",
                "Treat grounded_inputs as authoritative facts from the main agent.",
                "You are the document sub-agent: decide how to modify the document content yourself while respecting grounded_inputs.constraints.",
                "Use grounded_inputs.resolved_facts for dates and fixed facts, and grounded_inputs.source_materials for external research or QQ context.",
                "If the instruction says today/今天/日期写今天, use grounded_inputs.resolved_facts.current_date and current_date_mmdd; never infer the date from old document entries.",
                "When adding a dated log-style entry, preserve the document's nearby style and write one clean standalone entry instead of copying raw command fragments.",
                "Preserve the overall structure unless the user explicitly asks to restructure it.",
                "Do not add explanations outside the rewritten content.",
            ],
            payload={
                "instruction": compact["user_text"],
                "source_path": source_path,
                "recent_context": compact["recent_context"],
                "grounded_inputs": compact_grounded_inputs,
                "preserve_structure": preserve_structure,
                "original_text": self._compact_text(original_text, max_chars=9000),
            },
            model=self.response_model,
        )
        content = self._normalize_nullable_string(payload.get("content")) or original_text
        summary = self._normalize_nullable_string(payload.get("summary"))
        return {"content": content, "summary": summary}
