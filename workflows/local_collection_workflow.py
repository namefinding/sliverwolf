from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from local_agent.intent.models import LocalCollectionIntent
from local_agent.protocol.models import (
    DecisionType,
    KnowledgeRequestIntent,
    OutputKind,
    RiskLevel,
    TaskGoal,
    ToolCallResult,
    ToolDecision,
)
from local_agent.utils.workspace_path import WorkspacePathNormalizer


class LocalCollectionWorkflow:
    _CATEGORY_PATTERNS = {
        "image": ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.webp", "*.heic", "*.tif", "*.tiff"],
        "document": ["*.docx", "*.pdf", "*.md", "*.txt", "*.log"],
        "spreadsheet": ["*.xlsx", "*.csv"],
        "slides": ["*.pptx"],
        "video": ["*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm"],
        "audio": ["*.mp3", "*.wav", "*.flac", "*.m4a", "*.aac"],
        "archive": ["*.zip", "*.rar", "*.7z", "*.tar", "*.gz"],
    }
    _ALLOWED_CATEGORIES = set(_CATEGORY_PATTERNS)
    _MOVE_TERMS = (
        "\u79fb\u5230",
        "\u79fb\u52a8\u5230",
        "\u6536\u5230",
        "\u6536\u8fdb",
        "\u5f52\u5230",
        "\u653e\u5230",
        "\u653e\u8fdb",
        "\u632a\u5230",
        "\u6574\u7406\u5230",
        "move to",
        "move into",
        "move them to",
        "move them into",
    )
    _COPY_TERMS = (
        "\u590d\u5236\u5230",
        "\u62f7\u8d1d\u5230",
        "copy to",
        "copy into",
        "copy them to",
        "copy them into",
    )
    _DESTINATION_PATTERNS = (
        re.compile(
            r"(?:\u6536\u5230|\u6536\u8fdb|\u79fb\u5230|\u79fb\u52a8\u5230|\u632a\u5230|\u653e\u5230|\u653e\u8fdb|"
            r"\u5f52\u5230|\u6574\u7406\u5230|\u590d\u5236\u5230|\u62f7\u8d1d\u5230)\s+(.+?)(?=$|[,.!?;\u3002\uff0c\uff1f\uff01])",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?:move|copy)\s+(?:them\s+)?(?:to|into)\s+(.+?)(?=$|[,.!?;\u3002\uff0c\uff1f\uff01])",
            flags=re.IGNORECASE,
        ),
    )

    def __init__(self, workspace_root: str, llm_client: Any | None = None) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.path_normalizer = WorkspacePathNormalizer(str(self.workspace_root))
        self.llm_client = llm_client

    def parse_request(
        self,
        user_text: str,
        knowledge_intent: KnowledgeRequestIntent | None,
        *,
        task_classification: Any | None = None,
        recent_context: str = "",
    ) -> LocalCollectionIntent | None:
        knowledge_type = "" if knowledge_intent is None else str(getattr(knowledge_intent, "knowledge_type", "") or "")
        if knowledge_type != "local_workspace":
            return None

        task_kind = str(getattr(task_classification, "task_kind", "") or "").strip().lower()
        planned = self._plan_intent_with_llm(
            user_text=user_text,
            task_kind=task_kind,
            recent_context=recent_context,
        )
        normalized = self._intent_from_payload(planned)
        if normalized is not None:
            return normalized

        if task_kind and task_kind != "collection_action":
            return None
        return self._fallback_parse_request(user_text)

    def build_next_decision(
        self,
        *,
        user_text: str,
        completed_outputs: list[OutputKind],
        tool_results: list[ToolCallResult],
        intent: LocalCollectionIntent | None,
    ) -> ToolDecision | None:
        if intent is None or intent.terminal_output in completed_outputs:
            return None

        latest_result = self._latest_result(tool_results)
        if latest_result is None:
            goal = TaskGoal(
                summary=f"Collect the local files matching {intent.selection_query or 'the described condition'} into {intent.destination}.",
                required_outputs=[intent.terminal_output],
            )
            if intent.use_directory_listing:
                return ToolDecision(
                    decision=DecisionType.TOOL_CALL,
                    intent="collect_local_matches",
                    reason="This is a local batch organization request, so enumerate matching files in the requested scope before moving them.",
                    selected_tool="file.list",
                    arguments={
                        "path": intent.source_scope,
                        "recursive": False,
                        "include_dirs": False,
                        "patterns": intent.patterns,
                    },
                    risk_level=RiskLevel.LOW,
                    overall_task_goal=goal,
                    expected_step_outputs=[OutputKind.DIRECTORY_ENTRIES],
                )

            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="search_local_collection_candidates",
                reason="This is a local batch organization request, so first retrieve all files that match the described condition.",
                selected_tool="retrieval.search_local_objects",
                arguments={
                    "query": intent.selection_query or user_text,
                    "target_kind": "file",
                    "top_k": 100,
                    "path_scope": intent.source_scope,
                    "extensions": intent.extensions,
                },
                risk_level=RiskLevel.LOW,
                overall_task_goal=goal,
                expected_step_outputs=[OutputKind.OBJECT_CANDIDATES],
            )

        source_paths = self._extract_source_paths(latest_result, intent.destination)
        if not source_paths:
            return ToolDecision(
                decision=DecisionType.CLARIFY,
                intent="clarify_empty_local_collection",
                reason="No reliable local files matched the described condition.",
                response_hint="I still have not found matching local files in the current scope. Please tell me a more specific keyword, folder, or likely filename.",
                overall_task_goal=latest_result.data.get("overall_task_goal") if isinstance(latest_result.data, dict) else None,
            )

        destination_root = self.path_normalizer.resolve(intent.destination)
        items: list[dict[str, object]] = []
        for raw_path in source_paths:
            source_path = Path(raw_path)
            if source_path == destination_root or destination_root in source_path.parents:
                continue
            items.append(
                {
                    "src_path": str(source_path),
                    "dest_path": str(destination_root / source_path.name),
                    "overwrite": False,
                }
            )
        if not items:
            return None

        tool_name = "file.move_many" if intent.action == "move" else "file.copy_many"
        return ToolDecision(
            decision=DecisionType.TOOL_CALL,
            intent=f"{intent.action}_local_collection",
            reason="The matching local files are ready, so execute the requested batch file operation now.",
            selected_tool=tool_name,
            arguments={"items": items, "continue_on_error": False},
            risk_level=RiskLevel.LOW,
            overall_task_goal=TaskGoal(
                summary=f"{intent.action.title()} the matching local files into {intent.destination}.",
                required_outputs=[intent.terminal_output],
            ),
            expected_step_outputs=[intent.terminal_output],
        )

    def _plan_intent_with_llm(
        self,
        *,
        user_text: str,
        task_kind: str,
        recent_context: str,
    ) -> dict[str, Any]:
        if self.llm_client is None or not hasattr(self.llm_client, "plan_local_collection_intent"):
            return {}
        try:
            payload = self.llm_client.plan_local_collection_intent(
                user_text=user_text,
                task_kind=task_kind or "collection_action",
                scope_hints=self._scope_hints(),
                recent_context=recent_context,
            )
        except TypeError:
            try:
                payload = self.llm_client.plan_local_collection_intent(user_text=user_text)
            except Exception:
                return {}
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _intent_from_payload(self, payload: dict[str, Any]) -> LocalCollectionIntent | None:
        if not payload or not bool(payload.get("should_handle")):
            return None

        action = str(payload.get("action", "") or "").strip().lower()
        if action not in {"move", "copy"}:
            return None

        destination = self._normalize_reference(payload.get("destination"))
        if not destination:
            return None

        source_scope = self._normalize_reference(payload.get("source_scope")) or "."
        selection_query = self._normalize_text(payload.get("selection_query")) or ""
        category = self._normalize_text(payload.get("category"), lowercase=True)
        if category not in self._ALLOWED_CATEGORIES:
            category = None
        patterns = self._normalize_string_list(payload.get("patterns"))
        extensions = self._normalize_extensions(payload.get("extensions"))

        if not patterns and category:
            patterns = list(self._patterns_for_category(category))
        if not extensions and category:
            extensions = self._extensions_for_category(category)

        use_directory_listing = bool(payload.get("use_directory_listing"))
        if not selection_query and patterns:
            use_directory_listing = True

        terminal_output = OutputKind.PATH_UPDATED if action == "move" else OutputKind.PATH_CREATED
        return LocalCollectionIntent(
            action=action,
            destination=destination,
            source_scope=source_scope,
            selection_query=selection_query,
            category=category,
            patterns=patterns,
            extensions=extensions,
            use_directory_listing=use_directory_listing,
            terminal_output=terminal_output,
        )

    def _fallback_parse_request(self, user_text: str) -> LocalCollectionIntent | None:
        action = self._fallback_action(user_text)
        if action is None:
            return None
        destination = self._fallback_destination(user_text)
        if destination is None:
            return None

        source_scope = self._fallback_scope(user_text)
        terminal_output = OutputKind.PATH_UPDATED if action == "move" else OutputKind.PATH_CREATED
        return LocalCollectionIntent(
            action=action,
            destination=destination,
            source_scope=source_scope,
            selection_query="",
            category=None,
            patterns=[],
            extensions=[],
            use_directory_listing=False,
            terminal_output=terminal_output,
        )

    @staticmethod
    def _latest_result(tool_results: list[ToolCallResult]) -> ToolCallResult | None:
        for result in reversed(tool_results):
            if result.status == "success" and result.tool_name in {"file.list", "retrieval.search_local_objects"}:
                return result
        return None

    def _extract_source_paths(self, result: ToolCallResult, destination: str) -> list[str]:
        destination_root = self.path_normalizer.resolve(destination)
        if result.tool_name == "file.list":
            entries = result.data.get("entries", [])
            return [
                str(Path(entry["path"]))
                for entry in entries
                if isinstance(entry, dict)
                and isinstance(entry.get("path"), str)
                and not bool(entry.get("is_dir"))
                and destination_root not in Path(entry["path"]).parents
            ]

        if result.tool_name == "retrieval.search_local_objects":
            candidates = result.data.get("candidates", [])
            selected: list[str] = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                path = candidate.get("path")
                if not isinstance(path, str) or not path.strip():
                    continue
                if str(candidate.get("object_kind", "file")) != "file":
                    continue
                try:
                    score = float(candidate.get("score", 0.0))
                except (TypeError, ValueError):
                    score = 0.0
                if score < 0.28:
                    continue
                candidate_path = Path(path)
                if destination_root in candidate_path.parents:
                    continue
                selected.append(str(candidate_path))
            return selected

        return []

    def _scope_hints(self) -> dict[str, str]:
        hints = {
            "workspace": str(self.workspace_root),
            "current_directory": str(self.workspace_root),
        }
        desktop = self._preferred_user_directory("Desktop")
        if desktop is not None:
            hints["desktop"] = str(desktop)
        downloads = self._preferred_user_directory("Downloads")
        if downloads is not None:
            hints["downloads"] = str(downloads)
        return hints

    def _fallback_scope(self, user_text: str) -> str:
        lowered = user_text.lower()
        if "\u684c\u9762" in user_text or "desktop" in lowered:
            desktop = self._preferred_user_directory("Desktop")
            if desktop is not None:
                return str(desktop)
        if "\u4e0b\u8f7d" in user_text or "downloads" in lowered or "download" in lowered:
            downloads = self._preferred_user_directory("Downloads")
            if downloads is not None:
                return str(downloads)
        return "."

    @classmethod
    def _fallback_action(cls, user_text: str) -> str | None:
        lowered = user_text.lower()
        if any(term in lowered for term in cls._MOVE_TERMS):
            return "move"
        if any(term in lowered for term in cls._COPY_TERMS):
            return "copy"
        return None

    def _fallback_destination(self, user_text: str) -> str | None:
        for pattern in self._DESTINATION_PATTERNS:
            match = pattern.search(user_text)
            if not match:
                continue
            destination = self._normalize_reference(match.group(1))
            if destination:
                return destination
        return None

    def _normalize_reference(self, value: Any) -> str | None:
        text = self._normalize_text(value)
        if text is None:
            return None
        return self.path_normalizer.normalize_reference(text)

    @staticmethod
    def _normalize_text(value: Any, *, lowercase: bool = False) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().strip("\"'")
        if not normalized:
            return None
        return normalized.lower() if lowercase else normalized

    @classmethod
    def _normalize_string_list(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        for item in value:
            text = cls._normalize_text(item)
            if text:
                normalized.append(text)
        return normalized

    @classmethod
    def _normalize_extensions(cls, value: Any) -> list[str]:
        normalized: list[str] = []
        for item in cls._normalize_string_list(value):
            normalized.append(item if item.startswith(".") else f".{item.lstrip('*.')}")
        return normalized

    @staticmethod
    def _preferred_user_directory(name: str) -> Path | None:
        user_profile = os.environ.get("USERPROFILE")
        if not user_profile:
            return None
        candidate = Path(user_profile) / name
        return candidate if candidate.exists() else None

    @classmethod
    def _patterns_for_category(cls, category: str | None) -> list[str]:
        return list(cls._CATEGORY_PATTERNS.get(category or "", []))

    @classmethod
    def _extensions_for_category(cls, category: str | None) -> list[str]:
        return [pattern.replace("*", "") for pattern in cls._patterns_for_category(category)]
