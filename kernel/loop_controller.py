from __future__ import annotations

import json

from local_agent.protocol.models import ToolCallResult, ToolDecision


class LoopController:
    def __init__(self, max_consecutive_failures: int = 2) -> None:
        self.max_consecutive_failures = max_consecutive_failures

    @staticmethod
    def _normalize_signature_value(value):
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        if isinstance(value, list):
            normalized_items = []
            for item in value:
                normalized_item = LoopController._normalize_signature_value(item)
                if normalized_item is None:
                    continue
                normalized_items.append(normalized_item)
            return normalized_items or None
        if isinstance(value, dict):
            normalized_dict = {}
            for key, item in value.items():
                normalized_item = LoopController._normalize_signature_value(item)
                if normalized_item is None:
                    continue
                normalized_dict[str(key)] = normalized_item
            return normalized_dict or None
        return value

    @classmethod
    def _canonicalize_arguments(cls, tool_name: str | None, arguments: dict) -> dict:
        normalized = cls._normalize_signature_value(arguments)
        if not isinstance(normalized, dict):
            normalized = {}

        # These are LLM planning hints rather than true execution identities.
        for hint_key in ("query_terms", "alias_terms"):
            normalized.pop(hint_key, None)

        if tool_name == "web.search":
            if normalized.get("max_results") == 5:
                normalized.pop("max_results", None)
        elif tool_name == "web.fetch":
            if normalized.get("max_chars") == 8000:
                normalized.pop("max_chars", None)
            if normalized.get("allow_insecure") is False:
                normalized.pop("allow_insecure", None)
            if normalized.get("prefer_browser") is False:
                normalized.pop("prefer_browser", None)
        elif tool_name == "web.research":
            if normalized.get("max_results") == 5:
                normalized.pop("max_results", None)
            if normalized.get("max_pages") == 2:
                normalized.pop("max_pages", None)
            if normalized.get("max_chars") == 4000:
                normalized.pop("max_chars", None)
            if normalized.get("allow_insecure") is False:
                normalized.pop("allow_insecure", None)
            if normalized.get("prefer_browser") is False:
                normalized.pop("prefer_browser", None)
        return normalized

    @classmethod
    def request_signature(cls, decision: ToolDecision) -> str:
        tool_name = decision.selected_tool
        return json.dumps(
            {
                "tool": tool_name,
                "arguments": cls._canonicalize_arguments(tool_name, decision.arguments),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def has_successful_result(tool_results: list[ToolCallResult]) -> bool:
        return any(result.status == "success" for result in tool_results)

    def should_stop_on_duplicate_request(
        self,
        decision: ToolDecision,
        previous_signatures: list[str],
        tool_results: list[ToolCallResult],
    ) -> str | None:
        if not self.has_successful_result(tool_results):
            return None
        signature = self.request_signature(decision)
        if previous_signatures and previous_signatures[-1] == signature:
            return "duplicate_tool_request_after_success"
        return None

    def should_stop_after_result(self, tool_results: list[ToolCallResult]) -> str | None:
        if len(tool_results) < self.max_consecutive_failures:
            return None
        recent = tool_results[-self.max_consecutive_failures :]
        if all(result.status == "error" for result in recent):
            return "too_many_consecutive_tool_failures"
        return None

    def should_finalize_after_decision_error(self, tool_results: list[ToolCallResult]) -> bool:
        return self.has_successful_result(tool_results)
