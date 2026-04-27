from __future__ import annotations

import re

from local_agent.llm.ollama_client import OllamaClient
from local_agent.protocol.models import DecisionReview, DecisionType, Message, ToolDecision, ToolManifest


class DecisionCritic:
    def __init__(self, llm_client: OllamaClient) -> None:
        self.llm_client = llm_client

    def review(
        self,
        messages: list[Message],
        decision: ToolDecision,
        tool_manifests: list[ToolManifest],
        observations: list[str],
    ) -> DecisionReview:
        try:
            review = self.llm_client.critique_decision(
                messages=messages,
                decision=decision,
                tool_manifests=tool_manifests,
                observations=observations,
            )
        except Exception as exc:  # noqa: BLE001
            return DecisionReview(
                approved=True,
                issues=[f"critic_fallback: {exc}"],
                summary="Critic output was invalid, so the planner decision was preserved.",
                suggested_decision=None,
            )

        return self._sanitize_review(review, decision, observations)

    def _sanitize_review(
        self,
        review: DecisionReview,
        planner_decision: ToolDecision,
        observations: list[str],
    ) -> DecisionReview:
        if review.approved:
            if review.suggested_decision is not None:
                return DecisionReview(
                    approved=True,
                    issues=[*review.issues, "critic_suggestion_ignored_for_approved_review"],
                    summary=review.summary,
                    suggested_decision=None,
                )
            return review

        if review.suggested_decision is None:
            if not review.issues and not review.summary.strip():
                return DecisionReview(
                    approved=True,
                    issues=["critic_empty_rejection_ignored"],
                    summary="Critic rejected the planner decision without rationale or a safe fix, so the planner decision was preserved.",
                    suggested_decision=None,
                )
            return review

        if not self._is_grounded_suggestion(review.suggested_decision, planner_decision, observations):
            return DecisionReview(
                approved=True,
                issues=[*review.issues, "critic_ungrounded_suggestion_ignored"],
                summary="Critic suggestion was ignored because it introduced an ungrounded path or unsupported rewrite. Planner decision preserved.",
                suggested_decision=None,
            )

        return review

    def _is_grounded_suggestion(
        self,
        suggestion: ToolDecision,
        planner_decision: ToolDecision,
        observations: list[str],
    ) -> bool:
        if suggestion.decision != DecisionType.TOOL_CALL:
            return True

        known_paths = self._extract_known_paths(planner_decision, observations)
        selected_tool = suggestion.selected_tool or ""

        if not self._is_allowed_tool_transition(planner_decision, selected_tool, known_paths):
            return False

        if selected_tool in {"file.read", "retrieval.inspect_local_candidate"}:
            candidate_paths = self._extract_suggestion_paths(suggestion)
            if candidate_paths and not all(path in known_paths for path in candidate_paths):
                return False

        if selected_tool == "file.read" and planner_decision.selected_tool == "file.list" and not known_paths:
            return False

        if selected_tool == "retrieval.inspect_local_candidate" and not known_paths:
            return False

        return True

    @staticmethod
    def _is_allowed_tool_transition(
        planner_decision: ToolDecision,
        suggested_tool: str,
        known_paths: set[str],
    ) -> bool:
        if not suggested_tool:
            return True

        if planner_decision.decision != DecisionType.TOOL_CALL:
            if known_paths and suggested_tool in {
                "retrieval.search_local_objects",
                "file.list",
                "file.search_text",
            }:
                return False
            return True

        planner_tool = planner_decision.selected_tool or ""
        allowed_transitions = {
            "retrieval.search_local_objects": {
                "retrieval.search_local_objects",
                "file.search_by_name",
                "file.list",
                "file.search_text",
            },
            "file.search_by_name": {
                "file.search_by_name",
                "file.list",
                "file.search_text",
                "file.read",
                "file.extract_text",
                "retrieval.inspect_local_candidate",
            },
            "file.list": {
                "file.list",
                "file.search_text",
                "file.search_by_name",
                "file.read",
                "file.extract_text",
                "retrieval.inspect_local_candidate",
            },
            "file.search_text": {
                "file.search_text",
                "file.search_by_name",
                "file.read",
                "file.extract_text",
                "file.write",
                "retrieval.inspect_local_candidate",
            },
            "retrieval.inspect_local_candidate": {
                "retrieval.inspect_local_candidate",
                "file.read",
                "file.extract_text",
                "file.write",
            },
            "file.read": {
                "file.read",
                "file.extract_text",
                "file.write",
            },
            "file.extract_text": {
                "file.extract_text",
                "file.read",
                "file.write",
            },
        }

        allowed = allowed_transitions.get(planner_tool)
        if allowed is None:
            return True
        return suggested_tool in allowed

    @staticmethod
    def _extract_known_paths(planner_decision: ToolDecision, observations: list[str]) -> set[str]:
        known_paths: set[str] = set()
        path_pattern = re.compile(r"[A-Za-z]:(?:\\|/)[^'\",\]\s]+(?: [^'\",\]\s]+)*")

        if planner_decision.selected_tool == "file.read":
            for value in planner_decision.arguments.get("paths", []):
                if isinstance(value, str) and value.strip():
                    known_paths.add(value.strip().strip("\"'"))

        if planner_decision.selected_tool == "retrieval.inspect_local_candidate":
            value = planner_decision.arguments.get("path")
            if isinstance(value, str) and value.strip():
                known_paths.add(value.strip().strip("\"'"))

        for observation in observations:
            for match in path_pattern.findall(observation):
                known_paths.add(match.strip().strip("\"'"))
        return known_paths

    @staticmethod
    def _extract_suggestion_paths(suggestion: ToolDecision) -> list[str]:
        if suggestion.selected_tool == "file.read":
            values = suggestion.arguments.get("paths", [])
            if isinstance(values, list):
                return [value.strip().strip("\"'") for value in values if isinstance(value, str) and value.strip()]
        if suggestion.selected_tool == "retrieval.inspect_local_candidate":
            value = suggestion.arguments.get("path")
            if isinstance(value, str) and value.strip():
                return [value.strip().strip("\"'")]
        return []
