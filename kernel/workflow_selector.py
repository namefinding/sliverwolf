from __future__ import annotations

from local_agent.intent.models import IntentBundle, WorkflowProposal
from local_agent.protocol.models import CandidateState, OutputKind, ToolDecision


class WorkflowSelector:
    @classmethod
    def choose(
        cls,
        *,
        proposals: list[WorkflowProposal],
        intent_bundle: IntentBundle,
        completed_outputs: list[OutputKind],
        candidate_state: CandidateState | None,
    ) -> ToolDecision | None:
        viable = [proposal for proposal in proposals if proposal.decision is not None]
        viable = cls._apply_task_envelope(viable, intent_bundle)
        if not viable:
            return None

        scored: list[tuple[int, WorkflowProposal]] = []
        for proposal in viable:
            score = proposal.priority
            score += cls._intent_bonus(proposal, intent_bundle, completed_outputs, candidate_state)
            scored.append((score, proposal))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1].decision

    @staticmethod
    def _apply_task_envelope(
        proposals: list[WorkflowProposal],
        intent_bundle: IntentBundle,
    ) -> list[WorkflowProposal]:
        envelope = getattr(intent_bundle, "task_envelope", None)
        if envelope is None:
            return proposals

        allowed = {
            str(family or "").strip()
            for family in getattr(envelope, "allowed_families", [])
            if str(family or "").strip()
        }
        blocked = {
            str(family or "").strip()
            for family in getattr(envelope, "blocked_families", [])
            if str(family or "").strip()
        }
        if not allowed and not blocked:
            return proposals

        filtered: list[WorkflowProposal] = []
        for proposal in proposals:
            family = str(proposal.family or "").strip()
            if family in blocked:
                continue
            if allowed and family not in allowed:
                continue
            filtered.append(proposal)
        return filtered

    @staticmethod
    def _intent_bonus(
        proposal: WorkflowProposal,
        intent_bundle: IntentBundle,
        completed_outputs: list[OutputKind],
        candidate_state: CandidateState | None,
    ) -> int:
        bonus = 0
        family = proposal.family
        knowledge = intent_bundle.knowledge_request
        document = intent_bundle.document_delivery
        site = intent_bundle.site_search
        classification = intent_bundle.task_classification
        envelope = getattr(intent_bundle, "task_envelope", None)
        preferred_tools = {
            str(tool or "").strip()
            for tool in getattr(envelope, "preferred_tools", []) if str(tool or "").strip()
        }

        if candidate_state is not None and family == "candidate_followup":
            bonus += 60
        if candidate_state is not None and family == "state_transition":
            bonus += 90
        if OutputKind.OBJECT_CANDIDATES in completed_outputs and family in {"local_lookup", "file_lookup"}:
            bonus -= 40
        if classification is not None:
            if family in set(classification.preferred_families):
                try:
                    order_bonus = max(0, 12 - classification.preferred_families.index(family) * 4)
                except ValueError:
                    order_bonus = 0
                bonus += 45 + order_bonus
            elif classification.preferred_families:
                if classification.domain == "local_workspace" and family.startswith("web"):
                    bonus -= 70
                if classification.domain == "web" and family in {"local_lookup", "file_lookup", "file_delivery", "document_summary"}:
                    bonus -= 35
                if classification.domain == "qq_history" and family in {
                    "local_collection",
                    "local_lookup",
                    "file_lookup",
                    "file_delivery",
                    "document_summary",
                }:
                    bonus -= 120
                if classification.domain == "qq_history" and family == "qq_history":
                    bonus += 65
        if preferred_tools and proposal.decision is not None:
            selected_tool = str(getattr(proposal.decision, "selected_tool", "") or "").strip()
            if selected_tool in preferred_tools:
                bonus += 28
        if knowledge.knowledge_type == "local_workspace":
            if family in {"local_collection", "local_lookup", "file_lookup", "document_summary", "file_delivery", "document_operation"}:
                bonus += 50
            if family.startswith("web"):
                bonus -= 80
        elif knowledge.knowledge_type == "qq_history":
            if family == "qq_history":
                bonus += 70
            if family in {"local_collection", "local_lookup", "file_lookup", "file_delivery", "document_summary", "document_operation"}:
                bonus -= 120
            if family.startswith("web"):
                bonus -= 80
        elif knowledge.needs_grounding or site.site:
            if family.startswith("web"):
                bonus += 35
            if family in {"local_lookup", "file_lookup"} and not document.save_output:
                bonus -= 20

        if document.save_output and family in {"document_summary", "file_delivery", "local_lookup"}:
            bonus += 20
        if site.site and family.startswith("web"):
            bonus += 15
        return bonus
