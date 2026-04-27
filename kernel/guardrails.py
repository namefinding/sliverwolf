from __future__ import annotations

from local_agent.protocol.models import ToolDecision


class Guardrails:
    def validate(self, decision: ToolDecision) -> None:
        if decision.decision.value != "tool_call":
            return
        if decision.selected_tool is None:
            raise ValueError("Tool decision is missing selected_tool.")
