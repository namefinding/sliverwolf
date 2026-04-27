from __future__ import annotations

from datetime import datetime
from typing import Any

from local_agent.modules.base import ToolRegistry
from local_agent.protocol.models import (
    CandidateState,
    DecisionType,
    OutputKind,
    TaskGoal,
    ToolDecision,
    WorkflowState,
)


class WorkflowArgumentPlanner:
    def __init__(self, llm_client, registry: ToolRegistry) -> None:
        self.llm_client = llm_client
        self.registry = registry

    @staticmethod
    def should_plan(
        decision: ToolDecision,
        *,
        decision_source: str,
        workflow_family: str,
    ) -> bool:
        if decision.decision != DecisionType.TOOL_CALL or not decision.selected_tool:
            return False
        if decision.selected_tool.startswith("system."):
            return False
        if decision_source in {
            "state_machine_direct",
            "workflow_policy_override",
            "state_machine_repair",
            "state_machine_repair_exception",
        }:
            return True
        return workflow_family != "generic" and not decision.arguments

    def plan(
        self,
        *,
        decision: ToolDecision,
        user_text: str,
        workflow_family: str,
        decision_source: str,
        observations: list[str],
        completed_outputs: list[OutputKind],
        overall_task_goal: TaskGoal | None,
        candidate_state: CandidateState | None,
        workflow_state: WorkflowState | None,
        task_envelope=None,
    ) -> tuple[ToolDecision, dict[str, Any] | None]:
        if not self.should_plan(
            decision,
            decision_source=decision_source,
            workflow_family=workflow_family,
        ):
            return decision, None

        if not hasattr(self.llm_client, "plan_workflow_tool_arguments"):
            return decision, None

        selected_tool = decision.selected_tool or ""
        if not self.registry.has_tool(selected_tool):
            return decision, None

        manifest = self.registry.get_manifest(selected_tool)
        now_local = datetime.now().astimezone()
        trace_payload: dict[str, Any] = {
            "selected_tool": selected_tool,
            "workflow_family": workflow_family,
            "decision_source": decision_source,
            "used": False,
        }
        if self._should_skip_for_complete_direct_arguments(
            decision,
            manifest=manifest,
            decision_source=decision_source,
        ):
            if selected_tool.startswith("web."):
                trace_payload["agent"] = "web_agent"
            trace_payload["skipped"] = "complete_direct_arguments"
            trace_payload["current_arguments"] = decision.arguments
            return decision, trace_payload
        if self._should_skip_for_kernel_write_followup(
            decision,
            manifest=manifest,
            decision_source=decision_source,
        ):
            trace_payload["skipped"] = "kernel_write_followup_arguments"
            trace_payload["current_arguments"] = decision.arguments
            return decision, trace_payload

        planner_kwargs = {
            "selected_tool": selected_tool,
            "tool_input_schema": manifest.input_schema,
            "user_text": user_text,
            "workflow_family": workflow_family,
            "step_intent": decision.intent,
            "step_reason": decision.reason,
            "current_time_iso": now_local.isoformat(),
            "timezone": str(now_local.tzinfo or "Asia/Shanghai"),
            "overall_task_goal": None
            if overall_task_goal is None
            else overall_task_goal.model_dump(mode="json"),
            "expected_step_outputs": [item.value for item in decision.expected_step_outputs],
            "completed_outputs": [item.value for item in completed_outputs],
            "candidate_state": None if candidate_state is None else candidate_state.model_dump(mode="json"),
            "workflow_state": None if workflow_state is None else workflow_state.model_dump(mode="json"),
            "current_arguments": decision.arguments,
            "observations": observations[-8:],
            "task_envelope": None
            if task_envelope is None
            else task_envelope.model_dump(mode="json"),
            "execution_brief": None
            if task_envelope is None
            else str(getattr(task_envelope, "delegated_execution_brief", "") or "").strip() or None,
        }
        try:
            if selected_tool.startswith("web.") and hasattr(self.llm_client, "plan_web_tool_arguments"):
                trace_payload["agent"] = "web_agent"
                planned_arguments = self.llm_client.plan_web_tool_arguments(**planner_kwargs)
            else:
                planned_arguments = self.llm_client.plan_workflow_tool_arguments(**planner_kwargs)
        except Exception as exc:  # noqa: BLE001
            trace_payload["error"] = str(exc)
            return decision, trace_payload

        if not isinstance(planned_arguments, dict) or not planned_arguments:
            trace_payload["planned_arguments"] = planned_arguments
            return decision, trace_payload

        replanned_decision = decision.model_copy(deep=True)
        replanned_decision.arguments = planned_arguments
        trace_payload["used"] = True
        trace_payload["planned_arguments"] = planned_arguments
        trace_payload["previous_arguments"] = decision.arguments
        return replanned_decision, trace_payload

    @staticmethod
    def _should_skip_for_complete_direct_arguments(
        decision: ToolDecision,
        *,
        manifest,
        decision_source: str,
    ) -> bool:
        if decision_source != "state_machine_direct":
            return False
        arguments = decision.arguments if isinstance(decision.arguments, dict) else {}
        if not arguments:
            return False
        if bool(getattr(manifest, "side_effect", False)):
            return False
        input_schema = getattr(manifest, "input_schema", {}) or {}
        required_fields = input_schema.get("required", []) if isinstance(input_schema, dict) else []
        if not required_fields:
            return True
        for field in required_fields:
            value = arguments.get(str(field))
            if value is None:
                return False
            if isinstance(value, str) and not value.strip():
                return False
            if isinstance(value, (list, tuple, set, dict)) and not value:
                return False
        return True

    @staticmethod
    def _should_skip_for_kernel_write_followup(
        decision: ToolDecision,
        *,
        manifest,
        decision_source: str,
    ) -> bool:
        if decision_source != "state_machine_repair":
            return False
        if not str(decision.intent or "").startswith("write_"):
            return False
        if OutputKind.FILE_WRITTEN not in decision.expected_step_outputs:
            return False
        if OutputKind.FILE_WRITTEN not in getattr(manifest, "produces", []):
            return False
        arguments = decision.arguments if isinstance(decision.arguments, dict) else {}
        path = arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            return False
        has_content = any(
            isinstance(arguments.get(field), value_type) and bool(arguments.get(field))
            for field, value_type in {
                "content": str,
                "paragraphs": list,
                "rows": list,
                "bullets": list,
            }.items()
        )
        return has_content
