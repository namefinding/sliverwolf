from __future__ import annotations

from typing import Any, Iterable

from local_agent.protocol.models import OutputKind, TaskGoal, ToolDecision, ToolExecutionContext


def output_values(outputs: Iterable[Any] | None) -> list[str]:
    return [
        str(item.value if isinstance(item, OutputKind) else item).strip()
        for item in (outputs or [])
        if str(item.value if isinstance(item, OutputKind) else item).strip()
    ]


def build_tool_execution_context(
    *,
    decision: ToolDecision,
    user_text: str,
    task_envelope: Any,
    overall_task_goal: TaskGoal | None,
    recent_context: str = "",
    workflow_family: str = "",
) -> ToolExecutionContext:
    required_outputs = list(getattr(overall_task_goal, "required_outputs", []) or [])
    if not required_outputs:
        required_outputs = list(getattr(task_envelope, "required_outputs", []) or [])
    if not required_outputs:
        required_outputs = list(decision.expected_step_outputs or [])

    execution_brief = str(getattr(task_envelope, "delegated_execution_brief", "") or "").strip()
    if not execution_brief:
        execution_brief = str(getattr(task_envelope, "primary_objective", "") or "").strip()
    if not execution_brief:
        execution_brief = decision.reason or decision.intent

    grounded_inputs = {
        "user_request": user_text,
        "primary_objective": str(getattr(task_envelope, "primary_objective", "") or "").strip(),
        "planning_focus_text": str(getattr(task_envelope, "planning_focus_text", "") or "").strip(),
        "selected_tool": decision.selected_tool or "",
        "step_intent": decision.intent,
        "workflow_family": workflow_family,
        "overall_task_goal": None if overall_task_goal is None else overall_task_goal.model_dump(mode="json"),
        "expected_step_outputs": output_values(decision.expected_step_outputs),
        "envelope_required_outputs": output_values(getattr(task_envelope, "required_outputs", []) or []),
        "recent_context_excerpt": recent_context[-1200:] if recent_context else "",
    }
    constraints = {
        "downstream_must_not_reclassify_task": True,
        "must_satisfy_required_outputs_before_success": True,
        "allowed_families": _clean_strings(getattr(task_envelope, "allowed_families", []) or []),
        "blocked_families": _clean_strings(getattr(task_envelope, "blocked_families", []) or []),
        "tool_order_constraints": _clean_strings(getattr(task_envelope, "tool_order_constraints", []) or []),
        "execution_notes": _clean_strings(getattr(task_envelope, "execution_notes", []) or []),
        "known_failure_avoidance": _clean_strings(getattr(task_envelope, "known_failure_avoidance", []) or []),
    }
    return ToolExecutionContext(
        execution_brief=execution_brief,
        required_outputs=required_outputs,
        grounded_inputs={key: value for key, value in grounded_inputs.items() if value not in ("", None, [])},
        constraints=constraints,
    )


def build_subagent_task_package(
    payload: Any,
    *,
    default_constraints: dict[str, Any] | None = None,
    fact_keys: Iterable[str] = (
        "current_date",
        "current_date_mmdd",
        "current_time_iso",
        "timezone",
        "target_path",
        "target_name",
    ),
) -> dict[str, Any]:
    package = dict(getattr(payload, "grounded_inputs", {}) or {})
    resolved_facts = dict(getattr(payload, "resolved_facts", {}) or {})
    source_materials = dict(getattr(payload, "source_materials", {}) or {})
    constraints = dict(getattr(payload, "constraints", {}) or {})
    style_hints = dict(getattr(payload, "style_hints", {}) or {})

    execution_brief = str(getattr(payload, "execution_brief", "") or "").strip()
    required_outputs = output_values(getattr(payload, "required_outputs", []) or [])

    if execution_brief:
        package.setdefault("execution_brief", execution_brief)
    if required_outputs:
        package.setdefault("required_outputs", required_outputs)

    for key in fact_keys:
        if key in package and key not in resolved_facts:
            resolved_facts[key] = package[key]

    constraints.setdefault("downstream_must_not_reclassify_task", True)
    constraints.setdefault("must_satisfy_required_outputs_before_success", True)
    for key, value in (default_constraints or {}).items():
        constraints.setdefault(key, value)

    package["resolved_facts"] = resolved_facts
    package["source_materials"] = source_materials
    package["constraints"] = constraints
    package["style_hints"] = style_hints
    return package


def _clean_strings(values: Iterable[Any]) -> list[str]:
    return [str(item).strip() for item in values if str(item).strip()]
