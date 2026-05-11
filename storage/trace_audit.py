from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


IMPORTANT_EVENTS = {
    "user_input",
    "main_agent_context",
    "workflow_contract_debug",
    "state_machine_debug",
    "decision_short_circuit",
    "decision_raw",
    "decision_review",
    "decision_effective",
    "decision_argument_repair",
    "decision_tool_reroute",
    "upstream_constraint_applied",
    "tool_request",
    "tool_result",
    "completion_check",
    "state_machine_progress",
    "loop_stop",
    "error",
    "final_response",
}


def load_trace_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def trace_id_for(event: dict[str, Any]) -> str | None:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    trace_id = payload.get("trace_id")
    return str(trace_id) if trace_id else None


def events_by_trace(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        trace_id = trace_id_for(event)
        if trace_id:
            grouped[trace_id].append(event)
    return grouped


def short_text(value: Any, limit: int = 180) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def elapsed_ms_between(start: Any, end: Any) -> float | None:
    try:
        started = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        ended = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return round((ended - started).total_seconds() * 1000, 2)


def output_values(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    rendered: list[str] = []
    for item in values:
        value = item.get("value") if isinstance(item, dict) else item
        if value is not None:
            rendered.append(str(value))
    return rendered


def workflow_rows(workflow: dict[str, Any] | None) -> list[str]:
    if not isinstance(workflow, dict):
        return []
    rows: list[str] = []
    for node in workflow.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        rows.append(
            "- `{node_id}` status=`{status}` tool=`{tool}` produces=`{produces}` requires=`{requires}` intent={intent}".format(
                node_id=node.get("node_id", ""),
                status=node.get("status", ""),
                tool=node.get("tool"),
                produces=", ".join(output_values(node.get("produces"))),
                requires=", ".join(output_values(node.get("requires"))),
                intent=short_text(node.get("intent"), 120),
            )
        )
    return rows


def collect_warnings(trace_events: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    contract_tools: set[str] = set()
    raw_tools: set[str] = set()
    errors: list[str] = []
    required_outputs: set[str] = set()
    completed_outputs: set[str] = set()
    user_text = ""

    for event in trace_events:
        event_type = event.get("event_type")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if event_type == "user_input":
            user_text = str(payload.get("text") or user_text)
        elif event_type == "main_agent_context":
            envelope = payload.get("task_envelope") if isinstance(payload.get("task_envelope"), dict) else {}
            required_outputs.update(output_values(envelope.get("required_outputs")))
            workflow = envelope.get("workflow_spec") if isinstance(envelope.get("workflow_spec"), dict) else {}
            for node in workflow.get("nodes") or []:
                if isinstance(node, dict) and node.get("tool"):
                    contract_tools.add(str(node.get("tool")))
        elif event_type == "decision_raw":
            decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
            if decision.get("selected_tool"):
                raw_tools.add(str(decision.get("selected_tool")))
        elif event_type == "completion_check":
            assessment = payload.get("assessment") if isinstance(payload.get("assessment"), dict) else {}
            completed_outputs.update(output_values(assessment.get("completed_outputs")))
        elif event_type == "error":
            errors.append(str(payload.get("error") or ""))

    if errors:
        warnings.append("execution_error: " + " | ".join(short_text(item, 120) for item in errors if item))
    if "message_sent" in required_outputs and "message_sent" not in completed_outputs:
        if not any(event.get("event_type") == "final_response" for event in trace_events):
            warnings.append("message_sent_required_but_no_final_response")
    if contract_tools and raw_tools and not raw_tools.issubset(contract_tools):
        warnings.append(
            "planner_selected_tool_outside_contract: contract="
            + ", ".join(sorted(contract_tools))
            + " raw="
            + ", ".join(sorted(raw_tools))
        )
    if "document_agent.read" in contract_tools and "qq" in user_text.lower():
        warnings.append("possible_task_misroute: QQ request routed to document_agent.read")
    return warnings


def render_trace_audit(trace_id: str, trace_events: list[dict[str, Any]]) -> str:
    trace_events = [event for event in trace_events if event.get("event_type") in IMPORTANT_EVENTS]
    lines = [f"# Trace Audit: {trace_id}", ""]
    if trace_events:
        lines.append(f"- first_event: `{trace_events[0].get('timestamp', '')}`")
        lines.append(f"- last_event: `{trace_events[-1].get('timestamp', '')}`")
        elapsed_ms = elapsed_ms_between(trace_events[0].get("timestamp"), trace_events[-1].get("timestamp"))
        if elapsed_ms is not None:
            lines.append(f"- elapsed_ms: `{elapsed_ms}`")
    warnings = collect_warnings(trace_events)
    if warnings:
        lines.append("- warnings: " + " ; ".join(warnings))

    for event in trace_events:
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if event_type == "user_input":
            lines.append(f"- user_input: {short_text(payload.get('text'), 260)}")
        elif event_type == "main_agent_context":
            envelope = payload.get("task_envelope") if isinstance(payload.get("task_envelope"), dict) else {}
            lines.extend(
                [
                    "",
                    "## LLM Contract",
                    f"- objective: {short_text(envelope.get('primary_objective'), 260)}",
                    f"- mode: `{envelope.get('mode')}` needs_grounding=`{envelope.get('needs_grounding')}`",
                    f"- allowed: `{', '.join(envelope.get('allowed_families') or [])}`",
                    f"- blocked: `{', '.join(envelope.get('blocked_families') or [])}`",
                    f"- required_outputs: `{', '.join(output_values(envelope.get('required_outputs')))}`",
                    f"- preferred_tools: `{', '.join(envelope.get('preferred_tools') or [])}`",
                ]
            )
            if envelope.get("planning_focus_text"):
                lines.append(f"- grounded_inputs: {short_text(envelope.get('planning_focus_text'), 300)}")
            workflow = envelope.get("workflow_spec") if isinstance(envelope.get("workflow_spec"), dict) else None
            rows = workflow_rows(workflow)
            if rows:
                lines.append("- workflow_nodes:")
                lines.extend(rows)
        elif event_type == "state_machine_debug":
            workflow = payload.get("workflow") if isinstance(payload.get("workflow"), dict) else {}
            selected = workflow.get("selected_decision") if isinstance(workflow.get("selected_decision"), dict) else {}
            lines.extend(["", f"## Step {payload.get('step')} State", f"- source: `{payload.get('source')}`"])
            if selected:
                lines.append(
                    "- selected: decision=`{decision}` tool=`{tool}` intent={intent} args={args}".format(
                        decision=selected.get("decision"),
                        tool=selected.get("selected_tool"),
                        intent=short_text(selected.get("intent"), 140),
                        args=short_text(selected.get("arguments"), 260),
                    )
                )
        elif event_type in {"decision_raw", "decision_effective"}:
            decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
            lines.append(
                "- {kind}: source=`{source}` decision=`{decision}` tool=`{tool}` intent={intent} args={args}".format(
                    kind=event_type,
                    source=payload.get("source"),
                    decision=decision.get("decision"),
                    tool=decision.get("selected_tool"),
                    intent=short_text(decision.get("intent"), 140),
                    args=short_text(decision.get("arguments"), 260),
                )
            )
        elif event_type in {"decision_argument_repair", "decision_tool_reroute", "upstream_constraint_applied"}:
            lines.append(f"- {event_type}: {short_text(payload.get('details'), 320)}")
        elif event_type == "decision_review":
            review = payload.get("review") if isinstance(payload.get("review"), dict) else {}
            lines.append(
                f"- review: source=`{payload.get('source')}` approved=`{review.get('approved')}` issues=`{', '.join(review.get('issues') or [])}` summary={short_text(review.get('summary'), 180)}"
            )
        elif event_type == "tool_request":
            request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
            lines.append(
                f"- tool_request: step=`{payload.get('step')}` tool=`{request.get('tool_name')}` args={short_text(request.get('arguments'), 320)}"
            )
        elif event_type == "tool_result":
            result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
            metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
            lines.append(
                f"- tool_result: step=`{payload.get('step')}` tool=`{result.get('tool_name')}` status=`{result.get('status')}` outputs=`{', '.join(output_values(result.get('produced_outputs')))}` latency_ms=`{metrics.get('latency_ms')}`"
            )
            if metrics.get("observation"):
                lines.append(f"  observation: {short_text(metrics.get('observation'), 320)}")
        elif event_type == "completion_check":
            assessment = payload.get("assessment") if isinstance(payload.get("assessment"), dict) else {}
            lines.append(
                f"- completion: done=`{assessment.get('done')}` completed=`{', '.join(output_values(assessment.get('completed_outputs')))}` missing=`{', '.join(output_values(assessment.get('missing_outputs')))}` reason={short_text(assessment.get('reason'), 160)}"
            )
        elif event_type == "loop_stop":
            lines.append(f"- loop_stop: reason=`{payload.get('reason')}` details={short_text(payload.get('details'), 180)}")
        elif event_type == "error":
            lines.append(f"- error: {short_text(payload.get('error'), 260)}")
        elif event_type == "final_response":
            lines.extend(["", "## Final", f"- response: {short_text(payload.get('text'), 500)}"])

    lines.append("")
    return "\n".join(lines)


def write_trace_audit_files(
    *,
    trace_path: str | Path,
    trace_id: str,
    reports_root: str | Path | None = None,
) -> dict[str, str]:
    trace_path = Path(trace_path)
    if reports_root is None:
        reports_root = trace_path.parent.parent / "reports"
    reports_root = Path(reports_root)
    events = load_trace_events(trace_path)
    grouped = events_by_trace(events)
    report = render_trace_audit(trace_id, grouped.get(trace_id, []))

    audit_dir = reports_root / "trace_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    per_trace_path = audit_dir / f"{trace_id}.md"
    latest_path = reports_root / "trace_audit_latest.md"
    per_trace_path.write_text(report, encoding="utf-8")
    latest_path.write_text(report, encoding="utf-8")
    return {"trace_report_path": str(per_trace_path), "latest_trace_report_path": str(latest_path)}
