from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from local_agent.kernel.completion_judge import CompletionJudge
from local_agent.protocol.models import CompletionAssessment, OutputKind, TaskGoal, ToolCallResult
from local_agent.protocol.tool_outputs import build_evidence, resolve_effective_outputs


@dataclass(frozen=True, slots=True)
class TraceReplayToolEvent:
    tool_name: str
    data: dict[str, Any]
    status: str = "success"
    manifest_outputs: tuple[OutputKind, ...] = ()


@dataclass(frozen=True, slots=True)
class TraceReplayCase:
    name: str
    required_outputs: tuple[OutputKind, ...]
    events: tuple[TraceReplayToolEvent, ...]
    completed_outputs: tuple[OutputKind, ...] = ()
    completion_mode: str = "outputs"
    metadata: dict[str, Any] = field(default_factory=dict)


def replay_required_outputs(case: TraceReplayCase) -> CompletionAssessment:
    completed_outputs = list(case.completed_outputs)
    tool_results: list[ToolCallResult] = []
    for index, event in enumerate(case.events, start=1):
        produced_outputs = []
        if event.status == "success":
            produced_outputs = resolve_effective_outputs(
                tool_name=event.tool_name,
                produced_outputs=list(event.manifest_outputs),
                data=event.data,
            )
        for output in produced_outputs:
            if output not in completed_outputs:
                completed_outputs.append(output)
        tool_results.append(
            ToolCallResult(
                request_id=f"replay_{index}",
                trace_id=case.name,
                tool_name=event.tool_name,
                status=event.status,
                data=event.data,
                produced_outputs=produced_outputs,
                evidence=build_evidence(
                    tool_name=event.tool_name,
                    data=event.data,
                    produced_outputs=produced_outputs,
                ),
            )
        )

    return CompletionJudge().assess(
        overall_task_goal=TaskGoal(
            summary=case.name,
            required_outputs=list(case.required_outputs),
            completion_mode=case.completion_mode,
        ),
        completed_outputs=completed_outputs,
        tool_results=tool_results,
    )
