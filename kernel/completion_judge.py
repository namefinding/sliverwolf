from __future__ import annotations

from local_agent.protocol.models import CompletionAssessment, OutputKind, TaskGoal, ToolCallResult
from local_agent.protocol.tool_outputs import resolve_effective_outputs


class CompletionJudge:
    @staticmethod
    def resolve_effective_outputs(
        tool_name: str,
        produced_outputs: list[OutputKind],
        result: ToolCallResult,
    ) -> list[OutputKind]:
        if result.status != "success":
            return []

        if result.produced_outputs:
            return list(result.produced_outputs)

        return resolve_effective_outputs(
            tool_name=tool_name,
            produced_outputs=produced_outputs,
            data=result.data,
        )

    def assess(
        self,
        overall_task_goal: TaskGoal | None,
        completed_outputs: list[OutputKind],
        tool_results: list[ToolCallResult],
    ) -> CompletionAssessment:
        normalized_outputs = list(dict.fromkeys(completed_outputs))
        if not tool_results:
            return CompletionAssessment(
                done=False,
                reason="No tool result available yet.",
                completed_outputs=normalized_outputs,
            )

        last_result = tool_results[-1]
        if last_result.status != "success":
            return CompletionAssessment(
                done=False,
                reason="Latest tool result is not successful.",
                completed_outputs=normalized_outputs,
            )
        tool_name = str(getattr(last_result, "tool_name", "") or "").strip()

        if tool_name in {"system.create_reminder", "system.create_scheduled_task"}:
            result_data = last_result.data if isinstance(last_result.data, dict) else {}
            if result_data.get("created") is not True:
                return CompletionAssessment(
                    done=False,
                    reason="Scheduled task creation did not actually succeed.",
                    completed_outputs=normalized_outputs,
                )
        completion_mode = "outputs" if overall_task_goal is None else overall_task_goal.completion_mode

        if completion_mode == "success":
            return CompletionAssessment(
                done=True,
                reason="Latest successful tool result completes the task by success contract.",
                completed_outputs=normalized_outputs,
            )

        if overall_task_goal is None or not overall_task_goal.required_outputs:
            return CompletionAssessment(
                done=False,
                reason="Overall task goal is missing or has no required outputs yet.",
                completed_outputs=normalized_outputs,
            )

        missing_outputs = [
            output_name for output_name in overall_task_goal.required_outputs if output_name not in normalized_outputs
        ]
        if not missing_outputs:
            return CompletionAssessment(
                done=True,
                reason="All required outputs for the overall task goal have been completed.",
                completed_outputs=normalized_outputs,
            )

        return CompletionAssessment(
            done=False,
            reason="The task still has missing required outputs.",
            completed_outputs=normalized_outputs,
            missing_outputs=missing_outputs,
        )
