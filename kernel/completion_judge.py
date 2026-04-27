from __future__ import annotations

from local_agent.protocol.models import CompletionAssessment, OutputKind, TaskGoal, ToolCallResult


class CompletionJudge:
    @staticmethod
    def resolve_effective_outputs(
        tool_name: str,
        produced_outputs: list[OutputKind],
        result: ToolCallResult,
    ) -> list[OutputKind]:
        if result.status != "success":
            return []

        effective_outputs: list[OutputKind] = []
        data = result.data
        for output_kind in produced_outputs:
            if CompletionJudge._is_output_satisfied(output_kind, tool_name, data):
                effective_outputs.append(output_kind)
        return effective_outputs

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

    @staticmethod
    def _is_output_satisfied(output_kind: OutputKind, tool_name: str, data: dict) -> bool:
        if output_kind == OutputKind.OBJECT_CANDIDATES:
            return bool(data.get("candidates"))
        if output_kind == OutputKind.CONTACT_CANDIDATES:
            return bool(data.get("candidates"))
        if output_kind == OutputKind.OBJECT_DETAILS:
            return (
                bool(data.get("path"))
                or bool(data.get("items"))
                or bool(data.get("results"))
                or bool(data.get("reply"))
                or bool(data.get("messages"))
                or bool(data.get("attachments"))
                or bool(data.get("current_target"))
                or bool(data.get("session_id"))
            )
        if output_kind == OutputKind.FILE_CONTENTS:
            return bool(data.get("files"))
        if output_kind == OutputKind.FILE_WRITTEN:
            return (bool(data.get("path")) and int(data.get("bytes_written", 0)) >= 0) or any(
                bool(item.get("path")) and item.get("ok", True) and int(item.get("bytes_written", 0)) >= 0
                for item in data.get("results", [])
                if isinstance(item, dict)
            )
        if output_kind == OutputKind.PATH_OPENED:
            return (bool(data.get("path")) and bool(data.get("opened"))) or any(
                bool(item.get("path")) and bool(item.get("opened"))
                for item in data.get("results", [])
                if isinstance(item, dict)
            )
        if output_kind == OutputKind.PATH_CREATED:
            return (bool(data.get("path")) and bool(data.get("created") or data.get("copied"))) or any(
                bool(item.get("path")) and bool(item.get("created") or item.get("copied"))
                for item in data.get("results", [])
                if isinstance(item, dict)
            )
        if output_kind == OutputKind.PATH_UPDATED:
            return (bool(data.get("path")) and bool(data.get("moved") or data.get("renamed"))) or any(
                bool(item.get("path")) and bool(item.get("moved") or item.get("renamed"))
                for item in data.get("results", [])
                if isinstance(item, dict)
            )
        if output_kind == OutputKind.PATH_DELETED:
            return (bool(data.get("path")) and bool(data.get("deleted"))) or any(
                bool(item.get("path")) and bool(item.get("deleted"))
                for item in data.get("results", [])
                if isinstance(item, dict)
            )
        if output_kind == OutputKind.WEB_CONTENT:
            return bool(data.get("content"))
        if output_kind == OutputKind.DIRECTORY_ENTRIES:
            return "entries" in data
        if output_kind == OutputKind.SEARCH_MATCHES:
            return bool(data.get("matches"))
        if output_kind == OutputKind.SEARCH_RESULTS:
            return bool(data.get("results"))
        if output_kind == OutputKind.MEMORY_ITEMS:
            return "items" in data
        if output_kind == OutputKind.MEMORY_SAVED:
            return bool(data.get("content") or data.get("stored"))
        if output_kind == OutputKind.MESSAGE_SENT:
            return (
                bool(data.get("sent"))
                or bool(data.get("message_id"))
                or bool(data.get("ok"))
                or bool(data.get("path"))
                or bool(data.get("file_path"))
                or bool(data.get("message"))
                or bool(data.get("speech_text"))
                or bool(data.get("audio_path"))
            )
        return tool_name != ""