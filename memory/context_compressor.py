from __future__ import annotations

from local_agent.protocol.models import Message, PendingTask, Role


class ContextCompressor:
    def __init__(
        self,
        *,
        keep_recent_messages: int = 10,
        summary_max_chars: int = 360,
        turn_excerpt_chars: int = 72,
        keep_summary_lines: int = 8,
    ) -> None:
        self.keep_recent_messages = keep_recent_messages
        self.summary_max_chars = summary_max_chars
        self.turn_excerpt_chars = turn_excerpt_chars
        self.keep_summary_lines = keep_summary_lines

    def build_session_summary(self, messages: list[Message]) -> str:
        summary, _ = self.update_session_summary(messages, previous_summary="", summarized_visible_count=0)
        return summary

    def update_session_summary(
        self,
        messages: list[Message],
        *,
        previous_summary: str = "",
        summarized_visible_count: int = 0,
    ) -> tuple[str, int]:
        visible_messages = [message for message in messages if message.role in {Role.USER, Role.ASSISTANT}]
        if len(visible_messages) <= self.keep_recent_messages:
            return "", 0

        older_messages = visible_messages[:-self.keep_recent_messages]
        target_count = len(older_messages)
        if target_count <= summarized_visible_count:
            return previous_summary, summarized_visible_count

        new_messages = older_messages[summarized_visible_count:]
        delta_lines = [self._summarize_message(message) for message in new_messages]
        delta_lines = [line for line in delta_lines if line]
        if not delta_lines:
            return previous_summary, target_count

        previous_lines = self._summary_body_lines(previous_summary)
        merged_lines = previous_lines + delta_lines
        merged_lines = self._compress_summary_lines(merged_lines)
        if not merged_lines:
            return "", target_count
        return self._format_summary(merged_lines), target_count

    @staticmethod
    def build_active_task_summary(pending_task: PendingTask | None) -> str:
        if pending_task is None:
            return ""

        parts = [f"当前待完成任务: {pending_task.summary or pending_task.intent}"]
        if pending_task.clarification_prompt:
            parts.append(f"等待补充: {pending_task.clarification_prompt}")
        if pending_task.selection_candidates:
            candidate_names = ", ".join(candidate.name for candidate in pending_task.selection_candidates[:3])
            if len(pending_task.selection_candidates) > 3:
                candidate_names += " ..."
            parts.append(f"等待选择候选: {candidate_names}")
        if pending_task.collected_slots:
            slot_summary = ", ".join(f"{key}={value}" for key, value in pending_task.collected_slots.items())
            parts.append(f"已收集信息: {slot_summary}")
        return "\n".join(parts)

    def _summarize_message(self, message: Message) -> str:
        prefix = "用户" if message.role == Role.USER else "助手"
        excerpt = self._clip(message.content, self.turn_excerpt_chars)
        if not excerpt:
            return ""
        return f"{prefix}: {excerpt}"

    def _compress_summary_lines(self, lines: list[str]) -> list[str]:
        selected: list[str] = []
        total_chars = 0
        for line in reversed(lines):
            projected = total_chars + len(line) + (1 if selected else 0)
            if selected and (len(selected) >= self.keep_summary_lines or projected > self.summary_max_chars):
                break
            if not selected and projected > self.summary_max_chars:
                selected.append(self._clip(line, self.summary_max_chars))
                break
            selected.append(line)
            total_chars = projected
        return list(reversed(selected))

    @staticmethod
    def _summary_body_lines(summary: str) -> list[str]:
        if not summary:
            return []
        normalized = summary.replace("此前对话摘要:\n", "", 1).strip()
        if not normalized:
            return []
        return [line.strip() for line in normalized.splitlines() if line.strip()]

    @staticmethod
    def _format_summary(lines: list[str]) -> str:
        if not lines:
            return ""
        return "此前对话摘要:\n" + "\n".join(lines)

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(limit - 1, 1)].rstrip() + "…"
