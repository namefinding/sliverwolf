from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from local_agent.modules.base import ToolRegistry

from local_agent.protocol.models import DecisionType, OutputKind, TaskGoal, ToolDecision
from local_agent.utils.workspace_path import WorkspacePathNormalizer


class DecisionValidationError(ValueError):
    """Raised when a decision is invalid or unsafe to execute."""


class DecisionValidator:
    def __init__(self, workspace_root: str, registry: ToolRegistry) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.path_normalizer = WorkspacePathNormalizer(str(self.workspace_root))
        self.registry = registry
        self._tool_validators = {
            "file.list": self._validate_file_list,
            "file.search_by_name": self._validate_file_search_by_name,
            "file.read": self._validate_file_read,
            "file.extract_text": self._validate_file_extract_text,
            "file.extract_structure": self._validate_file_extract_structure,
            "file.search_blocks": self._validate_file_search_blocks,
            "file.search_text": self._validate_file_search_text,
            "file.write": self._validate_file_write,
            "file.write_docx": self._validate_file_write_docx,
            "file.edit_docx": self._validate_file_edit_docx,
            "file.render_docx_from_template": self._validate_file_render_docx_from_template,
            "file.write_xlsx": self._validate_file_write_xlsx,
            "file.write_pptx": self._validate_file_write_pptx,
            "file.write_many": self._validate_file_write_many,
            "file.append": self._validate_file_append,
            "file.append_many": self._validate_file_append_many,
            "file.metadata": self._validate_file_metadata,
            "file.metadata_many": self._validate_file_metadata_many,
            "file.preview": self._validate_file_preview,
            "file.preview_many": self._validate_file_preview_many,
            "file.mkdir": self._validate_file_mkdir,
            "file.mkdir_many": self._validate_file_mkdir_many,
            "file.copy": self._validate_file_copy,
            "file.copy_many": self._validate_file_copy_many,
            "file.move": self._validate_file_move,
            "file.move_many": self._validate_file_move_many,
            "file.rename": self._validate_file_rename,
            "file.rename_many": self._validate_file_rename_many,
            "file.delete": self._validate_file_delete,
            "file.delete_many": self._validate_file_delete_many,
            "file.open_path": self._validate_file_open_like,
            "file.open_many": self._validate_file_open_many,
            "file.reveal_in_explorer": self._validate_file_open_like,
            "file.reveal_many": self._validate_file_reveal_many,
            "document_agent.summarize": self._validate_document_agent_summarize,
            "document_agent.read": self._validate_document_agent_read,
            "document_agent.inspect": self._validate_document_agent_inspect,
            "document_agent.compose": self._validate_document_agent_compose,
            "document_agent.edit": self._validate_document_agent_edit,
            "image.inspect": self._validate_image_inspect,
            "image.describe": self._validate_image_describe,
            "image.read_text": self._validate_image_read_text,
            "image.capture_screen": self._validate_image_capture_screen,
            "image.capture_region": self._validate_image_capture_region,
            "memory.remember": self._validate_memory_remember,
            "memory.recall": self._validate_memory_recall,
            "retrieval.rebuild_local_index": self._validate_retrieval_rebuild_local_index,
            "retrieval.sync_local_index": self._validate_retrieval_sync_local_index,
            "retrieval.search_local_objects": self._validate_retrieval_search_local_objects,
            "retrieval.inspect_local_candidate": self._validate_retrieval_inspect_local_candidate,
            "web.search": self._validate_web_search,
            "web.fetch": self._validate_web_fetch,
            "web.open_page": self._validate_web_open_page,
            "web.research": self._validate_web_research,
            "qq.get_current_context": self._validate_qq_get_current_context,
            "qq.get_recent_messages": self._validate_qq_get_recent_messages,
            "qq.get_last_reply": self._validate_qq_get_last_reply,
            "qq.search_history": self._validate_qq_search_history,
            "qq.get_recent_attachments": self._validate_qq_get_recent_attachments,
            "qq.search_contacts": self._validate_qq_search_contacts,
            "qq.send_text": self._validate_qq_send_text,
            "qq.send_file": self._validate_qq_send_file,
            "qq.send_voice": self._validate_qq_send_voice,
            "system.get_time": self._validate_system_get_time,
            "system.create_reminder": self._validate_system_create_reminder,
            "system.create_scheduled_task": self._validate_system_create_scheduled_task,
            "system.list_reminders": self._validate_system_list_reminders,
            "system.cancel_reminder": self._validate_system_cancel_reminder,
        }

    def validate(self, decision: ToolDecision) -> None:
        self._validate_decision_consistency(decision)
        self._validate_goal_fields(decision)
        if decision.decision != DecisionType.TOOL_CALL:
            return

        tool_name = decision.selected_tool or ""
        if not self.registry.has_tool(tool_name):
            raise DecisionValidationError(f"Unknown tool: {tool_name}")

        manifest = self.registry.get_manifest(tool_name)
        if decision.expected_step_outputs and any(
            output_kind not in manifest.produces for output_kind in decision.expected_step_outputs
        ):
            raise DecisionValidationError("expected_step_outputs must be produced by the selected tool")

        validator = self._tool_validators.get(tool_name)
        # skill 和未注册 validator 的工具自动放行
        if validator is not None:
            validator(decision.arguments)

    def _validate_system_get_time(self, arguments: dict) -> None:
        kind = arguments.get("kind", "datetime")
        timezone = arguments.get("timezone")

        if "kind" in arguments and (not isinstance(kind, str) or not kind.strip()):
            raise DecisionValidationError("system.get_time kind must be a non-empty string when provided")

        if str(kind).strip().lower() not in {"datetime", "date", "time", "weekday"}:
            raise DecisionValidationError("system.get_time kind must be one of: datetime, date, time, weekday")

        if "timezone" in arguments and timezone is not None and not isinstance(timezone, str):
            raise DecisionValidationError("system.get_time timezone must be a string when provided")

    def _validate_system_create_reminder(self, arguments: dict) -> None:
        from datetime import datetime

        when_iso = arguments.get("when_iso")
        timezone = arguments.get("timezone")
        message = arguments.get("message")
        session_id = arguments.get("session_id")

        if not isinstance(when_iso, str) or not when_iso.strip():
            raise DecisionValidationError("system.create_reminder requires non-empty when_iso")

        if not isinstance(timezone, str) or not timezone.strip():
            raise DecisionValidationError("system.create_reminder requires non-empty timezone")

        if not isinstance(message, str) or not message.strip():
            raise DecisionValidationError("system.create_reminder requires non-empty message")

        if "session_id" in arguments and session_id is not None and not isinstance(session_id, str):
            raise DecisionValidationError("system.create_reminder session_id must be a string when provided")

        try:
            parsed = datetime.fromisoformat(when_iso)
        except ValueError as exc:
            raise DecisionValidationError("system.create_reminder when_iso must be a valid ISO datetime") from exc

        if parsed.tzinfo is None:
            raise DecisionValidationError("system.create_reminder when_iso must include timezone info")

    def _validate_system_create_scheduled_task(self, arguments: dict) -> None:
        from datetime import datetime

        task_type = arguments.get("task_type")
        when_iso = arguments.get("when_iso")
        timezone = arguments.get("timezone")
        session_id = arguments.get("session_id")
        channel = arguments.get("channel")
        message = arguments.get("message", "")
        task_payload = arguments.get("task_payload", {})

        if not isinstance(task_type, str) or not task_type.strip():
            raise DecisionValidationError("system.create_scheduled_task requires non-empty task_type")

        normalized_task_type = task_type.strip().lower()
        if normalized_task_type not in {"notify", "deferred_agent_task"}:
            raise DecisionValidationError(
                "system.create_scheduled_task task_type must be one of: notify, deferred_agent_task"
            )

        if not isinstance(when_iso, str) or not when_iso.strip():
            raise DecisionValidationError("system.create_scheduled_task requires non-empty when_iso")

        if not isinstance(timezone, str) or not timezone.strip():
            raise DecisionValidationError("system.create_scheduled_task requires non-empty timezone")

        if "session_id" in arguments and session_id is not None and not isinstance(session_id, str):
            raise DecisionValidationError("system.create_scheduled_task session_id must be a string when provided")

        if "channel" in arguments and channel is not None and not isinstance(channel, str):
            raise DecisionValidationError("system.create_scheduled_task channel must be a string when provided")

        if "message" in arguments and not isinstance(message, str):
            raise DecisionValidationError("system.create_scheduled_task message must be a string when provided")

        if "task_payload" in arguments and not isinstance(task_payload, dict):
            raise DecisionValidationError("system.create_scheduled_task task_payload must be an object when provided")

        try:
            parsed = datetime.fromisoformat(when_iso)
        except ValueError as exc:
            raise DecisionValidationError(
                "system.create_scheduled_task when_iso must be a valid ISO datetime"
            ) from exc

        if parsed.tzinfo is None:
            raise DecisionValidationError("system.create_scheduled_task when_iso must include timezone info")

        if normalized_task_type == "notify" and not str(message).strip():
            text_hint = task_payload.get("text") if isinstance(task_payload, dict) else None
            if not isinstance(text_hint, str) or not text_hint.strip():
                raise DecisionValidationError(
                    "system.create_scheduled_task notify tasks require a non-empty message or task_payload.text"
                )

        if normalized_task_type == "deferred_agent_task":
            instruction_text = task_payload.get("instruction_text") if isinstance(task_payload, dict) else None
            if not isinstance(instruction_text, str) or not instruction_text.strip():
                raise DecisionValidationError(
                    "system.create_scheduled_task deferred_agent_task requires task_payload.instruction_text"
                )

    def _validate_system_list_reminders(self, arguments: dict) -> None:
        status = arguments.get("status", "scheduled")
        session_id = arguments.get("session_id")

        if "status" in arguments and (not isinstance(status, str) or not status.strip()):
            raise DecisionValidationError("system.list_reminders status must be a non-empty string when provided")

        if str(status).strip().lower() not in {"scheduled", "cancelled", "fired"}:
            raise DecisionValidationError("system.list_reminders status must be one of: scheduled, cancelled, fired")

        if "session_id" in arguments and session_id is not None and not isinstance(session_id, str):
            raise DecisionValidationError("system.list_reminders session_id must be a string when provided")

    def _validate_system_cancel_reminder(self, arguments: dict) -> None:
        reminder_id = arguments.get("reminder_id")
        if not isinstance(reminder_id, str) or not reminder_id.strip():
            raise DecisionValidationError("system.cancel_reminder requires non-empty reminder_id")


    def validate_against_task_state(
            self,
            decision: ToolDecision,
            *,
            overall_task_goal: TaskGoal | None,
            completed_outputs: list[OutputKind],
    ) -> None:
        if decision.decision not in {DecisionType.RESPOND, DecisionType.FINISH}:
            return

        if overall_task_goal is None or not overall_task_goal.required_outputs:
            return

        missing_outputs = [
            output_kind
            for output_kind in overall_task_goal.required_outputs
            if output_kind not in completed_outputs
        ]
        if missing_outputs:
            missing_names = [item.value for item in missing_outputs]
            raise DecisionValidationError(
                f"Respond is not allowed before required outputs are completed: {missing_names}"
            )

    def _validate_decision_consistency(self, decision: ToolDecision) -> None:
        if decision.decision == DecisionType.TOOL_CALL:
            if not decision.selected_tool:
                raise DecisionValidationError("tool_call requires selected_tool")
            if not isinstance(decision.arguments, dict):
                raise DecisionValidationError("tool_call requires arguments to be an object")
            return

        if decision.selected_tool is not None:
            raise DecisionValidationError("Non-tool decisions must not include selected_tool")
        if decision.arguments != {}:
            raise DecisionValidationError("Non-tool decisions must use empty arguments")

    def _validate_goal_fields(self, decision: ToolDecision) -> None:
        if decision.overall_task_goal is not None:
            self._validate_output_list(
                decision.overall_task_goal.required_outputs,
                "overall_task_goal.required_outputs",
            )
        self._validate_output_list(decision.expected_step_outputs, "expected_step_outputs")

    @staticmethod
    def _validate_output_list(outputs: list[OutputKind], field_name: str) -> None:
        if not isinstance(outputs, list):
            raise DecisionValidationError(f"{field_name} must be a list")
        if not all(isinstance(item, OutputKind) for item in outputs):
            raise DecisionValidationError(f"{field_name} must contain valid output kinds")

    def _resolve_workspace_path(self, raw_path: str) -> Path:
        target = self.path_normalizer.resolve(raw_path)
        return target

    def _validate_file_list(self, arguments: dict) -> None:
        path = arguments.get("path")
        recursive = arguments.get("recursive")
        include_dirs = arguments.get("include_dirs")
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("file.list requires a non-empty path")
        if "recursive" in arguments and not isinstance(recursive, bool):
            raise DecisionValidationError("file.list recursive must be a boolean when provided")
        if "include_dirs" in arguments and not isinstance(include_dirs, bool):
            raise DecisionValidationError("file.list include_dirs must be a boolean when provided")
        target = self._resolve_workspace_path(path)
        if not target.exists():
            raise DecisionValidationError(f"file.list path does not exist: {target}")
        if not target.is_dir():
            raise DecisionValidationError(f"file.list path must be a directory: {target}")

    def _validate_file_read(self, arguments: dict) -> None:
        if "paths" not in arguments and isinstance(arguments.get("path"), str):
            arguments["paths"] = [arguments["path"]]
        paths = arguments.get("paths")
        encoding = arguments.get("encoding")
        max_bytes = arguments.get("max_bytes")
        if not isinstance(paths, list) or not paths:
            raise DecisionValidationError("file.read requires a non-empty paths list")
        if "encoding" in arguments and not isinstance(encoding, str):
            raise DecisionValidationError("file.read encoding must be a string when provided")
        if "max_bytes" in arguments and (not isinstance(max_bytes, int) or max_bytes <= 0):
            raise DecisionValidationError("file.read max_bytes must be a positive integer when provided")
        for raw_path in paths:
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise DecisionValidationError("file.read paths must be non-empty strings")
            target = self._resolve_workspace_path(raw_path)
            if not target.exists():
                raise DecisionValidationError(f"file.read path does not exist: {target}")
            if not target.is_file():
                raise DecisionValidationError(f"file.read path must be a file: {target}")

    def _validate_file_search_by_name(self, arguments: dict) -> None:
        path = arguments.get("path", ".")
        query = arguments.get("query")
        recursive = arguments.get("recursive")
        target_kind = arguments.get("target_kind", "any")
        top_k = arguments.get("top_k", 8)
        include_dirs = arguments.get("include_dirs")
        extensions = arguments.get("extensions", [])
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("file.search_by_name requires a non-empty path")
        if not isinstance(query, str) or not query.strip():
            raise DecisionValidationError("file.search_by_name requires a non-empty query")
        if "recursive" in arguments and not isinstance(recursive, bool):
            raise DecisionValidationError("file.search_by_name recursive must be a boolean when provided")
        if "include_dirs" in arguments and not isinstance(include_dirs, bool):
            raise DecisionValidationError("file.search_by_name include_dirs must be a boolean when provided")
        if str(target_kind).strip().lower() not in {"any", "file", "folder"}:
            raise DecisionValidationError("file.search_by_name target_kind must be any, file, or folder")
        if not isinstance(top_k, int) or top_k <= 0:
            raise DecisionValidationError("file.search_by_name top_k must be a positive integer")
        if "extensions" in arguments and (
            not isinstance(extensions, list) or not all(isinstance(item, str) and item.strip() for item in extensions)
        ):
            raise DecisionValidationError("file.search_by_name extensions must be a string list when provided")
        target = self._resolve_workspace_path(path)
        if not target.exists():
            raise DecisionValidationError(f"file.search_by_name path does not exist: {target}")
        if not target.is_dir():
            raise DecisionValidationError(f"file.search_by_name path must be a directory: {target}")

    def _validate_file_extract_text(self, arguments: dict) -> None:
        if "paths" not in arguments and isinstance(arguments.get("path"), str):
            arguments["paths"] = [arguments["path"]]
        paths = arguments.get("paths")
        encoding = arguments.get("encoding")
        max_chars = arguments.get("max_chars")
        max_rows_per_sheet = arguments.get("max_rows_per_sheet")
        if not isinstance(paths, list) or not paths:
            raise DecisionValidationError("file.extract_text requires a non-empty paths list")
        if "encoding" in arguments and not isinstance(encoding, str):
            raise DecisionValidationError("file.extract_text encoding must be a string when provided")
        if "max_chars" in arguments and (not isinstance(max_chars, int) or max_chars <= 0):
            raise DecisionValidationError("file.extract_text max_chars must be a positive integer when provided")
        if "max_rows_per_sheet" in arguments and (
            not isinstance(max_rows_per_sheet, int) or max_rows_per_sheet <= 0
        ):
            raise DecisionValidationError("file.extract_text max_rows_per_sheet must be a positive integer when provided")
        for raw_path in paths:
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise DecisionValidationError("file.extract_text paths must be non-empty strings")
            target = self._resolve_workspace_path(raw_path)
            if not target.exists():
                raise DecisionValidationError(f"file.extract_text path does not exist: {target}")
            if not target.is_file():
                raise DecisionValidationError(f"file.extract_text path must be a file: {target}")

    def _validate_file_extract_structure(self, arguments: dict) -> None:
        path = arguments.get("path")
        include_text = arguments.get("include_text")
        max_blocks = arguments.get("max_blocks")
        max_chars_per_block = arguments.get("max_chars_per_block")
        max_rows_per_sheet = arguments.get("max_rows_per_sheet")
        encoding = arguments.get("encoding")
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("file.extract_structure requires a non-empty path")
        if "encoding" in arguments and not isinstance(encoding, str):
            raise DecisionValidationError("file.extract_structure encoding must be a string when provided")
        if "include_text" in arguments and not isinstance(include_text, bool):
            raise DecisionValidationError("file.extract_structure include_text must be a boolean when provided")
        if "max_blocks" in arguments and (not isinstance(max_blocks, int) or max_blocks <= 0):
            raise DecisionValidationError("file.extract_structure max_blocks must be a positive integer when provided")
        if "max_chars_per_block" in arguments and (
            not isinstance(max_chars_per_block, int) or max_chars_per_block <= 0
        ):
            raise DecisionValidationError(
                "file.extract_structure max_chars_per_block must be a positive integer when provided"
            )
        if "max_rows_per_sheet" in arguments and (not isinstance(max_rows_per_sheet, int) or max_rows_per_sheet <= 0):
            raise DecisionValidationError(
                "file.extract_structure max_rows_per_sheet must be a positive integer when provided"
            )
        target = self._resolve_workspace_path(path)
        if not target.exists():
            raise DecisionValidationError(f"file.extract_structure path does not exist: {target}")
        if not target.is_file():
            raise DecisionValidationError(f"file.extract_structure path must be a file: {target}")

    def _validate_file_search_blocks(self, arguments: dict) -> None:
        path = arguments.get("path")
        query = arguments.get("query")
        terms = arguments.get("terms")
        max_matches = arguments.get("max_matches")
        max_blocks = arguments.get("max_blocks")
        max_chars_per_block = arguments.get("max_chars_per_block")
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("file.search_blocks requires a non-empty path")
        if "query" in arguments and query is not None and not isinstance(query, str):
            raise DecisionValidationError("file.search_blocks query must be a string when provided")
        if "terms" in arguments and (
            not isinstance(terms, list) or not all(isinstance(item, str) and item.strip() for item in terms)
        ):
            raise DecisionValidationError("file.search_blocks terms must be a non-empty string list when provided")
        if not (isinstance(query, str) and query.strip()) and not (
            isinstance(terms, list) and any(isinstance(item, str) and item.strip() for item in terms)
        ):
            raise DecisionValidationError("file.search_blocks requires query or terms")
        if "max_matches" in arguments and (not isinstance(max_matches, int) or max_matches <= 0):
            raise DecisionValidationError("file.search_blocks max_matches must be a positive integer when provided")
        if "max_blocks" in arguments and (not isinstance(max_blocks, int) or max_blocks <= 0):
            raise DecisionValidationError("file.search_blocks max_blocks must be a positive integer when provided")
        if "max_chars_per_block" in arguments and (
            not isinstance(max_chars_per_block, int) or max_chars_per_block <= 0
        ):
            raise DecisionValidationError(
                "file.search_blocks max_chars_per_block must be a positive integer when provided"
            )
        target = self._resolve_workspace_path(path)
        if not target.exists():
            raise DecisionValidationError(f"file.search_blocks path does not exist: {target}")
        if not target.is_file():
            raise DecisionValidationError(f"file.search_blocks path must be a file: {target}")

    def _validate_file_search_text(self, arguments: dict) -> None:
        path = arguments.get("path")
        query = arguments.get("query")
        terms = arguments.get("terms")
        recursive = arguments.get("recursive")
        max_matches = arguments.get("max_matches")
        match_mode = arguments.get("match_mode")
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("file.search_text requires a non-empty path")
        if "query" in arguments and query is not None and not isinstance(query, str):
            raise DecisionValidationError("file.search_text query must be a string when provided")
        if "terms" in arguments:
            if not isinstance(terms, list) or not all(isinstance(item, str) and item.strip() for item in terms):
                raise DecisionValidationError("file.search_text terms must be a non-empty string list when provided")
        if not (isinstance(query, str) and query.strip()) and not (
            isinstance(terms, list) and any(isinstance(item, str) and item.strip() for item in terms)
        ):
            raise DecisionValidationError("file.search_text requires a non-empty query or terms list")
        if "match_mode" in arguments and match_mode not in {"any", "all"}:
            raise DecisionValidationError("file.search_text match_mode must be 'any' or 'all' when provided")
        if "recursive" in arguments and not isinstance(recursive, bool):
            raise DecisionValidationError("file.search_text recursive must be a boolean when provided")
        if "max_matches" in arguments and (not isinstance(max_matches, int) or max_matches <= 0):
            raise DecisionValidationError("file.search_text max_matches must be a positive integer when provided")
        target = self._resolve_workspace_path(path)
        if not target.exists():
            raise DecisionValidationError(f"file.search_text path does not exist: {target}")

    def _validate_file_write(self, arguments: dict) -> None:
        path = arguments.get("path")
        content = arguments.get("content")
        encoding = arguments.get("encoding")
        overwrite = arguments.get("overwrite")
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("file.write requires a non-empty path")
        if not isinstance(content, str):
            raise DecisionValidationError("file.write requires content to be a string")
        if "encoding" in arguments and not isinstance(encoding, str):
            raise DecisionValidationError("file.write encoding must be a string when provided")
        if "overwrite" in arguments and not isinstance(overwrite, bool):
            raise DecisionValidationError("file.write overwrite must be a boolean when provided")
        if isinstance(content, str) and content.strip() in {"{paths}", "{candidates}", "{results}", "{candidate_paths}"}:
            raise DecisionValidationError("file.write content must not be an unresolved placeholder")
        self._resolve_workspace_path(path)

    def _validate_file_write_docx(self, arguments: dict) -> None:
        path = arguments.get("path")
        content = arguments.get("content", "")
        title = arguments.get("title")
        paragraphs = arguments.get("paragraphs")
        overwrite = arguments.get("overwrite")
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("file.write_docx requires a non-empty path")
        if path and not path.lower().endswith(".docx"):
            raise DecisionValidationError("file.write_docx path must end with .docx")
        if not isinstance(content, str):
            raise DecisionValidationError("file.write_docx content must be a string")
        if "title" in arguments and title is not None and not isinstance(title, str):
            raise DecisionValidationError("file.write_docx title must be a string when provided")
        if "paragraphs" in arguments and (
            not isinstance(paragraphs, list) or not all(isinstance(item, str) for item in paragraphs)
        ):
            raise DecisionValidationError("file.write_docx paragraphs must be a string list when provided")
        if "overwrite" in arguments and not isinstance(overwrite, bool):
            raise DecisionValidationError("file.write_docx overwrite must be a boolean when provided")
        self._resolve_workspace_path(path)

    def _validate_document_agent_compose(self, arguments: dict) -> None:
        instruction = arguments.get("instruction")
        output_path = arguments.get("output_path")
        title = arguments.get("title")
        recent_context = arguments.get("recent_context")
        source_materials = arguments.get("source_materials")
        resolved_facts = arguments.get("resolved_facts")
        style_hints = arguments.get("style_hints")
        max_chars = arguments.get("max_chars")
        if not isinstance(instruction, str) or not instruction.strip():
            raise DecisionValidationError("document_agent.compose requires a non-empty instruction")
        if "output_path" in arguments and output_path is not None:
            if not isinstance(output_path, str) or not output_path.strip():
                raise DecisionValidationError("document_agent.compose output_path must be a non-empty string when provided")
            self._resolve_workspace_path(output_path)
        if "title" in arguments and title is not None and not isinstance(title, str):
            raise DecisionValidationError("document_agent.compose title must be a string when provided")
        if "recent_context" in arguments and recent_context is not None and not isinstance(recent_context, str):
            raise DecisionValidationError("document_agent.compose recent_context must be a string when provided")
        if "source_materials" in arguments and source_materials is not None and not isinstance(source_materials, dict):
            raise DecisionValidationError("document_agent.compose source_materials must be an object when provided")
        if "resolved_facts" in arguments and resolved_facts is not None and not isinstance(resolved_facts, dict):
            raise DecisionValidationError("document_agent.compose resolved_facts must be an object when provided")
        if "style_hints" in arguments and style_hints is not None and not isinstance(style_hints, dict):
            raise DecisionValidationError("document_agent.compose style_hints must be an object when provided")
        if "max_chars" in arguments and (not isinstance(max_chars, int) or max_chars <= 0):
            raise DecisionValidationError("document_agent.compose max_chars must be a positive integer when provided")

    def _validate_file_edit_docx(self, arguments: dict) -> None:
        source_path = arguments.get("source_path")
        output_path = arguments.get("output_path")
        edits = arguments.get("edits")
        overwrite = arguments.get("overwrite")
        if not isinstance(source_path, str) or not source_path.strip():
            raise DecisionValidationError("file.edit_docx requires a non-empty source_path")
        if not isinstance(output_path, str) or not output_path.strip():
            raise DecisionValidationError("file.edit_docx requires a non-empty output_path")
        if not source_path.lower().endswith(".docx") or not output_path.lower().endswith(".docx"):
            raise DecisionValidationError("file.edit_docx source_path and output_path must end with .docx")
        if not isinstance(edits, list) or not edits:
            raise DecisionValidationError("file.edit_docx requires a non-empty edits list")
        for item in edits:
            if not isinstance(item, dict):
                raise DecisionValidationError("file.edit_docx edits must be objects")
            block_id = item.get("block_id")
            action = item.get("action", "replace")
            text = item.get("text", "")
            if not isinstance(block_id, str) or not block_id.strip():
                raise DecisionValidationError("file.edit_docx edit block_id must be a non-empty string")
            if action not in {"replace", "delete", "insert_after"}:
                raise DecisionValidationError("file.edit_docx action must be replace, delete, or insert_after")
            if "text" in item and not isinstance(text, str):
                raise DecisionValidationError("file.edit_docx edit text must be a string when provided")
        if "overwrite" in arguments and not isinstance(overwrite, bool):
            raise DecisionValidationError("file.edit_docx overwrite must be a boolean when provided")
        source_target = self._resolve_workspace_path(source_path)
        self._resolve_workspace_path(output_path)
        if not source_target.exists():
            raise DecisionValidationError(f"file.edit_docx source_path does not exist: {source_target}")

    def _validate_file_render_docx_from_template(self, arguments: dict) -> None:
        template_path = arguments.get("template_path")
        output_path = arguments.get("output_path")
        source_path = arguments.get("source_path")
        content = arguments.get("content", "")
        paragraphs = arguments.get("paragraphs")
        title = arguments.get("title")
        overwrite = arguments.get("overwrite")
        if not isinstance(template_path, str) or not template_path.strip():
            raise DecisionValidationError("file.render_docx_from_template requires a non-empty template_path")
        if not isinstance(output_path, str) or not output_path.strip():
            raise DecisionValidationError("file.render_docx_from_template requires a non-empty output_path")
        if not template_path.lower().endswith(".docx") or not output_path.lower().endswith(".docx"):
            raise DecisionValidationError(
                "file.render_docx_from_template template_path and output_path must end with .docx"
            )
        if "source_path" in arguments and source_path is not None and not isinstance(source_path, str):
            raise DecisionValidationError("file.render_docx_from_template source_path must be a string when provided")
        if "content" in arguments and not isinstance(content, str):
            raise DecisionValidationError("file.render_docx_from_template content must be a string when provided")
        if "paragraphs" in arguments and (
            not isinstance(paragraphs, list) or not all(isinstance(item, str) for item in paragraphs)
        ):
            raise DecisionValidationError(
                "file.render_docx_from_template paragraphs must be a string list when provided"
            )
        if "title" in arguments and title is not None and not isinstance(title, str):
            raise DecisionValidationError("file.render_docx_from_template title must be a string when provided")
        if "overwrite" in arguments and not isinstance(overwrite, bool):
            raise DecisionValidationError("file.render_docx_from_template overwrite must be a boolean when provided")
        has_source_path = isinstance(source_path, str) and source_path.strip()
        has_content = isinstance(content, str) and content.strip()
        has_paragraphs = isinstance(paragraphs, list) and any(isinstance(item, str) and item.strip() for item in paragraphs)
        if not (has_source_path or has_content or has_paragraphs):
            raise DecisionValidationError(
                "file.render_docx_from_template requires source_path, content, or paragraphs"
            )
        template_target = self._resolve_workspace_path(template_path)
        self._resolve_workspace_path(output_path)
        if not template_target.exists():
            raise DecisionValidationError(
                f"file.render_docx_from_template template_path does not exist: {template_target}"
            )
        if has_source_path:
            source_target = self._resolve_workspace_path(source_path)
            if not source_target.exists():
                raise DecisionValidationError(
                    f"file.render_docx_from_template source_path does not exist: {source_target}"
                )

    def _validate_document_agent_summarize(self, arguments: dict) -> None:
        source_path = arguments.get("source_path")
        instruction = arguments.get("instruction")
        recent_context = arguments.get("recent_context")
        grounded_inputs = arguments.get("grounded_inputs")
        max_chars = arguments.get("max_chars")
        if not isinstance(source_path, str) or not source_path.strip():
            raise DecisionValidationError("document_agent.summarize requires a non-empty source_path")
        if not isinstance(instruction, str) or not instruction.strip():
            raise DecisionValidationError("document_agent.summarize requires a non-empty instruction")
        if "recent_context" in arguments and recent_context is not None and not isinstance(recent_context, str):
            raise DecisionValidationError("document_agent.summarize recent_context must be a string when provided")
        if "grounded_inputs" in arguments and grounded_inputs is not None and not isinstance(grounded_inputs, dict):
            raise DecisionValidationError("document_agent.summarize grounded_inputs must be an object when provided")
        if "max_chars" in arguments and (not isinstance(max_chars, int) or max_chars <= 0):
            raise DecisionValidationError("document_agent.summarize max_chars must be a positive integer when provided")
        target = self._resolve_workspace_path(source_path)
        if not target.exists():
            raise DecisionValidationError(f"document_agent.summarize source_path does not exist: {target}")
        if not target.is_file():
            raise DecisionValidationError(f"document_agent.summarize source_path must be a file: {target}")

    def _validate_document_agent_read(self, arguments: dict) -> None:
        source_path = arguments.get("source_path")
        instruction = arguments.get("instruction")
        recent_context = arguments.get("recent_context")
        grounded_inputs = arguments.get("grounded_inputs")
        max_chars = arguments.get("max_chars")
        if not isinstance(source_path, str) or not source_path.strip():
            raise DecisionValidationError("document_agent.read requires a non-empty source_path")
        if "instruction" in arguments and instruction is not None and not isinstance(instruction, str):
            raise DecisionValidationError("document_agent.read instruction must be a string when provided")
        if "recent_context" in arguments and recent_context is not None and not isinstance(recent_context, str):
            raise DecisionValidationError("document_agent.read recent_context must be a string when provided")
        if "grounded_inputs" in arguments and grounded_inputs is not None and not isinstance(grounded_inputs, dict):
            raise DecisionValidationError("document_agent.read grounded_inputs must be an object when provided")
        if "max_chars" in arguments and (not isinstance(max_chars, int) or max_chars <= 0):
            raise DecisionValidationError("document_agent.read max_chars must be a positive integer when provided")
        target = self._resolve_workspace_path(source_path)
        if not target.exists():
            raise DecisionValidationError(f"document_agent.read source_path does not exist: {target}")
        if not target.is_file():
            raise DecisionValidationError(f"document_agent.read source_path must be a file: {target}")

    def _validate_document_agent_inspect(self, arguments: dict) -> None:
        source_path = arguments.get("source_path")
        instruction = arguments.get("instruction")
        recent_context = arguments.get("recent_context")
        grounded_inputs = arguments.get("grounded_inputs")
        max_blocks = arguments.get("max_blocks")
        max_chars_per_block = arguments.get("max_chars_per_block")
        max_matches = arguments.get("max_matches")
        if not isinstance(source_path, str) or not source_path.strip():
            raise DecisionValidationError("document_agent.inspect requires a non-empty source_path")
        if not isinstance(instruction, str) or not instruction.strip():
            raise DecisionValidationError("document_agent.inspect requires a non-empty instruction")
        if "recent_context" in arguments and recent_context is not None and not isinstance(recent_context, str):
            raise DecisionValidationError("document_agent.inspect recent_context must be a string when provided")
        if "grounded_inputs" in arguments and grounded_inputs is not None and not isinstance(grounded_inputs, dict):
            raise DecisionValidationError("document_agent.inspect grounded_inputs must be an object when provided")
        if "max_blocks" in arguments and (not isinstance(max_blocks, int) or max_blocks <= 0):
            raise DecisionValidationError("document_agent.inspect max_blocks must be a positive integer when provided")
        if "max_chars_per_block" in arguments and (
            not isinstance(max_chars_per_block, int) or max_chars_per_block <= 0
        ):
            raise DecisionValidationError(
                "document_agent.inspect max_chars_per_block must be a positive integer when provided"
            )
        if "max_matches" in arguments and (not isinstance(max_matches, int) or max_matches <= 0):
            raise DecisionValidationError("document_agent.inspect max_matches must be a positive integer when provided")
        target = self._resolve_workspace_path(source_path)
        if not target.exists():
            raise DecisionValidationError(f"document_agent.inspect source_path does not exist: {target}")
        if not target.is_file():
            raise DecisionValidationError(f"document_agent.inspect source_path must be a file: {target}")

    def _validate_document_agent_edit(self, arguments: dict) -> None:
        source_path = arguments.get("source_path")
        output_path = arguments.get("output_path")
        instruction = arguments.get("instruction")
        recent_context = arguments.get("recent_context")
        grounded_inputs = arguments.get("grounded_inputs")
        allow_overwrite = arguments.get("allow_overwrite")
        preserve_structure = arguments.get("preserve_structure")
        preserve_style = arguments.get("preserve_style")
        max_chars = arguments.get("max_chars")
        max_blocks = arguments.get("max_blocks")
        max_chars_per_block = arguments.get("max_chars_per_block")
        if not isinstance(source_path, str) or not source_path.strip():
            raise DecisionValidationError("document_agent.edit requires a non-empty source_path")
        if not isinstance(instruction, str) or not instruction.strip():
            raise DecisionValidationError("document_agent.edit requires a non-empty instruction")
        if "output_path" in arguments and output_path is not None and (not isinstance(output_path, str) or not output_path.strip()):
            raise DecisionValidationError("document_agent.edit output_path must be a non-empty string when provided")
        if "recent_context" in arguments and recent_context is not None and not isinstance(recent_context, str):
            raise DecisionValidationError("document_agent.edit recent_context must be a string when provided")
        if "grounded_inputs" in arguments and grounded_inputs is not None and not isinstance(grounded_inputs, dict):
            raise DecisionValidationError("document_agent.edit grounded_inputs must be an object when provided")
        if "allow_overwrite" in arguments and not isinstance(allow_overwrite, bool):
            raise DecisionValidationError("document_agent.edit allow_overwrite must be a boolean when provided")
        if "preserve_structure" in arguments and not isinstance(preserve_structure, bool):
            raise DecisionValidationError("document_agent.edit preserve_structure must be a boolean when provided")
        if "preserve_style" in arguments and not isinstance(preserve_style, bool):
            raise DecisionValidationError("document_agent.edit preserve_style must be a boolean when provided")
        if "max_chars" in arguments and (not isinstance(max_chars, int) or max_chars <= 0):
            raise DecisionValidationError("document_agent.edit max_chars must be a positive integer when provided")
        if "max_blocks" in arguments and (not isinstance(max_blocks, int) or max_blocks <= 0):
            raise DecisionValidationError("document_agent.edit max_blocks must be a positive integer when provided")
        if "max_chars_per_block" in arguments and (
            not isinstance(max_chars_per_block, int) or max_chars_per_block <= 0
        ):
            raise DecisionValidationError(
                "document_agent.edit max_chars_per_block must be a positive integer when provided"
            )
        source_target = self._resolve_workspace_path(source_path)
        if not source_target.exists():
            raise DecisionValidationError(f"document_agent.edit source_path does not exist: {source_target}")
        if not source_target.is_file():
            raise DecisionValidationError(f"document_agent.edit source_path must be a file: {source_target}")
        if isinstance(output_path, str) and output_path.strip():
            self._resolve_workspace_path(output_path)

    def _validate_file_write_xlsx(self, arguments: dict) -> None:
        path = arguments.get("path")
        content = arguments.get("content", "")
        title = arguments.get("title")
        rows = arguments.get("rows")
        sheet_name = arguments.get("sheet_name")
        overwrite = arguments.get("overwrite")
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("file.write_xlsx requires a non-empty path")
        if path and not path.lower().endswith(".xlsx"):
            raise DecisionValidationError("file.write_xlsx path must end with .xlsx")
        if not isinstance(content, str):
            raise DecisionValidationError("file.write_xlsx content must be a string")
        if "title" in arguments and title is not None and not isinstance(title, str):
            raise DecisionValidationError("file.write_xlsx title must be a string when provided")
        if "sheet_name" in arguments and sheet_name is not None and not isinstance(sheet_name, str):
            raise DecisionValidationError("file.write_xlsx sheet_name must be a string when provided")
        if "rows" in arguments:
            if not isinstance(rows, list) or not all(isinstance(row, list) for row in rows):
                raise DecisionValidationError("file.write_xlsx rows must be a list of rows when provided")
        if "overwrite" in arguments and not isinstance(overwrite, bool):
            raise DecisionValidationError("file.write_xlsx overwrite must be a boolean when provided")
        self._resolve_workspace_path(path)

    def _validate_file_write_pptx(self, arguments: dict) -> None:
        path = arguments.get("path")
        content = arguments.get("content", "")
        title = arguments.get("title")
        bullets = arguments.get("bullets")
        overwrite = arguments.get("overwrite")
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("file.write_pptx requires a non-empty path")
        if path and not path.lower().endswith(".pptx"):
            raise DecisionValidationError("file.write_pptx path must end with .pptx")
        if not isinstance(content, str):
            raise DecisionValidationError("file.write_pptx content must be a string")
        if "title" in arguments and title is not None and not isinstance(title, str):
            raise DecisionValidationError("file.write_pptx title must be a string when provided")
        if "bullets" in arguments and (
            not isinstance(bullets, list) or not all(isinstance(item, str) for item in bullets)
        ):
            raise DecisionValidationError("file.write_pptx bullets must be a string list when provided")
        if "overwrite" in arguments and not isinstance(overwrite, bool):
            raise DecisionValidationError("file.write_pptx overwrite must be a boolean when provided")
        self._resolve_workspace_path(path)

    def _validate_file_append(self, arguments: dict) -> None:
        path = arguments.get("path")
        content = arguments.get("content")
        encoding = arguments.get("encoding")
        create = arguments.get("create")
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("file.append requires a non-empty path")
        if not isinstance(content, str):
            raise DecisionValidationError("file.append requires content to be a string")
        if "encoding" in arguments and not isinstance(encoding, str):
            raise DecisionValidationError("file.append encoding must be a string when provided")
        if "create" in arguments and not isinstance(create, bool):
            raise DecisionValidationError("file.append create must be a boolean when provided")
        self._resolve_workspace_path(path)

    @staticmethod
    def _validate_continue_on_error(arguments: dict, tool_name: str) -> None:
        if "continue_on_error" in arguments and not isinstance(arguments.get("continue_on_error"), bool):
            raise DecisionValidationError(f"{tool_name} continue_on_error must be a boolean when provided")

    @staticmethod
    def _require_non_empty_items(arguments: dict, tool_name: str) -> list[dict]:
        items = arguments.get("items")
        if not isinstance(items, list) or not items:
            raise DecisionValidationError(f"{tool_name} requires a non-empty items list")
        if not all(isinstance(item, dict) for item in items):
            raise DecisionValidationError(f"{tool_name} items must be objects")
        return items

    @staticmethod
    def _require_non_empty_paths(arguments: dict, tool_name: str) -> list[str]:
        paths = arguments.get("paths")
        if not isinstance(paths, list) or not paths:
            raise DecisionValidationError(f"{tool_name} requires a non-empty paths list")
        if not all(isinstance(item, str) and item.strip() for item in paths):
            raise DecisionValidationError(f"{tool_name} paths must be non-empty strings")
        return paths

    def _validate_file_write_many(self, arguments: dict) -> None:
        self._validate_continue_on_error(arguments, "file.write_many")
        items = self._require_non_empty_items(arguments, "file.write_many")
        for item in items:
            self._validate_file_write(item)

    def _validate_file_append_many(self, arguments: dict) -> None:
        self._validate_continue_on_error(arguments, "file.append_many")
        items = self._require_non_empty_items(arguments, "file.append_many")
        for item in items:
            self._validate_file_append(item)

    def _validate_file_metadata_many(self, arguments: dict) -> None:
        self._validate_continue_on_error(arguments, "file.metadata_many")
        for raw_path in self._require_non_empty_paths(arguments, "file.metadata_many"):
            self._validate_file_metadata({"path": raw_path})

    def _validate_file_preview_many(self, arguments: dict) -> None:
        self._validate_continue_on_error(arguments, "file.preview_many")
        preview_args = {
            "encoding": arguments.get("encoding"),
            "max_chars": arguments.get("max_chars"),
            "max_children": arguments.get("max_children"),
        }
        for raw_path in self._require_non_empty_paths(arguments, "file.preview_many"):
            self._validate_file_preview({"path": raw_path, **{k: v for k, v in preview_args.items() if v is not None}})

    def _validate_file_mkdir_many(self, arguments: dict) -> None:
        self._validate_continue_on_error(arguments, "file.mkdir_many")
        mkdir_args = {
            "exist_ok": arguments.get("exist_ok"),
            "parents": arguments.get("parents"),
        }
        for raw_path in self._require_non_empty_paths(arguments, "file.mkdir_many"):
            self._validate_file_mkdir({"path": raw_path, **{k: v for k, v in mkdir_args.items() if v is not None}})

    def _validate_file_copy_many(self, arguments: dict) -> None:
        self._validate_continue_on_error(arguments, "file.copy_many")
        items = self._require_non_empty_items(arguments, "file.copy_many")
        for item in items:
            self._validate_file_copy(item)

    def _validate_file_move_many(self, arguments: dict) -> None:
        self._validate_continue_on_error(arguments, "file.move_many")
        items = self._require_non_empty_items(arguments, "file.move_many")
        for item in items:
            self._validate_file_move(item)

    def _validate_file_rename_many(self, arguments: dict) -> None:
        self._validate_continue_on_error(arguments, "file.rename_many")
        items = self._require_non_empty_items(arguments, "file.rename_many")
        for item in items:
            self._validate_file_rename(item)

    def _validate_file_delete_many(self, arguments: dict) -> None:
        self._validate_continue_on_error(arguments, "file.delete_many")
        delete_args = {
            "recursive": arguments.get("recursive"),
            "missing_ok": arguments.get("missing_ok"),
        }
        for raw_path in self._require_non_empty_paths(arguments, "file.delete_many"):
            self._validate_file_delete({"path": raw_path, **{k: v for k, v in delete_args.items() if v is not None}})

    def _validate_file_open_many(self, arguments: dict) -> None:
        self._validate_continue_on_error(arguments, "file.open_many")
        for raw_path in self._require_non_empty_paths(arguments, "file.open_many"):
            self._validate_file_open_like({"path": raw_path})

    def _validate_file_reveal_many(self, arguments: dict) -> None:
        self._validate_continue_on_error(arguments, "file.reveal_many")
        for raw_path in self._require_non_empty_paths(arguments, "file.reveal_many"):
            self._validate_file_open_like({"path": raw_path})

    def _validate_file_metadata(self, arguments: dict) -> None:
        path = arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("file.metadata requires a non-empty path")
        target = self._resolve_workspace_path(path)
        if not target.exists():
            raise DecisionValidationError(f"file.metadata path does not exist: {target}")

    def _validate_file_preview(self, arguments: dict) -> None:
        path = arguments.get("path")
        max_chars = arguments.get("max_chars")
        max_children = arguments.get("max_children")
        encoding = arguments.get("encoding")
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("file.preview requires a non-empty path")
        if "encoding" in arguments and not isinstance(encoding, str):
            raise DecisionValidationError("file.preview encoding must be a string when provided")
        if "max_chars" in arguments and (not isinstance(max_chars, int) or max_chars <= 0):
            raise DecisionValidationError("file.preview max_chars must be a positive integer when provided")
        if "max_children" in arguments and (not isinstance(max_children, int) or max_children <= 0):
            raise DecisionValidationError("file.preview max_children must be a positive integer when provided")
        target = self._resolve_workspace_path(path)
        if not target.exists():
            raise DecisionValidationError(f"file.preview path does not exist: {target}")

    def _validate_file_mkdir(self, arguments: dict) -> None:
        path = arguments.get("path")
        exist_ok = arguments.get("exist_ok")
        parents = arguments.get("parents")
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("file.mkdir requires a non-empty path")
        if "exist_ok" in arguments and not isinstance(exist_ok, bool):
            raise DecisionValidationError("file.mkdir exist_ok must be a boolean when provided")
        if "parents" in arguments and not isinstance(parents, bool):
            raise DecisionValidationError("file.mkdir parents must be a boolean when provided")
        self._resolve_workspace_path(path)

    def _validate_file_copy(self, arguments: dict) -> None:
        src_path = arguments.get("src_path")
        dest_path = arguments.get("dest_path")
        overwrite = arguments.get("overwrite")
        if not isinstance(src_path, str) or not src_path.strip():
            raise DecisionValidationError("file.copy requires a non-empty src_path")
        if not isinstance(dest_path, str) or not dest_path.strip():
            raise DecisionValidationError("file.copy requires a non-empty dest_path")
        if "overwrite" in arguments and not isinstance(overwrite, bool):
            raise DecisionValidationError("file.copy overwrite must be a boolean when provided")
        src = self._resolve_workspace_path(src_path)
        self._resolve_workspace_path(dest_path)
        if not src.exists():
            raise DecisionValidationError(f"file.copy src_path does not exist: {src}")

    def _validate_file_move(self, arguments: dict) -> None:
        src_path = arguments.get("src_path")
        dest_path = arguments.get("dest_path")
        overwrite = arguments.get("overwrite")
        if not isinstance(src_path, str) or not src_path.strip():
            raise DecisionValidationError("file.move requires a non-empty src_path")
        if not isinstance(dest_path, str) or not dest_path.strip():
            raise DecisionValidationError("file.move requires a non-empty dest_path")
        if "overwrite" in arguments and not isinstance(overwrite, bool):
            raise DecisionValidationError("file.move overwrite must be a boolean when provided")
        src = self._resolve_workspace_path(src_path)
        self._resolve_workspace_path(dest_path)
        if not src.exists():
            raise DecisionValidationError(f"file.move src_path does not exist: {src}")
        if src == self.workspace_root:
            raise DecisionValidationError("file.move cannot target the workspace root")

    def _validate_file_rename(self, arguments: dict) -> None:
        path = arguments.get("path")
        new_name = arguments.get("new_name")
        overwrite = arguments.get("overwrite")
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("file.rename requires a non-empty path")
        if not isinstance(new_name, str) or not new_name.strip():
            raise DecisionValidationError("file.rename requires a non-empty new_name")
        if any(sep in new_name for sep in ("/", "\\")):
            raise DecisionValidationError("file.rename new_name must not contain path separators")
        if "overwrite" in arguments and not isinstance(overwrite, bool):
            raise DecisionValidationError("file.rename overwrite must be a boolean when provided")
        target = self._resolve_workspace_path(path)
        if not target.exists():
            raise DecisionValidationError(f"file.rename path does not exist: {target}")
        if target == self.workspace_root:
            raise DecisionValidationError("file.rename cannot target the workspace root")

    def _validate_file_delete(self, arguments: dict) -> None:
        path = arguments.get("path")
        recursive = arguments.get("recursive")
        missing_ok = arguments.get("missing_ok")
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("file.delete requires a non-empty path")
        if "recursive" in arguments and not isinstance(recursive, bool):
            raise DecisionValidationError("file.delete recursive must be a boolean when provided")
        if "missing_ok" in arguments and not isinstance(missing_ok, bool):
            raise DecisionValidationError("file.delete missing_ok must be a boolean when provided")
        target = self._resolve_workspace_path(path)
        if target == self.workspace_root:
            raise DecisionValidationError("file.delete cannot target the workspace root")
        if not target.exists() and not bool(missing_ok):
            raise DecisionValidationError(f"file.delete path does not exist: {target}")

    def _validate_file_open_like(self, arguments: dict) -> None:
        path = arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("file.open_path requires a non-empty path")
        target = self._resolve_workspace_path(path)
        if not target.exists():
            raise DecisionValidationError(f"file.open_path path does not exist: {target}")

    def _validate_image_inspect(self, arguments: dict) -> None:
        path = arguments.get("path")
        include_ocr = arguments.get("include_ocr")
        ocr_max_chars = arguments.get("ocr_max_chars")
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("image.inspect requires a non-empty path")
        if "include_ocr" in arguments and not isinstance(include_ocr, bool):
            raise DecisionValidationError("image.inspect include_ocr must be a boolean when provided")
        if "ocr_max_chars" in arguments and (not isinstance(ocr_max_chars, int) or ocr_max_chars <= 0):
            raise DecisionValidationError("image.inspect ocr_max_chars must be a positive integer when provided")
        target = self._resolve_workspace_path(path)
        if not target.exists():
            raise DecisionValidationError(f"image.inspect path does not exist: {target}")
        if not target.is_file():
            raise DecisionValidationError(f"image.inspect path must be a file: {target}")

    def _validate_image_describe(self, arguments: dict) -> None:
        path = arguments.get("path")
        focus = arguments.get("focus")
        include_ocr = arguments.get("include_ocr")
        ocr_max_chars = arguments.get("ocr_max_chars")
        max_description_chars = arguments.get("max_description_chars")
        prompt = arguments.get("prompt")
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("image.describe requires a non-empty path")
        if "focus" in arguments and focus not in {"general", "ui", "ocr"}:
            raise DecisionValidationError("image.describe focus must be one of: general, ui, ocr")
        if "include_ocr" in arguments and not isinstance(include_ocr, bool):
            raise DecisionValidationError("image.describe include_ocr must be a boolean when provided")
        if "ocr_max_chars" in arguments and (not isinstance(ocr_max_chars, int) or ocr_max_chars <= 0):
            raise DecisionValidationError("image.describe ocr_max_chars must be a positive integer when provided")
        if "max_description_chars" in arguments and (
            not isinstance(max_description_chars, int) or max_description_chars <= 0
        ):
            raise DecisionValidationError("image.describe max_description_chars must be a positive integer when provided")
        if "prompt" in arguments and prompt is not None and not isinstance(prompt, str):
            raise DecisionValidationError("image.describe prompt must be a string when provided")
        target = self._resolve_workspace_path(path)
        if not target.exists():
            raise DecisionValidationError(f"image.describe path does not exist: {target}")
        if not target.is_file():
            raise DecisionValidationError(f"image.describe path must be a file: {target}")

    def _validate_image_read_text(self, arguments: dict) -> None:
        paths = arguments.get("paths")
        max_chars = arguments.get("max_chars")
        if not isinstance(paths, list) or not paths:
            raise DecisionValidationError("image.read_text requires a non-empty paths list")
        if "max_chars" in arguments and (not isinstance(max_chars, int) or max_chars <= 0):
            raise DecisionValidationError("image.read_text max_chars must be a positive integer when provided")
        for raw_path in paths:
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise DecisionValidationError("image.read_text paths must be non-empty strings")
            target = self._resolve_workspace_path(raw_path)
            if not target.exists():
                raise DecisionValidationError(f"image.read_text path does not exist: {target}")
            if not target.is_file():
                raise DecisionValidationError(f"image.read_text path must be a file: {target}")

    def _validate_image_capture_screen(self, arguments: dict) -> None:
        output_path = arguments.get("output_path")
        delay_ms = arguments.get("delay_ms")
        all_screens = arguments.get("all_screens")
        if not isinstance(output_path, str) or not output_path.strip():
            raise DecisionValidationError("image.capture_screen requires a non-empty output_path")
        if "delay_ms" in arguments and (not isinstance(delay_ms, int) or delay_ms < 0):
            raise DecisionValidationError("image.capture_screen delay_ms must be a non-negative integer when provided")
        if "all_screens" in arguments and not isinstance(all_screens, bool):
            raise DecisionValidationError("image.capture_screen all_screens must be a boolean when provided")
        self._resolve_workspace_path(output_path)

    def _validate_image_capture_region(self, arguments: dict) -> None:
        output_path = arguments.get("output_path")
        for field in ("x", "y", "width", "height"):
            value = arguments.get(field)
            if not isinstance(value, int):
                raise DecisionValidationError(f"image.capture_region {field} must be an integer")
        if arguments.get("width", 0) <= 0 or arguments.get("height", 0) <= 0:
            raise DecisionValidationError("image.capture_region width and height must be positive integers")
        delay_ms = arguments.get("delay_ms")
        if not isinstance(output_path, str) or not output_path.strip():
            raise DecisionValidationError("image.capture_region requires a non-empty output_path")
        if "delay_ms" in arguments and (not isinstance(delay_ms, int) or delay_ms < 0):
            raise DecisionValidationError("image.capture_region delay_ms must be a non-negative integer when provided")
        self._resolve_workspace_path(output_path)

    def _validate_memory_remember(self, arguments: dict) -> None:
        content = arguments.get("content")
        if not isinstance(content, str) or not content.strip():
            raise DecisionValidationError("memory.remember requires non-empty content")

    def _validate_memory_recall(self, arguments: dict) -> None:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise DecisionValidationError("memory.recall requires non-empty query")

    def _validate_web_search(self, arguments: dict) -> None:
        query = arguments.get("query")
        max_results = arguments.get("max_results", 5)
        domains = arguments.get("domains", [])
        recency_days = arguments.get("recency_days")
        language = arguments.get("language")
        if not isinstance(query, str) or not query.strip():
            raise DecisionValidationError("web.search requires non-empty query")
        if "max_results" in arguments and (not isinstance(max_results, int) or max_results <= 0):
            raise DecisionValidationError("web.search max_results must be a positive integer when provided")
        self._validate_web_domains(domains, tool_name="web.search")
        if "recency_days" in arguments and recency_days is not None and (
            not isinstance(recency_days, int) or recency_days < 0
        ):
            raise DecisionValidationError("web.search recency_days must be a non-negative integer when provided")
        if "language" in arguments and language is not None and not isinstance(language, str):
            raise DecisionValidationError("web.search language must be a string when provided")

    def _validate_web_fetch(self, arguments: dict) -> None:
        url = arguments.get("url")
        max_chars = arguments.get("max_chars", 8000)
        allow_insecure = arguments.get("allow_insecure")
        prefer_browser = arguments.get("prefer_browser")
        if not isinstance(url, str) or not url.strip():
            raise DecisionValidationError("web.fetch requires non-empty url")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise DecisionValidationError(f"web.fetch received invalid url: {url}")
        if "max_chars" in arguments and (not isinstance(max_chars, int) or max_chars <= 0):
            raise DecisionValidationError("web.fetch max_chars must be a positive integer when provided")
        if "allow_insecure" in arguments and not isinstance(allow_insecure, bool):
            raise DecisionValidationError("web.fetch allow_insecure must be a boolean when provided")
        if "prefer_browser" in arguments and not isinstance(prefer_browser, bool):
            raise DecisionValidationError("web.fetch prefer_browser must be a boolean when provided")

    def _validate_web_open_page(self, arguments: dict) -> None:
        url = arguments.get("url")
        if not isinstance(url, str) or not url.strip():
            raise DecisionValidationError("web.open_page requires non-empty url")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise DecisionValidationError(f"web.open_page received invalid url: {url}")

    def _validate_web_research(self, arguments: dict) -> None:
        query = arguments.get("query")
        max_results = arguments.get("max_results", 5)
        max_pages = arguments.get("max_pages", 3)
        max_chars = arguments.get("max_chars", 4000)
        domains = arguments.get("domains", [])
        recency_days = arguments.get("recency_days")
        language = arguments.get("language")
        allow_insecure = arguments.get("allow_insecure")
        prefer_browser = arguments.get("prefer_browser")
        if not isinstance(query, str) or not query.strip():
            raise DecisionValidationError("web.research requires non-empty query")
        if "max_results" in arguments and (not isinstance(max_results, int) or max_results <= 0):
            raise DecisionValidationError("web.research max_results must be a positive integer when provided")
        if "max_pages" in arguments and (not isinstance(max_pages, int) or max_pages <= 0):
            raise DecisionValidationError("web.research max_pages must be a positive integer when provided")
        if "max_chars" in arguments and (not isinstance(max_chars, int) or max_chars <= 0):
            raise DecisionValidationError("web.research max_chars must be a positive integer when provided")
        self._validate_web_domains(domains, tool_name="web.research")
        if "recency_days" in arguments and recency_days is not None and (
            not isinstance(recency_days, int) or recency_days < 0
        ):
            raise DecisionValidationError("web.research recency_days must be a non-negative integer when provided")
        if "language" in arguments and language is not None and not isinstance(language, str):
            raise DecisionValidationError("web.research language must be a string when provided")
        if "allow_insecure" in arguments and not isinstance(allow_insecure, bool):
            raise DecisionValidationError("web.research allow_insecure must be a boolean when provided")
        if "prefer_browser" in arguments and not isinstance(prefer_browser, bool):
            raise DecisionValidationError("web.research prefer_browser must be a boolean when provided")

    @staticmethod
    def _validate_web_domains(domains: object, *, tool_name: str) -> None:
        if domains is None:
            return
        if not isinstance(domains, list) or not all(isinstance(item, str) and item.strip() for item in domains):
            raise DecisionValidationError(f"{tool_name} domains must be a non-empty string list when provided")

    def _validate_retrieval_rebuild_local_index(self, arguments: dict) -> None:
        if arguments not in ({}, None):
            raise DecisionValidationError("retrieval.rebuild_local_index does not accept arguments")

    def _validate_retrieval_sync_local_index(self, arguments: dict) -> None:
        if arguments not in ({}, None):
            raise DecisionValidationError("retrieval.sync_local_index does not accept arguments")

    def _validate_retrieval_search_local_objects(self, arguments: dict) -> None:
        query = arguments.get("query")
        target_kind = arguments.get("target_kind", "any")
        top_k = arguments.get("top_k", 8)
        path_scope = arguments.get("path_scope", ".")
        scope_mode = arguments.get("scope_mode", "subtree")
        extensions = arguments.get("extensions", [])
        rebuild_if_missing = arguments.get("rebuild_if_missing", True)

        normalized_target_kind = str(target_kind).strip().lower()
        normalized_target_kind = {
            "dir": "folder",
            "dirs": "folder",
            "directory": "folder",
            "directories": "folder",
            "folder": "folder",
            "folders": "folder",
            "file": "file",
            "files": "file",
            "any": "any",
        }.get(normalized_target_kind, normalized_target_kind)

        if not isinstance(query, str) or not query.strip():
            raise DecisionValidationError("retrieval.search_local_objects requires non-empty query")
        if normalized_target_kind not in {"any", "file", "folder"}:
            raise DecisionValidationError("retrieval.search_local_objects target_kind must be any, file, or folder")
        if not isinstance(top_k, int) or top_k <= 0:
            raise DecisionValidationError("retrieval.search_local_objects top_k must be a positive integer")
        if not isinstance(path_scope, str) or not path_scope.strip():
            raise DecisionValidationError("retrieval.search_local_objects path_scope must be a non-empty string")
        self._resolve_workspace_path(path_scope)
        if "scope_mode" in arguments and str(scope_mode).strip().lower() not in {"subtree", "shallow_first"}:
            raise DecisionValidationError("retrieval.search_local_objects scope_mode must be subtree or shallow_first")
        if "extensions" in arguments and (
            not isinstance(extensions, list) or not all(isinstance(item, str) and item.strip() for item in extensions)
        ):
            raise DecisionValidationError("retrieval.search_local_objects extensions must be a string list when provided")
        if "rebuild_if_missing" in arguments and not isinstance(rebuild_if_missing, bool):
            raise DecisionValidationError("retrieval.search_local_objects rebuild_if_missing must be boolean when provided")

    def _validate_retrieval_inspect_local_candidate(self, arguments: dict) -> None:
        path = arguments.get("path")
        max_chars = arguments.get("max_chars", 1200)
        max_children = arguments.get("max_children", 12)
        if not isinstance(path, str) or not path.strip():
            raise DecisionValidationError("retrieval.inspect_local_candidate requires a non-empty path")
        target = self._resolve_workspace_path(path)
        if not target.exists():
            raise DecisionValidationError(f"retrieval.inspect_local_candidate path does not exist: {target}")
        if "max_chars" in arguments and (not isinstance(max_chars, int) or max_chars <= 0):
            raise DecisionValidationError("retrieval.inspect_local_candidate max_chars must be a positive integer")
        if "max_children" in arguments and (not isinstance(max_children, int) or max_children <= 0):
            raise DecisionValidationError("retrieval.inspect_local_candidate max_children must be a positive integer")

    def _validate_qq_get_current_context(self, arguments: dict) -> None:
        if arguments not in ({}, None):
            raise DecisionValidationError("qq.get_current_context does not accept arguments")

    def _validate_qq_get_recent_messages(self, arguments: dict) -> None:
        limit = arguments.get("limit", 8)
        include_assistant = arguments.get("include_assistant")
        if "limit" in arguments and (not isinstance(limit, int) or limit <= 0):
            raise DecisionValidationError("qq.get_recent_messages limit must be a positive integer when provided")
        if "include_assistant" in arguments and not isinstance(include_assistant, bool):
            raise DecisionValidationError("qq.get_recent_messages include_assistant must be boolean when provided")

    def _validate_qq_get_last_reply(self, arguments: dict) -> None:
        contact_query = arguments.get("contact_query")
        if "contact_query" in arguments and contact_query is not None and not isinstance(contact_query, str):
            raise DecisionValidationError("qq.get_last_reply contact_query must be a string when provided")

    def _validate_qq_search_history(self, arguments: dict) -> None:
        query = arguments.get("query")
        contact_query = arguments.get("contact_query")
        limit = arguments.get("limit", 5)
        if "query" in arguments and query is not None and not isinstance(query, str):
            raise DecisionValidationError("qq.search_history query must be a string when provided")
        if "contact_query" in arguments and contact_query is not None and not isinstance(contact_query, str):
            raise DecisionValidationError("qq.search_history contact_query must be a string when provided")
        if not (isinstance(query, str) and query.strip()) and not (isinstance(contact_query, str) and contact_query.strip()):
            raise DecisionValidationError("qq.search_history requires a non-empty query or contact_query")
        if "limit" in arguments and (not isinstance(limit, int) or limit <= 0):
            raise DecisionValidationError("qq.search_history limit must be a positive integer when provided")

    def _validate_qq_get_recent_attachments(self, arguments: dict) -> None:
        contact_query = arguments.get("contact_query")
        kind = str(arguments.get("kind", "any")).strip().lower()
        limit = arguments.get("limit", 5)
        if "contact_query" in arguments and contact_query is not None and not isinstance(contact_query, str):
            raise DecisionValidationError("qq.get_recent_attachments contact_query must be a string when provided")
        if kind not in {"any", "image", "file", "audio"}:
            raise DecisionValidationError("qq.get_recent_attachments kind must be any, image, file, or audio")
        if "limit" in arguments and (not isinstance(limit, int) or limit <= 0):
            raise DecisionValidationError("qq.get_recent_attachments limit must be a positive integer when provided")

    def _validate_qq_search_contacts(self, arguments: dict) -> None:
        query = arguments.get("query")
        target_kind = str(arguments.get("target_kind", "any")).strip().lower()
        limit = arguments.get("limit", 5)
        exclude_sender = arguments.get("exclude_sender")
        if not isinstance(query, str) or not query.strip():
            raise DecisionValidationError("qq.search_contacts requires a non-empty query")
        if target_kind not in {"any", "friend", "group"}:
            raise DecisionValidationError("qq.search_contacts target_kind must be any, friend, or group")
        if "limit" in arguments and (not isinstance(limit, int) or limit <= 0):
            raise DecisionValidationError("qq.search_contacts limit must be a positive integer when provided")
        if "exclude_sender" in arguments and not isinstance(exclude_sender, bool):
            raise DecisionValidationError("qq.search_contacts exclude_sender must be boolean when provided")

    def _validate_qq_send_text(self, arguments: dict) -> None:
        message = arguments.get("message")
        target_kind = str(arguments.get("target_kind", "current")).strip().lower()
        target_id = arguments.get("target_id")
        if not isinstance(message, str) or not message.strip():
            raise DecisionValidationError("qq.send_text requires a non-empty message")
        self._validate_qq_target(target_kind, target_id, tool_name="qq.send_text")

    def _validate_qq_send_file(self, arguments: dict) -> None:
        file_path = arguments.get("file_path")
        target_kind = str(arguments.get("target_kind", "current")).strip().lower()
        target_id = arguments.get("target_id")
        if not isinstance(file_path, str) or not file_path.strip():
            raise DecisionValidationError("qq.send_file requires a non-empty file_path")
        candidate = Path(file_path).expanduser().resolve()
        if not candidate.is_file():
            raise DecisionValidationError(f"qq.send_file file_path does not exist: {candidate}")
        self._validate_qq_target(target_kind, target_id, tool_name="qq.send_file")

    def _validate_qq_send_voice(self, arguments: dict) -> None:
        speech_text = arguments.get("speech_text")
        audio_path = arguments.get("audio_path")
        target_kind = str(arguments.get("target_kind", "current")).strip().lower()
        target_id = arguments.get("target_id")
        if not (isinstance(speech_text, str) and speech_text.strip()) and not (
            isinstance(audio_path, str) and audio_path.strip()
        ):
            raise DecisionValidationError("qq.send_voice requires speech_text or audio_path")
        if isinstance(audio_path, str) and audio_path.strip():
            candidate = Path(audio_path).expanduser().resolve()
            if not candidate.is_file():
                raise DecisionValidationError(f"qq.send_voice audio_path does not exist: {candidate}")
        self._validate_qq_target(target_kind, target_id, tool_name="qq.send_voice")

    @staticmethod
    def _validate_qq_target(target_kind: str, target_id: object, *, tool_name: str) -> None:
        if target_kind not in {"current", "friend", "group"}:
            raise DecisionValidationError(f"{tool_name} target_kind must be current, friend, or group")
        if target_kind == "current":
            if target_id is not None:
                raise DecisionValidationError(f"{tool_name} must not set target_id when target_kind is current")
            return
        if isinstance(target_id, str) and target_id.isdigit():
            target_id = int(target_id)
        if not isinstance(target_id, int) or target_id <= 0:
            raise DecisionValidationError(f"{tool_name} requires a positive integer target_id for explicit targets")
