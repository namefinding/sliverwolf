from __future__ import annotations

from local_agent.protocol.models import Message, Role, ToolCallResult


class ContextBuilder:
    @staticmethod
    def build_observations(tool_results: list[ToolCallResult]) -> list[str]:
        observations: list[str] = []
        for result in tool_results:
            if result.status == "success":
                observations.append(ContextBuilder._format_success_observation(result))
            else:
                observations.append(f"{result.request_id} error={result.error.message if result.error else 'unknown'}")
        return observations

    @staticmethod
    def build_prompt_messages(
        messages: list[Message],
        *,
        keep_last: int = 12,
        session_summary: str = "",
        active_task_summary: str = "",
        warm_memory_summary: str = "",
        learning_memory_summary: str = "",
        cold_memory_summary: str = "",
    ) -> list[Message]:
        prompt_messages = list(messages[-keep_last:])
        summary_parts: list[str] = []
        if session_summary.strip():
            summary_parts.append(session_summary.strip())
        if active_task_summary.strip():
            summary_parts.append(active_task_summary.strip())
        if warm_memory_summary.strip():
            summary_parts.append(warm_memory_summary.strip())
        if learning_memory_summary.strip():
            summary_parts.append(learning_memory_summary.strip())
        if cold_memory_summary.strip():
            summary_parts.append(cold_memory_summary.strip())
        if not summary_parts:
            return prompt_messages

        summary_message = Message(role=Role.SYSTEM, content="Session hot context:\n" + "\n\n".join(summary_parts))
        if prompt_messages and prompt_messages[0].role == Role.SYSTEM:
            return [prompt_messages[0], summary_message, *prompt_messages[1:]]
        return [summary_message, *prompt_messages]

    @staticmethod
    def _format_success_observation(result: ToolCallResult) -> str:
        tool_name = result.tool_name or ""
        data = result.data

        if tool_name == "retrieval.search_local_objects":
            candidates = data.get("candidates", [])
            candidate_paths = [candidate.get("path", "") for candidate in candidates[:5] if candidate.get("path")]
            return (
                f"{result.request_id} retrieval.search_local_objects "
                f"query={data.get('query', '')!r} "
                f"candidate_count={len(candidates)} "
                f"candidate_paths={candidate_paths}"
            )

        if tool_name == "file.write":
            return (
                f"{result.request_id} file.write "
                f"path={data.get('path', '')!r} "
                f"bytes_written={data.get('bytes_written', 0)}"
            )

        if tool_name == "file.append":
            return (
                f"{result.request_id} file.append "
                f"path={data.get('path', '')!r} "
                f"bytes_written={data.get('bytes_written', 0)}"
            )

        if tool_name == "file.read":
            files = data.get("files", [])
            file_paths = [item.get("path", "") for item in files[:3] if item.get("path")]
            return f"{result.request_id} file.read file_count={len(files)} file_paths={file_paths}"

        if tool_name == "file.extract_text":
            files = data.get("files", [])
            file_paths = [item.get("path", "") for item in files[:3] if item.get("path")]
            extraction_types = [item.get("extraction_type", "") for item in files[:3]]
            return (
                f"{result.request_id} file.extract_text "
                f"file_count={len(files)} file_paths={file_paths} extraction_types={extraction_types}"
            )

        if tool_name == "file.list":
            entries = data.get("entries", [])
            entry_names = [item.get("name", "") for item in entries[:5] if item.get("name")]
            return f"{result.request_id} file.list entry_count={len(entries)} sample_entries={entry_names}"

        if tool_name == "file.search_text":
            matches = data.get("matches", [])
            match_paths = [item.get("path", "") for item in matches[:5] if item.get("path")]
            return (
                f"{result.request_id} file.search_text "
                f"match_count={len(matches)} "
                f"match_paths={match_paths}"
            )

        if tool_name == "file.search_by_name":
            candidates = data.get("candidates", [])
            candidate_paths = [item.get("path", "") for item in candidates[:5] if item.get("path")]
            return (
                f"{result.request_id} file.search_by_name "
                f"candidate_count={len(candidates)} candidate_paths={candidate_paths}"
            )

        if tool_name in {"file.metadata", "file.preview"}:
            return (
                f"{result.request_id} {tool_name} "
                f"path={data.get('path', '')!r} "
                f"is_dir={data.get('is_dir', False)}"
            )

        if tool_name in {"file.mkdir", "file.copy", "file.move", "file.rename", "file.delete"}:
            return (
                f"{result.request_id} {tool_name} "
                f"path={data.get('path', '')!r}"
            )

        if tool_name in {
            "file.write_many",
            "file.append_many",
            "file.metadata_many",
            "file.preview_many",
            "file.mkdir_many",
            "file.copy_many",
            "file.move_many",
            "file.rename_many",
            "file.delete_many",
            "file.open_many",
            "file.reveal_many",
        }:
            paths = [item.get("path", "") for item in data.get("results", [])[:5] if isinstance(item, dict) and item.get("path")]
            return (
                f"{result.request_id} {tool_name} "
                f"success_count={data.get('success_count', 0)} "
                f"failure_count={data.get('failure_count', 0)} "
                f"paths={paths}"
            )

        if tool_name in {"file.open_path", "file.reveal_in_explorer"}:
            return (
                f"{result.request_id} {tool_name} "
                f"path={data.get('path', '')!r} "
                f"opened={data.get('opened', False)}"
            )

        if tool_name == "retrieval.inspect_local_candidate":
            return (
                f"{result.request_id} retrieval.inspect_local_candidate "
                f"path={data.get('path', '')!r} "
                f"object_kind={data.get('object_kind', '')!r}"
            )

        if tool_name == "web.search":
            results = data.get("results", [])
            urls = [item.get("url", "") for item in results[:3] if isinstance(item, dict) and item.get("url")]
            return (
                f"{result.request_id} web.search "
                f"query={data.get('query', '')!r} result_count={len(results)} urls={urls}"
            )

        if tool_name == "web.fetch":
            return (
                f"{result.request_id} web.fetch "
                f"url={data.get('final_url', data.get('url', ''))!r} "
                f"status_code={data.get('status_code', 0)} "
                f"extractor={data.get('extractor', '')!r}"
            )

        if tool_name == "web.open_page":
            return (
                f"{result.request_id} web.open_page "
                f"url={data.get('url', data.get('path', ''))!r} "
                f"opened={data.get('opened', False)} "
                f"opener={data.get('opener', '')!r}"
            )

        if tool_name == "web.research":
            results = data.get("results", [])
            sources = data.get("sources", [])
            urls = [item.get("url", "") for item in sources[:3] if isinstance(item, dict) and item.get("url")]
            return (
                f"{result.request_id} web.research "
                f"query={data.get('query', '')!r} result_count={len(results)} source_count={len(sources)} urls={urls}"
            )

        return f"{result.request_id} {data}"
