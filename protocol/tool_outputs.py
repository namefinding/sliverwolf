from __future__ import annotations

from typing import Any

from local_agent.protocol.models import OutputKind


def resolve_effective_outputs(
    *,
    tool_name: str,
    produced_outputs: list[OutputKind],
    data: dict[str, Any],
) -> list[OutputKind]:
    return [
        output_kind
        for output_kind in produced_outputs
        if is_output_satisfied(output_kind, tool_name, data)
    ]


def is_output_satisfied(output_kind: OutputKind, tool_name: str, data: dict[str, Any]) -> bool:
    if output_kind in {OutputKind.OBJECT_CANDIDATES, OutputKind.CONTACT_CANDIDATES}:
        return bool(data.get("candidates"))
    if output_kind == OutputKind.OBJECT_DETAILS:
        return any(
            bool(data.get(key))
            for key in (
                "path",
                "items",
                "results",
                "reply",
                "messages",
                "attachments",
                "current_target",
                "session_id",
                "blocks",
                "matches",
                "formatted",
                "date",
                "time",
                "weekday",
                "reminder",
                "task",
                "count",
                "cancelled",
            )
        )
    if output_kind == OutputKind.FILE_CONTENTS:
        return bool(data.get("files"))
    if output_kind == OutputKind.FILE_WRITTEN:
        return _path_flag(data, "path", "bytes_written") or _any_result_path_flag(data, "bytes_written")
    if output_kind == OutputKind.PATH_OPENED:
        return _path_flag(data, "path", "opened") or _any_result_path_flag(data, "opened")
    if output_kind == OutputKind.PATH_CREATED:
        return _path_flag(data, "path", "created", "copied") or _any_result_path_flag(data, "created", "copied")
    if output_kind == OutputKind.PATH_UPDATED:
        return _path_flag(data, "path", "moved", "renamed") or _any_result_path_flag(data, "moved", "renamed")
    if output_kind == OutputKind.PATH_DELETED:
        return _path_flag(data, "path", "deleted") or _any_result_path_flag(data, "deleted")
    if output_kind == OutputKind.WEB_CONTENT:
        return bool(data.get("content"))
    if output_kind == OutputKind.DIRECTORY_ENTRIES:
        return "entries" in data
    if output_kind == OutputKind.SEARCH_MATCHES:
        return bool(data.get("matches"))
    if output_kind == OutputKind.SEARCH_RESULTS:
        return bool(data.get("results"))
    if output_kind == OutputKind.MEMORY_ITEMS:
        return "items" in data or "reminders" in data
    if output_kind == OutputKind.MEMORY_SAVED:
        return bool(data.get("content") or data.get("stored") or data.get("created") or data.get("reminder") or data.get("task"))
    if output_kind == OutputKind.MESSAGE_SENT:
        return any(
            bool(data.get(key))
            for key in ("sent", "message_id", "ok", "path", "file_path", "message", "speech_text", "audio_path")
        )
    return tool_name != ""


def build_evidence(
    *,
    tool_name: str,
    data: dict[str, Any],
    produced_outputs: list[OutputKind],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    output_values = [item.value for item in produced_outputs]

    def add(kind: str, value: Any, **extra: Any) -> None:
        text = "" if value is None else str(value).strip()
        if not text:
            return
        item = {"kind": kind, "value": text, "tool_name": tool_name, "outputs": output_values}
        item.update({key: val for key, val in extra.items() if val not in (None, "", [])})
        if item not in evidence:
            evidence.append(item)

    for key in ("path", "file_path", "output_path"):
        add("path", data.get(key), field=key)
    for key in ("url", "final_url"):
        add("source", data.get(key), field=key, title=data.get("title"))
    add("message_id", data.get("message_id"))
    add("session_id", data.get("session_id"))
    add("reminder_id", data.get("reminder_id"))
    add("formatted", data.get("formatted"))

    for nested_key in ("reminder", "task"):
        nested = data.get(nested_key)
        if isinstance(nested, dict):
            add("reminder_id", nested.get("id") or nested.get("reminder_id"), field=nested_key)
            add("scheduled_time", nested.get("when_iso") or nested.get("scheduled_for"), field=nested_key)
            add("message", nested.get("message"), field=nested_key)

    for collection_key in ("results", "sources", "files", "items", "reminders", "candidates", "messages", "attachments"):
        values = data.get(collection_key)
        if not isinstance(values, list):
            continue
        for item in values[:5]:
            if not isinstance(item, dict):
                continue
            add("source", item.get("url") or item.get("final_url"), collection=collection_key, title=item.get("title"))
            add("path", item.get("path") or item.get("file_path"), collection=collection_key)
            add("message_id", item.get("message_id") or item.get("id"), collection=collection_key)
            add("reminder_id", item.get("reminder_id") or item.get("id"), collection=collection_key)
            add("scheduled_time", item.get("when_iso") or item.get("scheduled_for"), collection=collection_key)
    return evidence


def build_observation(
    *,
    request_id: str,
    tool_name: str,
    status: str,
    data: dict[str, Any],
    error_message: str = "",
) -> str:
    if status != "success":
        return f"{request_id} {tool_name} error={error_message or 'unknown'}"

    if data.get("truncated") is True:
        return (
            f"{request_id} {tool_name} result_truncated=True "
            f"original_size_chars={data.get('original_size_chars', 0)} "
            f"preview={str(data.get('preview', ''))[:240]!r}"
        )

    parts = [f"{request_id} {tool_name}"]
    for key in (
        "query",
        "path",
        "file_path",
        "output_path",
        "url",
        "final_url",
        "target_kind",
        "kind",
        "opened",
        "sent",
        "created",
        "cancelled",
        "count",
    ):
        if key in data and data.get(key) not in (None, "", []):
            parts.append(f"{key}={data.get(key)!r}")

    for collection_key in (
        "candidates",
        "results",
        "sources",
        "files",
        "entries",
        "matches",
        "messages",
        "attachments",
        "items",
        "reminders",
    ):
        values = data.get(collection_key)
        if not isinstance(values, list):
            continue
        samples: list[str] = []
        for item in values[:3]:
            if isinstance(item, dict):
                sample = item.get("path") or item.get("url") or item.get("name") or item.get("title") or item.get("content")
                if sample:
                    samples.append(str(sample))
            elif item is not None:
                samples.append(str(item))
        parts.append(f"{collection_key}_count={len(values)}")
        if samples:
            parts.append(f"{collection_key}_sample={samples}")

    if len(parts) == 2:
        compact = {
            key: value
            for key, value in data.items()
            if key not in {"content", "text"} and value not in (None, "", [])
        }
        parts.append(f"data={str(compact)[:500]}")
    return " ".join(parts)


def _path_flag(data: dict[str, Any], path_key: str, *flag_keys: str) -> bool:
    if not data.get(path_key):
        return False
    if flag_keys == ("bytes_written",):
        return int(data.get("bytes_written", 0)) >= 0
    return any(bool(data.get(flag_key)) for flag_key in flag_keys)


def _any_result_path_flag(data: dict[str, Any], *flag_keys: str) -> bool:
    for item in data.get("results", []):
        if not isinstance(item, dict) or not item.get("path") or item.get("ok", True) is False:
            continue
        if flag_keys == ("bytes_written",):
            return int(item.get("bytes_written", 0)) >= 0
        if any(bool(item.get(flag_key)) for flag_key in flag_keys):
            return True
    return False
