from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, model_validator

from local_agent.app.qq_runtime import QQRuntimeRegistry
from local_agent.protocol.models import OutputKind, ToolManifest


class QQContextInput(BaseModel):
    pass


class QQRecentMessagesInput(BaseModel):
    limit: int = 8
    include_assistant: bool = True


class QQLastReplyInput(BaseModel):
    contact_query: str | None = None


class QQSearchHistoryInput(BaseModel):
    query: str | None = None
    contact_query: str | None = None
    limit: int = 5
    reply_after_last_outbound: bool = False

    @model_validator(mode="after")
    def validate_lookup(self) -> "QQSearchHistoryInput":
        query = "" if self.query is None else self.query.strip()
        contact_query = "" if self.contact_query is None else self.contact_query.strip()
        if not query and not contact_query and not self.reply_after_last_outbound:
            raise ValueError("query, contact_query, or reply_after_last_outbound must be provided")
        self.query = query or None
        self.contact_query = contact_query or None
        return self


class QQRecentAttachmentsInput(BaseModel):
    contact_query: str | None = None
    kind: str = "any"
    limit: int = 5

    @model_validator(mode="after")
    def normalize_kind(self) -> "QQRecentAttachmentsInput":
        normalized = self.kind.strip().lower() or "any"
        aliases = {"images": "image", "files": "file", "audios": "audio", "voice": "audio"}
        self.kind = aliases.get(normalized, normalized)
        return self


class QQSearchContactsInput(BaseModel):
    query: str
    target_kind: str = "any"
    limit: int = 5
    exclude_sender: bool = False

    @model_validator(mode="after")
    def normalize_kind(self) -> "QQSearchContactsInput":
        normalized = self.target_kind.strip().lower()
        aliases = {"user": "friend", "private": "friend", "dm": "friend", "chat": "any"}
        self.target_kind = aliases.get(normalized, normalized or "any")
        return self


class QQSendTextInput(BaseModel):
    message: str
    target_kind: str = "current"
    target_id: int | None = None

    @model_validator(mode="after")
    def normalize_target(self) -> "QQSendTextInput":
        self.target_kind = self.target_kind.strip().lower() or "current"
        return self


class QQSendFileInput(BaseModel):
    file_path: str
    target_kind: str = "current"
    target_id: int | None = None

    @model_validator(mode="after")
    def normalize_target(self) -> "QQSendFileInput":
        self.target_kind = self.target_kind.strip().lower() or "current"
        return self


class QQSendVoiceInput(BaseModel):
    speech_text: str | None = None
    audio_path: str | None = None
    target_kind: str = "current"
    target_id: int | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "QQSendVoiceInput":
        self.target_kind = self.target_kind.strip().lower() or "current"
        speech = "" if self.speech_text is None else self.speech_text.strip()
        audio = "" if self.audio_path is None else self.audio_path.strip()
        if not speech and not audio:
            raise ValueError("speech_text or audio_path must be provided")
        self.speech_text = speech or None
        self.audio_path = audio or None
        return self


class QQModule:
    def __init__(self, runtime_context: dict[str, Any] | None = None) -> None:
        self.runtime_context = runtime_context or {}
        self.runtime_id = str(self.runtime_context.get("runtime_id", "")).strip() or None
        self.runtime = QQRuntimeRegistry.get(self.runtime_id)
        self.current_target = self.runtime_context.get("current_target")
        if self.runtime is None and isinstance(self.current_target, dict):
            # Scheduled QQ tasks may survive a gateway restart; when that happens
            # the persisted runtime_id is stale, but the current target is still usable.
            self.runtime = QQRuntimeRegistry.get_any()
        self.sender_id = self._normalize_optional_string(self.runtime_context.get("sender_id"))
        access_policy = self.runtime_context.get("access_policy")
        self.access_policy = access_policy if isinstance(access_policy, dict) else {}

    def manifests(self) -> list[ToolManifest]:
        return [
            ToolManifest(
                tool_name="qq.get_current_context",
                module="qq",
                description="Read the current QQ session context, including sender, session, target, and channel permissions.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=QQContextInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"channel": {"type": "string"}, "session_id": {"type": "string"}}},
            ),
            ToolManifest(
                tool_name="qq.get_recent_messages",
                module="qq",
                description="Read a short window of recent messages from the current QQ session. Best for rebuilding immediate conversational context, not for targeted history lookup when a more specific QQ history tool applies.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=QQRecentMessagesInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"session_id": {"type": "string"}, "messages": {"type": "array"}}},
            ),
            ToolManifest(
                tool_name="qq.get_last_reply",
                module="qq",
                description="Read the latest inbound QQ reply for the current session or for a resolved contact. This is a narrow convenience lookup; prefer qq.search_history when the user wants the reply after a specific exchange or wants thread evidence.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=QQLastReplyInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"session_id": {"type": "string"}, "reply": {"type": "object"}}},
            ),
            ToolManifest(
                tool_name="qq.search_history",
                module="qq",
                description="Search stored QQ conversation history by current session, contact, keyword, or both. It can also return the latest inbound reply after our most recent outbound message by setting reply_after_last_outbound=true. Prefer this when the user asks what was discussed earlier, how the other side replied, or wants a summary of prior chat content.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=QQSearchHistoryInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"session_id": {"type": "string"}, "messages": {"type": "array"}}},
            ),
            ToolManifest(
                tool_name="qq.get_recent_attachments",
                module="qq",
                description="List recent QQ images, files, or audio attachments for the current session or a resolved contact. Prefer this when the user asks about a previously sent file, picture, voice message, or other attachment.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=QQRecentAttachmentsInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"session_id": {"type": "string"}, "attachments": {"type": "array"}}},
            ),
            ToolManifest(
                tool_name="qq.search_contacts",
                module="qq",
                description="Search QQ friends or groups by name, alias, or numeric ID.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.CONTACT_CANDIDATES],
                input_schema=QQSearchContactsInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"candidates": {"type": "array"}}},
            ),
            ToolManifest(
                tool_name="qq.send_text",
                module="qq",
                description="Send a QQ text message to the current session or to a specific friend or group.",
                side_effect=True,
                idempotent=False,
                produces=[OutputKind.MESSAGE_SENT],
                input_schema=QQSendTextInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"sent": {"type": "boolean"}, "target": {"type": "object"}}},
            ),
            ToolManifest(
                tool_name="qq.send_file",
                module="qq",
                description="Send a local file to the current QQ session or to a specific friend or group.",
                side_effect=True,
                idempotent=False,
                produces=[OutputKind.MESSAGE_SENT],
                input_schema=QQSendFileInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"sent": {"type": "boolean"}, "path": {"type": "string"}}},
            ),
            ToolManifest(
                tool_name="qq.send_voice",
                module="qq",
                description="Send a QQ voice message by synthesizing speech text or uploading an existing audio file.",
                side_effect=True,
                idempotent=False,
                produces=[OutputKind.MESSAGE_SENT],
                input_schema=QQSendVoiceInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"sent": {"type": "boolean"}, "target": {"type": "object"}}},
            ),
        ]

    def executor_map(self) -> dict[str, Any]:
        return {
            "qq.get_current_context": self.get_current_context,
            "qq.get_recent_messages": self.get_recent_messages,
            "qq.get_last_reply": self.get_last_reply,
            "qq.search_history": self.search_history,
            "qq.get_recent_attachments": self.get_recent_attachments,
            "qq.search_contacts": self.search_contacts,
            "qq.send_text": self.send_text,
            "qq.send_file": self.send_file,
            "qq.send_voice": self.send_voice,
        }

    def get_current_context(self, arguments: dict[str, Any]) -> dict[str, Any]:
        QQContextInput.model_validate(arguments)
        self._require_runtime()
        return {
            "channel": "onebot_v11",
            "session_id": self._normalize_optional_string(self.runtime_context.get("session_id")),
            "sender_id": self.sender_id,
            "current_target": self.current_target,
            "access_policy": self.access_policy,
        }

    def get_recent_messages(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = QQRecentMessagesInput.model_validate(arguments)
        runtime = self._require_runtime()
        session_id = self._normalize_optional_string(self.runtime_context.get("session_id"))
        messages = runtime.get_recent_messages(
            session_id=session_id,
            limit=payload.limit,
            include_assistant=payload.include_assistant,
        )
        return {
            "channel": "onebot_v11",
            "session_id": session_id,
            "messages": messages,
        }

    def get_last_reply(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = QQLastReplyInput.model_validate(arguments)
        runtime = self._require_runtime()
        session_id = self._normalize_optional_string(self.runtime_context.get("session_id"))
        reply = runtime.get_last_reply(
            session_id=session_id,
            contact_query=payload.contact_query,
        )
        return {
            "channel": "onebot_v11",
            "session_id": session_id,
            "contact_query": payload.contact_query,
            "found": reply is not None,
            "reply": reply,
        }

    def search_history(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = QQSearchHistoryInput.model_validate(arguments)
        runtime = self._require_runtime()
        session_id = self._normalize_optional_string(self.runtime_context.get("session_id"))
        messages = runtime.search_history(
            session_id=session_id,
            contact_query=payload.contact_query,
            query=payload.query,
            limit=payload.limit,
            reply_after_last_outbound=payload.reply_after_last_outbound,
        )
        return {
            "channel": "onebot_v11",
            "session_id": session_id,
            "contact_query": payload.contact_query,
            "query": payload.query,
            "reply_after_last_outbound": payload.reply_after_last_outbound,
            "messages": messages,
        }

    def get_recent_attachments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = QQRecentAttachmentsInput.model_validate(arguments)
        runtime = self._require_runtime()
        session_id = self._normalize_optional_string(self.runtime_context.get("session_id"))
        attachments = runtime.get_recent_attachments(
            session_id=session_id,
            contact_query=self._normalize_optional_string(payload.contact_query),
            kind=payload.kind,
            limit=payload.limit,
        )
        return {
            "channel": "onebot_v11",
            "session_id": session_id,
            "contact_query": self._normalize_optional_string(payload.contact_query),
            "kind": payload.kind,
            "attachments": attachments,
        }

    def search_contacts(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = QQSearchContactsInput.model_validate(arguments)
        runtime = self._require_runtime()
        self._require_full_access("qq.search_contacts requires a full-access QQ account.")
        candidates = runtime.search_contacts(
            payload.query,
            target_kind=payload.target_kind,
            limit=payload.limit,
            exclude_sender=payload.exclude_sender,
            sender_id=self.sender_id,
        )
        return {
            "query": payload.query,
            "target_kind": payload.target_kind,
            "candidates": candidates,
        }

    def send_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = QQSendTextInput.model_validate(arguments)
        runtime = self._require_runtime()
        self._require_target_access(payload.target_kind, tool_name="qq.send_text")
        result = runtime.send_text(
            payload.message,
            target_kind=payload.target_kind,
            target_id=payload.target_id,
            current_target=self.current_target if isinstance(self.current_target, dict) else None,
        )
        normalized = dict(result or {})
        normalized.setdefault("sent", bool(normalized.get("ok", True)))
        normalized.setdefault("message", payload.message)
        return normalized

    def send_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = QQSendFileInput.model_validate(arguments)
        runtime = self._require_runtime()
        self._require_full_access("qq.send_file requires a full-access QQ account.")
        self._require_target_access(payload.target_kind, tool_name="qq.send_file")
        target_path = Path(payload.file_path).expanduser().resolve()
        if not target_path.is_file():
            raise FileNotFoundError(f"QQ file path does not exist: {target_path}")
        result = runtime.send_file(
            str(target_path),
            target_kind=payload.target_kind,
            target_id=payload.target_id,
            current_target=self.current_target if isinstance(self.current_target, dict) else None,
        )
        normalized = dict(result or {})
        normalized.setdefault("sent", bool(normalized.get("ok", True)))
        normalized.setdefault("path", str(target_path))
        return normalized

    def send_voice(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = QQSendVoiceInput.model_validate(arguments)
        runtime = self._require_runtime()
        self._require_target_access(payload.target_kind, tool_name="qq.send_voice")
        if payload.audio_path:
            target_audio = Path(payload.audio_path).expanduser().resolve()
            if not target_audio.is_file():
                raise FileNotFoundError(f"QQ audio path does not exist: {target_audio}")
            audio_path = str(target_audio)
        else:
            audio_path = None
        result = runtime.send_voice(
            speech_text=payload.speech_text,
            audio_path=audio_path,
            target_kind=payload.target_kind,
            target_id=payload.target_id,
            current_target=self.current_target if isinstance(self.current_target, dict) else None,
        )
        normalized = dict(result or {})
        normalized.setdefault("sent", bool(normalized.get("ok", True)))
        if payload.speech_text:
            normalized.setdefault("speech_text", payload.speech_text)
        if audio_path:
            normalized.setdefault("audio_path", audio_path)
        return normalized

    def _require_runtime(self):
        if self.runtime is None:
            raise RuntimeError("QQ runtime is unavailable in the current session.")
        return self.runtime

    def _require_target_access(self, target_kind: str, *, tool_name: str) -> None:
        normalized = target_kind.strip().lower()
        if normalized == "current":
            return
        self._require_full_access(f"{tool_name} can only target other QQ contacts for a full-access account.")

    def _require_full_access(self, message: str) -> None:
        if bool(self.access_policy.get("allow_local_tools", False)):
            return
        raise PermissionError(message)

    @staticmethod
    def _normalize_optional_string(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None
