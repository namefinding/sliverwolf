from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, model_validator

from local_agent.app.qq_runtime import QQRuntimeRegistry
from local_agent.protocol.models import OutputKind, ToolExecutionContextFields, ToolManifest


class QQContextInput(ToolExecutionContextFields):
    pass


class QQRecentMessagesInput(ToolExecutionContextFields):
    limit: int = 8
    include_assistant: bool = True


class QQLastReplyInput(ToolExecutionContextFields):
    contact_query: str | None = None


class QQSearchHistoryInput(ToolExecutionContextFields):
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


class QQRecentAttachmentsInput(ToolExecutionContextFields):
    contact_query: str | None = None
    kind: str = "any"
    limit: int = 5

    @model_validator(mode="after")
    def normalize_kind(self) -> "QQRecentAttachmentsInput":
        normalized = self.kind.strip().lower() or "any"
        aliases = {"images": "image", "files": "file", "audios": "audio", "voice": "audio"}
        self.kind = aliases.get(normalized, normalized)
        return self


class QQSearchContactsInput(ToolExecutionContextFields):
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


class QQSendTextInput(ToolExecutionContextFields):
    message: str
    target_kind: str = "current"
    target_id: int | None = None

    @model_validator(mode="after")
    def normalize_target(self) -> "QQSendTextInput":
        self.target_kind = self.target_kind.strip().lower() or "current"
        return self


class QQSendFileInput(ToolExecutionContextFields):
    file_path: str
    target_kind: str = "current"
    target_id: int | None = None

    @model_validator(mode="after")
    def normalize_target(self) -> "QQSendFileInput":
        self.target_kind = self.target_kind.strip().lower() or "current"
        return self


class QQSendVoiceInput(ToolExecutionContextFields):
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


class QQGroupMembersInput(ToolExecutionContextFields):
    group_id: int | None = None

    @model_validator(mode="after")
    def normalize(self) -> "QQGroupMembersInput":
        self.group_id = self.group_id if isinstance(self.group_id, int) else None
        return self


class QQGroupBanInput(ToolExecutionContextFields):
    user_id: int
    group_id: int | None = None
    duration_seconds: int = 60

    @model_validator(mode="after")
    def normalize(self) -> "QQGroupBanInput":
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be >= 0")
        return self


class QQGroupCardInput(ToolExecutionContextFields):
    user_id: int
    card: str
    group_id: int | None = None


class QQGroupInfoInput(ToolExecutionContextFields):
    group_id: int | None = None


class QQSendImageInput(ToolExecutionContextFields):
    image_path: str
    target_kind: str = "current"
    target_id: int | None = None

    @model_validator(mode="after")
    def validate_path(self) -> "QQSendImageInput":
        raw = str(self.image_path or "").strip()
        p = Path(raw).expanduser().resolve()
        if not p.is_file():
            # 可能只是 sticker 名字，自动在 stickers 目录里找
            sticker_dir = Path("data/stickers")
            for ext in ("", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
                candidate = sticker_dir / f"{raw}{ext}"
                if candidate.is_file():
                    p = candidate.resolve()
                    break
            else:
                raise ValueError(f"image_path is not a readable file and not a known sticker: {raw}")
        self.image_path = str(p)
        return self


class QQSendStickerInput(ToolExecutionContextFields):
    sticker_name: str

    @model_validator(mode="after")
    def validate_sticker(self) -> "QQSendStickerInput":
        normalized = str(self.sticker_name or "").strip()
        if not normalized:
            raise ValueError("sticker_name must not be empty")
        p = QQModule.resolve_sticker_path(normalized)
        if p is None:
            raise ValueError(f"unknown sticker: {normalized}")
        self.sticker_name = p.stem
        return self


class QQSentimentInput(ToolExecutionContextFields):
    expression: str = "smile"

    @model_validator(mode="after")
    def normalize(self) -> "QQSentimentInput":
        valid = {"smile", "sad", "angry", "surprise", "sweat", "cry", "funny", "tongue", "shy", "cool", "sleep", "awkward", "happy", "excited", "bored", "fear", "annoyed"}
        self.expression = str(self.expression or "").strip().lower() or "smile"
        if self.expression not in valid:
            self.expression = "smile"
        return self


class QQReplyInput(ToolExecutionContextFields):
    message_id: int
    reply_text: str

    @model_validator(mode="after")
    def check_text(self) -> "QQReplyInput":
        if not self.reply_text.strip():
            raise ValueError("reply_text must not be empty")
        return self


class QQPokeInput(ToolExecutionContextFields):
    user_id: int
    group_id: int | None = None

    @model_validator(mode="after")
    def normalize(self) -> "QQGroupBanInput":
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be >= 0")
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
                description="Send a QQ text message to the CURRENT chat session only. To send a message to a DIFFERENT contact (代发), use skill.send_message instead.",
                side_effect=True,
                idempotent=False,
                requires_confirmation=True,
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
                requires_confirmation=True,
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
                requires_confirmation=True,
                produces=[OutputKind.MESSAGE_SENT],
                input_schema=QQSendVoiceInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"sent": {"type": "boolean"}, "target": {"type": "object"}}},
            ),
            ToolManifest(
                tool_name="qq.get_group_members",
                module="qq",
                description="Get the member list of the current QQ group or a specific group.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=QQGroupMembersInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"members": {"type": "array"}}},
            ),
            ToolManifest(
                tool_name="qq.get_group_info",
                module="qq",
                description="Get basic info for the current or a specific QQ group (name, member count).",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=QQGroupInfoInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"name": {"type": "string"}, "member_count": {"type": "integer"}}},
            ),
            ToolManifest(
                tool_name="qq.set_group_card",
                module="qq",
                description="Set a group member's nickname/card. Requires admin permissions.",
                side_effect=True,
                idempotent=False,
                requires_confirmation=True,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=QQGroupCardInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"card_set": {"type": "boolean"}}},
            ),
            ToolManifest(
                tool_name="qq.set_group_ban",
                module="qq",
                description="Ban or unban a group member from speaking. Requires admin permissions. duration_seconds=0 means unban.",
                side_effect=True,
                idempotent=False,
                requires_confirmation=True,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=QQGroupBanInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"banned": {"type": "boolean"}, "duration_seconds": {"type": "integer"}}},
            ),
            ToolManifest(
                tool_name="qq.poke",
                module="qq",
                description="Poke/shake a group member or friend.",
                side_effect=True,
                idempotent=False,
                requires_confirmation=True,
                produces=[OutputKind.MESSAGE_SENT],
                input_schema=QQPokeInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"poked": {"type": "boolean"}}},
            ),
            ToolManifest(
                tool_name="qq.send_like",
                module="qq",
                description="Give a like to the current user or a specific user (thumb up).",
                side_effect=True,
                idempotent=False,
                requires_confirmation=False,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema={"type": "object", "properties": {"user_id": {"type": "integer", "description": "user to like, defaults to current sender"}}},
                output_schema={"type": "object", "properties": {"liked": {"type": "boolean"}}},
            ),
            ToolManifest(
                tool_name="qq.send_image",
                module="qq",
                description="Send a local image file as a QQ message. Use qq.send_sticker for expressive stickers; reserve this for user-requested image sending.",
                side_effect=True,
                idempotent=False,
                requires_confirmation=True,
                timeout_ms=20_000,
                produces=[OutputKind.MESSAGE_SENT],
                input_schema=QQSendImageInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"sent": {"type": "boolean"}, "path": {"type": "string"}}},
            ),
            ToolManifest(
                tool_name="qq.send_sticker",
                module="qq",
                description=(
                    "Send one local sticker to the current QQ session as optional chat expression. "
                    "Use only when it naturally improves tone, never as the answer itself, and never make it a required task output."
                ),
                side_effect=True,
                idempotent=False,
                requires_confirmation=False,
                timeout_ms=20_000,
                produces=[OutputKind.MESSAGE_SENT],
                input_schema=QQSendStickerInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"sent": {"type": "boolean"}, "sticker_name": {"type": "string"}, "path": {"type": "string"}, "skipped": {"type": "boolean"}}},
            ),
            ToolManifest(
                tool_name="qq.list_stickers",
                module="qq",
                description="List all available sticker names from the local sticker repository. Use this before choosing qq.send_sticker.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema={"type": "object", "properties": {}},
                output_schema={"type": "object", "properties": {"names": {"type": "array"}, "count": {"type": "integer"}}},
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
            "qq.get_group_members": self.get_group_members,
            "qq.get_group_info": self.get_group_info,
            "qq.set_group_card": self.set_group_card,
            "qq.set_group_ban": self.set_group_ban,
            "qq.poke": self.poke,
            "qq.send_like": self.send_like,
            "qq.send_image": self.send_image,
            "qq.send_sticker": self.send_sticker,
            "qq.list_stickers": self.list_stickers,
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
        effective_limit = payload.limit
        if self._is_current_group_context(session_id=session_id):
            effective_limit = max(effective_limit, 12)
        messages = runtime.get_recent_messages(
            session_id=session_id,
            limit=effective_limit,
            include_assistant=payload.include_assistant,
        )
        return {
            "channel": "onebot_v11",
            "session_id": session_id,
            "messages": messages,
        }

    def _is_current_group_context(self, *, session_id: str | None = None) -> bool:
        if str(session_id or "").startswith("onebot_group_"):
            return True
        target = self.current_target
        if not isinstance(target, dict):
            return False
        message_type = str(target.get("message_type") or "").strip().lower()
        target_kind = str(target.get("target_kind") or target.get("kind") or "").strip().lower()
        return message_type == "group" or target_kind == "group" or target.get("group_id") is not None

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

    def get_group_members(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = QQGroupMembersInput.model_validate(arguments)
        runtime = self._require_runtime()
        self._require_full_access("qq.get_group_members requires a full-access QQ account.")
        group_id = payload.group_id or self._current_group_id()
        if group_id is None:
            raise RuntimeError("No group_id available. Call this from a group session or provide group_id.")
        members = runtime.get_group_members(group_id=group_id)
        return {"members": members}

    def get_group_info(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = QQGroupInfoInput.model_validate(arguments)
        runtime = self._require_runtime()
        group_id = payload.group_id or self._current_group_id()
        if group_id is None:
            raise RuntimeError("No group_id available. Call this from a group session or provide group_id.")
        info = runtime.get_group_info(group_id=group_id)
        return {"info": info} if info else {"info": None}

    def set_group_card(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = QQGroupCardInput.model_validate(arguments)
        runtime = self._require_runtime()
        self._require_full_access("qq.set_group_card requires a full-access QQ account.")
        group_id = payload.group_id or self._current_group_id()
        if group_id is None:
            raise RuntimeError("No group_id available.")
        result = runtime.set_group_card(group_id=group_id, user_id=payload.user_id, card=payload.card)
        return result

    def set_group_ban(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = QQGroupBanInput.model_validate(arguments)
        runtime = self._require_runtime()
        self._require_full_access("qq.set_group_ban requires a full-access QQ account.")
        group_id = payload.group_id or self._current_group_id()
        if group_id is None:
            raise RuntimeError("No group_id available.")
        result = runtime.set_group_ban(
            group_id=group_id,
            user_id=payload.user_id,
            duration_seconds=payload.duration_seconds,
        )
        return result

    def send_expression(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = QQSentimentInput.model_validate(arguments)
        runtime = self._require_runtime()
        self._require_target_access("current", tool_name="qq.send_expression")
        # 构造 CQ:face 消息
        face_id = self._expression_face_id(payload.expression)
        message = f"[CQ:face,id={face_id}]"
        result = runtime.send_text(
            message,
            target_kind="current",
            target_id=None,
            current_target=self.current_target if isinstance(self.current_target, dict) else None,
        )
        normalized = dict(result or {})
        normalized.setdefault("expression", payload.expression)
        normalized.setdefault("face_id", face_id)
        return normalized

    def poke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = QQPokeInput.model_validate(arguments)
        runtime = self._require_runtime()
        self._require_full_access("qq.poke requires a full-access QQ account.")
        group_id = payload.group_id or self._current_group_id()
        result = runtime.poke(user_id=payload.user_id, group_id=group_id)
        return {"poked": bool(result.get("ok", True) if isinstance(result, dict) else True), "user_id": payload.user_id}

    def send_like(self, arguments: dict[str, Any]) -> dict[str, Any]:
        runtime = self._require_runtime()
        user_id = arguments.get("user_id")
        if not isinstance(user_id, int):
            user_id = int(self.sender_id) if self.sender_id and self.sender_id.isdigit() else None
        if user_id is None:
            raise ValueError("No user_id to like.")
        result = runtime.send_like(user_id=user_id)
        return {"liked": True, "user_id": user_id}

    @staticmethod
    def _expression_face_id(expression: str) -> int:
        mapping = {
            "smile": 14, "sad": 5, "angry": 1, "surprise": 0, "sweat": 27, "cry": 9,
            "funny": 13, "tongue": 10, "shy": 8, "cool": 3, "sleep": 16,
            "awkward": 7, "happy": 2, "excited": 4, "bored": 6, "fear": 11, "annoyed": 12,
        }
        return mapping.get(expression, 14)

    def send_image(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = QQSendImageInput.model_validate(arguments)
        runtime = self._require_runtime()
        self._require_target_access(payload.target_kind, tool_name="qq.send_image")
        result = runtime.send_image(
            image_path=payload.image_path,
            target_kind=payload.target_kind,
            target_id=payload.target_id,
            current_target=self.current_target if isinstance(self.current_target, dict) else None,
        )
        normalized = dict(result or {})
        normalized.setdefault("sent", True)
        normalized.setdefault("path", payload.image_path)
        return normalized

    def send_sticker(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = QQSendStickerInput.model_validate(arguments)
        runtime = self._require_runtime()
        self._require_target_access("current", tool_name="qq.send_sticker")
        sticker_path = self.resolve_sticker_path(payload.sticker_name)
        if sticker_path is None:
            raise FileNotFoundError(f"Sticker does not exist: {payload.sticker_name}")
        if hasattr(runtime, "send_sticker"):
            result = runtime.send_sticker(
                sticker_name=payload.sticker_name,
                image_path=str(sticker_path),
                current_target=self.current_target if isinstance(self.current_target, dict) else None,
            )
        else:
            result = runtime.send_image(
                image_path=str(sticker_path),
                target_kind="current",
                target_id=None,
                current_target=self.current_target if isinstance(self.current_target, dict) else None,
            )
        normalized = dict(result or {})
        normalized.setdefault("sent", bool(normalized.get("ok", True)))
        normalized.setdefault("sticker_name", payload.sticker_name)
        if normalized.get("sent") is not False:
            normalized.setdefault("path", str(sticker_path))
        return normalized

    def list_stickers(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """列出 data/stickers/ 下所有图片文件名（只返回名字，不返回路径）。"""
        names = [path.stem for path in self.list_sticker_paths()]
        return {"names": names, "count": len(names)}

    @staticmethod
    def list_sticker_paths() -> list[Path]:
        sticker_dir = Path("data/stickers")
        sticker_dir.mkdir(parents=True, exist_ok=True)
        image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
        return [
            path.resolve()
            for path in sorted(sticker_dir.iterdir())
            if path.is_file() and path.suffix.lower() in image_exts
        ]

    @staticmethod
    def resolve_sticker_path(sticker_name: str) -> Path | None:
        normalized = str(sticker_name or "").strip()
        if not normalized:
            return None
        direct = Path(normalized).expanduser()
        if direct.is_file():
            return direct.resolve()
        sticker_dir = Path("data/stickers")
        for ext in ("", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
            candidate = sticker_dir / f"{normalized}{ext}"
            if candidate.is_file():
                return candidate.resolve()
        normalized_lower = normalized.casefold()
        for path in QQModule.list_sticker_paths():
            if path.stem.casefold() == normalized_lower:
                return path
        return None

    def _current_group_id(self) -> int | None:
        target = self.current_target
        if isinstance(target, dict):
            gid = target.get("group_id")
            if isinstance(gid, int):
                return gid
        return None

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
