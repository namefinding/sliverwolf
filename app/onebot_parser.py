from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from local_agent.app.onebot_models import (
    OneBotAudioAttachment,
    OneBotImageAttachment,
    OneBotInboundMessage,
    OneBotTarget,
)


def extract_text(payload: dict[str, Any]) -> str:
    message = payload.get("message")
    if isinstance(message, list):
        parts: list[str] = []
        for item in message:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                continue
            data = item.get("data")
            if isinstance(data, dict):
                text = data.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        if parts:
            return "".join(parts).strip()

    for candidate in (payload.get("text"), payload.get("content"), payload.get("raw_message")):
        if isinstance(candidate, str) and candidate.strip():
            cleaned = _strip_cq_segments(candidate)
            if not cleaned:
                continue
            return cleaned
    return ""


def extract_sender_id(payload: dict[str, Any]) -> str | None:
    for candidate in (payload.get("sender_id"), payload.get("user_id"), payload.get("session_id")):
        if isinstance(candidate, (str, int)) and str(candidate).strip():
            return str(candidate).strip()

    sender = payload.get("sender")
    if isinstance(sender, dict):
        for key in ("user_id", "nickname", "card"):
            value = sender.get(key)
            if isinstance(value, (str, int)) and str(value).strip():
                return str(value).strip()
    return None


def extract_audio_attachments(payload: dict[str, Any]) -> tuple[OneBotAudioAttachment, ...]:
    attachments: list[OneBotAudioAttachment] = []
    message = payload.get("message")
    if isinstance(message, list):
        for item in message:
            if not isinstance(item, dict) or item.get("type") != "record":
                continue
            data = item.get("data")
            if not isinstance(data, dict):
                continue

            path_value = data.get("path")
            file_value = data.get("file")
            remote_url = data.get("url")
            mime_type = data.get("mimetype") or data.get("mime") or "audio/wav"
            local_path, remote = _resolve_audio_locations(path_value, file_value, remote_url)

            if local_path or remote:
                attachments.append(
                    OneBotAudioAttachment(
                        local_path=local_path,
                        remote_url=remote,
                        mime_type=str(mime_type).strip() or None,
                    )
                )

    raw_message = payload.get("raw_message")
    if isinstance(raw_message, str) and "[CQ:record," in raw_message:
        attachments.extend(_extract_cq_record_attachments(raw_message))

    return tuple(attachments)


def extract_image_attachments(payload: dict[str, Any]) -> tuple[OneBotImageAttachment, ...]:
    attachments: list[OneBotImageAttachment] = []
    message = payload.get("message")
    if isinstance(message, list):
        for item in message:
            if not isinstance(item, dict):
                continue
            segment_type = str(item.get("type", "")).strip().lower()
            if segment_type not in {"image", "mface", "face"}:
                continue
            data = item.get("data")
            if not isinstance(data, dict):
                data = {}

            path_value = data.get("path")
            file_value = data.get("file")
            remote_url = data.get("url")
            mime_type = data.get("mimetype") or data.get("mime")
            summary = _normalize_attachment_value(data.get("summary"))
            image_type = _normalize_attachment_value(data.get("sub_type") or data.get("subType") or data.get("type"))
            local_path, remote, file_id = _resolve_image_locations(path_value, file_value, remote_url)
            is_emoji = _is_emoji_image(segment_type=segment_type, summary=summary, image_type=image_type)

            if local_path or remote or file_id or is_emoji:
                attachments.append(
                    OneBotImageAttachment(
                        local_path=local_path,
                        remote_url=remote,
                        file_id=file_id,
                        mime_type=str(mime_type).strip() if isinstance(mime_type, str) and mime_type.strip() else None,
                        summary=summary,
                        image_type=image_type,
                        is_emoji=is_emoji,
                    )
                )

    raw_message = payload.get("raw_message")
    if isinstance(raw_message, str) and "[CQ:image," in raw_message:
        attachments.extend(_extract_cq_image_attachments(raw_message))

    return tuple(attachments)


def _extract_cq_record_attachments(raw_message: str) -> list[OneBotAudioAttachment]:
    attachments: list[OneBotAudioAttachment] = []
    for match in re.finditer(r"\[CQ:record,([^\]]+)\]", raw_message):
        fields = _parse_cq_fields(match.group(1))
        path_value = fields.get("path")
        file_value = fields.get("file")
        remote_url = fields.get("url")
        mime_type = fields.get("mimetype") or fields.get("mime") or "audio/amr"
        local_path, remote = _resolve_audio_locations(path_value, file_value, remote_url)

        if local_path or remote:
            attachments.append(
                OneBotAudioAttachment(
                    local_path=local_path,
                    remote_url=remote,
                    mime_type=mime_type,
                )
            )
    return attachments


def _extract_cq_image_attachments(raw_message: str) -> list[OneBotImageAttachment]:
    attachments: list[OneBotImageAttachment] = []
    for match in re.finditer(r"\[CQ:image,([^\]]+)\]", raw_message):
        fields = _parse_cq_fields(match.group(1))
        path_value = fields.get("path")
        file_value = fields.get("file")
        remote_url = fields.get("url")
        mime_type = fields.get("mimetype") or fields.get("mime")
        summary = _normalize_attachment_value(fields.get("summary"))
        image_type = _normalize_attachment_value(fields.get("sub_type") or fields.get("subType") or fields.get("type"))
        local_path, remote, file_id = _resolve_image_locations(path_value, file_value, remote_url)
        is_emoji = _is_emoji_image(segment_type="image", summary=summary, image_type=image_type)

        if local_path or remote or file_id or is_emoji:
            attachments.append(
                OneBotImageAttachment(
                    local_path=local_path,
                    remote_url=remote,
                    file_id=file_id,
                    mime_type=mime_type,
                    summary=summary,
                    image_type=image_type,
                    is_emoji=is_emoji,
                )
            )
    return attachments


def _resolve_audio_locations(
    path_value: Any,
    file_value: Any,
    remote_url: Any,
) -> tuple[str | None, str | None]:
    local_path: str | None = None
    remote: str | None = None

    normalized_path = _normalize_attachment_value(path_value)
    normalized_file = _normalize_attachment_value(file_value)
    normalized_remote = _normalize_attachment_value(remote_url)

    if normalized_remote and normalized_remote.startswith(("http://", "https://")):
        remote = normalized_remote
    elif normalized_remote and _looks_like_local_audio_path(normalized_remote):
        local_path = normalized_remote

    if normalized_path and _looks_like_local_audio_path(normalized_path):
        local_path = normalized_path
    elif not local_path and normalized_path and not normalized_path.startswith(("http://", "https://")):
        local_path = normalized_path

    if normalized_file:
        if normalized_file.startswith(("http://", "https://")):
            remote = normalized_file
        elif not local_path:
            local_path = normalized_file

    return local_path, remote


def _resolve_image_locations(
    path_value: Any,
    file_value: Any,
    remote_url: Any,
) -> tuple[str | None, str | None, str | None]:
    local_path: str | None = None
    remote: str | None = None
    file_id: str | None = None

    normalized_path = _normalize_attachment_value(path_value)
    normalized_file = _normalize_attachment_value(file_value)
    normalized_remote = _normalize_attachment_value(remote_url)

    if normalized_remote and normalized_remote.startswith(("http://", "https://")):
        remote = normalized_remote
    elif normalized_remote and _looks_like_local_image_path(normalized_remote):
        local_path = normalized_remote

    if normalized_path and _looks_like_local_image_path(normalized_path):
        local_path = normalized_path
    elif normalized_path and normalized_path.startswith(("http://", "https://")):
        remote = normalized_path

    if normalized_file:
        if normalized_file.startswith(("http://", "https://")):
            remote = normalized_file
        elif _looks_like_local_image_path(normalized_file):
            local_path = normalized_file
        else:
            file_id = normalized_file

    return local_path, remote, file_id


def _normalize_attachment_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _looks_like_local_audio_path(value: str) -> bool:
    if value.startswith(("http://", "https://")):
        return False
    if "\\" in value or "/" in value:
        return True
    if Path(value).is_absolute():
        return True
    return False


def _looks_like_local_image_path(value: str) -> bool:
    if value.startswith(("http://", "https://")):
        return False
    if "\\" in value or "/" in value:
        return True
    if Path(value).is_absolute():
        return True
    return False


def _is_emoji_image(*, segment_type: str, summary: str | None, image_type: str | None) -> bool:
    lowered_type = str(segment_type or "").strip().lower()
    lowered_image_type = str(image_type or "").strip().lower()
    normalized_summary = str(summary or "")
    return (
        lowered_type in {"mface", "face"}
        or lowered_image_type in {"emoji", "sticker", "face", "mface"}
        or "\u8868\u60c5" in normalized_summary
    )


def _strip_cq_segments(raw_message: str) -> str:
    cleaned = re.sub(r"\[CQ:[^\]]+\]", "", raw_message)
    return cleaned.strip()


def _parse_cq_fields(field_blob: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in field_blob.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            result[key] = value
    return result


def extract_scope_root(payload: dict[str, Any]) -> str | None:
    scope_root = payload.get("scope_root")
    return scope_root.strip() if isinstance(scope_root, str) and scope_root.strip() else None


def extract_mode(payload: dict[str, Any]) -> str:
    mode = payload.get("mode")
    if isinstance(mode, str) and mode in {"auto", "chat", "agent"}:
        return mode
    return "auto"


def is_onebot_event(payload: dict[str, Any]) -> bool:
    return (
        isinstance(payload.get("post_type"), str)
        and isinstance(payload.get("self_id"), (int, str))
        and payload.get("post_type") in {"message", "message_sent"}
    )


def extract_onebot_typing_notice(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not (
        isinstance(payload.get("post_type"), str)
        and isinstance(payload.get("self_id"), (int, str))
        and payload.get("post_type") == "notice"
    ):
        return None

    raw_blob = json.dumps(payload, ensure_ascii=False).lower()
    if not any(marker in raw_blob for marker in ("input_status", "typing", "正在输入", "输入状态")):
        return None

    message_type = str(payload.get("message_type", "private") or "private").strip().lower()
    if message_type not in {"private", "group"}:
        message_type = "private"
    sender_id = extract_sender_id(payload)
    if sender_id is None:
        return None

    active = _infer_typing_active(payload, raw_blob)
    if active is None:
        return None

    if message_type == "group":
        group_id = payload.get("group_id")
        if group_id is None:
            return None
        session_id = f"onebot_group_{group_id}"
    else:
        session_id = f"onebot_private_{sender_id}"

    return {
        "session_id": session_id,
        "sender_id": sender_id,
        "message_type": message_type,
        "group_id": payload.get("group_id"),
        "active": active,
        "raw_payload": payload,
    }


def is_auth_failure(payload: dict[str, Any]) -> bool:
    return (
        payload.get("status") == "failed"
        and payload.get("retcode") == 1403
        and "token" in str(payload.get("message", "")).lower()
    )


def message_mentions_self(payload: dict[str, Any]) -> bool:
    self_id = payload.get("self_id")
    if self_id is None:
        return False
    normalized_self_id = str(self_id).strip()
    if not normalized_self_id:
        return False

    message = payload.get("message")
    if isinstance(message, list):
        for item in message:
            if not isinstance(item, dict) or item.get("type") != "at":
                continue
            data = item.get("data")
            if not isinstance(data, dict):
                continue
            mentioned = data.get("qq", data.get("user_id"))
            if mentioned is not None and str(mentioned).strip() == normalized_self_id:
                return True

    raw_message = payload.get("raw_message")
    if isinstance(raw_message, str):
        if f"[CQ:at,qq={normalized_self_id}]" in raw_message:
            return True
        # NapCat may encode @ as "QQ####" or just include the number
        if str(normalized_self_id) in raw_message and "@" in raw_message:
            return True
    return False


def is_self_message(payload: dict[str, Any]) -> bool:
    if payload.get("post_type") == "message_sent":
        return True

    self_id = payload.get("self_id")
    sender_id = extract_sender_id(payload)
    if self_id is None or sender_id is None:
        return False
    return str(self_id).strip() == str(sender_id).strip()


# 群聊主动插话计数器：每个群每收到 N 条未 @ 的消息，放一条让 agent 自行决定是否回复
def message_mentions_assistant_name(payload: dict[str, Any], assistant_aliases: tuple[str, ...] | list[str] | set[str] = ()) -> bool:
    text = extract_text(payload)
    normalized_text = _normalize_address_text(text)
    if not normalized_text:
        return False
    for alias in assistant_aliases:
        normalized_alias = _normalize_address_text(str(alias or ""))
        if not normalized_alias:
            continue
        # 消息中包含银狼名字 → 提到它
        if normalized_alias in normalized_text:
            return True
    return False


def _normalize_address_text(value: str) -> str:
    return re.sub(
        r"[\s`'\"\u201c\u201d\u2018\u2019\u300c\u300d\uff0c\u3002\uff01\uff1f?.!?:\uff1a\uff1b\uff08\uff09()\[\]{}_-]+",
        "",
        str(value or "").strip().lower(),
    )


def should_respond(
    payload: dict[str, Any],
    *,
    assistant_aliases: tuple[str, ...] | list[str] | set[str] = (),
) -> bool:
    message_type = str(payload.get("message_type", "")).strip()
    if message_type == "private":
        return True
    if message_type == "group":
        # 群聊：只有 @ 银狼 才触发指定回复
        if message_mentions_self(payload):
            return True
        return False
    return False


def extract_onebot_message(
    payload: dict[str, Any],
    *,
    assistant_aliases: tuple[str, ...] | list[str] | set[str] = (),
) -> OneBotInboundMessage | None:
    if not is_onebot_event(payload):
        return None
    if is_self_message(payload):
        return None
    if not should_respond(payload, assistant_aliases=assistant_aliases):
        return None

    message_type = str(payload.get("message_type", "")).strip()
    if message_type not in {"group", "private"}:
        return None

    text = extract_text(payload)
    audio_attachments = extract_audio_attachments(payload)
    image_attachments = extract_image_attachments(payload)
    if not text and not audio_attachments and not image_attachments:
        return None

    sender_id = extract_sender_id(payload) or "anonymous"
    if message_type == "group":
        group_id = payload.get("group_id")
        if group_id is None:
            return None
        session_id = f"onebot_group_{group_id}"
        target = OneBotTarget(message_type="group", group_id=int(group_id))
    else:
        session_id = f"onebot_private_{sender_id}"
        target = OneBotTarget(message_type="private", user_id=int(sender_id))

    mentioned_self = message_mentions_self(payload)
    mentioned_assistant_name = message_mentions_assistant_name(payload, assistant_aliases)
    metadata = {
        "raw_payload": payload,
        "platform": "onebot_v11",
        "addressed_to_assistant": bool(mentioned_self or mentioned_assistant_name),
        "mentioned_self": bool(mentioned_self),
        "mentioned_assistant_name": bool(mentioned_assistant_name),
    }

    return OneBotInboundMessage(
        session_id=session_id,
        text=text,
        sender_id=sender_id,
        mode=extract_mode(payload),
        scope_root=extract_scope_root(payload),
        target=target,
        metadata=metadata,
        audio_attachments=audio_attachments,
        image_attachments=image_attachments,
    )


def _infer_typing_active(payload: dict[str, Any], raw_blob: str) -> bool | None:
    for key in ("typing", "is_typing", "inputting", "active"):
        value = payload.get(key)
        parsed = _parse_typing_bool(value)
        if parsed is not None:
            return parsed

    for key in ("status", "state", "text", "message", "wording", "title"):
        value = payload.get(key)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if any(marker in normalized for marker in ("停止输入", "停止", "ended", "stop", "stopped", "inactive")):
                return False
            if any(marker in normalized for marker in ("正在输入", "typing", "inputting", "composing")):
                return True

    if "正在输入" in raw_blob or "inputting" in raw_blob or "composing" in raw_blob:
        return True
    if any(marker in raw_blob for marker in ('"typing":true', '"is_typing":true', '"inputting":true', '"active":true')):
        return True
    if any(marker in raw_blob for marker in ('"typing":false', '"is_typing":false', '"inputting":false', '"active":false')):
        return False
    if "stopped typing" in raw_blob or "停止输入" in raw_blob:
        return False
    return None


def _parse_typing_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on", "typing", "active", "inputting"}:
            return True
        if normalized in {"false", "0", "no", "off", "inactive", "idle", "stopped"}:
            return False
    return None
