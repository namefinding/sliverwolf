"""代发消息的通用 skill —— 查联系人 + 发送文字/表情包/文件，一步完成。

后续需要加新内容类型（如语音、图片）时，扩展 SendToContactInput + execute 即可。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from local_agent.protocol.models import OutputKind, ToolManifest
from local_agent.skills.base import Skill


class SendToContactInput(BaseModel):
    contact_name: str = Field(description="contact name, alias, or QQ number. E.g. '小明', 'Catch.Y'")
    message: str = Field(default="", description="text message to send. Optional.")
    sticker_name: str | None = Field(default=None, description="sticker name to send. Optional. E.g. '打招呼', '点赞'")
    file_path: str | None = Field(default=None, description="local file path to send as attachment. Optional. E.g. 'C:/Users/namef/Desktop/report.docx'")
    is_group: bool = Field(default=False, description="true if contact is a group")

    @model_validator(mode="after")
    def validate(self) -> "SendToContactInput":
        if not self.contact_name.strip():
            raise ValueError("contact_name must not be empty")
        has_content = bool(self.message.strip() or self.sticker_name or self.file_path)
        if not has_content:
            raise ValueError("at least one of message, sticker_name, or file_path must be provided")
        return self


class SendMessageSkill(Skill):
    def manifest(self) -> ToolManifest:
        return ToolManifest(
            tool_name="skill.send_to_contact",
            module="skill",
            description=(
                "Send a message, sticker, and/or file to a specific QQ contact (friend or group). "
                "Handles contact lookup → sending in one shot. "
                "E.g. '给小明发你好' → contact_name='小明', message='你好'. "
                "E.g. '给小明发个打招呼表情' → contact_name='小明', sticker_name='打招呼'. "
                "E.g. '把报告.docx发给小明' → contact_name='小明', file_path='C:/.../报告.docx'. "
                "E.g. '给小明发你好再发文件再发个点赞表情' → all three fields. "
                "Do NOT use qq.send_text, qq.send_image, qq.send_file, qq.search_contacts separately for this."
            ),
            side_effect=True,
            idempotent=False,
            requires_confirmation=True,
            timeout_ms=30_000,
            produces=[OutputKind.MESSAGE_SENT, OutputKind.OBJECT_DETAILS, OutputKind.CONTACT_CANDIDATES],
            input_schema=SendToContactInput.model_json_schema(),
            output_schema={
                "type": "object",
                "properties": {
                    "sent": {"type": "boolean"},
                    "contact": {"type": "string"},
                    "sent_text": {"type": "boolean"},
                    "sent_sticker": {"type": "boolean"},
                    "sent_file": {"type": "boolean"},
                },
            },
        )

    # ── execute ──────────────────────────────────────────────

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = SendToContactInput.model_validate(arguments)
        runtime = self._get_runtime()
        if runtime is None:
            return {"sent": False, "error": "QQ runtime not available"}

        contact = self._resolve_contact(runtime, payload.contact_name)
        if contact is None:
            return {"sent": False, "error": f"contact not found: {payload.contact_name}"}

        sent_text = self._send_text(runtime, contact, payload.message)
        sent_sticker = self._send_sticker(runtime, contact, payload.sticker_name)
        sent_file = self._send_file(runtime, contact, payload.file_path)

        return {
            "sent": any([sent_text, sent_sticker, sent_file]),
            "contact": contact["name"],
            "target_id": contact["target_id"],
            "target_kind": contact["kind"],
            "sent_text": sent_text,
            "sent_sticker": sent_sticker,
            "sent_file": sent_file,
        }

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _get_runtime():
        from local_agent.app.qq_runtime import QQRuntimeRegistry
        return QQRuntimeRegistry.get_any()

    @staticmethod
    def _resolve_contact(runtime, name: str) -> dict[str, Any] | None:
        candidates = runtime.search_contacts(name, target_kind="any", limit=3)
        if not candidates:
            return None
        best = candidates[0]
        tid = best.get("target_id")
        if not isinstance(tid, int):
            return None
        return {
            "target_id": tid,
            "kind": best.get("kind", "friend"),
            "name": best.get("name", name),
        }

    @staticmethod
    def _send_text(runtime, contact: dict, text: str) -> bool:
        if not text.strip():
            return False
        result = runtime.send_text(text.strip(), target_kind=contact["kind"], target_id=contact["target_id"], current_target=None)
        return bool(result.get("sent", result.get("ok", True)))

    @staticmethod
    def _send_sticker(runtime, contact: dict, sticker_name: str | None) -> bool:
        if not sticker_name:
            return False
        sticker_dir = Path("data/stickers")
        for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
            candidate = sticker_dir / f"{sticker_name}{ext}"
            if candidate.is_file():
                result = runtime.send_image(image_path=str(candidate.resolve()), target_kind=contact["kind"], target_id=contact["target_id"], current_target=None)
                return bool(result.get("sent", True))
        return False

    @staticmethod
    def _send_file(runtime, contact: dict, file_path: str | None) -> bool:
        if not file_path:
            return False
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            return False
        result = runtime.send_file(file_path=str(path), target_kind=contact["kind"], target_id=contact["target_id"], current_target=None)
        return bool(result.get("sent", result.get("ok", True)))
