"""编辑/改写文档的 skill —— 把 document_agent.edit 包装成 skill。

这是一个"内部调用 LLM"的 skill 示例。
主 agent 只需传 source_path + instruction，skill 内部处理全部细节。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from local_agent.protocol.models import OutputKind, ToolManifest
from local_agent.skills.base import Skill


class EditDocumentInput(BaseModel):
    source_path: str = Field(description="path to the file to edit")
    instruction: str = Field(description="what to change: append, replace, update date, add record, etc.")
    allow_overwrite: bool = Field(default=True, description="allow overwriting the original file")


class EditDocumentSkill(Skill):
    def manifest(self) -> ToolManifest:
        return ToolManifest(
            tool_name="skill.edit_document",
            module="skill",
            description=(
                "Edit/rewrite a document (.docx/.txt/.md) with a natural language instruction. "
                "This skill handles everything internally: reads the full file, extracts text, "
                "sends content + instruction to the LLM, receives rewritten content, and writes back. "
                "No need to inspect or read the file first — pass path + instruction directly. "
                "Use for: appending records, changing dates, updating content, reformatting."
            ),
            side_effect=True,
            idempotent=False,
            requires_confirmation=True,
            produces=[OutputKind.OBJECT_DETAILS, OutputKind.FILE_CONTENTS, OutputKind.FILE_WRITTEN],
            input_schema=EditDocumentInput.model_json_schema(),
            output_schema={"type": "object", "properties": {"path": {"type": "string"}, "updated": {"type": "boolean"}}},
        )

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = EditDocumentInput.model_validate(arguments)
        source_path = Path(payload.source_path).expanduser().resolve()
        if not source_path.is_file():
            return {"updated": False, "error": f"file not found: {source_path}"}

        suffix = source_path.suffix.lower()

        # 1. 读文件内容
        from local_agent.modules.file.service import FileModule
        file_module = FileModule(workspace_root=str(source_path.parent))

        if suffix in {".docx", ".xlsx", ".pptx"}:
            extracted = file_module.extract_text({"paths": [str(source_path)], "max_chars": 50000})
        else:
            extracted = file_module.read_files({"paths": [str(source_path)], "encoding": "utf-8", "max_bytes": 50000})

        files = extracted.get("files", [])
        original_text = str(files[0].get("content", "")) if files else ""
        if not original_text.strip():
            return {"updated": False, "error": "could not extract text from file"}

        # 2. LLM 改写
        from local_agent.llm.ollama_client import OllamaClient
        llm = _get_llm_client()
        if llm is None:
            return {"updated": False, "error": "LLM client not available"}

        rewritten = llm.rewrite_text_document_for_agent(
            instruction=payload.instruction,
            source_path=str(source_path),
            original_text=original_text,
            recent_context="",
            grounded_inputs={"instruction": payload.instruction},
            preserve_structure=True,
        )
        new_content = str(rewritten.get("content", "") or "").strip()
        if not new_content:
            return {"updated": False, "error": "LLM produced empty content"}

        # 3. 写回
        if suffix == ".docx":
            paragraphs = [b.strip() for b in new_content.split("\n\n") if b.strip()]
            file_module.write_docx({
                "path": str(source_path),
                "title": str(rewritten.get("title", "") or "") or None,
                "paragraphs": paragraphs,
                "overwrite": payload.allow_overwrite,
            })
        else:
            file_module.write_file({
                "path": str(source_path),
                "content": new_content,
                "encoding": "utf-8",
                "overwrite": payload.allow_overwrite,
            })

        return {"updated": True, "path": str(source_path), "summary": str(rewritten.get("summary", "") or "").strip()}


_LLM_CLIENT = None


def _get_llm_client():
    global _LLM_CLIENT
    if _LLM_CLIENT is not None:
        return _LLM_CLIENT
    try:
        import os
        from local_agent.llm.ollama_client import OllamaClient
        _LLM_CLIENT = OllamaClient(
            base_url="http://127.0.0.1:11434",
            model="deepseek-v4-flash",
            timeout_seconds=120,
            provider="deepseek",
            api_base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
        )
        return _LLM_CLIENT
    except Exception:
        return None
