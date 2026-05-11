"""移动文件的 skill —— 搜文件 + 确认 + 移动，一步完成。

这是一个纯确定性 skill（不调 LLM）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from local_agent.modules.file.service import FileModule
from local_agent.protocol.models import OutputKind, ToolManifest
from local_agent.skills.base import Skill


class MoveFileInput(BaseModel):
    file_query: str = Field(description="file name, keyword, or extension to search for. Accepts patterns like '图片' or '.png' or 'png'")
    dest_folder: str = Field(description="destination folder name or path")
    search_path: str = Field(default="C:/Users/namef/Desktop", description="directory to search in")
    extensions: list[str] | None = Field(default=None, description="optional: filter by file extensions like ['.png', '.jpg']")
    target_kind: str = Field(default="file", description="'file' or 'any'")
    overwrite: bool = Field(default=False)

    @model_validator(mode="after")
    def validate(self) -> "MoveFileInput":
        if not self.file_query.strip():
            raise ValueError("file_query must not be empty")
        if not self.dest_folder.strip():
            raise ValueError("dest_folder must not be empty")
        return self


class MoveFileSkill(Skill):
    def manifest(self) -> ToolManifest:
        return ToolManifest(
            tool_name="skill.move_file",
            module="skill",
            description=(
                "Move files matching a query or extension to a destination folder. "
                "Handles searching + moving in one step — supports batch move by keyword or extension. "
                "E.g. 'move all images to folder X' → file_query='image', extensions=['.png','.jpg','.jpeg','.gif']. "
                "E.g. 'move 银狼开发日志 to 项目文件夹' → file_query='银狼开发日志'. "
                "Do NOT try to search and move step-by-step — use this skill directly."
            ),
            side_effect=True,
            idempotent=False,
            requires_confirmation=True,
            produces=[OutputKind.OBJECT_DETAILS, OutputKind.PATH_UPDATED],
            input_schema=MoveFileInput.model_json_schema(),
            output_schema={"type": "object", "properties": {"moved": {"type": "boolean"}, "count": {"type": "integer"}}},
        )

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = MoveFileInput.model_validate(arguments)

        file_module = FileModule(workspace_root=str(Path(payload.search_path).resolve()))

        # 1. 搜索文件
        search_args: dict[str, Any] = {
            "path": payload.search_path,
            "query": payload.file_query,
            "query_terms": [payload.file_query],
            "recursive": True,
            "include_dirs": False,
            "scope_mode": "subtree",
            "target_kind": payload.target_kind,
            "top_k": 50,  # 批量移动最多 50 个
        }
        if payload.extensions:
            search_args["extensions"] = [e if e.startswith(".") else f".{e}" for e in payload.extensions]
        search_result = file_module.search_by_name(search_args)

        candidates = search_result.get("candidates", [])
        if not candidates:
            return {"moved": False, "error": f"no files found matching '{payload.file_query}'", "count": 0}

        # 2. 确认目标文件夹存在
        dest_path = Path(payload.dest_folder)
        if not dest_path.is_absolute():
            dest_path = Path(payload.search_path) / dest_path
        if not dest_path.is_dir():
            # 尝试创建
            file_module.create_directory({
                "path": str(dest_path),
                "parents": True,
                "exist_ok": True,
            })

        # 3. 移动文件
        items = []
        for c in candidates:
            src = c.get("path", "")
            if not src:
                continue
            src_name = Path(src).name
            items.append({
                "src_path": src,
                "dest_path": str(dest_path / src_name),
                "overwrite": payload.overwrite,
            })

        if not items:
            return {"moved": False, "error": "no valid source paths", "count": 0}

        result = file_module.move_many({
            "items": items,
            "continue_on_error": True,
        })

        return {
            "moved": result.get("success_count", 0) > 0,
            "count": result.get("success_count", 0),
            "failure_count": result.get("failure_count", 0),
            "dest_folder": str(dest_path),
        }
