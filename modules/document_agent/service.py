from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from local_agent.llm.ollama_client import OllamaClient
from local_agent.modules.file.service import FileModule
from local_agent.protocol.models import OutputKind, ToolManifest


class DocumentAgentSummaryInput(BaseModel):
    source_path: str
    instruction: str
    recent_context: str = ""
    grounded_inputs: dict[str, Any] = Field(default_factory=dict)
    resolved_facts: dict[str, Any] = Field(default_factory=dict)
    source_materials: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    style_hints: dict[str, Any] = Field(default_factory=dict)
    max_chars: int = 12000


class DocumentAgentReadInput(BaseModel):
    source_path: str
    instruction: str = ""
    recent_context: str = ""
    grounded_inputs: dict[str, Any] = Field(default_factory=dict)
    resolved_facts: dict[str, Any] = Field(default_factory=dict)
    source_materials: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    style_hints: dict[str, Any] = Field(default_factory=dict)
    max_chars: int = 12000


class DocumentAgentInspectInput(BaseModel):
    source_path: str
    instruction: str
    recent_context: str = ""
    grounded_inputs: dict[str, Any] = Field(default_factory=dict)
    resolved_facts: dict[str, Any] = Field(default_factory=dict)
    source_materials: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    style_hints: dict[str, Any] = Field(default_factory=dict)
    max_blocks: int = 200
    max_chars_per_block: int = 1200
    max_matches: int = 8


class DocumentAgentEditInput(BaseModel):
    source_path: str
    instruction: str
    output_path: str | None = None
    recent_context: str = ""
    grounded_inputs: dict[str, Any] = Field(default_factory=dict)
    resolved_facts: dict[str, Any] = Field(default_factory=dict)
    source_materials: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    style_hints: dict[str, Any] = Field(default_factory=dict)
    allow_overwrite: bool = True
    preserve_structure: bool = True
    preserve_style: bool = True
    max_chars: int = 12000
    max_blocks: int = 200
    max_chars_per_block: int = 1200


class DocumentAgentModule:
    def __init__(self, *, file_module: FileModule, llm_client: OllamaClient) -> None:
        self.file_module = file_module
        self.llm_client = llm_client

    def manifests(self) -> list[ToolManifest]:
        return [
            ToolManifest(
                tool_name="document_agent.summarize",
                module="document_agent",
                description="Delegate local document summarization to the document sub-agent.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.OBJECT_DETAILS, OutputKind.FILE_CONTENTS],
                input_schema=DocumentAgentSummaryInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"path": {"type": "string"}, "files": {"type": "array"}}},
            ),
            ToolManifest(
                tool_name="document_agent.read",
                module="document_agent",
                description="Delegate local document content extraction to the document sub-agent.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.FILE_CONTENTS],
                input_schema=DocumentAgentReadInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"path": {"type": "string"}, "files": {"type": "array"}}},
            ),
            ToolManifest(
                tool_name="document_agent.inspect",
                module="document_agent",
                description="Delegate local document inspection or block search to the document sub-agent.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=DocumentAgentInspectInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"path": {"type": "string"}, "blocks": {"type": "array"}}},
            ),
            ToolManifest(
                tool_name="document_agent.edit",
                module="document_agent",
                description="Delegate local document editing to the document sub-agent.",
                side_effect=True,
                idempotent=False,
                produces=[OutputKind.OBJECT_DETAILS, OutputKind.FILE_CONTENTS, OutputKind.FILE_WRITTEN],
                input_schema=DocumentAgentEditInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"path": {"type": "string"}, "files": {"type": "array"}}},
            ),
        ]

    def executor_map(self) -> dict[str, Any]:
        return {
            "document_agent.summarize": self.summarize,
            "document_agent.read": self.read,
            "document_agent.inspect": self.inspect,
            "document_agent.edit": self.edit,
        }

    def summarize(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = DocumentAgentSummaryInput.model_validate(arguments)
        source_path = self._resolve_source_path(payload.source_path)
        extracted = self.file_module.extract_text({"paths": [str(source_path)], "max_chars": payload.max_chars})
        file_entry = self._first_file_entry(extracted)
        summary_payload = self.llm_client.summarize_document_for_agent(
            instruction=payload.instruction,
            source_path=str(source_path),
            extracted_text=str(file_entry.get("content", "") or ""),
            recent_context=payload.recent_context,
            grounded_inputs=self._task_package(payload),
        )
        summary_text = str(summary_payload.get("summary", "") or "").strip()
        if not summary_text:
            raise ValueError("document_agent.summarize produced an empty summary")
        return {
            "path": str(source_path),
            "task_kind": "document_summary",
            "summary": summary_text,
            "speech_text": str(summary_payload.get("speech_text", "") or "").strip(),
            "file_type": file_entry.get("extraction_type"),
            "files": [
                {
                    "path": str(source_path),
                    "content": summary_text,
                    "source_excerpt": str(file_entry.get("content", "") or "")[:1200],
                    "extraction_type": file_entry.get("extraction_type"),
                }
            ],
        }

    def read(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = DocumentAgentReadInput.model_validate(arguments)
        source_path = self._resolve_source_path(payload.source_path)
        extracted = self.file_module.extract_text({"paths": [str(source_path)], "max_chars": payload.max_chars})
        file_entry = self._first_file_entry(extracted)
        return {
            "path": str(source_path),
            "task_kind": "document_read",
            "file_type": file_entry.get("extraction_type"),
            "files": extracted.get("files", []),
        }

    def inspect(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = DocumentAgentInspectInput.model_validate(arguments)
        source_path = self._resolve_source_path(payload.source_path)
        extracted = self.file_module.extract_text({"paths": [str(source_path)], "max_chars": 4000})
        file_entry = self._first_file_entry(extracted)
        inspection_plan = self.llm_client.plan_document_inspection(
            instruction=payload.instruction,
            source_path=str(source_path),
            extracted_text=str(file_entry.get("content", "") or ""),
            recent_context=payload.recent_context,
            grounded_inputs=self._task_package(payload),
        )
        mode = str(inspection_plan.get("mode", "") or "").strip().lower()
        query = str(inspection_plan.get("query", "") or "").strip()
        if mode == "search_blocks" and query:
            result = self.file_module.search_blocks(
                {
                    "path": str(source_path),
                    "query": query,
                    "max_matches": payload.max_matches,
                    "max_blocks": payload.max_blocks,
                    "max_chars_per_block": payload.max_chars_per_block,
                }
            )
            result["inspection_mode"] = "search_blocks"
            result["task_kind"] = "document_inspect"
            return result

        result = self.file_module.extract_structure(
            {
                "path": str(source_path),
                "include_text": True,
                "max_blocks": payload.max_blocks,
                "max_chars_per_block": payload.max_chars_per_block,
                "max_rows_per_sheet": 10,
            }
        )
        result["inspection_mode"] = "structure"
        result["task_kind"] = "document_inspect"
        return result

    def edit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = DocumentAgentEditInput.model_validate(arguments)
        source_path = self._resolve_source_path(payload.source_path)
        output_path = self.file_module.resolve_path(payload.output_path or payload.source_path)
        self.file_module.ensure_workspace_path(output_path)

        suffix = source_path.suffix.lower()
        if suffix == ".docx":
            return self._edit_docx(payload=payload, source_path=source_path, output_path=output_path)
        if suffix in {".txt", ".md"}:
            return self._edit_text_document(payload=payload, source_path=source_path, output_path=output_path)
        raise ValueError(f"document_agent.edit does not support this file type yet: {suffix or 'unknown'}")

    def _edit_docx(
        self,
        *,
        payload: DocumentAgentEditInput,
        source_path: Path,
        output_path: Path,
    ) -> dict[str, Any]:
        structure = self.file_module.extract_structure(
            {
                "path": str(source_path),
                "include_text": True,
                "max_blocks": payload.max_blocks,
                "max_chars_per_block": payload.max_chars_per_block,
                "max_rows_per_sheet": 10,
            }
        )
        extracted = self.file_module.extract_text({"paths": [str(source_path)], "max_chars": payload.max_chars})
        file_entry = self._first_file_entry(extracted)
        allowed_block_ids = {
            str(block.get("block_id", "")).strip()
            for block in structure.get("blocks", [])
            if isinstance(block, dict)
        }
        append_anchor_block_id = self._last_paragraph_block_id(structure)
        planning_payload = payload.model_copy(
            update={
                "grounded_inputs": self._with_docx_style_hints(
                    self._task_package(payload),
                    structure=structure,
                )
            }
        )
        plan, edits = self._plan_docx_edits_with_fallback(
            payload=planning_payload,
            source_path=source_path,
            extracted_text=str(file_entry.get("content", "") or ""),
            structure_blocks=self._prompt_blocks(structure),
            allowed_block_ids=allowed_block_ids,
            suggested_append_anchor_block_id=append_anchor_block_id,
        )
        if not edits:
            raise ValueError("document_agent.edit produced no valid docx edits")

        write_result = self.file_module.edit_docx(
            {
                "source_path": str(source_path),
                "output_path": str(output_path),
                "overwrite": payload.allow_overwrite,
                "edits": edits,
            }
        )
        updated = self.file_module.extract_text({"paths": [str(output_path)], "max_chars": payload.max_chars})
        updated_file = self._first_file_entry(updated)
        return {
            **write_result,
            "task_kind": "document_edit",
            "summary": str(plan.get("summary", "") or "").strip(),
            "file_type": structure.get("file_type"),
            "block_count": structure.get("block_count"),
            "files": updated.get("files", []),
            "source_excerpt": str(file_entry.get("content", "") or "")[:1200],
            "updated_excerpt": str(updated_file.get("content", "") or "")[:1200],
        }

    def _edit_text_document(
        self,
        *,
        payload: DocumentAgentEditInput,
        source_path: Path,
        output_path: Path,
    ) -> dict[str, Any]:
        extracted = self.file_module.read_files({"paths": [str(source_path)], "encoding": "utf-8", "max_bytes": payload.max_chars})
        file_entry = self._first_file_entry(extracted)
        rewritten = self.llm_client.rewrite_text_document_for_agent(
            instruction=payload.instruction,
            source_path=str(source_path),
            original_text=str(file_entry.get("content", "") or ""),
            recent_context=payload.recent_context,
            grounded_inputs=self._task_package(payload),
            preserve_structure=payload.preserve_structure,
        )
        new_content = str(rewritten.get("content", "") or "")
        if not new_content.strip():
            raise ValueError("document_agent.edit produced empty text content")
        write_result = self.file_module.write_file(
            {
                "path": str(output_path),
                "content": new_content,
                "encoding": "utf-8",
                "overwrite": payload.allow_overwrite,
            }
        )
        return {
            **write_result,
            "task_kind": "document_edit",
            "summary": str(rewritten.get("summary", "") or "").strip(),
            "file_type": output_path.suffix.lower().lstrip("."),
            "files": [{"path": str(output_path), "content": new_content, "extraction_type": output_path.suffix.lower().lstrip(".")}],
            "source_excerpt": str(file_entry.get("content", "") or "")[:1200],
            "updated_excerpt": new_content[:1200],
        }

    def _resolve_source_path(self, raw_path: str) -> Path:
        source_path = self.file_module.resolve_path(raw_path)
        self.file_module.ensure_workspace_path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")
        if not source_path.is_file():
            raise IsADirectoryError(f"Source path must be a file: {source_path}")
        return source_path

    @staticmethod
    def _first_file_entry(result: dict[str, Any]) -> dict[str, Any]:
        files = result.get("files")
        if not isinstance(files, list) or not files or not isinstance(files[0], dict):
            raise ValueError("Document agent expected one extracted file entry")
        return files[0]

    @staticmethod
    def _task_package(payload: Any) -> dict[str, Any]:
        package = dict(getattr(payload, "grounded_inputs", {}) or {})
        resolved_facts = dict(getattr(payload, "resolved_facts", {}) or {})
        source_materials = dict(getattr(payload, "source_materials", {}) or {})
        constraints = dict(getattr(payload, "constraints", {}) or {})
        style_hints = dict(getattr(payload, "style_hints", {}) or {})

        for key in ("current_date", "current_date_mmdd", "current_time_iso", "timezone", "target_path", "target_name"):
            if key in package and key not in resolved_facts:
                resolved_facts[key] = package[key]

        constraints.setdefault("document_agent_decides_edit_operations", True)
        constraints.setdefault("main_agent_does_not_select_blocks", True)
        constraints.setdefault("must_not_change_unrelated_content", True)

        package["resolved_facts"] = resolved_facts
        package["source_materials"] = source_materials
        package["constraints"] = constraints
        package["style_hints"] = style_hints
        return package

    @staticmethod
    def _prompt_blocks(structure: dict[str, Any], *, limit: int = 80) -> list[dict[str, str]]:
        prompt_blocks: list[dict[str, str]] = []
        for block in structure.get("blocks", [])[:limit]:
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("block_id", "") or "").strip()
            if not block_id:
                continue
            prompt_blocks.append(
                {
                    "block_id": block_id,
                    "block_type": str(block.get("block_type", "") or "").strip(),
                    "text": str(block.get("text", "") or "").strip()[:400],
                }
            )
        return prompt_blocks

    @staticmethod
    def _last_paragraph_block_id(structure: dict[str, Any]) -> str | None:
        for block in reversed(structure.get("blocks", [])):
            if not isinstance(block, dict):
                continue
            if str(block.get("block_type", "")).strip().lower() != "paragraph":
                continue
            block_id = str(block.get("block_id", "") or "").strip()
            if block_id:
                return block_id
        return None

    @staticmethod
    def _with_docx_style_hints(
        grounded_inputs: dict[str, Any],
        *,
        structure: dict[str, Any],
    ) -> dict[str, Any]:
        enriched = dict(grounded_inputs or {})
        paragraphs: list[str] = []
        date_prefixed: list[str] = []
        for block in structure.get("blocks", []):
            if not isinstance(block, dict):
                continue
            if str(block.get("block_type", "")).strip().lower() != "paragraph":
                continue
            text = str(block.get("text", "") or "").strip()
            if not text:
                continue
            paragraphs.append(text[:220])
            if re.match(r"^\d{4}[：:]", text):
                date_prefixed.append(text[:220])

        if paragraphs:
            enriched["recent_document_paragraphs"] = paragraphs[-6:]
        if date_prefixed:
            enriched["date_entry_style_examples"] = date_prefixed[-4:]
            enriched["preferred_date_entry_prefix"] = "MMDD："
        return enriched

    @staticmethod
    def _normalize_docx_edits(raw_edits: object, *, allowed_block_ids: set[str]) -> list[dict[str, str]]:
        valid_actions = {"replace", "insert_after", "delete"}
        action_aliases = {
            "replace": "replace",
            "rewrite": "replace",
            "overwrite": "replace",
            "update": "replace",
            "set": "replace",
            "delete": "delete",
            "remove": "delete",
            "drop": "delete",
            "insert_after": "insert_after",
            "append": "insert_after",
            "add": "insert_after",
            "insert": "insert_after",
            "append_after": "insert_after",
            "add_after": "insert_after",
            "after": "insert_after",
        }
        normalized: list[dict[str, str]] = []
        if not isinstance(raw_edits, list):
            return normalized
        for item in raw_edits:
            if not isinstance(item, dict):
                continue
            block_id = str(
                item.get("block_id")
                or item.get("target_block_id")
                or item.get("anchor_block_id")
                or ""
            ).strip()
            raw_action = str(item.get("action", "") or "").strip().lower()
            action = action_aliases.get(raw_action, raw_action)
            text = str(item.get("text", "") or "")
            if block_id not in allowed_block_ids or action not in valid_actions:
                continue
            if action != "delete" and not text.strip():
                continue
            normalized.append({"block_id": block_id, "action": action, "text": text})
        return normalized

    def _plan_docx_edits_with_fallback(
        self,
        *,
        payload: DocumentAgentEditInput,
        source_path: Path,
        extracted_text: str,
        structure_blocks: list[dict[str, str]],
        allowed_block_ids: set[str],
        suggested_append_anchor_block_id: str | None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        plan = self.llm_client.plan_document_docx_edits(
            instruction=payload.instruction,
            source_path=str(source_path),
            extracted_text=extracted_text,
            structure_blocks=structure_blocks,
            recent_context=payload.recent_context,
            grounded_inputs=payload.grounded_inputs,
            preserve_structure=payload.preserve_structure,
            preserve_style=payload.preserve_style,
            suggested_append_anchor_block_id=suggested_append_anchor_block_id,
        )
        edits = self._normalize_docx_edits_with_anchor_fallback(
            plan.get("edits"),
            allowed_block_ids=allowed_block_ids,
            suggested_append_anchor_block_id=suggested_append_anchor_block_id,
        )
        primary_model = str(getattr(self.llm_client, "model", "") or "").strip()
        response_model = str(getattr(self.llm_client, "response_model", "") or "").strip()
        if edits or not primary_model or primary_model == response_model:
            return plan, edits

        retry_plan = self.llm_client.plan_document_docx_edits(
            instruction=payload.instruction,
            source_path=str(source_path),
            extracted_text=extracted_text,
            structure_blocks=structure_blocks,
            recent_context=payload.recent_context,
            grounded_inputs=payload.grounded_inputs,
            preserve_structure=payload.preserve_structure,
            preserve_style=payload.preserve_style,
            suggested_append_anchor_block_id=suggested_append_anchor_block_id,
            model_override=primary_model,
        )
        retry_edits = self._normalize_docx_edits_with_anchor_fallback(
            retry_plan.get("edits"),
            allowed_block_ids=allowed_block_ids,
            suggested_append_anchor_block_id=suggested_append_anchor_block_id,
        )
        if retry_edits:
            return retry_plan, retry_edits
        return plan, edits

    @classmethod
    def _normalize_docx_edits_with_anchor_fallback(
        cls,
        raw_edits: object,
        *,
        allowed_block_ids: set[str],
        suggested_append_anchor_block_id: str | None,
    ) -> list[dict[str, str]]:
        normalized = cls._normalize_docx_edits(raw_edits, allowed_block_ids=allowed_block_ids)
        if normalized or not isinstance(raw_edits, list):
            return normalized

        anchor = str(suggested_append_anchor_block_id or "").strip()
        if not anchor or anchor not in allowed_block_ids:
            return normalized

        recovered: list[dict[str, str]] = []
        for item in raw_edits:
            if not isinstance(item, dict):
                continue
            raw_action = str(item.get("action", "") or "").strip().lower()
            if raw_action not in {"append", "add", "insert", "append_after", "add_after", "after", "insert_after"}:
                continue
            text = str(item.get("text", "") or "")
            if not text.strip():
                continue
            recovered.append({"block_id": anchor, "action": "insert_after", "text": text})
        return recovered
