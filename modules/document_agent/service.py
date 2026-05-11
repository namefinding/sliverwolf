from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from local_agent.llm.ollama_client import OllamaClient
from local_agent.modules.file.service import FileModule
from local_agent.protocol.execution_contract import build_subagent_task_package
from local_agent.protocol.models import OutputKind, ToolExecutionContextFields, ToolManifest


class DocumentAgentSummaryInput(ToolExecutionContextFields):
    source_path: str
    instruction: str
    recent_context: str = ""
    resolved_facts: dict[str, Any] = Field(default_factory=dict)
    source_materials: dict[str, Any] = Field(default_factory=dict)
    style_hints: dict[str, Any] = Field(default_factory=dict)
    max_chars: int = 12000


class DocumentAgentReadInput(ToolExecutionContextFields):
    source_path: str
    instruction: str = ""
    recent_context: str = ""
    resolved_facts: dict[str, Any] = Field(default_factory=dict)
    source_materials: dict[str, Any] = Field(default_factory=dict)
    style_hints: dict[str, Any] = Field(default_factory=dict)
    max_chars: int = 12000


class DocumentAgentInspectInput(ToolExecutionContextFields):
    source_path: str
    instruction: str
    recent_context: str = ""
    resolved_facts: dict[str, Any] = Field(default_factory=dict)
    source_materials: dict[str, Any] = Field(default_factory=dict)
    style_hints: dict[str, Any] = Field(default_factory=dict)
    max_chars: int = 16000
    max_blocks: int = 200
    max_chars_per_block: int = 1200
    max_matches: int = 8


class DocumentAgentEditInput(ToolExecutionContextFields):
    source_path: str
    instruction: str
    output_path: str | None = None
    recent_context: str = ""
    resolved_facts: dict[str, Any] = Field(default_factory=dict)
    source_materials: dict[str, Any] = Field(default_factory=dict)
    style_hints: dict[str, Any] = Field(default_factory=dict)
    allow_overwrite: bool = True
    preserve_structure: bool = True
    preserve_style: bool = True
    max_chars: int = 12000
    max_blocks: int = 200
    max_chars_per_block: int = 1200


class DocumentAgentComposeInput(ToolExecutionContextFields):
    instruction: str
    output_path: str | None = None
    title: str | None = None
    recent_context: str = ""
    resolved_facts: dict[str, Any] = Field(default_factory=dict)
    source_materials: dict[str, Any] = Field(default_factory=dict)
    style_hints: dict[str, Any] = Field(default_factory=dict)
    max_chars: int = 12000


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
                description="Delegate local document inspection, evidence gathering, and content judgment to the document sub-agent.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=DocumentAgentInspectInput.model_json_schema(),
                output_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "answer": {"type": "string"},
                        "findings": {"type": "array"},
                        "evidence": {"type": "array"},
                        "blocks": {"type": "array"},
                    },
                },
            ),
            ToolManifest(
                tool_name="document_agent.compose",
                module="document_agent",
                description="Delegate document composition from raw grounded materials to the document sub-agent before writing a file.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.FILE_CONTENTS, OutputKind.OBJECT_DETAILS],
                input_schema=DocumentAgentComposeInput.model_json_schema(),
                output_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "paragraphs": {"type": "array"},
                        "files": {"type": "array"},
                        "quality": {"type": "object"},
                    },
                },
            ),
            ToolManifest(
                tool_name="document_agent.edit",
                module="document_agent",
                description="Edit/rewrite a document (.docx/.txt/.md) by giving it source_path + editing instruction. This sub-agent handles everything internally: reads the full file, sends content + instruction to LLM, writes back the rewritten result. No need to inspect/read the file separately first — just pass the path and instruction directly.",
                side_effect=True,
                idempotent=False,
                requires_confirmation=True,
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
            "document_agent.compose": self.compose,
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
        extracted = self.file_module.extract_text({"paths": [str(source_path)], "max_chars": max(4000, payload.max_chars)})
        file_entry = self._first_file_entry(extracted)
        extracted_text = str(file_entry.get("content", "") or "")
        analyzer = getattr(self.llm_client, "analyze_document_for_agent", None)
        if callable(analyzer):
            analysis_payload = analyzer(
                instruction=payload.instruction,
                source_path=str(source_path),
                extracted_text=extracted_text,
                recent_context=payload.recent_context,
                grounded_inputs=self._task_package(payload),
            )
            analysis = analysis_payload if isinstance(analysis_payload, dict) else {}
            answer = str(analysis.get("answer") or analysis.get("summary") or "").strip()
            summary = str(analysis.get("summary") or answer or "").strip()
            findings = analysis.get("findings") if isinstance(analysis.get("findings"), list) else []
            evidence = analysis.get("evidence") if isinstance(analysis.get("evidence"), list) else []
            if not answer:
                answer = "Document inspection completed, but no clear conclusion was produced."
            return {
                "path": str(source_path),
                "task_kind": "document_inspect",
                "inspection_mode": str(analysis.get("inspection_mode") or "analysis"),
                "answer": answer,
                "summary": summary or answer,
                "speech_text": str(analysis.get("speech_text") or "").strip(),
                "verdict": analysis.get("verdict") if isinstance(analysis.get("verdict"), dict) else {},
                "findings": findings,
                "evidence": evidence,
                "analysis": analysis,
                "file_type": file_entry.get("extraction_type"),
                "source_excerpt": extracted_text[:1200],
                "tail_excerpt": self._tail_excerpt(extracted_text),
                "files": [
                    {
                        "path": str(source_path),
                        "content": answer,
                        "source_excerpt": extracted_text[:1200],
                        "tail_excerpt": self._tail_excerpt(extracted_text),
                        "extraction_type": file_entry.get("extraction_type"),
                    }
                ],
            }

        inspection_plan = self.llm_client.plan_document_inspection(
            instruction=payload.instruction,
            source_path=str(source_path),
            extracted_text=extracted_text,
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

    def compose(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = DocumentAgentComposeInput.model_validate(arguments)
        source_materials = dict(payload.source_materials or {})
        composer = getattr(self.llm_client, "compose_document_for_agent", None)
        if callable(composer):
            composed = composer(
                instruction=payload.instruction,
                title=payload.title,
                output_path=payload.output_path,
                source_materials=source_materials,
                recent_context=payload.recent_context,
                grounded_inputs=self._task_package(payload),
                style_hints=payload.style_hints,
                max_chars=payload.max_chars,
            )
        else:
            composed = self._compose_from_plain_materials(payload=payload, source_materials=source_materials)

        title = str(composed.get("title") or payload.title or "").strip()
        content = self._clean_generated_document_text(str(composed.get("content") or "").strip())
        if not content:
            content = self._clean_generated_document_text(self._fallback_material_text(source_materials))
        if not content:
            raise ValueError("document_agent.compose produced empty document content")

        quality = self._inspect_composed_text(content)
        paragraphs = self._split_paragraphs(content)
        file_entry = {
            "path": str(payload.output_path or ""),
            "content": content,
            "title": title,
            "extraction_type": "composed_document",
        }
        return {
            "task_kind": "document_compose",
            "title": title,
            "content": content,
            "paragraphs": paragraphs,
            "quality": quality,
            "source_material_keys": sorted(source_materials.keys()),
            "files": [file_entry],
        }

    def edit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = DocumentAgentEditInput.model_validate(arguments)
        source_path = self._resolve_source_path(payload.source_path)
        output_path = self.file_module.resolve_path(payload.output_path or payload.source_path)
        self.file_module.ensure_workspace_path(output_path)

        suffix = source_path.suffix.lower()
        # .docx .txt .md 统一走全文 LLM 重写——读全文 + 编辑指令 → LLM 返回新内容 → 覆盖写入
        if suffix in {".docx", ".txt", ".md"}:
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
        prompt_blocks = self._prompt_blocks(structure)
        plan, edits = self._plan_docx_edits_with_fallback(
            payload=planning_payload,
            source_path=source_path,
            extracted_text=str(file_entry.get("content", "") or ""),
            structure_blocks=prompt_blocks,
            allowed_block_ids=allowed_block_ids,
            suggested_append_anchor_block_id=append_anchor_block_id,
        )
        if not edits:
            raise ValueError("document_agent.edit produced no valid docx edits")
        edit_safety = self._validate_docx_edit_safety(
            plan=plan,
            edits=edits,
            structure_blocks=prompt_blocks,
        )

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
            "edit_intent": plan.get("edit_intent") if isinstance(plan.get("edit_intent"), dict) else {},
            "file_type": structure.get("file_type"),
            "block_count": structure.get("block_count"),
            "files": updated.get("files", []),
            "source_excerpt": str(file_entry.get("content", "") or "")[:1200],
            "updated_excerpt": str(updated_file.get("content", "") or "")[:1200],
            "edit_safety": edit_safety,
        }

    def _edit_text_document(
        self,
        *,
        payload: DocumentAgentEditInput,
        source_path: Path,
        output_path: Path,
    ) -> dict[str, Any]:
        # .docx 等 Office 文档用 extract_text 读，纯文本用 read_files
        suffix = source_path.suffix.lower()
        if suffix in {".docx", ".xlsx", ".pptx"}:
            extracted = self.file_module.extract_text({"paths": [str(source_path)], "max_chars": payload.max_chars})
        else:
            extracted = self.file_module.read_files({"paths": [str(source_path)], "encoding": "utf-8", "max_bytes": payload.max_chars})
        file_entry = self._first_file_entry(extracted)
        original_text = str(file_entry.get("content", "") or "").strip()
        if not original_text:
            raise ValueError(f"document_agent.edit could not extract text from: {source_path}")
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
        # .docx 不能用裸 text 写回，走 write_docx 结构写回
        if suffix == ".docx":
            paragraphs = [block.strip() for block in new_content.split("\n\n") if block.strip()]
            write_result = self.file_module.write_docx(
                {
                    "path": str(output_path),
                    "title": str(rewritten.get("title", "") or "") or None,
                    "paragraphs": paragraphs,
                    "overwrite": payload.allow_overwrite,
                }
            )
        else:
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
    def _extract_research_bundle(source_materials: dict[str, Any]) -> dict[str, Any]:
        for key in ("research_bundle", "web_research", "web", "web_result"):
            value = source_materials.get(key)
            if isinstance(value, dict) and value:
                return value
        tool_results = source_materials.get("tool_results")
        if isinstance(tool_results, list):
            for item in reversed(tool_results):
                if not isinstance(item, dict):
                    continue
                if item.get("tool_name") in {"web.research", "web.fetch"}:
                    data = item.get("data")
                    if isinstance(data, dict) and data:
                        return data
        return {}

    def _compose_from_plain_materials(
        self,
        *,
        payload: DocumentAgentComposeInput,
        source_materials: dict[str, Any],
    ) -> dict[str, str]:
        fallback_text = self._fallback_material_text(source_materials)
        summarizer = getattr(self.llm_client, "summarize_document_for_agent", None)
        if callable(summarizer):
            result = summarizer(
                instruction=payload.instruction,
                source_path=payload.output_path or payload.title or "composed_document",
                extracted_text=fallback_text,
                recent_context=payload.recent_context,
                grounded_inputs=self._task_package(payload),
            )
            return {
                "title": payload.title or str(result.get("title") or "").strip(),
                "content": str(result.get("summary") or result.get("content") or "").strip(),
            }
        return {"title": payload.title or "", "content": fallback_text}

    @classmethod
    def _fallback_material_text(cls, source_materials: dict[str, Any]) -> str:
        chunks: list[str] = []
        for key in ("summary", "content", "text", "body"):
            value = source_materials.get(key)
            if isinstance(value, str) and value.strip():
                chunks.append(value.strip())
        for collection_key in ("sources", "results", "items"):
            collection = source_materials.get(collection_key)
            if not isinstance(collection, list):
                continue
            for item in collection[:8]:
                if not isinstance(item, dict):
                    continue
                parts = [
                    str(item.get("title") or "").strip(),
                    str(item.get("url") or "").strip(),
                    str(item.get("snippet") or item.get("excerpt") or item.get("content") or "").strip(),
                ]
                joined = "\n".join(part for part in parts if part)
                if joined:
                    chunks.append(joined)
        return cls._clean_generated_document_text("\n\n".join(chunks))

    @staticmethod
    def _looks_like_tool_observation_text(text: str) -> bool:
        compact = str(text or "").strip()
        if not compact:
            return False
        return bool(re.match(r"^req_[0-9a-f]+\s+\w+\.", compact)) or " results_sample=" in compact

    @staticmethod
    def _looks_like_mojibake_text(text: str) -> bool:
        sample = str(text or "")[:400]
        if not sample:
            return False
        markers = ("氓", "忙", "莽", "盲", "猫", "茅", "脗", "陇", "職", "聫", "聢", "聦")
        marker_count = sum(sample.count(marker) for marker in markers)
        return marker_count >= 8

    @classmethod
    def _clean_generated_document_text(cls, text: str) -> str:
        lines: list[str] = []
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if cls._looks_like_tool_observation_text(line):
                continue
            if cls._looks_like_mojibake_text(line):
                continue
            lines.append(line)
        return "\n\n".join(lines).strip()

    @staticmethod
    def _split_paragraphs(content: str) -> list[str]:
        return [part.strip() for part in re.split(r"\n{2,}", str(content or "")) if part.strip()]

    @classmethod
    def _inspect_composed_text(cls, content: str) -> dict[str, Any]:
        issues: list[str] = []
        if cls._looks_like_tool_observation_text(content):
            issues.append("contains_tool_observation")
        if cls._looks_like_mojibake_text(content):
            issues.append("contains_mojibake")
        if len(str(content or "").strip()) < 40:
            issues.append("very_short_content")
        return {
            "ok": not issues,
            "issues": issues,
            "char_count": len(str(content or "")),
            "paragraph_count": len(cls._split_paragraphs(content)),
        }

    @staticmethod
    def _task_package(payload: Any) -> dict[str, Any]:
        return build_subagent_task_package(
            payload,
            default_constraints={
                "document_agent_decides_edit_operations": True,
                "main_agent_does_not_select_blocks": True,
                "must_not_change_unrelated_content": True,
                "destructive_document_edits_require_conservative_plan": True,
                "delete_edits_must_preserve_or_replace_requested_information": True,
                "deduplication_edits_must_keep_one_representative": True,
                "prefer_merge_or_replace_over_delete_when_uncertain": True,
            },
        )

    @staticmethod
    def _tail_excerpt(text: str, *, max_chars: int = 2000) -> str:
        cleaned = str(text or "")
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[-max_chars:]

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

    @classmethod
    def _validate_docx_edit_safety(
        cls,
        *,
        plan: dict[str, Any],
        edits: list[dict[str, str]],
        structure_blocks: list[dict[str, str]],
    ) -> dict[str, Any]:
        delete_ids = [
            str(edit.get("block_id", "") or "").strip()
            for edit in edits
            if str(edit.get("action", "") or "").strip().lower() == "delete"
        ]
        delete_ids = [block_id for block_id in delete_ids if block_id]
        if not delete_ids:
            return {"destructive": False, "delete_count": 0}

        non_delete_edits = [
            edit
            for edit in edits
            if str(edit.get("action", "") or "").strip().lower() in {"replace", "insert_after"}
            and str(edit.get("text", "") or "").strip()
        ]
        if len(delete_ids) == 1:
            return {"destructive": True, "delete_count": 1, "status": "single_delete"}
        if non_delete_edits:
            return {
                "destructive": True,
                "delete_count": len(delete_ids),
                "status": "delete_with_replacement_or_merge",
                "replacement_count": len(non_delete_edits),
            }

        block_text_by_id = {
            str(block.get("block_id", "") or "").strip(): str(block.get("text", "") or "")
            for block in structure_blocks
            if isinstance(block, dict) and str(block.get("block_id", "") or "").strip()
        }
        deleted = set(delete_ids)
        safety = cls._extract_docx_safety_review(plan)
        if safety.get("user_requested_total_removal"):
            return {
                "destructive": True,
                "delete_count": len(delete_ids),
                "status": "explicit_total_removal",
            }

        preserved_ids = [
            block_id
            for block_id in cls._extract_preserved_block_ids(plan)
            if block_id in block_text_by_id and block_id not in deleted
        ]
        covered_pairs: list[dict[str, Any]] = []
        uncovered_delete_ids: list[str] = []
        for delete_id in delete_ids:
            deleted_text = cls._normalize_compare_text(block_text_by_id.get(delete_id, ""))
            if not deleted_text:
                continue
            best_pair: dict[str, Any] | None = None
            for preserved_id in preserved_ids:
                preserved_text = cls._normalize_compare_text(block_text_by_id.get(preserved_id, ""))
                if not preserved_text:
                    continue
                score = cls._text_similarity(deleted_text, preserved_text)
                if score >= 0.82 or deleted_text in preserved_text or preserved_text in deleted_text:
                    if best_pair is None or score > float(best_pair["similarity"]):
                        best_pair = {
                            "deleted_block_id": delete_id,
                            "preserved_block_id": preserved_id,
                            "similarity": round(score, 3),
                        }
            if best_pair is None:
                uncovered_delete_ids.append(delete_id)
            else:
                covered_pairs.append(best_pair)

        if uncovered_delete_ids:
            raise ValueError(
                "Unsafe destructive docx edit plan: multiple delete operations need a surviving equivalent block "
                "or a replacement/merge edit before writing. Uncovered delete block_ids: "
                + ", ".join(uncovered_delete_ids)
            )

        return {
            "destructive": True,
            "delete_count": len(delete_ids),
            "status": "surviving_equivalent_verified",
            "preserved_pairs": covered_pairs,
        }

    @staticmethod
    def _extract_docx_safety_review(plan: dict[str, Any]) -> dict[str, Any]:
        for key in ("safety", "safety_review", "edit_safety", "destructive_edit_review"):
            value = plan.get(key)
            if isinstance(value, dict):
                return value
        return {}

    @classmethod
    def _extract_preserved_block_ids(cls, plan: dict[str, Any]) -> list[str]:
        ids: list[str] = []
        safety = cls._extract_docx_safety_review(plan)
        for key in (
            "preserved_block_ids",
            "surviving_block_ids",
            "representative_block_ids",
            "preserved_equivalent_block_ids",
        ):
            value = safety.get(key)
            if isinstance(value, list):
                ids.extend(str(item).strip() for item in value if str(item).strip())
            elif isinstance(value, str) and value.strip():
                ids.append(value.strip())

        raw_edits = plan.get("edits")
        if isinstance(raw_edits, dict):
            raw_items = [raw_edits]
        elif isinstance(raw_edits, list):
            raw_items = raw_edits
        else:
            raw_items = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            for key in (
                "preserved_equivalent_block_id",
                "surviving_block_id",
                "representative_block_id",
                "preserved_block_id",
            ):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    ids.append(value.strip())
        return list(dict.fromkeys(ids))

    @staticmethod
    def _normalize_compare_text(value: str) -> str:
        return re.sub(r"\s+", "", str(value or "")).lower()

    @staticmethod
    def _text_similarity(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        return SequenceMatcher(None, left, right).ratio()
