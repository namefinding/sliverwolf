from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from local_agent.protocol.models import OutputKind, ToolManifest
from local_agent.retrieval.hybrid_index import HybridIndexService


class RebuildLocalIndexInput(BaseModel):
    pass


class SyncLocalIndexInput(BaseModel):
    pass


class SearchLocalObjectsInput(BaseModel):
    query: str
    target_kind: str = "any"
    top_k: int = 8
    path_scope: str = "."
    scope_mode: str = "subtree"
    extensions: list[str] = Field(default_factory=list)
    rebuild_if_missing: bool = True

    @model_validator(mode="after")
    def normalize_target_kind(self) -> "SearchLocalObjectsInput":
        aliases = {
            "dir": "folder",
            "dirs": "folder",
            "directory": "folder",
            "directories": "folder",
            "folder": "folder",
            "folders": "folder",
            "file": "file",
            "files": "file",
            "any": "any",
        }
        self.target_kind = aliases.get(self.target_kind.strip().lower(), self.target_kind.strip().lower())
        return self


class InspectLocalCandidateInput(BaseModel):
    path: str
    max_chars: int = 1200
    max_children: int = 12


class RetrievalModule:
    def __init__(self, index_service: HybridIndexService) -> None:
        self.index_service = index_service

    def manifests(self) -> list[ToolManifest]:
        return [
            ToolManifest(
                tool_name="retrieval.rebuild_local_index",
                module="retrieval",
                description="Build or rebuild the local hybrid index for files and folders in the workspace.",
                side_effect=True,
                idempotent=False,
                produces=[],
                input_schema=RebuildLocalIndexInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"indexed_total": {"type": "integer"}}},
            ),
            ToolManifest(
                tool_name="retrieval.sync_local_index",
                module="retrieval",
                description="Incrementally sync the local hybrid index by updating only added, changed, or deleted files and folders.",
                side_effect=True,
                idempotent=False,
                produces=[],
                input_schema=SyncLocalIndexInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"added": {"type": "integer"}, "updated": {"type": "integer"}, "deleted": {"type": "integer"}}},
            ),
            ToolManifest(
                tool_name="retrieval.search_local_objects",
                module="retrieval",
                description=(
                    "Find likely local files or folders from fuzzy descriptions using hybrid retrieval. "
                    "Use this when the user remembers what an object does more than its exact path or name."
                ),
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.OBJECT_CANDIDATES],
                input_schema=SearchLocalObjectsInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"candidates": {"type": "array"}}},
            ),
            ToolManifest(
                tool_name="retrieval.inspect_local_candidate",
                module="retrieval",
                description=(
                    "Inspect a retrieved local file or folder candidate to confirm whether it matches the user's intent. "
                    "Use this after retrieval.search_local_objects when top candidates need closer verification."
                ),
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=InspectLocalCandidateInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"path": {"type": "string"}, "object_kind": {"type": "string"}}},
            ),
        ]

    def executor_map(self) -> dict[str, Any]:
        return {
            "retrieval.rebuild_local_index": self.rebuild_local_index,
            "retrieval.sync_local_index": self.sync_local_index,
            "retrieval.search_local_objects": self.search_local_objects,
            "retrieval.inspect_local_candidate": self.inspect_local_candidate,
        }

    def rebuild_local_index(self, arguments: dict[str, Any]) -> dict[str, Any]:
        RebuildLocalIndexInput.model_validate(arguments)
        return self.index_service.rebuild_filesystem_index()

    def sync_local_index(self, arguments: dict[str, Any]) -> dict[str, Any]:
        SyncLocalIndexInput.model_validate(arguments)
        return self.index_service.sync_filesystem_index()

    def search_local_objects(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = SearchLocalObjectsInput.model_validate(arguments)
        return self.index_service.search(
            query=payload.query,
            target_kind=payload.target_kind,
            top_k=payload.top_k,
            path_scope=payload.path_scope,
            scope_mode=payload.scope_mode,
            extensions=payload.extensions,
            rebuild_if_missing=payload.rebuild_if_missing,
        )

    def inspect_local_candidate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = InspectLocalCandidateInput.model_validate(arguments)
        return self.index_service.inspect(
            raw_path=payload.path,
            max_chars=payload.max_chars,
            max_children=payload.max_children,
        )
