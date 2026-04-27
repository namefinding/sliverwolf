from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterable, Protocol

import requests

from local_agent.utils.workspace_path import WorkspacePathNormalizer


TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".csv",
    ".html",
    ".htm",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".css",
}

STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "path",
    "file",
    "folder",
    "directory",
    "agent",
    "module",
    "service",
    "tool",
    "local",
    "current",
    "project",
    "workspace",
    "当前",
    "工作区",
    "文件",
    "文件夹",
    "目录",
    "路径",
    "那个",
    "这个",
    "相关",
    "负责",
    "请",
    "帮我",
}

_DB_INIT_LOCKS: dict[str, Lock] = {}
_DB_INIT_LOCKS_GUARD = Lock()

QUERY_REWRITE_MAP = {
    "语音": ["voice", "audio", "tts", "gptsovits", "speech"],
    "播报": ["voice", "tts", "speech"],
    "银狼": ["silverwolf"],
    "规划": ["planner", "planning"],
    "规划器": ["planner"],
    "路由": ["router", "routing"],
    "配置": ["config", "settings"],
    "记忆": ["memory"],
    "搜索": ["search", "retrieval"],
    "检索": ["search", "retrieval"],
    "模型": ["model"],
    "权重": ["weights", "checkpoint"],
    "写入": ["write", "save"],
    "保存": ["write", "save"],
    "文件夹": ["folder", "directory"],
    "目录": ["folder", "directory"],
    "文件": ["file"],
    "会话": ["session"],
    "会话管理": ["session", "store", "state"],
    "常驻": ["resident", "server"],
    "服务": ["server", "service", "http"],
    "网页": ["web", "http"],
}

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{1,8}")


@dataclass
class IndexRecord:
    module: str
    object_kind: str
    path: str
    name: str
    parent_path: str
    ext: str
    summary: str
    keywords: list[str]
    searchable_text: str
    embedding_text: str
    metadata: dict[str, str | int | float | bool | None]
    size_bytes: int
    mtime: float


@dataclass
class SearchQueryPlan:
    raw_query: str
    target_kind: str
    rewritten_terms: list[str]
    name_terms: list[str]
    intent_terms: list[str]
    prefer_project_source: bool
    prefer_folders: bool
    prefer_documents: bool
    prefer_code: bool
    mentions_tests: bool


class EmbeddingProvider(Protocol):
    name: str
    model_name: str | None

    def embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        ...


class RerankerProvider(Protocol):
    name: str

    def rerank(
        self,
        plan: SearchQueryPlan,
        candidates: list[dict[str, object]],
        top_k: int,
    ) -> list[dict[str, object]]:
        ...


class HashEmbeddingProvider:
    name = "hash"
    model_name = None

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return vector
        return [round(value / norm, 6) for value in vector]


class OllamaEmbeddingProvider:
    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_seconds: int = 30,
        batch_size: int = 64,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        if not texts:
            return []

        result: list[list[float] | None] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            result.extend(self._embed_batch(batch))
        return result

    def _embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model_name, "input": texts},
            timeout=self.timeout_seconds,
        )
        if response.ok:
            payload = response.json()
            embeddings = payload.get("embeddings")
            if isinstance(embeddings, list) and embeddings:
                return embeddings

        if len(texts) == 1:
            return [None]

        midpoint = max(1, len(texts) // 2)
        left = self._embed_batch(texts[:midpoint])
        right = self._embed_batch(texts[midpoint:])
        return [*left, *right]


class HeuristicRerankerProvider:
    name = "heuristic"

    def rerank(
        self,
        plan: SearchQueryPlan,
        candidates: list[dict[str, object]],
        top_k: int,
    ) -> list[dict[str, object]]:
        ranked = []
        for candidate in candidates:
            candidate = dict(candidate)
            rerank_boost = self._rerank_boost(plan, candidate)
            candidate["rerank_score"] = round(rerank_boost, 4)
            candidate["score"] = round(float(candidate.get("score", 0.0)) + rerank_boost, 4)
            ranked.append(candidate)

        ranked.sort(
            key=lambda item: (
                float(item["score"]),
                float(item.get("embedding_score", 0.0)),
                float(item.get("intent_score", 0.0)),
                float(item.get("name_score", 0.0)),
            ),
            reverse=True,
        )
        return ranked[:top_k]

    def _rerank_boost(self, plan: SearchQueryPlan, hit: dict[str, object]) -> float:
        path = str(hit["path"]).lower()
        name = str(hit["name"]).lower()
        ext = str(hit.get("ext", "")).lower()
        boost = 0.0

        if plan.prefer_project_source and "\\src\\" in path:
            boost += 0.18
        if plan.prefer_project_source and "\\app\\" in path:
            boost += 0.08
        if plan.prefer_documents and ext in {".md", ".txt", ".doc", ".docx"}:
            boost += 0.08
        if plan.prefer_documents and any(segment in path for segment in ("\\docs\\", "\\文本文件\\", "\\word文档\\")):
            boost += 0.06
        if plan.prefer_code and ext in {".py", ".js", ".ts", ".tsx", ".jsx"}:
            boost += 0.06
        if "\\.venv\\" in path or "site-packages" in path:
            boost -= 0.18
        if any(segment in path for segment in ("\\__pycache__\\", "\\node_modules\\", "\\dist\\", "\\build\\")):
            boost -= 0.18
        if "\\data\\" in path:
            boost -= 0.14
        if "\\tests\\" in path and not plan.mentions_tests:
            boost -= 0.08
        if name.startswith("test_") and not plan.mentions_tests:
            boost -= 0.06
        if ext in {".json", ".jsonl", ".log", ".tmp"}:
            boost -= 0.10
        if any(token in name for token in ("smoke", "debug", "trace", "result", "results", "output", "tmp", "temp")):
            boost -= 0.18
        if name.startswith("_"):
            boost -= 0.03
        if plan.prefer_folders and hit["object_kind"] == "folder":
            boost += 0.12
        if plan.target_kind == "file" and hit["object_kind"] == "file":
            boost += 0.06
        if plan.target_kind == "folder" and hit["object_kind"] == "folder":
            boost += 0.06
        if any(term in name for term in plan.name_terms):
            boost += 0.08
        return round(boost, 4)


class OllamaJudgeRerankerProvider:
    name = "ollama_judge"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_seconds: int = 30,
        rerank_top_n: int = 5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.rerank_top_n = max(1, rerank_top_n)

    def rerank(
        self,
        plan: SearchQueryPlan,
        candidates: list[dict[str, object]],
        top_k: int,
    ) -> list[dict[str, object]]:
        if not candidates:
            return []

        seeded = sorted(
            [dict(candidate) for candidate in candidates],
            key=lambda item: (
                float(item.get("score", 0.0)),
                float(item.get("embedding_score", 0.0)),
                float(item.get("intent_score", 0.0)),
                float(item.get("name_score", 0.0)),
            ),
            reverse=True,
        )
        rerank_count = min(len(seeded), top_k, self.rerank_top_n)
        head = seeded[:rerank_count]
        tail = seeded[rerank_count:]
        try:
            ranking = self._score_candidates(plan, head)
        except Exception:
            ranking = []
        if ranking:
            reranked_head = []
            seen_indexes: set[int] = set()
            for item in ranking:
                index = int(item["index"])
                candidate = dict(head[index - 1])
                candidate["rerank_score"] = round(float(item["score"]), 4)
                candidate["score"] = round(float(candidate.get("score", 0.0)) + float(item["score"]) * 0.35, 4)
                reranked_head.append(candidate)
                seen_indexes.add(index)
            for index, candidate in enumerate(head, start=1):
                if index in seen_indexes:
                    continue
                reranked_head.append(candidate)
            reranked_head.sort(key=lambda item: float(item["score"]), reverse=True)
            return [*reranked_head, *tail][:top_k]

        heuristic = HeuristicRerankerProvider()
        return heuristic.rerank(plan, seeded, top_k)

    def _score_candidates(self, plan: SearchQueryPlan, candidates: list[dict[str, object]]) -> list[dict[str, float | int]]:
        prompt_lines = [
            "You are a retrieval reranker.",
            "Rank the candidates by how well they satisfy the query.",
            "Return JSON only in the form {\"ranking\":[{\"index\":1,\"score\":0.95}, ...]}.",
            "Scores must be between 0 and 1.",
            f"Query: {plan.raw_query}",
            "Candidates:",
        ]
        for idx, candidate in enumerate(candidates, start=1):
            metadata = candidate.get("metadata", {}) or {}
            relative_path = metadata.get("relative_path", "")
            prompt_lines.append(
                f"{idx}. path={candidate.get('path','')} | name={candidate.get('name','')} | "
                f"kind={candidate.get('object_kind','')} | ext={candidate.get('ext','')} | "
                f"summary={str(candidate.get('summary',''))[:240]} | relative_path={relative_path}"
            )

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": "\n".join(prompt_lines),
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        raw_text = str(payload.get("response", "")).strip()
        parsed = self._extract_ranking(raw_text, len(candidates))
        return parsed

    @staticmethod
    def _extract_ranking(raw_text: str, candidate_count: int) -> list[dict[str, float | int]]:
        match = re.search(r"\{[\s\S]*\}", raw_text)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

        ranking = payload.get("ranking", [])
        if not isinstance(ranking, list):
            return []

        cleaned: list[dict[str, float | int]] = []
        seen: set[int] = set()
        for item in ranking:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            score = item.get("score")
            if not isinstance(index, int) or not (1 <= index <= candidate_count) or index in seen:
                continue
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                continue
            seen.add(index)
            cleaned.append({"index": index, "score": max(0.0, min(1.0, numeric_score))})

        if not cleaned:
            return []
        return cleaned


class FileSystemIndexProvider:
    def __init__(self, workspace_root: str, ignored_paths: set[Path] | None = None) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.ignored_paths = {path.resolve() for path in (ignored_paths or set())}

    def iter_records(self) -> Iterable[IndexRecord]:
        for item in self.workspace_root.rglob("*"):
            if self._should_skip(item):
                continue
            if item.is_dir():
                yield self._build_directory_record(item)
            elif item.is_file():
                yield self._build_file_record(item)

    def _should_skip(self, item: Path) -> bool:
        parts = set(item.parts)
        resolved = item.resolve()
        if ".git" in parts or "__pycache__" in parts:
            return True
        if resolved in self.ignored_paths:
            return True
        resolved_text = str(resolved)
        return any(resolved_text in {f"{ignored}-wal", f"{ignored}-shm", f"{ignored}-journal"} for ignored in self.ignored_paths)

    def _build_directory_record(self, directory: Path) -> IndexRecord:
        child_names = []
        try:
            child_names = sorted(child.name for child in directory.iterdir())[:12]
        except OSError:
            child_names = []
        relative_path = str(directory.relative_to(self.workspace_root))
        keywords = _dedupe_preserve_order(_tokenize(directory.name) + _tokenize(relative_path))
        summary = f"Folder under workspace. Contains: {', '.join(child_names[:8])}" if child_names else "Folder in workspace."
        searchable_text = " ".join(
            part for part in [directory.name, relative_path, summary, " ".join(child_names)] if part
        )
        stat = directory.stat()
        return IndexRecord(
            module="filesystem",
            object_kind="folder",
            path=str(directory),
            name=directory.name,
            parent_path=str(directory.parent),
            ext="",
            summary=summary,
            keywords=keywords,
            searchable_text=searchable_text,
            embedding_text=" ".join(
                part for part in [directory.name, relative_path, summary, " ".join(child_names[:8])] if part
            ),
            metadata={"child_sample": ", ".join(child_names[:8]), "relative_path": relative_path},
            size_bytes=0,
            mtime=stat.st_mtime,
        )

    def _build_file_record(self, file_path: Path) -> IndexRecord:
        stat = file_path.stat()
        ext = file_path.suffix.lower()
        relative_path = str(file_path.relative_to(self.workspace_root))
        summary = f"{ext or 'file'} in workspace."
        snippet = ""
        if ext in TEXT_EXTENSIONS and stat.st_size <= 512_000:
            try:
                snippet = file_path.read_text(encoding="utf-8", errors="ignore")[:2000]
                summary = _make_file_summary(relative_path, snippet)
            except OSError:
                snippet = ""
        keywords = _dedupe_preserve_order(
            _tokenize(file_path.stem)
            + _tokenize(relative_path)
            + _tokenize(summary)
        )
        searchable_text = " ".join(
            part for part in [file_path.name, relative_path, summary, snippet] if part
        )
        return IndexRecord(
            module="filesystem",
            object_kind="file",
            path=str(file_path),
            name=file_path.name,
            parent_path=str(file_path.parent),
            ext=ext,
            summary=summary,
            keywords=keywords,
            searchable_text=searchable_text,
            embedding_text=" ".join(
                part for part in [file_path.name, relative_path, summary, snippet[:1200]] if part
            ),
            metadata={"relative_path": relative_path},
            size_bytes=stat.st_size,
            mtime=stat.st_mtime,
        )


class HybridIndexService:
    def __init__(
        self,
        db_path: str,
        workspace_root: str,
        embedding_provider: EmbeddingProvider | None = None,
        reranker_provider: RerankerProvider | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.workspace_root = Path(workspace_root).resolve()
        self.path_normalizer = WorkspacePathNormalizer(str(self.workspace_root))
        self.embedding_provider = embedding_provider or HashEmbeddingProvider()
        self.reranker_provider = reranker_provider or HeuristicRerankerProvider()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def rebuild_filesystem_index(self) -> dict[str, int | str]:
        provider = FileSystemIndexProvider(str(self.workspace_root), ignored_paths={self.db_path})
        records = list(provider.iter_records())
        with self._connect() as conn:
            conn.execute("DELETE FROM retrieval_objects")
            conn.execute("DELETE FROM retrieval_objects_fts")
            conn.execute("DELETE FROM retrieval_object_embeddings")

            row_mappings: list[tuple[int, str]] = []
            for record in records:
                cursor = conn.execute(
                    """
                    INSERT INTO retrieval_objects (
                        module, object_kind, path, name, parent_path, ext, summary,
                        keywords_json, searchable_text, metadata_json, size_bytes, mtime
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.module,
                        record.object_kind,
                        record.path,
                        record.name,
                        record.parent_path,
                        record.ext,
                        record.summary,
                        json.dumps(record.keywords, ensure_ascii=False),
                        record.searchable_text,
                        json.dumps(record.metadata, ensure_ascii=False),
                        record.size_bytes,
                        record.mtime,
                    ),
                )
                row_id = cursor.lastrowid
                row_mappings.append((row_id, record.embedding_text))
                conn.execute(
                    "INSERT INTO retrieval_objects_fts(rowid, searchable_text) VALUES (?, ?)",
                    (row_id, record.searchable_text),
                )

            embedding_status = "disabled"
            if self.embedding_provider is not None and row_mappings:
                try:
                    embeddings = self.embedding_provider.embed_texts([text for _, text in row_mappings])
                    inserted = 0
                    skipped = 0
                    for (row_id, _), embedding in zip(row_mappings, embeddings, strict=False):
                        if embedding is None:
                            skipped += 1
                            continue
                        conn.execute(
                            """
                            INSERT INTO retrieval_object_embeddings (
                                object_id, provider_name, model_name, embedding_json
                            ) VALUES (?, ?, ?, ?)
                            """,
                            (
                                row_id,
                                self.embedding_provider.name,
                                getattr(self.embedding_provider, "model_name", None),
                                json.dumps(embedding),
                            ),
                        )
                        inserted += 1
                    if inserted and skipped:
                        embedding_status = "partial"
                    elif inserted:
                        embedding_status = "ready"
                    else:
                        embedding_status = "failed"
                except Exception:
                    embedding_status = "failed"

            conn.commit()
        file_count = sum(1 for record in records if record.object_kind == "file")
        folder_count = sum(1 for record in records if record.object_kind == "folder")
        return {
            "indexed_total": len(records),
            "indexed_files": file_count,
            "indexed_folders": folder_count,
            "embedding_provider": self.embedding_provider.name if self.embedding_provider is not None else "none",
            "embedding_status": embedding_status,
            "reranker_provider": self.reranker_provider.name,
        }

    def sync_filesystem_index(self) -> dict[str, int | str]:
        provider = FileSystemIndexProvider(str(self.workspace_root), ignored_paths={self.db_path})
        current_records = {record.path: record for record in provider.iter_records()}

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            existing_rows = conn.execute(
                """
                SELECT id, path, object_kind, name, parent_path, ext, size_bytes, mtime
                FROM retrieval_objects
                """
            ).fetchall()
            existing_by_path = {str(row["path"]): row for row in existing_rows}

            deleted_paths = [path for path in existing_by_path if path not in current_records]
            added_records: list[IndexRecord] = []
            updated_records: list[IndexRecord] = []
            unchanged = 0

            for path, record in current_records.items():
                existing = existing_by_path.get(path)
                if existing is None:
                    added_records.append(record)
                    continue
                if self._record_needs_update(record, existing):
                    updated_records.append(record)
                else:
                    unchanged += 1

            for path in deleted_paths:
                row_id = int(existing_by_path[path]["id"])
                conn.execute("DELETE FROM retrieval_object_embeddings WHERE object_id = ?", (row_id,))
                conn.execute("DELETE FROM retrieval_objects_fts WHERE rowid = ?", (row_id,))
                conn.execute("DELETE FROM retrieval_objects WHERE id = ?", (row_id,))

            for record in updated_records:
                existing = existing_by_path[record.path]
                row_id = int(existing["id"])
                conn.execute("DELETE FROM retrieval_object_embeddings WHERE object_id = ?", (row_id,))
                conn.execute("DELETE FROM retrieval_objects_fts WHERE rowid = ?", (row_id,))
                conn.execute("DELETE FROM retrieval_objects WHERE id = ?", (row_id,))

            inserted = self._insert_records(conn, [*added_records, *updated_records])
            conn.commit()

        return {
            "added": len(added_records),
            "updated": len(updated_records),
            "deleted": len(deleted_paths),
            "unchanged": unchanged,
            "embedding_provider": self.embedding_provider.name if self.embedding_provider is not None else "none",
            "embedding_status": inserted["embedding_status"],
            "indexed_total": len(current_records),
        }

    def search(
        self,
        query: str,
        target_kind: str = "any",
        top_k: int = 8,
        path_scope: str = ".",
        scope_mode: str = "subtree",
        extensions: list[str] | None = None,
        rebuild_if_missing: bool = True,
    ) -> dict[str, object]:
        if rebuild_if_missing and not self.has_rows():
            self.rebuild_filesystem_index()

        normalized_scope = self._normalize_scope(path_scope)
        ext_filter = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in (extensions or [])]
        plan = analyze_query(query, target_kind)
        rows = self._fetch_candidate_rows(plan.target_kind, normalized_scope, ext_filter)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                "fts": executor.submit(self._fts_search, plan, normalized_scope, ext_filter, top_k * 4),
                "name": executor.submit(self._name_path_search, plan, rows, top_k * 4),
                "intent": executor.submit(self._intent_search, plan, rows, top_k * 4),
                "folder": executor.submit(self._folder_search, plan, rows, top_k * 4),
                "embedding": executor.submit(self._embedding_search, plan, rows, top_k * 4),
            }
            hit_sets = {name: future.result() for name, future in futures.items()}

        merged = self._merge_hit_sets(hit_sets)
        candidates = self._score_candidates(merged)
        reranked = self.reranker_provider.rerank(plan, candidates, top_k)
        reranked = self._apply_scope_mode(
            reranked,
            plan=plan,
            scope_root=Path(normalized_scope),
            scope_mode=scope_mode,
            top_k=top_k,
        )
        return {
            "query": query,
            "rewritten_terms": plan.rewritten_terms,
            "candidates": reranked,
            "debug": {
                "embedding_provider": self.embedding_provider.name if self.embedding_provider is not None else "none",
                "reranker_provider": self.reranker_provider.name,
                "scope_mode": scope_mode,
                "recall_counts": {name: len(hits) for name, hits in hit_sets.items()},
            },
        }

    def inspect(self, raw_path: str, max_chars: int = 1200, max_children: int = 12) -> dict[str, object]:
        target = self._resolve_workspace_path(raw_path)
        if target.is_dir():
            children = []
            try:
                for child in sorted(target.iterdir(), key=lambda item: item.name.lower())[:max_children]:
                    children.append({"name": child.name, "path": str(child), "is_dir": child.is_dir()})
            except OSError:
                children = []
            return {
                "path": str(target),
                "object_kind": "folder",
                "name": target.name,
                "summary": f"Folder under workspace with {len(children)} sampled children.",
                "children": children,
            }

        ext = target.suffix.lower()
        snippet = ""
        if ext in TEXT_EXTENSIONS:
            try:
                snippet = target.read_text(encoding="utf-8", errors="ignore")[:max_chars]
            except OSError:
                snippet = ""
        return {
            "path": str(target),
            "object_kind": "file",
            "name": target.name,
            "ext": ext,
            "size": target.stat().st_size,
            "summary": _make_file_summary(str(target.relative_to(self.workspace_root)), snippet),
            "snippet": snippet,
        }

    def has_rows(self) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM retrieval_objects").fetchone()
        return bool(row and row[0] > 0)

    def has_embedding_rows(self) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM retrieval_object_embeddings").fetchone()
        return bool(row and row[0] > 0)

    def _record_needs_update(self, record: IndexRecord, existing: sqlite3.Row) -> bool:
        return (
            record.object_kind != existing["object_kind"]
            or record.name != existing["name"]
            or record.parent_path != existing["parent_path"]
            or record.ext != existing["ext"]
            or record.size_bytes != existing["size_bytes"]
            or abs(float(record.mtime) - float(existing["mtime"])) > 1e-6
        )

    def _insert_records(self, conn: sqlite3.Connection, records: list[IndexRecord]) -> dict[str, str | int]:
        row_mappings: list[tuple[int, str]] = []
        for record in records:
            cursor = conn.execute(
                """
                INSERT INTO retrieval_objects (
                    module, object_kind, path, name, parent_path, ext, summary,
                    keywords_json, searchable_text, metadata_json, size_bytes, mtime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.module,
                    record.object_kind,
                    record.path,
                    record.name,
                    record.parent_path,
                    record.ext,
                    record.summary,
                    json.dumps(record.keywords, ensure_ascii=False),
                    record.searchable_text,
                    json.dumps(record.metadata, ensure_ascii=False),
                    record.size_bytes,
                    record.mtime,
                ),
            )
            row_id = cursor.lastrowid
            row_mappings.append((row_id, record.embedding_text))
            conn.execute(
                "INSERT INTO retrieval_objects_fts(rowid, searchable_text) VALUES (?, ?)",
                (row_id, record.searchable_text),
            )

        embedding_status = "disabled"
        if self.embedding_provider is not None and row_mappings:
            try:
                embeddings = self.embedding_provider.embed_texts([text for _, text in row_mappings])
                inserted = 0
                skipped = 0
                for (row_id, _), embedding in zip(row_mappings, embeddings, strict=False):
                    if embedding is None:
                        skipped += 1
                        continue
                    conn.execute(
                        """
                        INSERT INTO retrieval_object_embeddings (
                            object_id, provider_name, model_name, embedding_json
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            row_id,
                            self.embedding_provider.name,
                            getattr(self.embedding_provider, "model_name", None),
                            json.dumps(embedding),
                        ),
                    )
                    inserted += 1
                if inserted and skipped:
                    embedding_status = "partial"
                elif inserted:
                    embedding_status = "ready"
                else:
                    embedding_status = "failed"
            except Exception:
                embedding_status = "failed"
        return {"inserted": len(row_mappings), "embedding_status": embedding_status}

    def _fetch_candidate_rows(self, target_kind: str, path_scope: str, extensions: list[str]) -> list[sqlite3.Row]:
        where_clauses = ["path LIKE ?"]
        params: list[object] = [f"{path_scope}%"]
        if target_kind in {"file", "folder"}:
            where_clauses.append("object_kind = ?")
            params.append(target_kind)
        if extensions:
            placeholders = ",".join("?" for _ in extensions)
            where_clauses.append(f"ext IN ({placeholders})")
            params.extend(extensions)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT id, path, name, parent_path, object_kind, summary, ext, module, searchable_text, keywords_json, metadata_json
                FROM retrieval_objects
                WHERE {' AND '.join(where_clauses)}
                """,
                params,
            ).fetchall()
        return rows

    def _fts_search(
        self,
        plan: SearchQueryPlan,
        path_scope: str,
        extensions: list[str],
        limit: int,
    ) -> list[dict[str, object]]:
        tokens = [term for term in plan.rewritten_terms if term]
        if not tokens:
            return []

        where_clauses = ["o.path LIKE ?"]
        params: list[object] = [f"{path_scope}%"]
        if plan.target_kind in {"file", "folder"}:
            where_clauses.append("o.object_kind = ?")
            params.append(plan.target_kind)
        if extensions:
            placeholders = ",".join("?" for _ in extensions)
            where_clauses.append(f"o.ext IN ({placeholders})")
            params.extend(extensions)

        match_expr = " OR ".join(f'"{token}"' for token in tokens)
        query = f"""
            SELECT
                o.id,
                o.path,
                o.name,
                o.parent_path,
                o.object_kind,
                o.summary,
                o.ext,
                o.module,
                o.metadata_json,
                bm25(retrieval_objects_fts) AS score
            FROM retrieval_objects_fts
            JOIN retrieval_objects o ON o.id = retrieval_objects_fts.rowid
            WHERE retrieval_objects_fts MATCH ?
              AND {' AND '.join(where_clauses)}
            ORDER BY score
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(query, [match_expr, *params, limit]).fetchall()

        hits = []
        for row in rows:
            hits.append(
                self._make_hit(
                    row,
                    fts_score=1 / (1 + abs(float(row[9]))),
                )
            )
        return hits

    def _name_path_search(self, plan: SearchQueryPlan, rows: list[sqlite3.Row], limit: int) -> list[dict[str, object]]:
        hits = []
        term_set = set(plan.name_terms)
        for row in rows:
            if not term_set:
                continue
            name_tokens = set(_tokenize(row["name"]))
            path_tokens = set(_tokenize(row["path"]))
            parent_tokens = set(_tokenize(row["parent_path"]))
            name_score = len(term_set & name_tokens) / max(len(term_set), 1)
            path_score = len(term_set & path_tokens) / max(len(term_set), 1)
            parent_score = len(term_set & parent_tokens) / max(len(term_set), 1)
            score = min(1.0, name_score * 0.55 + path_score * 0.3 + parent_score * 0.15)
            if score <= 0:
                continue
            hits.append(self._make_hit(row, name_score=round(score, 4)))
        hits.sort(key=lambda item: item["name_score"], reverse=True)
        return hits[:limit]

    def _intent_search(self, plan: SearchQueryPlan, rows: list[sqlite3.Row], limit: int) -> list[dict[str, object]]:
        hits = []
        term_set = set(plan.intent_terms)
        for row in rows:
            summary_tokens = set(_tokenize(row["summary"] or ""))
            searchable_tokens = set(_tokenize(row["searchable_text"] or ""))
            keyword_tokens = set(json.loads(row["keywords_json"] or "[]"))
            summary_score = len(term_set & summary_tokens) / max(len(term_set), 1)
            body_score = len(term_set & searchable_tokens) / max(len(term_set), 1)
            keyword_score = len(term_set & keyword_tokens) / max(len(term_set), 1)
            score = min(1.0, summary_score * 0.45 + body_score * 0.35 + keyword_score * 0.2)
            if score <= 0:
                continue
            hits.append(self._make_hit(row, intent_score=round(score, 4)))
        hits.sort(key=lambda item: item["intent_score"], reverse=True)
        return hits[:limit]

    def _folder_search(self, plan: SearchQueryPlan, rows: list[sqlite3.Row], limit: int) -> list[dict[str, object]]:
        if plan.target_kind == "file":
            return []

        hits = []
        term_set = set(plan.intent_terms)
        for row in rows:
            if row["object_kind"] != "folder":
                continue
            metadata = json.loads(row["metadata_json"] or "{}")
            child_sample = metadata.get("child_sample", "")
            child_tokens = set(_tokenize(child_sample))
            name_tokens = set(_tokenize(row["name"]))
            summary_tokens = set(_tokenize(row["summary"] or ""))
            score = min(
                1.0,
                (len(term_set & name_tokens) / max(len(term_set), 1)) * 0.35
                + (len(term_set & summary_tokens) / max(len(term_set), 1)) * 0.25
                + (len(term_set & child_tokens) / max(len(term_set), 1)) * 0.4,
            )
            if score <= 0:
                continue
            hits.append(self._make_hit(row, folder_score=round(score, 4)))
        hits.sort(key=lambda item: item["folder_score"], reverse=True)
        return hits[:limit]

    def _embedding_search(self, plan: SearchQueryPlan, rows: list[sqlite3.Row], limit: int) -> list[dict[str, object]]:
        if self.embedding_provider is None or not rows or not self.has_embedding_rows():
            return []

        try:
            query_vector = self.embedding_provider.embed_texts(
                [" ".join([plan.raw_query, *plan.rewritten_terms])]
            )[0]
        except Exception:
            return []

        row_ids = [int(row["id"]) for row in rows]
        placeholders = ",".join("?" for _ in row_ids)
        with self._connect() as conn:
            embedding_rows = conn.execute(
                f"""
                SELECT object_id, embedding_json
                FROM retrieval_object_embeddings
                WHERE object_id IN ({placeholders})
                """,
                row_ids,
            ).fetchall()

        embedding_map = {int(object_id): json.loads(embedding_json) for object_id, embedding_json in embedding_rows}
        hits = []
        for row in rows:
            vector = embedding_map.get(int(row["id"]))
            if not vector:
                continue
            score = _cosine_similarity(query_vector, vector)
            if score <= 0:
                continue
            hits.append(self._make_hit(row, embedding_score=round(score, 4)))
        hits.sort(key=lambda item: item["embedding_score"], reverse=True)
        return hits[:limit]

    def _merge_hit_sets(self, hit_sets: dict[str, list[dict[str, object]]]) -> dict[int, dict[str, object]]:
        merged: dict[int, dict[str, object]] = {}
        for hits in hit_sets.values():
            for hit in hits:
                existing = merged.get(hit["id"])
                if existing is None:
                    merged[hit["id"]] = dict(hit)
                    continue
                for key in ("fts_score", "name_score", "intent_score", "folder_score", "embedding_score"):
                    existing[key] = max(float(existing.get(key, 0.0)), float(hit.get(key, 0.0)))
        return merged

    def _score_candidates(self, merged: dict[int, dict[str, object]]) -> list[dict[str, object]]:
        candidates = []
        for hit in merged.values():
            hit = dict(hit)
            score_sources = {
                "fts": round(float(hit.get("fts_score", 0.0)), 4),
                "name": round(float(hit.get("name_score", 0.0)), 4),
                "intent": round(float(hit.get("intent_score", 0.0)), 4),
                "folder": round(float(hit.get("folder_score", 0.0)), 4),
                "embedding": round(float(hit.get("embedding_score", 0.0)), 4),
            }
            base_score = (
                score_sources["fts"] * 0.22
                + score_sources["name"] * 0.22
                + score_sources["intent"] * 0.22
                + score_sources["folder"] * 0.14
                + score_sources["embedding"] * 0.20
            )
            hit["score_sources"] = score_sources
            hit["score"] = round(base_score, 4)
            candidates.append(hit)
        return candidates

    def _make_hit(
        self,
        row: sqlite3.Row | tuple,
        *,
        fts_score: float = 0.0,
        name_score: float = 0.0,
        intent_score: float = 0.0,
        folder_score: float = 0.0,
        embedding_score: float = 0.0,
    ) -> dict[str, object]:
        if isinstance(row, sqlite3.Row):
            metadata = json.loads(row["metadata_json"] or "{}")
            return {
                "id": row["id"],
                "path": row["path"],
                "name": row["name"],
                "object_kind": row["object_kind"],
                "summary": row["summary"],
                "ext": row["ext"],
                "module": row["module"],
                "metadata": metadata,
                "fts_score": fts_score,
                "name_score": name_score,
                "intent_score": intent_score,
                "folder_score": folder_score,
                "embedding_score": embedding_score,
            }
        metadata = json.loads(row[8] or "{}")
        return {
            "id": row[0],
            "path": row[1],
            "name": row[2],
            "object_kind": row[4],
            "summary": row[5],
            "ext": row[6],
            "module": row[7],
            "metadata": metadata,
            "fts_score": fts_score,
            "name_score": name_score,
            "intent_score": intent_score,
            "folder_score": folder_score,
            "embedding_score": embedding_score,
        }

    def _normalize_scope(self, raw_scope: str) -> str:
        target = self._resolve_workspace_path(raw_scope)
        return str(target)

    @staticmethod
    def _apply_scope_mode(
        candidates: list[dict[str, object]],
        *,
        plan: SearchQueryPlan,
        scope_root: Path,
        scope_mode: str,
        top_k: int,
    ) -> list[dict[str, object]]:
        normalized_mode = str(scope_mode or "subtree").strip().lower()
        adjusted: list[dict[str, object]] = []
        for item in candidates:
            candidate = dict(item)
            raw_path = str(candidate.get("path", "") or "")
            try:
                relative = Path(raw_path).resolve().relative_to(scope_root.resolve())
                depth = max(len(relative.parts) - 1, 0)
            except Exception:
                depth = 99
            scope_bonus = max(0.0, 0.24 - min(depth, 6) * 0.05) if normalized_mode == "shallow_first" else 0.0
            title_match = _title_match_score(plan, candidate)
            title_bonus = _title_match_bonus(title_match)
            candidate["scope_depth"] = depth
            candidate["scope_bonus"] = round(scope_bonus, 4)
            candidate["title_match_score"] = round(title_match, 4)
            ranking_signal = (
                float(candidate.get("rerank_score", candidate.get("score", 0.0)) or 0.0)
                + scope_bonus
                + title_bonus
            )
            candidate["_scope_rank"] = ranking_signal
            adjusted.append(candidate)

        adjusted.sort(
            key=lambda item: (
                float(item.get("_scope_rank", 0.0)),
                float(item.get("title_match_score", 0.0)),
                -float(item.get("scope_depth", 99)),
                float(item.get("rerank_score", item.get("score", 0.0)) or 0.0),
            ),
            reverse=True,
        )
        for item in adjusted:
            item.pop("_scope_rank", None)
        return adjusted[:top_k]

    def _resolve_workspace_path(self, raw_path: str) -> Path:
        target = self.path_normalizer.resolve(raw_path)
        try:
            target.relative_to(self.workspace_root)
        except ValueError as exc:
            raise PermissionError(f"Path is outside workspace: {target}") from exc
        return target

    def _init_db(self) -> None:
        db_key = str(self.db_path.resolve())
        with _DB_INIT_LOCKS_GUARD:
            init_lock = _DB_INIT_LOCKS.setdefault(db_key, Lock())

        with init_lock:
            last_error: sqlite3.OperationalError | None = None
            for _ in range(6):
                try:
                    with sqlite3.connect(self.db_path, timeout=30) as conn:
                        conn.execute("PRAGMA journal_mode=WAL")
                        conn.execute("PRAGMA busy_timeout = 30000")
                        conn.execute(
                            """
                            CREATE TABLE IF NOT EXISTS retrieval_objects (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                module TEXT NOT NULL,
                                object_kind TEXT NOT NULL,
                                path TEXT NOT NULL UNIQUE,
                                name TEXT NOT NULL,
                                parent_path TEXT NOT NULL,
                                ext TEXT NOT NULL,
                                summary TEXT NOT NULL,
                                keywords_json TEXT NOT NULL,
                                searchable_text TEXT NOT NULL,
                                metadata_json TEXT NOT NULL,
                                size_bytes INTEGER NOT NULL,
                                mtime REAL NOT NULL
                            )
                            """
                        )
                        conn.execute(
                            """
                            CREATE TABLE IF NOT EXISTS retrieval_object_embeddings (
                                object_id INTEGER PRIMARY KEY,
                                provider_name TEXT NOT NULL,
                                model_name TEXT,
                                embedding_json TEXT NOT NULL,
                                FOREIGN KEY(object_id) REFERENCES retrieval_objects(id) ON DELETE CASCADE
                            )
                            """
                        )
                        conn.execute(
                            """
                            CREATE VIRTUAL TABLE IF NOT EXISTS retrieval_objects_fts
                            USING fts5(searchable_text)
                            """
                        )
                        conn.commit()
                    return
                except sqlite3.OperationalError as exc:
                    last_error = exc
                    if "locked" not in str(exc).lower():
                        raise
                    time.sleep(0.4)
            if last_error is not None:
                raise last_error


def analyze_query(query: str, target_kind: str) -> SearchQueryPlan:
    lowered_query = query.lower()
    rewritten_terms = rewrite_query(query)
    name_terms = [term for term in rewritten_terms if len(term) <= 24]
    intent_terms = _dedupe_preserve_order(rewritten_terms + _tokenize(lowered_query))
    prefer_project_source = any(token in lowered_query for token in ("项目", "project", "当前项目", "源码", "代码"))
    prefer_folders = target_kind == "folder" or any(
        token in lowered_query for token in ("文件夹", "目录", "folder", "directory")
    )
    prefer_documents = any(
        token in lowered_query
        for token in ("markdown", "md", "doc", "docx", "document", "readme", "鏂囨。", "鏋舵瀯", "缁撴瀯", "璇存槑", "鎵嬪唽")
    )
    prefer_code = any(
        token in lowered_query
        for token in ("python", ".py", "code", "planner", "router", "session", "鍚姩", "瑙勫垝", "璺敱", "浠ｇ爜")
    )
    mentions_tests = any(token in lowered_query for token in ("test", "tests", "娴嬭瘯"))
    return SearchQueryPlan(
        raw_query=query,
        target_kind=target_kind,
        rewritten_terms=rewritten_terms,
        name_terms=name_terms,
        intent_terms=intent_terms,
        prefer_project_source=prefer_project_source,
        prefer_folders=prefer_folders,
        prefer_documents=prefer_documents,
        prefer_code=prefer_code,
        mentions_tests=mentions_tests,
    )


def rewrite_query(query: str) -> list[str]:
    lowered_query = query.lower()
    tokens = _tokenize(lowered_query)
    expanded = list(tokens)
    for phrase, aliases in QUERY_REWRITE_MAP.items():
        if phrase in query or phrase in lowered_query:
            expanded.append(phrase)
            expanded.extend(aliases)
    for token in tokens:
        expanded.extend(QUERY_REWRITE_MAP.get(token, []))
    return _dedupe_preserve_order([token for token in expanded if token not in STOP_WORDS])


def _make_file_summary(relative_hint: str, snippet: str) -> str:
    lines = [line.strip() for line in snippet.splitlines() if line.strip()]
    head = " ".join(lines[:4])[:240]
    if head:
        return f"Source file under {relative_hint}. Leading content: {head}"
    return f"File under {relative_hint}."


def _tokenize(text: str) -> list[str]:
    tokens = []
    for match in TOKEN_PATTERN.findall(text.lower()):
        raw_token = match.strip().lower()
        if not raw_token:
            continue
        pieces = re.split(r"[_\-/\\\.]+", raw_token)
        for token in [raw_token, *pieces]:
            token = token.strip().lower()
            if not token or token in STOP_WORDS:
                continue
            tokens.append(token)
    return tokens


def _compact_title(text: str) -> str:
    return "".join(re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower()))


def _title_match_score(plan: SearchQueryPlan, candidate: dict[str, object]) -> float:
    raw_name = str(candidate.get("name", "") or "")
    name_stem = Path(raw_name).stem if raw_name else Path(str(candidate.get("path", "") or "")).stem
    query_compact = _compact_title(plan.raw_query)
    stem_compact = _compact_title(name_stem)
    if not query_compact or not stem_compact:
        return 0.0
    if stem_compact == query_compact:
        return 1.0
    if query_compact in stem_compact:
        return 0.92
    if stem_compact in query_compact and len(stem_compact) >= 4:
        return 0.82

    query_terms = [term for term in plan.name_terms if term and term not in STOP_WORDS]
    if not query_terms:
        query_terms = _tokenize(plan.raw_query)
    name_tokens = set(_tokenize(name_stem))
    if not query_terms or not name_tokens:
        return 0.0
    matched = sum(1 for term in query_terms if term in name_tokens or _compact_title(term) in stem_compact)
    if matched <= 0:
        return 0.0
    coverage = matched / max(len(query_terms), 1)
    if coverage >= 1.0:
        return 0.78
    return coverage * 0.62


def _title_match_bonus(title_match: float) -> float:
    if title_match >= 1.0:
        return 0.75
    if title_match >= 0.9:
        return 0.62
    if title_match >= 0.78:
        return 0.46
    return title_match * 0.28


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(l * r for l, r in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return round(dot / (left_norm * right_norm), 6)
