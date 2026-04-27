from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from local_agent.protocol.models import MemoryRecord


@dataclass(frozen=True)
class StoredMemory:
    memory_id: int
    record: MemoryRecord


class SQLiteMemoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_type TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL,
                    tags TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_type_scope_id ON memories(memory_type, scope, id DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_scope_importance ON memories(scope, importance DESC, id DESC)")
            conn.commit()

    def remember(self, record: MemoryRecord) -> None:
        tags = ",".join(record.tags)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (memory_type, scope, content, importance, tags, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.memory_type,
                    record.scope,
                    record.content,
                    record.importance,
                    tags,
                    record.created_at.isoformat(),
                ),
            )
            conn.commit()

    def recall(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        pattern = f"%{query}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT memory_type, scope, content, importance, tags, created_at
                FROM memories
                WHERE content LIKE ?
                ORDER BY importance DESC, id DESC
                LIMIT ?
                """,
                (pattern, limit),
            ).fetchall()

        results: list[MemoryRecord] = []
        for row in rows:
            results.append(
                MemoryRecord(
                    memory_type=row[0],
                    scope=row[1],
                    content=row[2],
                    importance=row[3],
                    tags=[tag for tag in row[4].split(",") if tag],
                    created_at=row[5],
                )
            )
        return results

    def recall_structured(
        self,
        query: str,
        limit: int = 5,
        *,
        memory_types: Iterable[str] | None = None,
        scopes: Iterable[str] | None = None,
    ) -> list[MemoryRecord]:
        query_terms = _tokenize_memory_text(query)
        rows = self._select_rows(
            query_terms=query_terms,
            memory_types=memory_types,
            scopes=scopes,
            limit=max(limit * 12, 80),
        )

        allowed_types = {item for item in (memory_types or []) if item}
        allowed_scopes = {item for item in (scopes or []) if item}
        scored: list[tuple[float, MemoryRecord]] = []
        for row in rows:
            record = _row_to_record(row)
            if allowed_types and record.memory_type not in allowed_types:
                continue
            if allowed_scopes and record.scope not in allowed_scopes:
                continue

            haystack_terms = _tokenize_memory_text(record.content)
            haystack_terms.update(tag.lower() for tag in record.tags)
            overlap = len(query_terms & haystack_terms) if query_terms else 0
            substring_bonus = 1.0 if query and query.lower() in record.content.lower() else 0.0
            score = float(record.importance) + overlap * 0.25 + substring_bonus * 0.4
            if query and overlap == 0 and substring_bonus == 0 and query.lower() not in record.content.lower():
                continue
            scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def list_records(
        self,
        *,
        memory_types: Iterable[str] | None = None,
        scopes: Iterable[str] | None = None,
        limit: int = 100,
    ) -> list[StoredMemory]:
        where_clauses: list[str] = []
        parameters: list[object] = []
        type_items = [item for item in (memory_types or []) if item]
        scope_items = [item for item in (scopes or []) if item]
        if type_items:
            placeholders = ",".join("?" for _ in type_items)
            where_clauses.append(f"memory_type IN ({placeholders})")
            parameters.extend(type_items)
        if scope_items:
            placeholders = ",".join("?" for _ in scope_items)
            where_clauses.append(f"scope IN ({placeholders})")
            parameters.extend(scope_items)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, memory_type, scope, content, importance, tags, created_at
                FROM memories
                {where_sql}
                ORDER BY id DESC
                LIMIT ?
                """,
                (*parameters, limit),
            ).fetchall()
        return [StoredMemory(memory_id=row[0], record=_row_to_record(row[1:])) for row in rows]

    def count_records(
        self,
        *,
        memory_types: Iterable[str] | None = None,
        scopes: Iterable[str] | None = None,
    ) -> int:
        where_clauses: list[str] = []
        parameters: list[object] = []
        type_items = [item for item in (memory_types or []) if item]
        scope_items = [item for item in (scopes or []) if item]
        if type_items:
            placeholders = ",".join("?" for _ in type_items)
            where_clauses.append(f"memory_type IN ({placeholders})")
            parameters.extend(type_items)
        if scope_items:
            placeholders = ",".join("?" for _ in scope_items)
            where_clauses.append(f"scope IN ({placeholders})")
            parameters.extend(scope_items)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM memories {where_sql}",
                parameters,
            ).fetchone()
        return 0 if row is None else int(row[0])

    def delete_ids(self, ids: Iterable[int]) -> int:
        id_list = [int(item) for item in ids]
        if not id_list:
            return 0
        placeholders = ",".join("?" for _ in id_list)
        with self._connect() as conn:
            cursor = conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", id_list)
            conn.commit()
            return int(cursor.rowcount or 0)

    def _select_rows(
        self,
        *,
        query_terms: set[str],
        memory_types: Iterable[str] | None,
        scopes: Iterable[str] | None,
        limit: int,
    ) -> list[tuple]:
        where_clauses: list[str] = []
        parameters: list[object] = []
        type_items = [item for item in (memory_types or []) if item]
        scope_items = [item for item in (scopes or []) if item]
        if type_items:
            placeholders = ",".join("?" for _ in type_items)
            where_clauses.append(f"memory_type IN ({placeholders})")
            parameters.extend(type_items)
        if scope_items:
            placeholders = ",".join("?" for _ in scope_items)
            where_clauses.append(f"scope IN ({placeholders})")
            parameters.extend(scope_items)
        like_terms = sorted(term for term in query_terms if term and len(term) >= 2)[:6]
        if like_terms:
            like_sql = " OR ".join("LOWER(content) LIKE ?" for _ in like_terms)
            where_clauses.append(f"({like_sql})")
            parameters.extend(f"%{term.lower()}%" for term in like_terms)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT memory_type, scope, content, importance, tags, created_at
                FROM memories
                {where_sql}
                ORDER BY importance DESC, id DESC
                LIMIT ?
                """,
                (*parameters, limit),
            ).fetchall()
        if rows or like_terms:
            return rows
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT memory_type, scope, content, importance, tags, created_at
                FROM memories
                ORDER BY importance DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()


def _tokenize_memory_text(text: str) -> set[str]:
    lowered = text.lower()
    tokens = {token for token in re.findall(r"[a-z0-9_]+", lowered) if token}
    cjk_chunks = re.findall(r"[\u4e00-\u9fff]+", text)
    for chunk in cjk_chunks:
        normalized = chunk.strip()
        if not normalized:
            continue
        tokens.add(normalized)
        if len(normalized) <= 2:
            tokens.add(normalized)
            continue
        for index in range(len(normalized) - 1):
            tokens.add(normalized[index : index + 2])
    return tokens


def _row_to_record(row: tuple) -> MemoryRecord:
    return MemoryRecord(
        memory_type=row[0],
        scope=row[1],
        content=row[2],
        importance=row[3],
        tags=[tag for tag in str(row[4]).split(",") if tag],
        created_at=row[5],
    )
