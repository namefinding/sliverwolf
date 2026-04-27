from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class QQHistoryStore:
    def __init__(
        self,
        db_path: str,
        *,
        max_messages: int = 20_000,
        max_attachments: int = 20_000,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_messages = max(500, int(max_messages))
        self.max_attachments = max(500, int(max_attachments))
        self._writes_since_prune = 0
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS qq_history_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    contact_id TEXT,
                    contact_name TEXT,
                    sender_id TEXT,
                    direction TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    text TEXT NOT NULL DEFAULT '',
                    attachment_count INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS qq_history_attachments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    contact_id TEXT,
                    contact_name TEXT,
                    direction TEXT NOT NULL,
                    attachment_kind TEXT NOT NULL,
                    local_path TEXT,
                    remote_url TEXT,
                    file_name TEXT,
                    mime_type TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(message_id) REFERENCES qq_history_messages(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_qq_history_messages_session_created ON qq_history_messages(session_id, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_qq_history_messages_contact_created ON qq_history_messages(contact_id, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_qq_history_attachments_session_created ON qq_history_attachments(session_id, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_qq_history_attachments_contact_created ON qq_history_attachments(contact_id, created_at DESC)"
            )
            conn.commit()

    def record_message(
        self,
        *,
        session_id: str,
        direction: str,
        message_type: str,
        text: str = "",
        sender_id: str | None = None,
        contact_id: str | None = None,
        contact_name: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: str,
    ) -> int:
        normalized_session_id = str(session_id or "").strip()
        if not normalized_session_id:
            raise ValueError("session_id is required")

        normalized_direction = str(direction or "").strip().lower() or "inbound"
        normalized_message_type = str(message_type or "").strip().lower() or "text"
        normalized_text = str(text or "").strip()
        normalized_contact_id = self._normalize_optional(contact_id)
        normalized_contact_name = self._normalize_optional(contact_name)
        normalized_sender_id = self._normalize_optional(sender_id)
        normalized_attachments = [
            self._normalize_attachment(item)
            for item in (attachments or [])
            if isinstance(item, dict)
        ]
        normalized_metadata = metadata if isinstance(metadata, dict) else {}

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO qq_history_messages (
                    session_id,
                    contact_id,
                    contact_name,
                    sender_id,
                    direction,
                    message_type,
                    text,
                    attachment_count,
                    metadata_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_session_id,
                    normalized_contact_id,
                    normalized_contact_name,
                    normalized_sender_id,
                    normalized_direction,
                    normalized_message_type,
                    normalized_text,
                    len(normalized_attachments),
                    json.dumps(normalized_metadata, ensure_ascii=False),
                    created_at,
                ),
            )
            message_id = int(cursor.lastrowid)
            for attachment in normalized_attachments:
                conn.execute(
                    """
                    INSERT INTO qq_history_attachments (
                        message_id,
                        session_id,
                        contact_id,
                        contact_name,
                        direction,
                        attachment_kind,
                        local_path,
                        remote_url,
                        file_name,
                        mime_type,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        normalized_session_id,
                        normalized_contact_id,
                        normalized_contact_name,
                        normalized_direction,
                        attachment["attachment_kind"],
                        attachment.get("local_path"),
                        attachment.get("remote_url"),
                        attachment.get("file_name"),
                        attachment.get("mime_type"),
                        created_at,
                    ),
                )
            self._writes_since_prune += 1
            if self._writes_since_prune >= 25:
                self._prune_locked(conn)
                self._writes_since_prune = 0
            conn.commit()
        return message_id

    def get_last_reply(
        self,
        *,
        session_id: str | None = None,
        contact_id: str | None = None,
        contact_query: str | None = None,
    ) -> dict[str, Any] | None:
        where, params = self._build_message_filters(
            session_id=session_id,
            contact_id=contact_id,
            contact_query=contact_query,
            direction="inbound",
        )
        query = """
            SELECT *
            FROM qq_history_messages
            {where_clause}
            ORDER BY created_at DESC, id DESC
            LIMIT 1
        """.format(where_clause=f"WHERE {' AND '.join(where)}" if where else "")
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
            if row is None:
                return None
            attachments = conn.execute(
                """
                SELECT attachment_kind, local_path, remote_url, file_name, mime_type, created_at
                FROM qq_history_attachments
                WHERE message_id = ?
                ORDER BY id ASC
                """,
                (int(row["id"]),),
            ).fetchall()
        return self._serialize_message_row(row, attachments)

    def search_messages(
        self,
        *,
        session_id: str | None = None,
        contact_id: str | None = None,
        contact_query: str | None = None,
        query: str | None = None,
        limit: int = 5,
        reply_after_last_outbound: bool = False,
    ) -> list[dict[str, Any]]:
        normalized_limit = max(1, int(limit or 1))
        where, params = self._build_message_filters(
            session_id=session_id,
            contact_id=contact_id,
            contact_query=contact_query,
        )
        if reply_after_last_outbound:
            cutoff = self._find_latest_outbound_timestamp(
                session_id=session_id,
                contact_id=contact_id,
                contact_query=contact_query,
            )
            where.append("direction = ?")
            params.append("inbound")
            if cutoff:
                where.append("created_at > ?")
                params.append(cutoff)
        normalized_query = self._normalize_optional(query)
        if normalized_query:
            like_value = f"%{normalized_query.lower()}%"
            where.append(
                "(LOWER(text) LIKE ? OR LOWER(COALESCE(contact_name, '')) LIKE ? OR LOWER(COALESCE(contact_id, '')) LIKE ?)"
            )
            params.extend([like_value, like_value, like_value])
        sql = """
            SELECT *
            FROM qq_history_messages
            {where_clause}
            ORDER BY created_at DESC, id DESC
            LIMIT ?
        """.format(where_clause=f"WHERE {' AND '.join(where)}" if where else "")
        params.append(normalized_limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            if not rows:
                return []
            message_ids = [int(row["id"]) for row in rows]
            placeholders = ", ".join("?" for _ in message_ids)
            attachment_rows = conn.execute(
                f"""
                SELECT message_id, attachment_kind, local_path, remote_url, file_name, mime_type, created_at
                FROM qq_history_attachments
                WHERE message_id IN ({placeholders})
                ORDER BY id ASC
                """,
                message_ids,
            ).fetchall()
        attachments_by_message: dict[int, list[sqlite3.Row]] = {}
        for row in attachment_rows:
            attachments_by_message.setdefault(int(row["message_id"]), []).append(row)
        return [self._serialize_message_row(row, attachments_by_message.get(int(row["id"]), [])) for row in rows]

    def _find_latest_outbound_timestamp(
        self,
        *,
        session_id: str | None = None,
        contact_id: str | None = None,
        contact_query: str | None = None,
    ) -> str | None:
        where, params = self._build_message_filters(
            session_id=session_id,
            contact_id=contact_id,
            contact_query=contact_query,
            direction="outbound",
        )
        sql = """
            SELECT created_at
            FROM qq_history_messages
            {where_clause}
            ORDER BY created_at DESC, id DESC
            LIMIT 1
        """.format(where_clause=f"WHERE {' AND '.join(where)}" if where else "")
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        if row is None:
            return None
        value = row["created_at"]
        return str(value).strip() or None

    def get_recent_attachments(
        self,
        *,
        session_id: str | None = None,
        contact_id: str | None = None,
        contact_query: str | None = None,
        kind: str = "any",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        normalized_limit = max(1, int(limit or 1))
        normalized_kind = str(kind or "any").strip().lower() or "any"
        where, params = self._build_attachment_filters(
            session_id=session_id,
            contact_id=contact_id,
            contact_query=contact_query,
        )
        if normalized_kind != "any":
            where.append("attachment_kind = ?")
            params.append(normalized_kind)
        sql = """
            SELECT *
            FROM qq_history_attachments
            {where_clause}
            ORDER BY created_at DESC, id DESC
            LIMIT ?
        """.format(where_clause=f"WHERE {' AND '.join(where)}" if where else "")
        params.append(normalized_limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._serialize_attachment_row(row) for row in rows]

    def _build_message_filters(
        self,
        *,
        session_id: str | None = None,
        contact_id: str | None = None,
        contact_query: str | None = None,
        direction: str | None = None,
    ) -> tuple[list[str], list[Any]]:
        where: list[str] = []
        params: list[Any] = []
        normalized_session_id = self._normalize_optional(session_id)
        normalized_contact_id = self._normalize_optional(contact_id)
        normalized_contact_query = self._normalize_optional(contact_query)
        if normalized_session_id:
            where.append("session_id = ?")
            params.append(normalized_session_id)
        if normalized_contact_id:
            where.append("contact_id = ?")
            params.append(normalized_contact_id)
        elif normalized_contact_query:
            like_value = f"%{normalized_contact_query.lower()}%"
            where.append("(LOWER(COALESCE(contact_name, '')) LIKE ? OR LOWER(COALESCE(contact_id, '')) LIKE ?)")
            params.extend([like_value, like_value])
        if direction:
            where.append("direction = ?")
            params.append(direction)
        return where, params

    def _build_attachment_filters(
        self,
        *,
        session_id: str | None = None,
        contact_id: str | None = None,
        contact_query: str | None = None,
    ) -> tuple[list[str], list[Any]]:
        where: list[str] = []
        params: list[Any] = []
        normalized_session_id = self._normalize_optional(session_id)
        normalized_contact_id = self._normalize_optional(contact_id)
        normalized_contact_query = self._normalize_optional(contact_query)
        if normalized_session_id:
            where.append("session_id = ?")
            params.append(normalized_session_id)
        if normalized_contact_id:
            where.append("contact_id = ?")
            params.append(normalized_contact_id)
        elif normalized_contact_query:
            like_value = f"%{normalized_contact_query.lower()}%"
            where.append("(LOWER(COALESCE(contact_name, '')) LIKE ? OR LOWER(COALESCE(contact_id, '')) LIKE ?)")
            params.extend([like_value, like_value])
        return where, params

    def _prune_locked(self, conn: sqlite3.Connection) -> None:
        total_messages = int(conn.execute("SELECT COUNT(*) FROM qq_history_messages").fetchone()[0])
        if total_messages > self.max_messages:
            conn.execute(
                """
                DELETE FROM qq_history_messages
                WHERE id NOT IN (
                    SELECT id
                    FROM qq_history_messages
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                )
                """,
                (self.max_messages,),
            )
        total_attachments = int(conn.execute("SELECT COUNT(*) FROM qq_history_attachments").fetchone()[0])
        if total_attachments > self.max_attachments:
            conn.execute(
                """
                DELETE FROM qq_history_attachments
                WHERE id NOT IN (
                    SELECT id
                    FROM qq_history_attachments
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                )
                """,
                (self.max_attachments,),
            )

    @staticmethod
    def _serialize_message_row(row: sqlite3.Row, attachments: list[sqlite3.Row]) -> dict[str, Any]:
        payload = {
            "message_id": int(row["id"]),
            "session_id": row["session_id"],
            "contact_id": row["contact_id"],
            "contact_name": row["contact_name"],
            "sender_id": row["sender_id"],
            "direction": row["direction"],
            "message_type": row["message_type"],
            "text": row["text"],
            "attachment_count": int(row["attachment_count"]),
            "created_at": row["created_at"],
            "attachments": [
                {
                    "kind": item["attachment_kind"],
                    "local_path": item["local_path"],
                    "remote_url": item["remote_url"],
                    "file_name": item["file_name"],
                    "mime_type": item["mime_type"],
                    "created_at": item["created_at"],
                }
                for item in attachments
            ],
        }
        payload["summary"] = QQHistoryStore._build_message_summary(payload)
        return payload

    @staticmethod
    def _serialize_attachment_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "attachment_id": int(row["id"]),
            "message_id": int(row["message_id"]),
            "session_id": row["session_id"],
            "contact_id": row["contact_id"],
            "contact_name": row["contact_name"],
            "direction": row["direction"],
            "kind": row["attachment_kind"],
            "local_path": row["local_path"],
            "remote_url": row["remote_url"],
            "file_name": row["file_name"],
            "mime_type": row["mime_type"],
            "created_at": row["created_at"],
        }

    @staticmethod
    def _normalize_optional(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @staticmethod
    def _normalize_attachment(payload: dict[str, Any]) -> dict[str, str | None]:
        attachment_kind = str(payload.get("kind") or payload.get("attachment_kind") or "file").strip().lower() or "file"
        local_path = QQHistoryStore._normalize_optional(payload.get("local_path"))
        remote_url = QQHistoryStore._normalize_optional(payload.get("remote_url"))
        file_name = QQHistoryStore._normalize_optional(payload.get("file_name"))
        mime_type = QQHistoryStore._normalize_optional(payload.get("mime_type"))
        if not file_name and local_path:
            file_name = Path(local_path).name
        return {
            "attachment_kind": attachment_kind,
            "local_path": local_path,
            "remote_url": remote_url,
            "file_name": file_name,
            "mime_type": mime_type,
        }

    @staticmethod
    def _build_message_summary(message: dict[str, Any]) -> str:
        contact = str(message.get("contact_name") or message.get("contact_id") or message.get("session_id") or "该联系人")
        direction = "对方" if message.get("direction") == "inbound" else "你"
        text = str(message.get("text") or "").strip()
        attachments = message.get("attachments") or []
        parts: list[str] = [f"{direction}在 {contact}"]
        if text:
            parts.append(f"说了：{text}")
        elif attachments:
            kinds = "、".join(str(item.get("kind") or "附件") for item in attachments[:3])
            parts.append(f"发送了附件：{kinds}")
        else:
            parts.append("发送了一条消息")
        return "".join(parts)
