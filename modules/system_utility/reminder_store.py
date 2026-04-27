from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ReminderStore:
    def __init__(self, db_path: str = "data/reminders.sqlite3") -> None:
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    reminder_id TEXT PRIMARY KEY,
                    message TEXT NOT NULL,
                    when_iso TEXT NOT NULL,
                    scheduled_for TEXT NOT NULL,
                    timezone TEXT NOT NULL,
                    status TEXT NOT NULL,
                    session_id TEXT,
                    created_at TEXT NOT NULL,
                    fired_at TEXT,
                    metadata_json TEXT NOT NULL,
                    task_type TEXT NOT NULL DEFAULT 'notify',
                    task_payload_json TEXT NOT NULL DEFAULT '{}',
                    channel TEXT
                )
                """
            )
            conn.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None

        keys = set(row.keys())

        when_iso = row["when_iso"] if "when_iso" in keys else (
            row["scheduled_for"] if "scheduled_for" in keys else None
        )
        timezone = row["timezone"] if "timezone" in keys else None

        return {
            "reminder_id": row["reminder_id"],
            "message": row["message"],
            "when_iso": when_iso,
            "timezone": timezone,
            "status": row["status"],
            "session_id": row["session_id"] if "session_id" in keys else None,
            "created_at": row["created_at"] if "created_at" in keys else None,
            "fired_at": row["fired_at"] if "fired_at" in keys else None,
            "task_type": row["task_type"] if "task_type" in keys else "notify",
            "task_payload": json.loads(row["task_payload_json"] or "{}") if "task_payload_json" in keys else {},
            "channel": row["channel"] if "channel" in keys else None,
            "metadata": json.loads(row["metadata_json"] or "{}") if "metadata_json" in keys else {},
        }

    def create_scheduled_task(
            self,
            *,
            task_type: str,
            message: str,
            when_iso: str,
            timezone_name: str,
            session_id: str | None = None,
            channel: str | None = None,
            task_payload: dict[str, Any] | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        reminder_id = f"rem_{uuid.uuid4().hex[:12]}"
        created_at = datetime.now(UTC).isoformat()

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO reminders (
                    reminder_id, message, when_iso, scheduled_for, timezone,
                    status, session_id, created_at, fired_at, metadata_json,
                    task_type, task_payload_json, channel
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reminder_id,
                    message,
                    when_iso,
                    when_iso,
                    timezone_name,
                    "scheduled",
                    session_id,
                    created_at,
                    None,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    task_type,
                    json.dumps(task_payload or {}, ensure_ascii=False),
                    channel,
                ),
            )
            conn.commit()

            row = conn.execute(
                "SELECT * FROM reminders WHERE reminder_id = ?",
                (reminder_id,),
            ).fetchone()
            result = self._row_to_dict(row)
            if result is None:
                raise RuntimeError("Failed to create scheduled task.")
            return result

    def create_reminder(
            self,
            *,
            message: str,
            when_iso: str,
            timezone_name: str,
            session_id: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.create_scheduled_task(
            task_type="notify",
            message=message,
            when_iso=when_iso,
            timezone_name=timezone_name,
            session_id=session_id,
            channel=None,
            task_payload={},
            metadata=metadata,
        )

    def list_reminders(
        self,
        *,
        status: str = "scheduled",
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM reminders WHERE status = ?"
        params: list[Any] = [status]
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        sql += " ORDER BY scheduled_for ASC"

        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_dict(row) for row in rows if row is not None]

    def cancel_reminder(self, reminder_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE reminders SET status = ? WHERE reminder_id = ? AND status = ?",
                ("cancelled", reminder_id, "scheduled"),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM reminders WHERE reminder_id = ?",
                (reminder_id,),
            ).fetchone()
            return self._row_to_dict(row)

    def get_due_reminders(
            self,
            *,
            now_utc: datetime | None = None,
            limit: int = 20,
    ) -> list[dict[str, Any]]:
        now_dt = now_utc or datetime.now(UTC)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=UTC)

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM reminders
                WHERE status = ?
                ORDER BY created_at ASC
                """,
                ("scheduled",),
            ).fetchall()

        due: list[dict[str, Any]] = []
        for row in rows:
            item = self._row_to_dict(row)
            if item is None:
                continue

            when_iso = str(item.get("when_iso", "") or "").strip()
            if not when_iso:
                continue

            try:
                scheduled_dt = datetime.fromisoformat(when_iso)
                if scheduled_dt.tzinfo is None:
                    continue
            except Exception:
                continue

            if scheduled_dt <= now_dt.astimezone(scheduled_dt.tzinfo):
                due.append(item)
                if len(due) >= limit:
                    break

        return due

    def mark_fired(self, reminder_id: str) -> dict[str, Any] | None:
        fired_at = datetime.now(UTC).isoformat()

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE reminders
                SET status = ?, fired_at = ?
                WHERE reminder_id = ? AND status = ?
                """,
                ("fired", fired_at, reminder_id, "scheduled"),
            )
            conn.commit()

            row = conn.execute(
                "SELECT * FROM reminders WHERE reminder_id = ?",
                (reminder_id,),
            ).fetchone()
            return self._row_to_dict(row)

