from __future__ import annotations

from datetime import UTC, datetime
from typing import Iterable

from local_agent.protocol.models import InstructionIntent, MemoryCandidateIntent, MemoryRecord
from local_agent.storage.memory_store import SQLiteMemoryStore


WARM_MEMORY_TYPES = {
    "user_fact",
    "preference",
    "correction",
    "alias",
    "workflow_lesson",
    "eval_lesson",
    "success_pattern",
    "failure_pattern",
    "lesson_digest",
}
USER_MEMORY_TYPES = {
    "user_fact",
    "preference",
    "correction",
    "alias",
}
LEARNING_MEMORY_TYPES = {
    "workflow_lesson",
    "eval_lesson",
    "success_pattern",
    "failure_pattern",
    "lesson_digest",
}
_PREFERENCE_TOKENS = (
    "\u8bb0\u4f4f",
    "\u4ee5\u540e",
    "\u9ed8\u8ba4",
    "remember",
    "default",
)
_CORRECTION_TOKENS = (
    "\u4e0d\u8981\u518d",
    "\u4e0d\u8981\u628a",
    "\u4ec5\u9650",
    "\u53ea\u6709",
    "don't",
    "do not",
    "only",
)


class WarmMemoryService:
    _LEARNING_TYPES = LEARNING_MEMORY_TYPES
    _TYPE_PRIORITY = {
        "user_fact": 0.58,
        "preference": 0.52,
        "correction": 0.46,
        "alias": 0.24,
        "failure_pattern": 0.22,
        "workflow_lesson": 0.08,
        "eval_lesson": 0.06,
        "success_pattern": 0.03,
        "lesson_digest": -0.04,
    }

    def __init__(self, store: SQLiteMemoryStore) -> None:
        self.store = store

    def recall_for_text(
        self,
        user_text: str,
        *,
        scope: str = "user",
        limit: int = 4,
        memory_types: Iterable[str] | None = None,
    ) -> list[MemoryRecord]:
        normalized = " ".join(user_text.split()).strip()
        if not normalized:
            return []
        scopes = {scope, "session", "global"}
        allowed_types = set(memory_types or WARM_MEMORY_TYPES)
        records = self.store.recall_structured(
            normalized,
            limit=max(limit * 4, 12),
            memory_types=allowed_types,
            scopes=scopes,
        )
        lowered = normalized.lower()
        prefers_local_docs = any(token in lowered for token in {"桌面", "desktop", "文档", "文件", "日志", "汇报", "图片", "截图"})
        if prefers_local_docs:
            supplemental = self.store.list_records(
                memory_types={"preference", "correction", "failure_pattern"},
                scopes=scopes,
                limit=64,
            )
            for item in supplemental:
                record = item.record
                record_text = record.content.lower()
                tags = {tag.lower() for tag in record.tags}
                if "desktop" in tags or any(token in record_text for token in {"桌面", "desktop", "testing", "文档", "文件"}):
                    records.append(record)

        def _score(record: MemoryRecord) -> tuple[float, float]:
            score = float(record.importance) + self._TYPE_PRIORITY.get(record.memory_type, 0.0)
            tags = {tag.lower() for tag in record.tags}
            if "desktop" in tags and ("桌面" in normalized or "desktop" in lowered):
                score += 0.30
            if prefers_local_docs and record.memory_type in {"preference", "correction"}:
                score += 0.72
            if prefers_local_docs and record.memory_type == "failure_pattern":
                score += 0.32
            if record.memory_type == "lesson_digest":
                score -= 0.05
            return score, float(record.importance)

        unique: list[MemoryRecord] = []
        seen = set()
        for item in records:
            key = (item.memory_type, item.content)
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        ranked = sorted(unique, key=_score, reverse=True)
        return ranked[:limit]

    def recall_user_memory_for_text(
        self,
        user_text: str,
        *,
        scope: str = "user",
        limit: int = 4,
    ) -> list[MemoryRecord]:
        return self.recall_for_text(
            user_text,
            scope=scope,
            limit=limit,
            memory_types=USER_MEMORY_TYPES,
        )

    def recall_learning_memory_for_text(
        self,
        user_text: str,
        *,
        scope: str = "user",
        limit: int = 3,
    ) -> list[MemoryRecord]:
        return self.recall_for_text(
            user_text,
            scope=scope,
            limit=limit,
            memory_types=LEARNING_MEMORY_TYPES,
        )

    def remember_preference(
        self,
        content: str,
        *,
        scope: str = "user",
        tags: Iterable[str] | None = None,
        importance: float = 0.85,
    ) -> None:
        self._remember("preference", content, scope=scope, tags=tags, importance=importance)

    def remember_correction(
        self,
        content: str,
        *,
        scope: str = "user",
        tags: Iterable[str] | None = None,
        importance: float = 0.9,
    ) -> None:
        self._remember("correction", content, scope=scope, tags=tags, importance=importance)

    def remember_workflow_lesson(
        self,
        content: str,
        *,
        scope: str = "global",
        tags: Iterable[str] | None = None,
        importance: float = 0.86,
    ) -> None:
        self._remember_unique("workflow_lesson", content, scope=scope, tags=tags, importance=importance)

    def remember_eval_lesson(
        self,
        content: str,
        *,
        scope: str = "global",
        tags: Iterable[str] | None = None,
        importance: float = 0.88,
    ) -> None:
        self._remember_unique("eval_lesson", content, scope=scope, tags=tags, importance=importance)

    def remember_success_pattern(
        self,
        content: str,
        *,
        scope: str = "global",
        tags: Iterable[str] | None = None,
        importance: float = 0.68,
    ) -> None:
        self._remember_unique("success_pattern", content, scope=scope, tags=tags, importance=importance)

    def remember_failure_pattern(
        self,
        content: str,
        *,
        scope: str = "global",
        tags: Iterable[str] | None = None,
        importance: float = 0.91,
    ) -> None:
        self._remember_unique("failure_pattern", content, scope=scope, tags=tags, importance=importance)

    def remember_lesson_digest(
        self,
        content: str,
        *,
        scope: str = "global",
        tags: Iterable[str] | None = None,
        importance: float = 0.72,
    ) -> None:
        self._remember_unique("lesson_digest", content, scope=scope, tags=tags, importance=importance)

    def compact_learning_memories(
        self,
        *,
        scope: str = "user",
        max_records: int = 80,
        keep_recent: int = 28,
    ) -> str | None:
        count = self.store.count_records(memory_types=self._LEARNING_TYPES - {"lesson_digest"}, scopes={scope})
        if count <= max_records:
            return None

        stored = self.store.list_records(
            memory_types=self._LEARNING_TYPES - {"lesson_digest"},
            scopes={scope},
            limit=max(count + 8, 120),
        )
        if len(stored) <= max_records:
            return None

        recent = stored[:keep_recent]
        older = stored[keep_recent:]
        if not older:
            return None

        digest_lines = ["学习摘要（自动压缩）："]
        seen_contents: set[str] = set()
        for item in older:
            content = item.record.content.strip()
            if not content or content in seen_contents:
                continue
            seen_contents.add(content)
            digest_lines.append(f"- [{item.record.memory_type}] {content}")
            if len(digest_lines) >= 7:
                break
        if len(digest_lines) == 1:
            return None

        digest = "\n".join(digest_lines)
        recent_tags = {tag for item in recent[:10] for tag in item.record.tags}
        self.remember_lesson_digest(digest, scope=scope, tags=sorted(recent_tags)[:10], importance=0.74)
        self.store.delete_ids(item.memory_id for item in older)
        return digest

    def maybe_capture_user_instruction(self, user_text: str, *, scope: str = "user") -> MemoryRecord | None:
        normalized = " ".join(user_text.split()).strip()
        if len(normalized) < 6 or len(normalized) > 160:
            return None

        if any(token in normalized for token in _PREFERENCE_TOKENS + _CORRECTION_TOKENS):
            memory_type = "preference" if any(token in normalized for token in _PREFERENCE_TOKENS) else "correction"
            record = MemoryRecord(
                memory_type=memory_type,
                scope=scope,
                content=normalized,
                importance=0.82 if memory_type == "preference" else 0.88,
                tags=self._with_bucket_tags("user_profile", self._infer_tags(normalized.lower())),
                created_at=datetime.now(UTC),
            )
            self.store.remember(record)
            return record
        return None

    def remember_instruction_intent(
        self,
        instruction_intent: InstructionIntent,
        *,
        default_scope: str = "user",
    ) -> MemoryRecord | None:
        if not bool(getattr(instruction_intent, "persist_memory", False)):
            return None

        content = str(
            getattr(instruction_intent, "memory_text", None)
            or getattr(instruction_intent, "normalized_instruction", None)
            or ""
        ).strip()
        if len(content) < 4:
            return None

        intent_scope = str(getattr(instruction_intent, "scope", "") or "").strip().lower()
        scope = "session" if intent_scope == "session" else default_scope
        kind = str(getattr(instruction_intent, "kind", "") or "").strip().lower()
        memory_type = self._instruction_memory_type(kind)
        tags = self._infer_tags(content.lower())
        tags.extend(tag for tag in ("instruction", kind, intent_scope) if tag and tag != "none")
        tags = self._with_bucket_tags("user_profile", tags)

        importance = 0.9 if memory_type in {"correction", "alias"} else 0.84
        existing = self.store.recall_structured(
            content,
            limit=4,
            memory_types={memory_type},
            scopes={scope},
        )
        if any(item.content.strip() == content for item in existing):
            return None

        record = MemoryRecord(
            memory_type=memory_type,
            scope=scope,
            content=content,
            importance=importance,
            tags=tags,
            created_at=datetime.now(UTC),
        )
        self.store.remember(record)
        return record

    def remember_memory_candidate_intent(
        self,
        memory_candidate: MemoryCandidateIntent,
        *,
        default_scope: str = "user",
    ) -> MemoryRecord | None:
        if not bool(getattr(memory_candidate, "is_memory_candidate", False)):
            return None
        if not bool(getattr(memory_candidate, "persist_memory", False)):
            return None
        if not bool(getattr(memory_candidate, "should_write_memory", False)):
            return None
        try:
            confidence = float(getattr(memory_candidate, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0.72:
            return None

        content = str(
            getattr(memory_candidate, "memory_text", None)
            or getattr(memory_candidate, "normalized_text", None)
            or ""
        ).strip()
        if len(content) < 4:
            return None

        intent_scope = str(getattr(memory_candidate, "scope", "") or "").strip().lower()
        scope = "session" if intent_scope == "session" else default_scope
        kind = str(getattr(memory_candidate, "kind", "") or "").strip().lower()
        memory_type = self._memory_candidate_type(kind)
        memory_key = str(getattr(memory_candidate, "memory_key", None) or "").strip().lower()
        canonical_value = getattr(memory_candidate, "canonical_value", {})
        if not isinstance(canonical_value, dict):
            canonical_value = {}
        content = self._normalize_memory_candidate_content(
            memory_type=memory_type,
            memory_key=memory_key,
            canonical_value=canonical_value,
            fallback_content=content,
        )
        if not self._is_valid_memory_candidate_content(
            memory_type=memory_type,
            memory_key=memory_key,
            canonical_value=canonical_value,
            content=content,
        ):
            return None

        tags = self._infer_tags(content.lower())
        tags.extend(
            tag
            for tag in (
                "memory_candidate",
                kind,
                intent_scope,
                memory_key,
            )
            if tag and tag != "none"
        )
        for canonical_key, canonical_entry in canonical_value.items():
            key_text = str(canonical_key or "").strip().lower()
            value_text = str(canonical_entry or "").strip().lower()
            if key_text:
                tags.append(f"field:{key_text}")
            if key_text and value_text:
                tags.append(f"{key_text}:{value_text[:48]}")
        tags = self._with_bucket_tags(
            "agent_learning" if memory_type in self._LEARNING_TYPES else "user_profile",
            tags,
        )

        if bool(getattr(memory_candidate, "overwrite_existing", False)):
            self._delete_conflicting_records(
                scope=scope,
                memory_type=memory_type,
                memory_key=memory_key,
                kind=kind,
                content=content,
            )

        existing = self.store.recall_structured(
            content,
            limit=6,
            memory_types={memory_type},
            scopes={scope},
        )
        if any(item.content.strip() == content for item in existing):
            return None

        importance = 0.92 if memory_type in {"user_fact", "correction"} else 0.86
        record = MemoryRecord(
            memory_type=memory_type,
            scope=scope,
            content=content,
            importance=importance,
            tags=tags,
            created_at=datetime.now(UTC),
        )
        self.store.remember(record)
        return record

    def purge_development_memories(self, *, scope: str = "user") -> int:
        stored = self.store.list_records(scopes={scope, "session", "global"}, limit=2000)
        delete_ids: list[int] = []
        for item in stored:
            record = item.record
            tags = {tag.lower() for tag in record.tags}
            content = record.content.strip()
            if record.memory_type in self._LEARNING_TYPES:
                delete_ids.append(item.memory_id)
                continue
            if record.memory_type == "episodic":
                delete_ids.append(item.memory_id)
                continue
            if record.memory_type in {"alias", "user_fact"} and (
                "user.birthday" in tags
                or "生日" in content
                or "birthday" in content.lower()
            ):
                delete_ids.append(item.memory_id)
                continue
        return self.store.delete_ids(delete_ids)

    @staticmethod
    def format_for_prompt(records: list[MemoryRecord], *, title: str = "相关温记忆") -> str:
        if not records:
            return ""
        lines = [f"{title}:"]
        for index, record in enumerate(records, start=1):
            lines.append(f"{index}. [{record.memory_type}] {record.content}")
        return "\n".join(lines)

    def _remember(
        self,
        memory_type: str,
        content: str,
        *,
        scope: str,
        tags: Iterable[str] | None,
        importance: float,
    ) -> None:
        bucket = "agent_learning" if memory_type in self._LEARNING_TYPES else "user_profile"
        record = MemoryRecord(
            memory_type=memory_type,
            scope=scope,
            content=content.strip(),
            importance=importance,
            tags=self._with_bucket_tags(bucket, tags),
            created_at=datetime.now(UTC),
        )
        self.store.remember(record)

    def _remember_unique(
        self,
        memory_type: str,
        content: str,
        *,
        scope: str,
        tags: Iterable[str] | None,
        importance: float,
    ) -> None:
        normalized = content.strip()
        if not normalized:
            return
        existing = self.store.recall_structured(
            normalized,
            limit=6,
            memory_types={memory_type},
            scopes={scope},
        )
        if any(item.content.strip() == normalized for item in existing):
            return
        self._remember(memory_type, normalized, scope=scope, tags=tags, importance=importance)

    @staticmethod
    def _infer_tags(lowered_text: str) -> list[str]:
        tags: list[str] = []
        if "qq" in lowered_text:
            tags.append("qq")
        if "\u684c\u9762" in lowered_text or "desktop" in lowered_text:
            tags.append("desktop")
        if "\u9ed8\u8ba4" in lowered_text:
            tags.append("default")
        if "\u4e0d\u8981" in lowered_text:
            tags.append("correction")
        return tags

    @staticmethod
    def _with_bucket_tags(bucket: str, tags: Iterable[str] | None) -> list[str]:
        merged = [tag for tag in (tags or []) if str(tag).strip()]
        merged.append(f"bucket:{bucket}")
        return list(dict.fromkeys(str(tag).strip() for tag in merged if str(tag).strip()))

    @staticmethod
    def _instruction_memory_type(kind: str) -> str:
        if kind == "naming":
            return "alias"
        if kind in {"correction", "boundary"}:
            return "correction"
        return "preference"

    @staticmethod
    def _memory_candidate_type(kind: str) -> str:
        if kind == "user_fact":
            return "user_fact"
        return WarmMemoryService._instruction_memory_type(kind)

    @staticmethod
    def _normalize_memory_candidate_content(
        *,
        memory_type: str,
        memory_key: str,
        canonical_value: dict[str, object],
        fallback_content: str,
    ) -> str:
        content = str(fallback_content or "").strip()
        if memory_type != "user_fact":
            return content

        if memory_key == "user.birthday":
            relative = str(
                canonical_value.get("relative_day")
                or canonical_value.get("birthday")
                or canonical_value.get("value")
                or ""
            ).strip().lower()
            mapping = {
                "today": "今天",
                "tomorrow": "明天",
                "yesterday": "昨天",
                "day_after_tomorrow": "后天",
                "the day after tomorrow": "后天",
            }
            if relative in mapping:
                return f"用户的生日是{mapping[relative]}。"
            absolute = str(
                canonical_value.get("date")
                or canonical_value.get("date_text")
                or canonical_value.get("month_day")
                or ""
            ).strip()
            if absolute:
                return f"用户的生日是{absolute}。"
        return content

    @staticmethod
    def _is_valid_memory_candidate_content(
        *,
        memory_type: str,
        memory_key: str,
        canonical_value: dict[str, object],
        content: str,
    ) -> bool:
        normalized = str(content or "").strip()
        if len(normalized) < 4:
            return False
        if memory_type != "user_fact":
            return True
        if memory_key == "user.birthday":
            if "用户的生日是" in normalized:
                return True
            relative = str(
                canonical_value.get("relative_day")
                or canonical_value.get("birthday")
                or canonical_value.get("value")
                or ""
            ).strip()
            absolute = str(
                canonical_value.get("date")
                or canonical_value.get("date_text")
                or canonical_value.get("month_day")
                or ""
            ).strip()
            return bool(relative or absolute)
        return "用户" in normalized or "我" in normalized

    def _delete_conflicting_records(
        self,
        *,
        scope: str,
        memory_type: str,
        memory_key: str,
        kind: str,
        content: str,
    ) -> None:
        stored = self.store.list_records(memory_types={memory_type}, scopes={scope}, limit=80)
        delete_ids: list[int] = []
        for item in stored:
            tags = {tag.lower() for tag in item.record.tags}
            if memory_key and memory_key in tags:
                delete_ids.append(item.memory_id)
                continue
            if kind and kind in tags and item.record.content.strip() != content:
                delete_ids.append(item.memory_id)
        if delete_ids:
            self.store.delete_ids(delete_ids)
