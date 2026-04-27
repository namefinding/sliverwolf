from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from local_agent.protocol.models import ToolManifest

from .reminder_store import ReminderStore
from .timezone_utils import resolve_timezone


class GetTimeInput(BaseModel):
    kind: str = "datetime"
    timezone_name: str | None = None


class CreateReminderInput(BaseModel):
    when_iso: str
    timezone: str = "Asia/Shanghai"
    message: str
    session_id: str | None = None
    channel: str | None = None
    task_payload: dict[str, Any] = Field(default_factory=dict)


class ListRemindersInput(BaseModel):
    status: str = "scheduled"
    session_id: str | None = None


class CancelReminderInput(BaseModel):
    reminder_id: str


class CreateScheduledTaskInput(BaseModel):
    task_type: str  # "notify" | "deferred_agent_task"
    when_iso: str
    timezone: str = "Asia/Shanghai"
    session_id: str | None = None
    channel: str | None = None
    message: str = ""
    task_payload: dict[str, Any] = Field(default_factory=dict)


class SystemUtilityModule:
    def __init__(
        self,
        reminder_store: ReminderStore,
        default_timezone: str = "Asia/Shanghai",
    ) -> None:
        self.reminder_store = reminder_store
        self.default_timezone = default_timezone

    def manifests(self) -> list[ToolManifest]:
        return [
            ToolManifest(
                tool_name="system.get_time",
                module="system_utility",
                description="Get the current local time, date, or weekday.",
                side_effect=False,
                idempotent=True,
                produces=[],
                input_schema=GetTimeInput.model_json_schema(),
                output_schema={"type": "object"},
            ),
            ToolManifest(
                tool_name="system.create_reminder",
                module="system_utility",
                description="Create a local reminder using fully structured execution parameters.",
                side_effect=True,
                idempotent=False,
                produces=[],
                input_schema=CreateReminderInput.model_json_schema(),
                output_schema={"type": "object"},
            ),
            ToolManifest(
                tool_name="system.create_scheduled_task",
                module="system_utility",
                description="Create a scheduled local task that will either notify the user or trigger a deferred agent action at the specified time.",
                side_effect=True,
                idempotent=False,
                produces=[],
                input_schema=CreateScheduledTaskInput.model_json_schema(),
                output_schema={"type": "object"},
            ),
            ToolManifest(
                tool_name="system.list_reminders",
                module="system_utility",
                description="List existing local reminders.",
                side_effect=False,
                idempotent=True,
                produces=[],
                input_schema=ListRemindersInput.model_json_schema(),
                output_schema={"type": "object"},
            ),
            ToolManifest(
                tool_name="system.cancel_reminder",
                module="system_utility",
                description="Cancel an existing local reminder.",
                side_effect=True,
                idempotent=False,
                produces=[],
                input_schema=CancelReminderInput.model_json_schema(),
                output_schema={"type": "object"},
            ),
        ]

    def executor_map(self) -> dict[str, Any]:
        return {
            "system.get_time": self.get_time,
            "system.create_reminder": self.create_reminder,
            "system.create_scheduled_task": self.create_scheduled_task,
            "system.list_reminders": self.list_reminders,
            "system.cancel_reminder": self.cancel_reminder,
        }

    def _resolve_timezone(self, timezone_name: str | None):
        return resolve_timezone(timezone_name, default_timezone=self.default_timezone)

    def get_time(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = GetTimeInput.model_validate(arguments)
        tz = self._resolve_timezone(payload.timezone_name)
        now = datetime.now(tz)

        weekday_map = {
            0: "\u661f\u671f\u4e00",
            1: "\u661f\u671f\u4e8c",
            2: "\u661f\u671f\u4e09",
            3: "\u661f\u671f\u56db",
            4: "\u661f\u671f\u4e94",
            5: "\u661f\u671f\u516d",
            6: "\u661f\u671f\u65e5",
        }

        kind = (payload.kind or "datetime").strip().lower()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        weekday_str = weekday_map[now.weekday()]

        if kind == "date":
            formatted = date_str
        elif kind == "time":
            formatted = time_str
        elif kind == "weekday":
            formatted = weekday_str
        else:
            formatted = f"{date_str} {time_str} {weekday_str}"

        return {
            "kind": kind,
            "iso": now.isoformat(),
            "formatted": formatted,
            "date": date_str,
            "time": time_str,
            "weekday": weekday_str,
            "timezone": str(tz),
        }

    def create_scheduled_task(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = CreateScheduledTaskInput.model_validate(arguments)

        try:
            scheduled_for = datetime.fromisoformat(payload.when_iso)
        except ValueError:
            return {
                "created": False,
                "error": "invalid_when_iso",
                "when_iso": payload.when_iso,
                "task_type": payload.task_type,
            }

        if scheduled_for.tzinfo is None:
            return {
                "created": False,
                "error": "when_iso_requires_timezone",
                "when_iso": payload.when_iso,
                "task_type": payload.task_type,
            }

        task_type = str(payload.task_type or "").strip().lower()
        if task_type not in {"notify", "deferred_agent_task"}:
            return {
                "created": False,
                "error": "invalid_task_type",
                "task_type": payload.task_type,
            }

        record = self.reminder_store.create_scheduled_task(
            task_type=payload.task_type,
            message=payload.message,
            when_iso=payload.when_iso,
            timezone_name=payload.timezone,
            session_id=payload.session_id,
            channel=payload.channel,
            task_payload=payload.task_payload or {},
            metadata={"source": "llm_structured_arguments"},
        )
        return {
            "created": True,
            "task": record,
        }

    def create_reminder(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = CreateReminderInput.model_validate(arguments)
        result = self.create_scheduled_task(
            {
                "task_type": "notify",
                "when_iso": payload.when_iso,
                "timezone": payload.timezone,
                "session_id": payload.session_id,
                "channel": payload.channel,
                "message": payload.message,
                "task_payload": payload.task_payload,
            }
        )
        if not result.get("created"):
            return result
        return {
            "created": True,
            "reminder": result.get("task"),
        }

    def list_reminders(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = ListRemindersInput.model_validate(arguments)
        reminders = self.reminder_store.list_reminders(
            status=payload.status,
            session_id=payload.session_id,
        )
        return {
            "count": len(reminders),
            "status": payload.status,
            "reminders": reminders,
        }

    def cancel_reminder(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = CancelReminderInput.model_validate(arguments)
        updated = self.reminder_store.cancel_reminder(payload.reminder_id)
        return {
            "cancelled": updated is not None,
            "reminder": updated,
        }
