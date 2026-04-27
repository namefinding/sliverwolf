from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from local_agent.app.chat_service import ChatService
from local_agent.app.channel_router import ChannelRouter
from local_agent.app.healthcheck import run_startup_healthcheck
from local_agent.app.main import build_kernel, build_retrieval_service
from local_agent.app.session_store import InMemorySessionStore
from local_agent.app.startup_jobs import StartupJobsRunner
from local_agent.app.task_service import TaskService
from local_agent.app.task_store import InMemoryTaskStore
from local_agent.app.config import load_config
from local_agent.protocol.models import AppConfig
from local_agent.modules.system_utility.reminder_store import ReminderStore
from local_agent.modules.system_utility.scheduler import ReminderScheduler


@dataclass
class BootstrappedResidentApp:
    config: AppConfig
    chat_service: ChatService
    channel_router: ChannelRouter
    task_service: TaskService
    health_report: dict[str, object]
    startup_jobs: StartupJobsRunner
    scope_options: list[str]
    ui_defaults: dict[str, object]


def _build_scope_options(config_path: str, configured_workspace: str) -> list[str]:
    configured = str(Path(configured_workspace).resolve())
    project_root = str(Path(config_path).resolve().parent)
    desktop = str(Path.home() / "Desktop")
    testing = str(Path(desktop) / "testing")

    options: list[str] = []
    for candidate in (configured, testing, desktop, project_root):
        if Path(candidate).is_dir() and candidate not in options:
            options.append(candidate)
    return options


def bootstrap_resident_app(
    *,
    config_path: str,
    session_ttl_minutes: int,
    max_sessions: int,
) -> BootstrappedResidentApp:
    config = load_config(config_path)
    session_store = InMemorySessionStore(ttl_minutes=session_ttl_minutes, max_sessions=max_sessions)
    task_store = InMemoryTaskStore(ttl_minutes=session_ttl_minutes * 4, max_tasks=max_sessions * 10)
    def kernel_factory(scope_root=None, runtime_settings=None):
        runtime_settings = runtime_settings or {}
        agent_overrides = runtime_settings.get("agent") if isinstance(runtime_settings.get("agent"), dict) else None
        voice_overrides = runtime_settings.get("voice") if isinstance(runtime_settings.get("voice"), dict) else None
        policy_overrides = runtime_settings.get("access_policy") if isinstance(runtime_settings.get("access_policy"), dict) else None
        channel_runtime = runtime_settings.get("channel_runtime") if isinstance(runtime_settings.get("channel_runtime"), dict) else None
        return build_kernel(
            config_path,
            workspace_root_override=scope_root,
            agent_overrides=agent_overrides,
            voice_overrides=voice_overrides,
            policy_overrides=policy_overrides,
            channel_runtime=channel_runtime,
        )

    chat_service = ChatService(
        session_store=session_store,
        kernel_factory=kernel_factory,
        configured_workspace=config.agent.workspace_root,
        project_root=str(Path(config_path).resolve().parent),
    )
    reminder_store = ReminderStore("data/reminders.sqlite3")

    def _notify_task(task: dict) -> None:
        chat_service.handle_scheduled_task(task=task)

    def _dispatch_task(task: dict) -> None:
        chat_service.handle_scheduled_task(task=task)

    scheduler = ReminderScheduler(
        reminder_store=reminder_store,
        notify_callback=_notify_task,
        dispatch_callback=_dispatch_task,
        poll_interval_seconds=1.0,
    )
    scheduler.start()

    chat_service._reminder_scheduler = scheduler
    channel_router = ChannelRouter(chat_service, onebot_config=config.onebot)
    task_service = TaskService(
        session_store=session_store,
        task_store=task_store,
        kernel_factory=kernel_factory,
    )

    health_report = run_startup_healthcheck(config)
    retrieval_service = build_retrieval_service(config)
    startup_jobs = StartupJobsRunner(
        has_index=retrieval_service.has_rows,
        rebuild_index=retrieval_service.rebuild_filesystem_index,
        sync_index=retrieval_service.sync_filesystem_index,
    )

    if health_report.get("overall_status") == "ok":
        if retrieval_service.has_rows():
            startup_jobs.start_async()
        else:
            startup_jobs.mark_skipped("No local index exists yet. The first local retrieval request will build it on demand.")
    else:
        startup_jobs.mark_skipped("Startup jobs were skipped because critical checks did not pass.")

    return BootstrappedResidentApp(
        config=config,
        chat_service=chat_service,
        channel_router=channel_router,
        task_service=task_service,
        health_report=health_report,
        startup_jobs=startup_jobs,
        scope_options=_build_scope_options(config_path, config.agent.workspace_root),
        ui_defaults={
            "agent": {
                "persona_name": config.agent.persona_name,
                "persona_profile": config.agent.persona_profile,
                "chat_style_prompt": config.agent.chat_style_prompt,
                "display_style_prompt": config.agent.display_style_prompt,
                "speech_style_prompt": config.agent.speech_style_prompt,
                "tool_speech_enabled": config.agent.tool_speech_enabled,
                "speech_max_chars": config.agent.speech_max_chars,
            },
            "voice": {
                "enabled": config.voice.enabled,
            },
        },
    )
