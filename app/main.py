from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from local_agent.app.config import load_config
from local_agent.kernel.agent_kernel import AgentKernel
from local_agent.llm.ollama_client import OllamaClient
from local_agent.modules.base import ToolRegistry
from local_agent.modules.computer_use.service import ComputerUseModule
from local_agent.modules.document_agent.service import DocumentAgentModule
from local_agent.modules.file.service import FileModule
from local_agent.modules.image.service import ImageModule
from local_agent.modules.memory.service import MemoryModule
from local_agent.modules.qq.service import QQModule
from local_agent.modules.retrieval.service import RetrievalModule
from local_agent.modules.web.service import WebModule
from local_agent.storage.memory_store import SQLiteMemoryStore
from local_agent.storage.trace_store import JsonlTraceStore
from local_agent.voice.gptsovits import GPTSoVITSAdapter
from local_agent.retrieval.hybrid_index import (
    HashEmbeddingProvider,
    HeuristicRerankerProvider,
    HybridIndexService,
    OllamaJudgeRerankerProvider,
    OllamaEmbeddingProvider,
)
from local_agent.modules.system_utility.service import SystemUtilityModule
from local_agent.modules.system_utility.reminder_store import ReminderStore
from local_agent.modules.system_utility.scheduler import ReminderScheduler


def apply_session_identity_overrides(agent_config, access_policy: dict | None) -> None:
    if not isinstance(access_policy, dict):
        return

    alias_candidates: list[str] = []
    configured_aliases = getattr(agent_config, "assistant_aliases", None) or []
    for candidate in (
        getattr(agent_config, "system_name", None),
        getattr(agent_config, "persona_name", None),
        *configured_aliases,
    ):
        if not isinstance(candidate, str):
            continue
        normalized = candidate.strip()
        if normalized and normalized not in alias_candidates:
            alias_candidates.append(normalized)

    identity_lines: list[str] = []
    if alias_candidates:
        identity_lines.append("当用户称呼" + "、".join(alias_candidates) + "时，默认是在叫你自己。")

    if bool(access_policy.get("is_owner")):
        sender_id = str(access_policy.get("sender_id") or "").strip()
        address_as = str(access_policy.get("address_as") or "主人").strip() or "主人"
        owner_line = "当前会话中的 QQ 用户"
        if sender_id:
            owner_line += sender_id
        owner_line += f"是你的主人。自然地把对方称作{address_as}，但不要每句话都重复。"
        identity_lines.append(owner_line)

    if not identity_lines:
        return

    current_profile = str(getattr(agent_config, "persona_profile", "") or "").strip()
    for line in identity_lines:
        if line in current_profile:
            continue
        current_profile = f"{current_profile}\n{line}".strip() if current_profile else line
    agent_config.persona_profile = current_profile


def _scoped_retrieval_db_path(config, workspace_root: str) -> str:
    workspace_path = str(Path(workspace_root).resolve())
    configured_root = str(Path(config.agent.workspace_root).resolve())
    if workspace_path == configured_root:
        return config.agent.retrieval_db_path

    base_path = Path(config.agent.retrieval_db_path)
    suffix = hashlib.sha1(workspace_path.encode("utf-8")).hexdigest()[:12]
    scoped_name = f"{base_path.stem}_{suffix}{base_path.suffix or '.sqlite3'}"
    return str(base_path.with_name(scoped_name))


def build_retrieval_service(
    config,
    workspace_root_override: str | None = None,
    agent_config_override=None,
) -> HybridIndexService:
    agent_config = agent_config_override or config.agent
    workspace_root = str(Path(workspace_root_override or agent_config.workspace_root).resolve())

    embedding_provider = None
    if agent_config.retrieval_embedding_enabled:
        provider_name = agent_config.retrieval_embedding_provider.strip().lower()
        if provider_name == "ollama" and agent_config.retrieval_embedding_model:
            embedding_provider = OllamaEmbeddingProvider(
                base_url=agent_config.ollama_base_url,
                model_name=agent_config.retrieval_embedding_model,
                timeout_seconds=agent_config.retrieval_embedding_timeout_seconds,
                batch_size=agent_config.retrieval_embedding_batch_size,
            )
        elif provider_name == "hash":
            embedding_provider = HashEmbeddingProvider(
                dimensions=agent_config.retrieval_embedding_dimensions,
            )

    reranker_provider_name = agent_config.retrieval_reranker_provider.strip().lower()
    if reranker_provider_name == "ollama" and agent_config.retrieval_reranker_model:
        reranker_provider = OllamaJudgeRerankerProvider(
            base_url=agent_config.ollama_base_url,
            model_name=agent_config.retrieval_reranker_model,
            timeout_seconds=agent_config.retrieval_reranker_timeout_seconds,
            rerank_top_n=agent_config.retrieval_reranker_top_n,
        )
    else:
        reranker_provider = HeuristicRerankerProvider()

    return HybridIndexService(
        agent_config.retrieval_db_path,
        workspace_root,
        embedding_provider=embedding_provider,
        reranker_provider=reranker_provider,
    )


def build_kernel(
    config_path: str,
    workspace_root_override: str | None = None,
    agent_overrides: dict | None = None,
    voice_overrides: dict | None = None,
    policy_overrides: dict | None = None,
    channel_runtime: dict | None = None,
) -> AgentKernel:
    config = load_config(config_path)
    workspace_root = str(Path(workspace_root_override or config.agent.workspace_root).resolve())
    agent_config = config.agent.model_copy(deep=True)
    agent_config.workspace_root = workspace_root
    agent_config.retrieval_db_path = _scoped_retrieval_db_path(config, workspace_root)
    if isinstance(agent_overrides, dict):
        for key, value in agent_overrides.items():
            if hasattr(agent_config, key):
                setattr(agent_config, key, value)
    if isinstance(policy_overrides, dict):
        apply_session_identity_overrides(agent_config, policy_overrides)
    memory_store = SQLiteMemoryStore(agent_config.memory_db_path)
    trace_store = JsonlTraceStore(agent_config.trace_path)
    retrieval_service = build_retrieval_service(
        config,
        workspace_root_override=workspace_root,
        agent_config_override=agent_config,
    )
    llm_client = OllamaClient(
        base_url=agent_config.ollama_base_url,
        model=agent_config.model,
        timeout_seconds=agent_config.request_timeout_seconds,
        chat_model=agent_config.chat_model,
        critic_model=agent_config.critic_model,
        response_model=agent_config.response_model,
        vision_model=agent_config.vision_model,
        keep_alive=agent_config.ollama_keep_alive,
    )
    voice_config = config.voice.model_copy(deep=True)
    if isinstance(voice_overrides, dict):
        for key, value in voice_overrides.items():
            if hasattr(voice_config, key):
                setattr(voice_config, key, value)
    voice_adapter = GPTSoVITSAdapter(voice_config)

    registry = ToolRegistry()
    file_module = FileModule(workspace_root=workspace_root)
    document_agent_module = DocumentAgentModule(file_module=file_module, llm_client=llm_client)
    image_module = ImageModule(
        workspace_root=workspace_root,
        vision_describer=llm_client.describe_image,
    )
    memory_module = MemoryModule(store=memory_store)
    web_module = WebModule(config=config.web)
    retrieval_module = RetrievalModule(index_service=retrieval_service)
    computer_use_module = ComputerUseModule(workspace_root=workspace_root)
    qq_runtime_context = dict(channel_runtime or {})

    reminder_store = ReminderStore("data/reminders.sqlite3")
    system_utility_module = SystemUtilityModule(
        reminder_store=reminder_store
    )

    if isinstance(policy_overrides, dict):
        qq_runtime_context["access_policy"] = dict(policy_overrides)
    qq_module = QQModule(runtime_context=qq_runtime_context)

    allow_local_tools = True if not isinstance(policy_overrides, dict) else bool(
        policy_overrides.get("allow_local_tools", True))
    modules = [memory_module, web_module]
    if allow_local_tools:
        modules = [
            file_module,
            document_agent_module,
            image_module,
            memory_module,
            web_module,
            retrieval_module,
            system_utility_module,
            computer_use_module,
        ]
    if qq_module.runtime is not None:
        modules.append(qq_module)
    for module in modules:
        executors = module.executor_map()
        for manifest in module.manifests():
            registry.register(manifest, executors[manifest.tool_name])

    kernel = AgentKernel(
        config=agent_config,
        llm_client=llm_client,
        registry=registry,
        memory_store=memory_store,
        trace_store=trace_store,
        voice_adapter=voice_adapter,
    )
    kernel._reminder_store = reminder_store
    return kernel



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local agent kernel.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    args = parser.parse_args()

    kernel = build_kernel(args.config)

    reminder_store = getattr(kernel, "_reminder_store", None)
    if reminder_store is None:
        reminder_store = ReminderStore("data/reminders.sqlite3")
        kernel._reminder_store = reminder_store

    def _notify_reminder(reminder: dict) -> None:
        print(
            "[reminder-fired]",
            reminder.get("reminder_id"),
            reminder.get("when_iso"),
            reminder.get("message"),
        )

    def _dispatch_scheduled_task(task: dict) -> None:
        task_payload = task.get("task_payload") or {}
        instruction_text = str(task_payload.get("instruction_text") or "").strip()

        if not instruction_text:
            print(
                "[scheduled-task-fired-but-missing-instruction]",
                task.get("reminder_id"),
                task_payload,
            )
            return

        try:
            kernel.handle_scheduled_task(task=task)
        except Exception as exc:
            print(f"[scheduled-task-dispatch-error] {task.get('reminder_id')}: {exc}")

    scheduler = ReminderScheduler(
        reminder_store=reminder_store,
        notify_callback=_notify_reminder,
        dispatch_callback=_dispatch_scheduled_task,
        poll_interval_seconds=1.0,
    )
    scheduler.start()
    kernel._reminder_scheduler = scheduler
    print("Local agent is ready. Type 'exit' to quit.")

    while True:
        user_text = input("You> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break
        artifacts = kernel.handle_user_input(user_text)
        print(f"Agent> {artifacts.final_response}")
        if artifacts.tts_dispatched:
            print("Agent> TTS dispatched.")


if __name__ == "__main__":
    main()
