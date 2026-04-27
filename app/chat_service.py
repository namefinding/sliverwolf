from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Literal

from local_agent.app.chat_models import ChatTurnResult
from local_agent.app.context_tasks import ContextTaskManager
from local_agent.app.learning_service import RealTurnLearningService
from local_agent.app.scope_resolver import infer_scope_root
from local_agent.app.session_store import InMemorySessionStore, SessionState
from local_agent.app.turn_assembly import TurnAssemblyService
from local_agent.kernel.agent_kernel import AgentKernel
from local_agent.kernel.context_builder import ContextBuilder
from local_agent.memory.cold_archive import ColdArchiveService
from local_agent.memory.context_compressor import ContextCompressor
from local_agent.memory.warm_memory import WarmMemoryService
from local_agent.protocol.models import (
    FollowUpAssessment,
    LiveTurnState,
    Message,
    PendingTask,
    Role,
    TurnCompletionDecision,
)
from local_agent.protocol.models import FinalizedTurn, TurnResolution
from local_agent.runners import AgentTurnRunner, ChatTurnRunner, FollowUpTurnRunner
from local_agent.runners.base import RunnerContext


ChatMode = Literal["chat", "agent", "auto"]

class ChatService:
    _ARITHMETIC_PATTERN = re.compile(r"^\s*(-?\d+)\s*([+\-*/xX×÷])\s*(-?\d+)\s*(等于几|是多少|=?\s*)?$")
    _DATE_QUERY_PATTERN = re.compile(r"(今天|明天|后天|昨天|前天).*(几号|日期|星期|周几|什么日子)|今天是什么日子", flags=re.IGNORECASE)

    def __init__(
        self,
        session_store: InMemorySessionStore,
        kernel_factory: Callable[[str | None, dict | None], AgentKernel],
        *,
        configured_workspace: str | None = None,
        project_root: str | None = None,
        home_dir: str | Path | None = None,
    ) -> None:
        self.session_store = session_store
        self.kernel_factory = kernel_factory
        self.configured_workspace = configured_workspace
        self.project_root = project_root
        self.home_dir = Path(home_dir) if home_dir is not None else None
        self.context_compressor = ContextCompressor()
        self.chat_runner = ChatTurnRunner()
        self.agent_runner = AgentTurnRunner()
        self.follow_up_runner = FollowUpTurnRunner()
        self.turn_assembler = TurnAssemblyService()

    def resolve_turn_input(
            self,
            session: SessionState,
            *,
            raw_user_turn_text: str,
            turn_id: str | None = None,
    ) -> TurnResolution:
        kernel = session.kernel
        recent_messages = [
                              f"{message.role.value}: {message.content}"
                              for message in kernel.history
                              if message.role in {Role.SYSTEM, Role.USER, Role.ASSISTANT}
                          ][-6:]

        pending_task_summary = ""
        if session.pending_task is not None:
            pending_task_summary = "\n".join(
                part
                for part in (
                    f"summary={session.pending_task.summary}",
                    f"request={session.pending_task.original_user_request}",
                    f"missing_slots={', '.join(session.pending_task.missing_slots[:4])}" if session.pending_task.missing_slots else "",
                )
                if part
            )

        turn_type = "fresh_turn"

        active_task_summary = getattr(kernel, "active_task_summary", "") or ""
        context_task_board = ContextTaskManager.format_context_task_board(session)
        if context_task_board:
            active_task_summary = (
                f"{active_task_summary}\n{context_task_board}".strip()
                if active_task_summary
                else context_task_board
            )
        retrieved_topic_summary = ""

        if turn_type == "fresh_turn":
            active_task_summary = ""
            pending_task_summary = ""

        return TurnResolution(
            session_id=session.session_id,
            turn_id=turn_id,
            raw_user_turn_text=raw_user_turn_text,
            turn_type=turn_type,
            planner_visible_user_text=raw_user_turn_text,
            recent_context="\n".join(recent_messages),
            active_task_summary=active_task_summary,
            pending_task_summary=pending_task_summary,
            retrieved_topic_summary=retrieved_topic_summary,
            rationale="rule_bootstrap",
        )

    def handle_message(
            self,
            text: str,
            session_id: str | None = None,
            mode: ChatMode = "auto",
            scope_root: str | None = None,
            runtime_settings: dict | None = None,
            progress_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
    ) -> ChatTurnResult:
        print("[chat_service.handle_message:start]", {
            "session_id": session_id,
            "mode": mode,
            "text": text,
            "has_runtime_settings": isinstance(runtime_settings, dict),
        })

        resolved_session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"

        incoming_text = str(text or "").strip()
        resolution_turn_id: str | None = None

        finalized_turn_payload: dict[str, Any] | None = None
        live_turn_payload: dict[str, Any] | None = None
        if isinstance(runtime_settings, dict):
            maybe_payload = runtime_settings.get("finalized_turn")
            if isinstance(maybe_payload, dict):
                finalized_turn_payload = maybe_payload
            maybe_live_turn = runtime_settings.get("live_turn")
            if isinstance(maybe_live_turn, dict):
                live_turn_payload = maybe_live_turn

        if finalized_turn_payload is not None:
            incoming_text = str(finalized_turn_payload.get("raw_user_turn_text") or incoming_text).strip()
            maybe_turn_id = str(finalized_turn_payload.get("turn_id") or "").strip()
            resolution_turn_id = maybe_turn_id or None
        if mode == "auto":
            forced_mode = self._forced_mode_from_live_turn_payload(
                live_turn_payload=live_turn_payload,
                finalized_turn_payload=finalized_turn_payload,
            )
            if forced_mode in {"chat", "agent"}:
                mode = forced_mode

        resolved_scope_root = self._resolve_scope_root(incoming_text, scope_root)
        session = self.session_store.get_or_create(
            resolved_session_id,
            self.kernel_factory,
            scope_root=resolved_scope_root,
            runtime_settings=runtime_settings,
        )

        resolution = self.resolve_turn_input(
            session,
            raw_user_turn_text=incoming_text,
            turn_id=resolution_turn_id,
        )
        if resolution.turn_type == "fresh_turn":
            session.pending_follow_up_assessment = None

        session.kernel.active_task_summary = resolution.active_task_summary

        text = resolution.planner_visible_user_text

        detached_pending_for_new_request = False
        if mode == "auto" and session.pending_task is not None and session.pending_follow_up_assessment is None:
            pending_assessment = self.classify_follow_up(session.kernel, session.pending_task, text)
            if pending_assessment.action in {"resume", "resume_with_correction", "cancel"}:
                session.pending_follow_up_assessment = pending_assessment
            else:
                ContextTaskManager.remember_context_task(session, session.pending_task)
                self.session_store.set_pending_task(session.session_id, None, mode=session.last_mode)
                session.pending_task = None
                session.active_context_task = None
                detached_pending_for_new_request = True

        if mode == "auto" and not detached_pending_for_new_request and session.pending_task is None and session.context_tasks:
            latent_pending = self._activate_context_task_for_message(session, text)
            if latent_pending is not None:
                self.session_store.set_pending_task(session.session_id, latent_pending, mode="agent")
                session.pending_task = latent_pending

        if (
            mode == "auto"
            and session.pending_task is not None
            and session.pending_follow_up_assessment is not None
        ):
            result = self.follow_up_runner.run(
                RunnerContext(service=self, session=session, text=text, progress_callback=progress_callback)
            )
            self._update_context_task_registry(session, text, result)
            return result

        self.prepare_turn_context(session, text, capture_user_instruction=False)
        fast_local_result = self.try_fast_local_answer(session, text)
        if fast_local_result is not None:
            self._update_context_task_registry(session, text, fast_local_result)
            self.learn_from_real_turn(session, text, fast_local_result)
            print("[chat_service.handle_message:done]", {
                "session_id": session.session_id,
                "selected_mode": "fast_local",
            })
            return fast_local_result

        selected_mode = self._resolve_mode(session, text, mode)
        context = RunnerContext(service=self, session=session, text=text, progress_callback=progress_callback)
        if selected_mode == "agent":
            result = self.agent_runner.run(context)
        else:
            result = self.chat_runner.run(context)
        self._update_context_task_registry(session, text, result)
        self.learn_from_real_turn(session, text, result)
        print("[chat_service.handle_message:done]", {
            "session_id": session.session_id,
            "selected_mode": selected_mode,
        })
        return result

    @staticmethod
    def _forced_mode_from_live_turn_payload(
        *,
        live_turn_payload: dict[str, Any] | None,
        finalized_turn_payload: dict[str, Any] | None,
    ) -> str | None:
        live_turn_payload = live_turn_payload if isinstance(live_turn_payload, dict) else {}
        finalized_turn_payload = finalized_turn_payload if isinstance(finalized_turn_payload, dict) else {}
        forced_mode = str(live_turn_payload.get("forced_mode") or "").strip().lower()
        if forced_mode in {"chat", "agent"}:
            return forced_mode

        turn_kind = str(
            live_turn_payload.get("turn_kind")
            or finalized_turn_payload.get("turn_kind")
            or ""
        ).strip().lower()
        understood_task = str(
            live_turn_payload.get("understood_task")
            or finalized_turn_payload.get("understood_task")
            or ""
        ).strip()
        should_ack_task = bool(live_turn_payload.get("should_ack_task"))

        if turn_kind in {"execute_task", "memory_update", "instruction_update", "direct_reply"}:
            return "agent"
        if should_ack_task and understood_task:
            return "agent"
        if understood_task and turn_kind not in {"", "chat", "uncertain"}:
            return "agent"
        return None

    def should_run_in_background(
        self,
        *,
        text: str,
        session_id: str | None = None,
        mode: ChatMode = "auto",
        scope_root: str | None = None,
        runtime_settings: dict | None = None,
    ) -> bool:
        resolved_session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
        resolved_scope_root = self._resolve_scope_root(text, scope_root)
        session = self.session_store.get_or_create(
            resolved_session_id,
            self.kernel_factory,
            scope_root=resolved_scope_root,
            runtime_settings=runtime_settings,
        )
        if mode == "chat":
            return False
        if session.pending_task is not None:
            return False
        if mode == "agent":
            return True
        selected_mode = self._resolve_mode(session, text, mode)
        return selected_mode == "agent"

    def capture_live_turn_event(
        self,
        *,
        session_id: str,
        text: str,
        channel: str = "unknown",
        attachment_refs: list[str] | None = None,
        metadata: dict | None = None,
    ) -> LiveTurnState:
        return self.turn_assembler.observe_event(
            session_id=session_id,
            channel=channel,
            text=text,
            attachment_refs=attachment_refs,
            metadata=metadata,
        )

    def get_live_turn_state(self, session_id: str) -> LiveTurnState | None:
        return self.turn_assembler.get_state(session_id)

    def capture_live_typing(
        self,
        *,
        session_id: str,
        channel: str = "unknown",
        active: bool,
        metadata: dict | None = None,
        scope_root: str | None = None,
        runtime_settings: dict | None = None,
    ) -> LiveTurnState:
        session = self.session_store.get_or_create(
            session_id,
            self.kernel_factory,
            scope_root=scope_root,
            runtime_settings=runtime_settings,
        )
        hold_ms = int(getattr(getattr(session.kernel, "config", None), "live_turn_typing_hold_ms", 10000) or 10000)
        return self.turn_assembler.observe_typing(
            session_id=session_id,
            channel=channel,
            active=active,
            metadata=metadata,
            hold_ms=hold_ms,
        )

    def assess_live_turn(
        self,
        *,
        session_id: str,
        scope_root: str | None = None,
        runtime_settings: dict | None = None,
        typing_active: bool = False,
    ) -> TurnCompletionDecision:
        state = self.turn_assembler.get_state(session_id)
        if state is None:
            return TurnCompletionDecision(finalize=False, confidence=0.0, wait_ms=0, reason="no_live_turn", source="rule")

        session = self.session_store.get_or_create(
            session_id,
            self.kernel_factory,
            scope_root=scope_root,
            runtime_settings=runtime_settings,
        )
        kernel = session.kernel
        config = getattr(kernel, "config", None)
        quiet_window_ms = int(getattr(config, "live_turn_quiet_window_ms", 650) or 650)
        max_wait_ms = int(getattr(config, "live_turn_max_wait_ms", 12000) or 12000)
        fragment_max_wait_ms = int(getattr(config, "live_turn_fragment_max_wait_ms", 15000) or 15000)
        incomplete_extra_ms = int(getattr(config, "live_turn_incomplete_extra_ms", 450) or 450)
        attachment_extra_ms = int(getattr(config, "live_turn_attachment_extra_ms", 450) or 450)
        use_llm_judge = bool(getattr(config, "live_turn_use_llm_judge", True))
        recent_messages = [
            f"{message.role.value}: {message.content}"
            for message in kernel.history
            if message.role in {Role.SYSTEM, Role.USER, Role.ASSISTANT}
        ][-6:]
        pending_task_summary = ""
        if session.pending_task is not None:
            pending_task_summary = "\n".join(
                part
                for part in (
                    f"summary={session.pending_task.summary}",
                    f"request={session.pending_task.original_user_request}",
                    f"missing_slots={', '.join(session.pending_task.missing_slots[:4])}" if session.pending_task.missing_slots else "",
                )
                if part
            )
        return self.turn_assembler.decide_completion(
            state=state,
            quiet_window_ms=quiet_window_ms,
            max_wait_ms=max_wait_ms,
            fragment_max_wait_ms=fragment_max_wait_ms,
            incomplete_extra_ms=incomplete_extra_ms,
            attachment_extra_ms=attachment_extra_ms,
            typing_active=typing_active,
            llm_client=getattr(kernel, "llm_client", None),
            recent_context="\n".join(recent_messages),
            hot_context_summary=getattr(kernel, "hot_context_summary", ""),
            pending_task_summary=pending_task_summary,
            persona_name=str(getattr(config, "persona_name", "") or ""),
            use_llm_judge=use_llm_judge,
        )

    def finalize_live_turn(
            self,
            session_id: str,
            *,
            expected_version: int | None = None,
            finalize_reason: str = "",
    ) -> FinalizedTurn | None:
        return self.turn_assembler.finalize_turn(
            session_id,
            expected_version=expected_version,
            finalize_reason=finalize_reason,
        )

    def mark_live_turn_followup_prompted(self, session_id: str, *, followup_text: str) -> None:
        self.turn_assembler.mark_followup_prompt_sent(session_id, followup_text=followup_text)

    def learn_live_turn_habit(
        self,
        *,
        session_id: str,
        raw_user_turn_text: str,
        decision: TurnCompletionDecision,
    ) -> None:
        normalized = " ".join(str(raw_user_turn_text or "").split()).strip()
        if len(normalized) < 6:
            return
        if not decision.ask_followup and decision.reason not in {"fragment_wait_extended", "typing_active"}:
            return

        session = self.session_store.get(session_id)
        if session is None:
            return
        memory_store = getattr(session.kernel, "memory_store", None)
        if memory_store is None:
            return

        try:
            warm_memory = WarmMemoryService(memory_store)
            warm_memory.remember_workflow_lesson(
                "该用户在任务型对话里经常分多条补充细节。涉及日期、样式、内容、范围等参数时，不要过早收口；优先继续等待，必要时先问一句是否还有补充。",
                scope="user",
                tags=["live_turn", "turn_completion", "fragmented_input", "followup_collection"],
                importance=0.93,
            )
            trace_store = getattr(session.kernel, "trace_store", None)
            if trace_store is not None:
                trace_store.append(
                    "live_turn_learning",
                    {
                        "session_id": session_id,
                        "raw_user_turn_text": normalized,
                        "reason": decision.reason,
                        "confidence": decision.confidence,
                        "ask_followup": decision.ask_followup,
                    },
                )
        except Exception:
            return

    @staticmethod
    def _normalize_text(text: str) -> tuple[str, str]:
        lowered = text.lower().strip()
        compact = re.sub(r"[\s`'\"，。！？?.!?:：；（）()\[\]{}]+", "", lowered)
        return lowered, compact

    def _resolve_mode(self, session: SessionState, text: str, mode: ChatMode) -> str:
        if mode in {"chat", "agent"}:
            return mode
        return "agent"

    def _resolve_scope_root(self, text: str, explicit_scope_root: str | None) -> str | None:
        if explicit_scope_root:
            return explicit_scope_root
        return infer_scope_root(
            text,
            configured_workspace=self.configured_workspace,
            project_root=self.project_root,
            home_dir=self.home_dir,
        )

    def handle_scheduled_task(
            self,
            *,
            task: dict,
    ) -> None:
        session_id = str(task.get("session_id") or "").strip()
        if not session_id:
            print("[scheduled-task] missing session_id:", task)
            return

        runtime_settings = None
        session = self.session_store.get(session_id)
        if session is not None:
            runtime_settings = getattr(session, "runtime_settings", None)

        if not runtime_settings:
            task_payload = task.get("task_payload") or {}
            if not isinstance(task_payload, dict):
                task_payload = {}
            channel_runtime = task_payload.get("channel_runtime")
            task_channel = str(task.get("channel") or "").strip() or "onebot_v11"
            if isinstance(channel_runtime, dict) and channel_runtime:
                runtime_settings = {
                    "channel_runtime": channel_runtime,
                    "channel": {
                        "name": task_channel,
                    },
                }

        entry = self.session_store.get_or_create(
            session_id,
            self.kernel_factory,
            scope_root=None,
            runtime_settings=runtime_settings,
        )
        kernel = entry.kernel
        trace_store = getattr(kernel, "trace_store", None)
        if trace_store is not None:
            try:
                trace_store.append(
                    "scheduled_task_dispatch_start",
                    {
                        "session_id": session_id,
                        "task": {
                            "reminder_id": str(task.get("reminder_id") or "").strip() or None,
                            "task_type": str(task.get("task_type") or "").strip() or None,
                            "channel": str(task.get("channel") or "").strip() or None,
                            "has_runtime_settings": bool(runtime_settings),
                        },
                    },
                )
            except Exception:
                pass

        try:
            artifacts = kernel.handle_scheduled_task(task=task)
            refreshed_session = self.session_store.get(session_id)
            if refreshed_session is not None and artifacts is not None:
                refreshed_session.touch(refreshed_session.last_mode)
            if trace_store is not None:
                try:
                    trace_store.append(
                        "scheduled_task_dispatch_success",
                        {
                            "session_id": session_id,
                            "task": {
                                "reminder_id": str(task.get("reminder_id") or "").strip() or None,
                                "task_type": str(task.get("task_type") or "").strip() or None,
                            },
                            "has_artifacts": artifacts is not None,
                        },
                    )
                except Exception:
                    pass
        except Exception as exc:
            if trace_store is not None:
                try:
                    trace_store.append(
                        "scheduled_task_dispatch_error",
                        {
                            "session_id": session_id,
                            "task": {
                                "reminder_id": str(task.get("reminder_id") or "").strip() or None,
                                "task_type": str(task.get("task_type") or "").strip() or None,
                            },
                            "error": str(exc),
                        },
                    )
                except Exception:
                    pass
            print(f"[scheduled-task] dispatch error for {session_id}: {exc}")

    @staticmethod
    def prune_chat_history(session: SessionState) -> list[Message]:
        return ContextBuilder.build_prompt_messages(
            session.kernel.history,
            session_summary=session.hot_context.session_summary,
            active_task_summary=session.hot_context.active_task_summary,
            warm_memory_summary=getattr(session.kernel, "warm_memory_summary", ""),
            learning_memory_summary=getattr(session.kernel, "learning_memory_summary", ""),
            cold_memory_summary=getattr(session.kernel, "cold_memory_summary", ""),
        )

    def refresh_hot_context(self, session: SessionState) -> None:
        if not hasattr(session.kernel, "history"):
            session.hot_context.active_task_summary = self.context_compressor.build_active_task_summary(session.pending_task)
            session.hot_context.touch()
            return
        previous_summary = session.hot_context.session_summary
        (
            session.hot_context.session_summary,
            session.hot_context.summarized_visible_count,
        ) = self.context_compressor.update_session_summary(
            session.kernel.history,
            previous_summary=session.hot_context.session_summary,
            summarized_visible_count=session.hot_context.summarized_visible_count,
        )
        session.hot_context.active_task_summary = self.context_compressor.build_active_task_summary(session.pending_task)
        self._archive_hot_context_if_needed(session, previous_summary)
        session.hot_context.touch()
        session.kernel.hot_context_summary = session.hot_context.session_summary
        session.kernel.active_task_summary = session.hot_context.active_task_summary

    def prepare_turn_context(
        self,
        session: SessionState,
        user_text: str,
        *,
        capture_user_instruction: bool = True,
    ) -> None:
        self.refresh_hot_context(session)
        memory_store = getattr(session.kernel, "memory_store", None)
        if memory_store is None:
            session.kernel.user_memory_summary = ""
            session.kernel.learning_memory_summary = ""
            session.kernel.warm_memory_summary = ""
            session.kernel.cold_memory_summary = ""
            return

        warm_memory = WarmMemoryService(memory_store)
        user_recalled = warm_memory.recall_user_memory_for_text(user_text)
        learning_recalled = warm_memory.recall_learning_memory_for_text(user_text)
        session.kernel.user_memory_summary = warm_memory.format_for_prompt(
            user_recalled,
            title="相关用户记忆",
        )
        session.kernel.learning_memory_summary = warm_memory.format_for_prompt(
            learning_recalled,
            title="相关执行经验",
        )
        session.kernel.warm_memory_summary = session.kernel.user_memory_summary
        if capture_user_instruction:
            memory_candidate_intent = None
            try:
                memory_candidate_intent = session.kernel.intent_service.analyze_memory_candidate(
                    user_text,
                    recent_context="\n".join(
                        f"{message.role.value}: {message.content}"
                        for message in session.kernel.history
                        if message.role in {Role.SYSTEM, Role.USER, Role.ASSISTANT}
                    )[-1200:],
                    hot_context_summary=session.kernel.hot_context_summary,
                    warm_memory_summary=session.kernel.warm_memory_summary,
                    learning_memory_summary=getattr(session.kernel, "learning_memory_summary", ""),
                    cold_memory_summary=session.kernel.cold_memory_summary,
                    active_task_summary=session.kernel.active_task_summary,
                )
            except Exception:
                memory_candidate_intent = None

            if memory_candidate_intent is not None:
                remembered = warm_memory.remember_memory_candidate_intent(memory_candidate_intent)
                if remembered is None:
                    instruction_intent = session.kernel.intent_service._derive_instruction_intent(memory_candidate_intent)
                    if bool(getattr(instruction_intent, "persist_memory", False)):
                        warm_memory.remember_instruction_intent(instruction_intent)
                refreshed_user = warm_memory.recall_user_memory_for_text(user_text)
                refreshed_learning = warm_memory.recall_learning_memory_for_text(user_text)
                session.kernel.user_memory_summary = warm_memory.format_for_prompt(
                    refreshed_user,
                    title="相关用户记忆",
                )
                session.kernel.learning_memory_summary = warm_memory.format_for_prompt(
                    refreshed_learning,
                    title="相关执行经验",
                )
                session.kernel.warm_memory_summary = session.kernel.user_memory_summary
            else:
                warm_memory.maybe_capture_user_instruction(user_text)

        cold_archive = ColdArchiveService(memory_store)
        if cold_archive.should_recall(user_text):
            cold_records = cold_archive.recall_for_text(user_text)
            session.kernel.cold_memory_summary = cold_archive.format_for_prompt(cold_records)
        else:
            session.kernel.cold_memory_summary = ""

    def try_fast_local_answer(self, session: SessionState, user_text: str) -> ChatTurnResult | None:
        response = self._build_fast_local_response(session, user_text)
        if response is None:
            return None
        session.kernel.history.append(Message(role=Role.USER, content=user_text))
        session.kernel.history.append(Message(role=Role.ASSISTANT, content=response))
        self.refresh_hot_context(session)
        tts_dispatched = self.dispatch_tts(session, response)
        session.touch("agent")
        return ChatTurnResult(
            session_id=session.session_id,
            mode="agent",
            response=response,
            speech_text=response,
            tts_dispatched=tts_dispatched,
            used_agent=True,
            scope_root=session.scope_root,
            metadata={
                "trace_id": "",
                "execution_summary": {
                    "task_status": "completed",
                    "completed_by": "fast_local_answer",
                },
            },
        )

    def record_external_event(self, session_id: str, event_text: str) -> bool:
        session = self.session_store.get(session_id)
        if session is None:
            return False
        normalized = str(event_text or "").strip()
        if not normalized:
            return False
        session.kernel.history.append(Message(role=Role.SYSTEM, content=f"[external_event] {normalized}"))
        self.refresh_hot_context(session)
        session.touch(session.last_mode)
        return True

    def learn_from_real_turn(self, session: SessionState, user_text: str, turn_result: ChatTurnResult) -> None:
        memory_store = getattr(session.kernel, "memory_store", None)
        if memory_store is None:
            return
        try:
            learning = RealTurnLearningService(
                memory_store,
                trace_store=getattr(session.kernel, "trace_store", None),
                llm_client=getattr(session.kernel, "llm_client", None),
                reflection_enabled=bool(getattr(session.kernel.config, "learning_reflection_enabled", True)),
            )
            learned = learning.learn_from_turn(
                user_text=user_text,
                turn_result=turn_result,
                scope="user",
            )
            if learned:
                warm_memory = WarmMemoryService(memory_store)
                user_recalled = warm_memory.recall_user_memory_for_text(user_text)
                learning_recalled = warm_memory.recall_learning_memory_for_text(user_text)
                session.kernel.user_memory_summary = warm_memory.format_for_prompt(
                    user_recalled,
                    title="相关用户记忆",
                )
                session.kernel.learning_memory_summary = warm_memory.format_for_prompt(
                    learning_recalled,
                    title="相关执行经验",
                )
                session.kernel.warm_memory_summary = session.kernel.user_memory_summary
        except Exception:
            return

    def _archive_hot_context_if_needed(self, session: SessionState, previous_summary: str) -> None:
        memory_store = getattr(session.kernel, "memory_store", None)
        if memory_store is None:
            return

        summary = session.hot_context.session_summary
        if not summary or summary == previous_summary:
            return
        if len(summary) < 24:
            return

        visible_message_count = len(
            [message for message in session.kernel.history if message.role in {Role.USER, Role.ASSISTANT}]
        )
        minimum_count = self.context_compressor.keep_recent_messages + 2
        if visible_message_count < minimum_count:
            return
        if visible_message_count < session.hot_context.last_archived_message_count + 4:
            return

        digest = hashlib.sha1(summary.encode("utf-8")).hexdigest()
        if digest == session.hot_context.last_archived_digest:
            return

        cold_archive = ColdArchiveService(memory_store)
        archived_digest = cold_archive.archive_session_summary(
            summary,
            scope="session",
            tags=["hot-context", "auto-archive"],
        )
        session.hot_context.last_archived_digest = archived_digest
        session.hot_context.last_archived_message_count = visible_message_count

    @staticmethod
    def classify_follow_up(kernel: AgentKernel, pending_task: PendingTask, text: str) -> FollowUpAssessment:
        return ContextTaskManager.classify_follow_up(kernel, pending_task, text)

    @staticmethod
    def _resolve_selection_follow_up(pending_task: PendingTask, text: str) -> FollowUpAssessment | None:
        return ContextTaskManager.resolve_selection_follow_up(pending_task, text)

    @staticmethod
    def merge_pending_request(pending_task: PendingTask, text: str) -> str:
        return ContextTaskManager.merge_pending_request(pending_task, text)

    @staticmethod
    def _activate_context_task_for_message(session: SessionState, text: str) -> PendingTask | None:
        return ContextTaskManager.activate_context_task_for_message(session, text)

    @staticmethod
    def _build_context_task_from_result(user_text: str, turn_result: ChatTurnResult) -> PendingTask | None:
        return ContextTaskManager.build_context_task_from_result(user_text, turn_result)

    @staticmethod
    def _update_context_task_registry(session: SessionState, user_text: str, turn_result: ChatTurnResult) -> None:
        ContextTaskManager.update_context_task_registry(session, user_text, turn_result)

    @staticmethod
    def dispatch_tts(session: SessionState, response: str) -> bool:
        runtime_settings = session.runtime_settings or {}
        channel_settings = runtime_settings.get("channel") if isinstance(runtime_settings.get("channel"), dict) else {}
        channel_name = str(channel_settings.get("name", "")).strip()
        if channel_name == "onebot_v11":
            return False
        try:
            return session.kernel.voice_adapter.dispatch(response)
        except Exception:
            return False

    def _build_fast_local_response(self, session: SessionState, user_text: str) -> str | None:
        normalized = " ".join(str(user_text or "").split()).strip()
        if not normalized:
            return None

        date_response = self._try_answer_local_date_question(normalized)
        if date_response:
            return date_response

        arithmetic_response = self._try_answer_simple_arithmetic(normalized)
        if arithmetic_response:
            return arithmetic_response

        memory_response = self._try_answer_from_warm_memory(session, normalized)
        if memory_response:
            return memory_response

        return None

    @classmethod
    def _try_answer_local_date_question(cls, user_text: str) -> str | None:
        lowered = user_text.lower()
        if not cls._DATE_QUERY_PATTERN.search(user_text) and not any(token in lowered for token in ("今天几号", "今天星期", "今天周几", "今天是什么日子")):
            return None
        now_local = datetime.now().astimezone()
        target = now_local
        if "明天" in user_text:
            target = now_local + timedelta(days=1)
        elif "后天" in user_text:
            target = now_local + timedelta(days=2)
        elif "昨天" in user_text:
            target = now_local - timedelta(days=1)
        elif "前天" in user_text:
            target = now_local - timedelta(days=2)
        weekdays = "一二三四五六日"
        weekday_text = weekdays[target.weekday()]
        label = "今天"
        if "明天" in user_text:
            label = "明天"
        elif "后天" in user_text:
            label = "后天"
        elif "昨天" in user_text:
            label = "昨天"
        elif "前天" in user_text:
            label = "前天"
        return f"{label}是{target.year}年{target.month}月{target.day}日，星期{weekday_text}。"

    @classmethod
    def _try_answer_simple_arithmetic(cls, user_text: str) -> str | None:
        match = cls._ARITHMETIC_PATTERN.match(user_text)
        if not match:
            return None
        left = int(match.group(1))
        operator = match.group(2)
        right = int(match.group(3))
        if operator in {"x", "X", "×"}:
            result = left * right
        elif operator == "+":
            result = left + right
        elif operator == "-":
            result = left - right
        elif operator in {"/", "÷"}:
            if right == 0:
                return "这个不能直接算，除数不能是 0。"
            result = left / right
            if float(result).is_integer():
                result = int(result)
        else:
            return None
        return f"{left}{operator}{right}等于{result}。"

    @staticmethod
    def _try_answer_from_warm_memory(session: SessionState, user_text: str) -> str | None:
        lowered = user_text.lower()
        memory_store = getattr(session.kernel, "memory_store", None)
        if memory_store is None:
            return None
        warm_memory = WarmMemoryService(memory_store)
        recalled = warm_memory.recall_for_text(user_text, limit=6)
        if not recalled:
            return None

        if "生日" in user_text:
            for item in recalled:
                tags = {tag.lower() for tag in item.tags}
                if item.memory_type == "user_fact" and ("user.birthday" in tags or "生日" in item.content):
                    return f"我记得，{item.content}"
        if any(token in lowered for token in ("怎么称呼我", "叫我什么", "称呼我")):
            for item in recalled:
                tags = {tag.lower() for tag in item.tags}
                if item.memory_type == "alias" and ("user.preferred_name" in tags or "称呼" in item.content):
                    return f"我记着，{item.content}"
        return None
