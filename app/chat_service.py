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
from local_agent.memory.warm_memory import (
    MEMORY_LAYER_LONG_TERM_INSTRUCTION,
    MEMORY_LAYER_TASK_EXPERIENCE,
    MEMORY_LAYER_USER_PREFERENCE,
    WarmMemoryService,
)
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
    _ARITHMETIC_PATTERN = re.compile(r"^\s*(-?\d+)\s*([+\-*/xX×÷])\s*(-?\d+)\s*(等于几|是多少|=)?\s*$")
    _DATE_QUERY_PATTERN = re.compile(
        r"(今天|明天|后天|昨天|前天).*(几号|日期|星期|周几|什么日子)|what.*(date|day)",
        flags=re.IGNORECASE,
    )

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
            runtime_settings: dict | None = None,
    ) -> TurnResolution:
        kernel = session.kernel
        recent_messages, has_prior_context = self._build_recent_conversation_window(
            kernel.history,
            raw_user_turn_text=raw_user_turn_text,
            runtime_settings=runtime_settings if isinstance(runtime_settings, dict) else session.runtime_settings,
            recent_turn_texts=session.recent_turn_texts,
            recent_artifacts=session.recent_artifacts,
        )

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

        turn_type = "contextual_turn" if has_prior_context else "fresh_turn"

        active_task_summary = getattr(kernel, "active_task_summary", "") or ""
        context_task_board = ContextTaskManager.format_context_task_board(session)
        if context_task_board:
            active_task_summary = (
                f"{active_task_summary}\n{context_task_board}".strip()
                if active_task_summary
                else context_task_board
            )
        retrieved_topic_summary = ""

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
        self._merge_recent_artifacts_into_runtime_settings(session)
        llm_metrics_start = len(getattr(getattr(session.kernel, "llm_client", None), "last_call_metrics", []) or [])

        resolution = self.resolve_turn_input(
            session,
            raw_user_turn_text=incoming_text,
            turn_id=resolution_turn_id,
            runtime_settings=session.runtime_settings,
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
            self._update_recent_artifacts(session, text, result)
            self._update_context_task_registry(session, text, result)
            self._attach_llm_metrics_to_result(session, result, llm_metrics_start)
            return result

        self.prepare_turn_context(session, text, capture_user_instruction=False)
        fast_local_result = self.try_fast_local_answer(session, text)
        if fast_local_result is not None:
            self._update_recent_artifacts(session, text, fast_local_result)
            self._update_context_task_registry(session, text, fast_local_result)
            self.learn_from_real_turn(session, text, fast_local_result)
            self._attach_llm_metrics_to_result(session, fast_local_result, llm_metrics_start)
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
        self._update_recent_artifacts(session, text, result)
        self._update_context_task_registry(session, text, result)
        self.learn_from_real_turn(session, text, result)
        self._attach_llm_metrics_to_result(session, result, llm_metrics_start)
        print("[chat_service.handle_message:done]", {
            "session_id": session.session_id,
            "selected_mode": selected_mode,
        })
        return result

    @classmethod
    def _build_recent_conversation_window(
        cls,
        history: list[Message],
        *,
        raw_user_turn_text: str,
        runtime_settings: dict | None = None,
        recent_turn_texts: list[str] | None = None,
        recent_artifacts: list[dict] | None = None,
        limit: int = 10,
    ) -> tuple[list[str], bool]:
        runtime_settings = runtime_settings if isinstance(runtime_settings, dict) else {}
        current_text = str(raw_user_turn_text or "").strip()
        entries: list[tuple[str, str, bool]] = []

        for message in history or []:
            if message.role not in {Role.USER, Role.ASSISTANT}:
                continue
            content = str(message.content or "").strip()
            if content:
                entries.append((message.role.value, content, True))

        for item in recent_turn_texts or []:
            text = str(item or "").strip()
            if text:
                entries.append(("user", text, cls._normalize_context_text(text) != cls._normalize_context_text(current_text)))

        channel_runtime = runtime_settings.get("channel_runtime")
        channel_runtime = channel_runtime if isinstance(channel_runtime, dict) else {}
        for item in channel_runtime.get("recent_user_messages") or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if text:
                entries.append(("user", text, cls._normalize_context_text(text) != cls._normalize_context_text(current_text)))

        for item in channel_runtime.get("group_context_messages") or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            sender = str(item.get("sender_name") or item.get("sender_id") or "").strip()
            role = str(item.get("role") or "").strip()
            label = sender or role or "群聊成员"
            entries.append(
                (
                    "system",
                    f"group_context {label}: {text}",
                    cls._normalize_context_text(text) != cls._normalize_context_text(current_text),
                )
            )

        finalized_turn = runtime_settings.get("finalized_turn")
        finalized_turn = finalized_turn if isinstance(finalized_turn, dict) else {}
        finalized_segments = []
        if isinstance(channel_runtime.get("finalized_turn_segments"), list):
            finalized_segments.extend(channel_runtime.get("finalized_turn_segments") or [])
        if isinstance(finalized_turn.get("message_segments"), list):
            finalized_segments.extend(finalized_turn.get("message_segments") or [])
        for item in finalized_segments:
            text = str(item or "").strip()
            if text:
                entries.append(("user", text, cls._normalize_context_text(text) != cls._normalize_context_text(current_text)))

        artifact_lines = cls._format_recent_artifact_lines(recent_artifacts or [], limit=4)
        for line in artifact_lines:
            entries.append(("system", line, True))

        if current_text:
            entries.append(("user", current_text, False))

        deduped: list[tuple[str, str, bool]] = []
        seen: set[tuple[str, str]] = set()
        for role, text, is_prior in entries:
            key = (role, cls._normalize_context_text(text))
            if not key[1] or key in seen:
                continue
            seen.add(key)
            deduped.append((role, text, is_prior))

        window = deduped[-max(1, limit):]
        has_prior_context = any(is_prior for _, _, is_prior in window)
        return [f"{role}: {text}" for role, text, _ in window], has_prior_context

    @staticmethod
    def _clip_context_text(text: str, *, limit: int = 220) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(0, limit - 3)] + "..."

    @classmethod
    def _format_recent_artifact_lines(cls, artifacts: list[dict], *, limit: int = 4) -> list[str]:
        lines: list[str] = []
        for item in list(artifacts or [])[-limit:]:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "artifact").strip()
            role = str(item.get("role") or "").strip()
            path = str(item.get("path") or "").strip()
            title = str(item.get("title") or "").strip()
            trace_id = str(item.get("trace_id") or "").strip()
            if not path and not title:
                continue
            parts = [f"kind={kind}"]
            if role:
                parts.append(f"role={role}")
            if title:
                parts.append(f"title={title}")
            if path:
                parts.append(f"path={path}")
            if trace_id:
                parts.append(f"trace_id={trace_id}")
            lines.append("recent_artifact: " + "; ".join(parts))
        return lines

    @classmethod
    def _format_channel_context_for_chat(cls, runtime_settings: dict | None, *, limit: int = 12) -> str:
        runtime_settings = runtime_settings if isinstance(runtime_settings, dict) else {}
        channel_runtime = runtime_settings.get("channel_runtime")
        channel_runtime = channel_runtime if isinstance(channel_runtime, dict) else {}
        current_target = channel_runtime.get("current_target")
        current_target = current_target if isinstance(current_target, dict) else {}
        is_group = str(current_target.get("message_type") or "").strip() == "group"
        group_context = channel_runtime.get("group_context_messages")
        if not is_group or not isinstance(group_context, list):
            return ""
        group_reply_policy = str(channel_runtime.get("group_reply_policy") or "").strip()
        group_context_review = bool(channel_runtime.get("group_context_review"))

        lines: list[str] = []
        for item in group_context[-max(1, limit):]:
            if not isinstance(item, dict):
                continue
            text = cls._clip_context_text(str(item.get("text") or ""), limit=260)
            if not text:
                continue
            sender = str(item.get("sender_name") or item.get("sender_id") or "").strip()
            role = str(item.get("role") or "").strip()
            label = sender or role or "群聊成员"
            created_at = str(item.get("created_at") or "").strip()
            prefix = f"- {label}"
            if created_at:
                prefix += f" @ {created_at}"
            lines.append(f"{prefix}: {text}")
        if not lines:
            return ""
        optional_reply = ""
        if group_context_review or group_reply_policy == "optional":
            optional_reply = "这是一次看群回复：如果这段群聊不需要你插话，请只输出 __NO_REPLY__。\n"
        return (
            "最近 QQ 群聊上下文：\n"
            + optional_reply
            + "当最新群消息提到或叫到助手时，结合这段最近群聊记录理解短句、代词和省略表达。"
            "只回应最新用户这一轮，不要引入群聊记录里没有出现的话题。\n"
            + "\n".join(lines)
        )

    @staticmethod
    def _normalize_context_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip().lower()

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
        recent_messages, has_prior_context = self._build_recent_conversation_window(
            kernel.history,
            raw_user_turn_text=state.raw_user_turn_text,
            runtime_settings=session.runtime_settings,
            recent_turn_texts=session.recent_turn_texts,
            recent_artifacts=session.recent_artifacts,
        )
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
            has_prior_context=has_prior_context,
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
                "For fragmented task-like chat turns, keep collecting supplemental details before execution; ask a brief follow-up when date, style, content, or scope may still be incomplete.",
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
        compact = re.sub(r"[\s`'\"“”‘’「」『』，。！？.!?:：；、（）()\[\]{}]+", "", lowered)
        return lowered, compact

    @staticmethod
    def _attach_llm_metrics_to_result(session: SessionState, result: ChatTurnResult, metrics_start: int) -> None:
        llm_client = getattr(session.kernel, "llm_client", None)
        metrics = list(getattr(llm_client, "last_call_metrics", []) or [])
        if metrics_start < 0:
            metrics_start = 0
        calls = [item for item in metrics[metrics_start:] if isinstance(item, dict)]
        if not calls:
            return
        total_tokens = 0
        total_elapsed_ms = 0.0
        for item in calls:
            usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
            try:
                total_tokens += int(usage.get("total_tokens") or 0)
            except (TypeError, ValueError):
                pass
            try:
                total_elapsed_ms += float(item.get("elapsed_ms") or 0.0)
            except (TypeError, ValueError):
                pass
        metadata = dict(result.metadata or {})
        metadata["llm_call_metrics"] = {
            "call_count": len(calls),
            "total_tokens": total_tokens,
            "total_elapsed_ms": round(total_elapsed_ms, 2),
            "calls": calls,
        }
        result.metadata = metadata
        trace_store = getattr(session.kernel, "trace_store", None)
        if trace_store is not None:
            try:
                trace_store.append(
                    "llm_call_metrics",
                    {
                        "session_id": session.session_id,
                        "mode": result.mode,
                        "used_agent": result.used_agent,
                        **metadata["llm_call_metrics"],
                    },
                )
            except Exception:
                pass

    def _resolve_mode(self, session: SessionState, text: str, mode: ChatMode) -> str:
        if mode in {"chat", "agent"}:
            return mode
        route_decision = self._classify_turn_route(session, text)
        if isinstance(route_decision, dict):
            selected_mode = str(route_decision.get("mode") or "").strip().lower()
            if selected_mode in {"chat", "agent"}:
                return selected_mode
        if self._looks_like_lightweight_chat(text):
            return "chat"
        return "agent"

    def _classify_turn_route(self, session: SessionState, text: str) -> dict[str, Any] | None:
        signature = self._route_signature(session, text)
        cached = session.last_route_decision if isinstance(session.last_route_decision, dict) else None
        if cached is not None and cached.get("signature") == signature:
            return cached

        llm_client = getattr(session.kernel, "llm_client", None)
        classifier = getattr(llm_client, "classify_turn_route", None)
        if not callable(classifier):
            return None

        recent_messages, _ = self._build_recent_conversation_window(
            getattr(session.kernel, "history", []) or [],
            raw_user_turn_text=text,
            runtime_settings=session.runtime_settings,
            recent_turn_texts=session.recent_turn_texts,
            recent_artifacts=session.recent_artifacts,
            limit=8,
        )
        try:
            decision = classifier(
                user_text=text,
                recent_context="\n".join(recent_messages),
                hot_context_summary=str(getattr(session.kernel, "hot_context_summary", "") or ""),
                warm_memory_summary=str(getattr(session.kernel, "warm_memory_summary", "") or ""),
                cold_memory_summary=str(getattr(session.kernel, "cold_memory_summary", "") or ""),
                active_task_summary=str(getattr(session.kernel, "active_task_summary", "") or ""),
            )
        except Exception as exc:  # noqa: BLE001
            decision = {
                "mode": "",
                "turn_kind": "route_classifier_error",
                "confidence": 0.0,
                "rationale": str(exc)[:240],
                "source": "route_classifier_error",
            }

        if not isinstance(decision, dict):
            return None
        normalized_mode = str(decision.get("mode") or "").strip().lower()
        try:
            confidence = max(0.0, min(1.0, float(decision.get("confidence", 0.0) or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        if normalized_mode not in {"chat", "agent"} or confidence < 0.45:
            return None

        route_payload = {
            "signature": signature,
            "text": text,
            "mode": normalized_mode,
            "confidence": confidence,
            "turn_kind": str(decision.get("turn_kind", "") or ""),
            "rationale": str(decision.get("rationale", "") or ""),
            "source": str(decision.get("source", "llm_route_classifier") or "llm_route_classifier"),
        }
        session.last_route_decision = route_payload
        trace_store = getattr(session.kernel, "trace_store", None)
        if trace_store is not None:
            trace_store.append(
                "turn_route_decision",
                {
                    "session_id": session.session_id,
                    "text": text,
                    "mode": route_payload["mode"],
                    "confidence": route_payload["confidence"],
                    "turn_kind": route_payload["turn_kind"],
                    "rationale": route_payload["rationale"],
                    "source": route_payload["source"],
                },
            )
        return route_payload

    @staticmethod
    def _route_signature(session: SessionState, text: str) -> str:
        raw = "\n".join(
            [
                str(text or "").strip(),
                str(len(getattr(session.kernel, "history", []) or [])),
                str(getattr(session, "last_finalized_turn_id", "") or ""),
                str(bool(getattr(session, "pending_task", None))),
                "|".join(str(item.get("path") or item.get("title") or "") for item in (getattr(session, "recent_artifacts", []) or [])[-3:] if isinstance(item, dict)),
            ]
        )
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

    @classmethod
    def _looks_like_lightweight_chat(cls, text: str) -> bool:
        lowered, compact = cls._normalize_text(text)
        if not compact or len(compact) > 24:
            return False
        task_markers = (
            "帮我",
            "查",
            "搜",
            "搜索",
            "看看",
            "读取",
            "打开",
            "创建",
            "生成",
            "写",
            "保存",
            "发给",
            "提醒",
            "记一下",
            "记住",
            "还记得",
            "生日",
            "文件",
            "文档",
            "word",
            "docx",
            "pdf",
            "qq",
            "http",
            "https",
        )
        if any(marker in lowered or marker in compact for marker in task_markers):
            return False
        greeting_markers = (
            "早上好",
            "上午好",
            "中午好",
            "下午好",
            "晚上好",
            "你好",
            "嗨",
            "hi",
            "hello",
            "在吗",
            "银狼好",
        )
        if any(marker in lowered or marker in compact for marker in greeting_markers):
            return True
        return False

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
        messages = ContextBuilder.build_prompt_messages(
            session.kernel.history,
            session_summary=session.hot_context.session_summary,
            active_task_summary=session.hot_context.active_task_summary,
            warm_memory_summary=getattr(session.kernel, "warm_memory_summary", ""),
            learning_memory_summary=getattr(session.kernel, "learning_memory_summary", ""),
            cold_memory_summary=getattr(session.kernel, "cold_memory_summary", ""),
        )
        channel_context = ChatService._format_channel_context_for_chat(session.runtime_settings)
        if not channel_context:
            return messages
        context_message = Message(role=Role.SYSTEM, content=channel_context)
        if messages and messages[0].role == Role.SYSTEM:
            return [messages[0], context_message, *messages[1:]]
        return [context_message, *messages]

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

    @staticmethod
    def _refresh_warm_memory_summaries(
        session: SessionState,
        warm_memory: WarmMemoryService,
        user_text: str,
    ) -> None:
        profile_layers = warm_memory.recall_user_profile_layers_for_text(user_text)
        learning_recalled = warm_memory.recall_memory_layer_for_text(
            user_text,
            layer=MEMORY_LAYER_TASK_EXPERIENCE,
            limit=3,
        )
        user_parts = [
            warm_memory.format_for_prompt(
                profile_layers.get(MEMORY_LAYER_USER_PREFERENCE, []),
                title="User preferences",
            ),
            warm_memory.format_for_prompt(
                profile_layers.get(MEMORY_LAYER_LONG_TERM_INSTRUCTION, []),
                title="Long-term instructions",
            ),
        ]
        session.kernel.user_memory_summary = "\n\n".join(part for part in user_parts if part.strip())
        session.kernel.learning_memory_summary = warm_memory.format_for_prompt(
            learning_recalled,
            title="Task experience",
        )
        session.kernel.warm_memory_summary = session.kernel.user_memory_summary

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
        self._refresh_warm_memory_summaries(session, warm_memory, user_text)
        if capture_user_instruction:
            session.last_memory_update = None
            memory_candidate_intent = None
            try:
                memory_candidate_intent = session.kernel.intent_service.analyze_memory_candidate(
                    user_text,
                    recent_context="\n".join(
                        self._build_recent_conversation_window(
                            session.kernel.history,
                            raw_user_turn_text=user_text,
                            runtime_settings=session.runtime_settings,
                            recent_turn_texts=session.recent_turn_texts,
                            recent_artifacts=session.recent_artifacts,
                        )[0]
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
                        remembered = warm_memory.remember_instruction_intent(instruction_intent)
                if remembered is not None:
                    session.last_memory_update = {
                        "memory_type": remembered.memory_type,
                        "scope": remembered.scope,
                        "content": remembered.content,
                        "tags": list(remembered.tags),
                    }
                self._refresh_warm_memory_summaries(session, warm_memory, user_text)
            else:
                remembered = warm_memory.maybe_capture_user_instruction(user_text)
                if remembered is not None:
                    session.last_memory_update = {
                        "memory_type": remembered.memory_type,
                        "scope": remembered.scope,
                        "content": remembered.content,
                        "tags": list(remembered.tags),
                    }

        cold_archive = ColdArchiveService(memory_store)
        if cold_archive.should_recall(user_text):
            cold_records = cold_archive.recall_for_text(user_text)
            session.kernel.cold_memory_summary = cold_archive.format_for_prompt(cold_records)
        else:
            session.kernel.cold_memory_summary = ""

    def try_fast_local_answer(self, session: SessionState, user_text: str) -> ChatTurnResult | None:
        base_response = self._build_fast_local_response(session, user_text)
        if base_response is None:
            return None
        response = self._style_fast_local_response(session, user_text, base_response) or base_response
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
                    "base_response": base_response,
                    "styled": response != base_response,
                },
            },
        )

    def try_memory_update_ack(self, session: SessionState, user_text: str) -> ChatTurnResult | None:
        memory_update = getattr(session, "last_memory_update", None)
        if not isinstance(memory_update, dict):
            return None
        content = str(memory_update.get("content") or "").strip()
        memory_type = str(memory_update.get("memory_type") or "").strip()
        if not content or memory_type not in {"user_fact", "preference", "alias", "correction"}:
            return None
        fact = self._render_user_fact_memory_response(content)
        if not fact:
            return None
        response = fact.rstrip("。") + "，我记住了。"
        session.kernel.history.append(Message(role=Role.USER, content=user_text))
        session.kernel.history.append(Message(role=Role.ASSISTANT, content=response))
        self.refresh_hot_context(session)
        tts_dispatched = self.dispatch_tts(session, response)
        session.touch("chat")
        session.last_memory_update = None
        return ChatTurnResult(
            session_id=session.session_id,
            mode="chat",
            response=response,
            speech_text=response,
            tts_dispatched=tts_dispatched,
            used_agent=False,
            scope_root=session.scope_root,
            metadata={
                "trace_id": "",
                "execution_summary": {
                    "task_status": "completed",
                    "completed_by": "memory_update_ack",
                    "memory_type": memory_type,
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
                self._refresh_warm_memory_summaries(session, warm_memory, user_text)
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

    @classmethod
    def _merge_recent_artifacts_into_runtime_settings(cls, session: SessionState) -> None:
        runtime_settings = session.runtime_settings if isinstance(session.runtime_settings, dict) else {}
        channel_runtime = runtime_settings.get("channel_runtime")
        channel_runtime = dict(channel_runtime) if isinstance(channel_runtime, dict) else {}
        compact_artifacts = cls._compact_recent_artifacts(session.recent_artifacts, limit=6)
        if compact_artifacts:
            channel_runtime["recent_artifacts"] = compact_artifacts
            last_file = next(
                (item for item in reversed(compact_artifacts) if item.get("kind") == "file" and item.get("path")),
                None,
            )
            if last_file is not None:
                channel_runtime["last_file_artifact"] = last_file
                channel_runtime.setdefault("last_written_file", last_file.get("path"))
                if str(last_file.get("role") or "") in {"sent", "written_and_sent"}:
                    channel_runtime.setdefault("last_sent_file", last_file.get("path"))
        runtime_settings = dict(runtime_settings)
        runtime_settings["channel_runtime"] = channel_runtime
        session.runtime_settings = runtime_settings

    @classmethod
    def _update_recent_artifacts(cls, session: SessionState, user_text: str, turn_result: ChatTurnResult) -> None:
        new_artifacts = cls._extract_recent_artifacts_from_result(user_text, turn_result)
        if not new_artifacts:
            return
        existing = list(session.recent_artifacts or [])
        for artifact in new_artifacts:
            key = (
                str(artifact.get("kind") or ""),
                str(artifact.get("path") or ""),
                str(artifact.get("title") or ""),
                str(artifact.get("role") or ""),
            )
            existing = [
                item for item in existing
                if (
                    str(item.get("kind") or ""),
                    str(item.get("path") or ""),
                    str(item.get("title") or ""),
                    str(item.get("role") or ""),
                ) != key
            ]
            existing.append(artifact)
        session.recent_artifacts = cls._compact_recent_artifacts(existing, limit=12)
        cls._merge_recent_artifacts_into_runtime_settings(session)

    @classmethod
    def _extract_recent_artifacts_from_result(cls, user_text: str, turn_result: ChatTurnResult) -> list[dict]:
        metadata = turn_result.metadata if isinstance(turn_result.metadata, dict) else {}
        execution_summary = metadata.get("execution_summary")
        if not isinstance(execution_summary, dict):
            return []
        trace_id = str(metadata.get("trace_id") or execution_summary.get("trace_id") or "").strip()
        artifacts: list[dict] = []
        for path in execution_summary.get("written_files") or []:
            if isinstance(path, str) and path.strip():
                artifacts.append(
                    cls._build_file_artifact(
                        path=path,
                        role="written",
                        trace_id=trace_id,
                        user_text=user_text,
                    )
                )
        for action in execution_summary.get("successful_actions") or []:
            if not isinstance(action, dict):
                continue
            tool_name = str(action.get("tool_name") or "").strip()
            data = action.get("data") if isinstance(action.get("data"), dict) else {}
            path = str(data.get("path") or data.get("file_path") or "").strip()
            if not path:
                continue
            role = "sent" if tool_name == "qq.send_file" else "written"
            if tool_name in {"file.write", "file.write_docx", "file.write_xlsx", "file.write_pptx", "file.edit_docx", "file.render_docx_from_template"}:
                role = "written"
            artifacts.append(
                cls._build_file_artifact(
                    path=path,
                    role=role,
                    trace_id=trace_id,
                    user_text=user_text,
                    tool_name=tool_name,
                )
            )
        return cls._compact_recent_artifacts(artifacts, limit=8)

    @staticmethod
    def _build_file_artifact(
        *,
        path: str,
        role: str,
        trace_id: str = "",
        user_text: str = "",
        tool_name: str = "",
    ) -> dict:
        title = Path(path).name if path else ""
        return {
            "kind": "file",
            "role": role,
            "path": path,
            "title": title,
            "trace_id": trace_id,
            "tool_name": tool_name,
            "source_user_text": ChatService._clip_context_text(user_text, limit=120),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }

    @staticmethod
    def _compact_recent_artifacts(artifacts: list[dict], *, limit: int = 8) -> list[dict]:
        compact: list[dict] = []
        for item in artifacts or []:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "").strip()
            title = str(item.get("title") or "").strip()
            if not path and not title:
                continue
            compact.append(
                {
                    key: value
                    for key, value in {
                        "kind": str(item.get("kind") or "artifact").strip(),
                        "role": str(item.get("role") or "").strip(),
                        "path": path,
                        "title": title,
                        "trace_id": str(item.get("trace_id") or "").strip(),
                        "tool_name": str(item.get("tool_name") or "").strip(),
                        "source_user_text": ChatService._clip_context_text(str(item.get("source_user_text") or ""), limit=120),
                        "created_at": str(item.get("created_at") or "").strip(),
                    }.items()
                    if value
                }
            )
        return compact[-max(1, limit):]

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

    def _style_fast_local_response(self, session: SessionState, user_text: str, base_response: str) -> str | None:
        config = getattr(session.kernel, "config", None)
        if not bool(getattr(config, "fast_response_style_enabled", False)):
            return None
        model = str(getattr(config, "fast_response_model", "") or "").strip()
        if not model:
            return None
        llm_client = getattr(session.kernel, "llm_client", None)
        style_method = getattr(llm_client, "style_fast_local_response", None)
        if not callable(style_method):
            return None
        try:
            styled = style_method(
                user_text=user_text,
                fact_response=base_response,
                persona_profile=str(getattr(config, "persona_profile", "") or ""),
                chat_style_prompt=str(getattr(config, "chat_style_prompt", "") or ""),
                model=model,
                timeout_seconds=int(getattr(config, "fast_response_timeout_seconds", 8) or 8),
            )
        except Exception:
            return None
        return self._accept_fast_local_style(base_response, styled)

    @classmethod
    def _accept_fast_local_style(cls, base_response: str, styled_response: str) -> str | None:
        styled = " ".join(str(styled_response or "").split()).strip().strip("\"'“”‘’")
        if not styled:
            return None
        if len(styled) > 100 or len(styled) > max(60, len(base_response) * 3):
            return None
        if not re.search(r"[?？]\s*$", str(base_response or "")) and re.search(r"[?？]\s*$", styled):
            return None
        if cls._contains_cjk(base_response) and not cls._contains_cjk(styled):
            return None
        if cls._contains_unrelated_identity_fact(base_response, styled):
            return None
        if not cls._preserves_fast_response_subject(base_response, styled):
            return None
        base_tokens = cls._extract_fast_response_fact_tokens(base_response)
        styled_compact = cls._compact_fact_text(styled)
        for token in base_tokens:
            if cls._compact_fact_text(token) not in styled_compact:
                return None
        return styled

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))

    @classmethod
    def _contains_unrelated_identity_fact(cls, base_response: str, styled_response: str) -> bool:
        base = cls._compact_subject_text(base_response)
        styled = cls._compact_subject_text(styled_response)
        identity_markers = ("你的名字叫", "我的名字叫", "角色", "设定", "银狼")
        return any(marker in styled and marker not in base for marker in identity_markers)

    @classmethod
    def _preserves_fast_response_subject(cls, base_response: str, styled_response: str) -> bool:
        base = cls._compact_subject_text(base_response)
        styled = cls._compact_subject_text(styled_response)
        if cls._is_user_addressed_fact(base) and cls._starts_with_first_person_fact_claim(styled):
            return False
        if "你的生日" in base:
            if re.search(r"(我的生日|我.{0,8}生日|生日.{0,8}我)", styled):
                return False
        if "你的称呼" in base or "叫你" in base:
            if re.search(r"(我的称呼|叫我)", styled):
                return False
        return True

    @staticmethod
    def _is_user_addressed_fact(compact_text: str) -> bool:
        text = str(compact_text or "")
        return text.startswith("你") or "你的" in text or "给你" in text or "关于你" in text

    @staticmethod
    def _starts_with_first_person_fact_claim(compact_text: str) -> bool:
        text = str(compact_text or "")
        if not text.startswith(("我", "我的")):
            return False
        allowed_action_prefixes = (
            "我记住",
            "我记得",
            "我知道",
            "我明白",
            "我会记",
            "我这边",
            "我帮你",
        )
        return not text.startswith(allowed_action_prefixes)

    @classmethod
    def _extract_fast_response_fact_tokens(cls, text: str) -> list[str]:
        normalized = str(text or "")
        patterns = [
            r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*[日号]?",
            r"\d{1,2}\s*月\s*\d{1,2}\s*[日号]?",
            r"星期[一二三四五六日天]",
            r"-?\d+(?:\.\d+)?",
        ]
        tokens: list[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, normalized):
                token = match.group(0).strip()
                if token and token not in tokens:
                    tokens.append(token)
        food_match = re.search(r"(?:喜欢吃|爱吃)([^，。；;]+)", normalized)
        if food_match:
            for item in re.split(r"[、,，和跟及与\s]+", food_match.group(1)):
                token = item.strip()
                if len(token) >= 2 and token not in tokens:
                    tokens.append(token)
        for marker in ("矫正牙齿", "吃不太动"):
            if marker in normalized and marker not in tokens:
                tokens.append(marker)
        return tokens

    @staticmethod
    def _compact_fact_text(text: str) -> str:
        return re.sub(r"\s+", "", str(text or "")).replace("日", "号")

    @staticmethod
    def _compact_subject_text(text: str) -> str:
        return re.sub(r"\s+", "", str(text or ""))

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
        if not cls._DATE_QUERY_PATTERN.search(user_text) and not any(
            token in lowered
            for token in ("今天几号", "今天星期", "今天周几", "今天是什么日子")
        ):
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

    @classmethod
    def _try_answer_from_warm_memory(cls, session: SessionState, user_text: str) -> str | None:
        lowered = user_text.lower()
        memory_store = getattr(session.kernel, "memory_store", None)
        if memory_store is None:
            return None
        warm_memory = WarmMemoryService(memory_store)
        recalled = warm_memory.recall_for_text(user_text, limit=6)
        if not recalled and not cls._is_food_preference_query(user_text):
            return None

        if "生日" in user_text:
            for item in recalled:
                tags = {tag.lower() for tag in item.tags}
                if item.memory_type == "user_fact" and ("user.birthday" in tags or "生日" in item.content):
                    return cls._render_user_fact_memory_response(item.content)
        if cls._is_food_preference_query(user_text):
            food_items = list(recalled)
            try:
                food_items.extend(
                    item.record
                    for item in memory_store.list_records(
                        memory_types={"user_fact", "preference"},
                        scopes={"user", "session"},
                        limit=40,
                    )
                )
            except Exception:
                pass
            seen: set[str] = set()
            for item in food_items:
                content = str(item.content or "").strip()
                if not content or content in seen:
                    continue
                seen.add(content)
                tags = {tag.lower() for tag in item.tags}
                if cls._looks_like_food_preference_memory(content, tags):
                    return cls._render_user_fact_memory_response(content)
        if any(token in lowered for token in ("怎么称呼我", "叫我什么", "称呼我")):
            for item in recalled:
                tags = {tag.lower() for tag in item.tags}
                if item.memory_type == "alias" and ("user.preferred_name" in tags or "称呼" in item.content):
                    return cls._render_user_fact_memory_response(item.content)
        return None

    @staticmethod
    def _is_food_preference_query(user_text: str) -> bool:
        normalized = str(user_text or "")
        return any(token in normalized for token in ("爱吃", "喜欢吃", "吃什么", "饮食偏好", "口味", "忌口"))

    @staticmethod
    def _looks_like_food_preference_memory(content: str, tags: set[str]) -> bool:
        normalized = str(content or "")
        if any(tag.startswith("user.food") or tag.startswith("user.diet") for tag in tags):
            return True
        return any(token in normalized for token in ("喜欢吃", "爱吃", "饮食", "忌口", "矫正牙齿", "吃不太动"))

    @staticmethod
    def _render_user_fact_memory_response(content: str) -> str:
        normalized = " ".join(str(content or "").split()).strip()
        if not normalized:
            return ""
        if normalized.startswith("我的生日是"):
            normalized = "你的生日是" + normalized[len("我的生日是") :]
        elif normalized.startswith("用户的生日是"):
            normalized = "你的生日是" + normalized[len("用户的生日是") :]
        elif normalized.startswith("主人的生日是"):
            normalized = "你的生日是" + normalized[len("主人的生日是") :]
        elif normalized.startswith("用户"):
            normalized = "你" + normalized[len("用户") :]
        elif normalized.startswith("我"):
            normalized = "你" + normalized[len("我") :]
        return normalized
