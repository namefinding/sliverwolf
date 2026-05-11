from __future__ import annotations

from datetime import UTC, datetime, timedelta
from threading import Lock
import re
from typing import Any
from uuid import uuid4

from local_agent.protocol.models import FinalizedTurn
from local_agent.protocol.models import LiveTurnEvent, LiveTurnState, TurnCompletionDecision


_LIGHT_INCOMPLETE_CUES = (
    "然后",
    "还有",
    "另外",
    "补充",
    "顺便",
    "以及",
    "再加",
    "就说",
    "日期写",
    "样式参考",
    "优先",
)

_LIGHT_VAGUE_STARTER_CUES = (
    "在吗",
    "等下",
    "帮我",
    "我想",
    "你先",
)

_COMPLETE_SUFFIXES = ("。", "！", "？", "!", "?", "~", "～", "…", "：", ":")


class TurnAssemblyService:
    def __init__(self) -> None:
        self._states: dict[str, LiveTurnState] = {}
        self._lock = Lock()

    def observe_event(
        self,
        *,
        session_id: str,
        channel: str,
        text: str,
        attachment_refs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> LiveTurnState:
        observed_at = now or datetime.now(UTC)
        normalized_text = str(text or "").strip()
        attachment_refs = [str(item).strip() for item in (attachment_refs or []) if str(item).strip()]
        event = LiveTurnEvent(
            event_type="user_message",
            text=normalized_text,
            attachment_refs=attachment_refs,
            metadata=dict(metadata or {}),
            created_at=observed_at,
        )

        with self._lock:
            state = self._states.get(session_id)
            if state is None:
                state = LiveTurnState(
                    session_id=session_id,
                    turn_id=f"turn_{uuid4().hex[:12]}",
                    channel=channel,
                    version=1,
                    raw_user_turn_text=normalized_text,
                    event_count=1,
                    attachment_refs=list(attachment_refs),
                    events=[event],
                    first_event_at=observed_at,
                    last_event_at=observed_at,
                    metadata=dict(metadata or {}),
                )
            else:
                raw_user_turn_text = self._merge_turn_text(state.raw_user_turn_text, normalized_text)
                merged_refs = list(state.attachment_refs)
                for item in attachment_refs:
                    if item not in merged_refs:
                        merged_refs.append(item)
                merged_metadata = dict(state.metadata)
                if metadata:
                    merged_metadata.update(metadata)
                state = state.model_copy(
                    update={
                        "channel": channel or state.channel,
                        "version": state.version + 1,
                        "raw_user_turn_text": raw_user_turn_text,
                        "event_count": state.event_count + 1,
                        "attachment_refs": merged_refs,
                        "events": [*state.events, event],
                        "last_event_at": observed_at,
                        "metadata": merged_metadata,
                    }
                )
            self._states[session_id] = state
            return state

    def get_state(self, session_id: str) -> LiveTurnState | None:
        with self._lock:
            return self._states.get(session_id)

    def observe_typing(
        self,
        *,
        session_id: str,
        channel: str,
        active: bool,
        metadata: dict[str, Any] | None = None,
        now: datetime | None = None,
        hold_ms: int = 1800,
    ) -> LiveTurnState:
        observed_at = now or datetime.now(UTC)
        hold_ms = max(200, int(hold_ms or 0))
        typing_expires_at = observed_at if not active else observed_at + timedelta(milliseconds=hold_ms)
        event = LiveTurnEvent(
            event_type="typing_status",
            text="",
            attachment_refs=[],
            metadata={"active": active, **dict(metadata or {})},
            created_at=observed_at,
        )

        with self._lock:
            state = self._states.get(session_id)
            if state is None:
                state = LiveTurnState(
                    session_id=session_id,
                    turn_id=f"turn_{uuid4().hex[:12]}",
                    channel=channel,
                    version=0,
                    raw_user_turn_text="",
                    event_count=0,
                    attachment_refs=[],
                    events=[event],
                    first_event_at=observed_at,
                    last_event_at=observed_at,
                    typing_active=active,
                    last_typing_at=observed_at,
                    typing_expires_at=typing_expires_at,
                    metadata=dict(metadata or {}),
                )
            else:
                merged_metadata = dict(state.metadata)
                if metadata:
                    merged_metadata.update(metadata)
                state = state.model_copy(
                    update={
                        "channel": channel or state.channel,
                        "version": state.version,
                        "events": [*state.events, event],
                        "typing_active": active,
                        "last_typing_at": observed_at,
                        "typing_expires_at": typing_expires_at,
                        "metadata": merged_metadata,
                    }
                )
            self._states[session_id] = state
            return state

    def mark_followup_prompt_sent(
        self,
        session_id: str,
        *,
        followup_text: str,
        now: datetime | None = None,
    ) -> LiveTurnState | None:
        observed_at = now or datetime.now(UTC)
        with self._lock:
            state = self._states.get(session_id)
            if state is None:
                return None
            metadata = dict(state.metadata)
            metadata["followup_prompt_sent"] = True
            metadata["followup_prompt_text"] = str(followup_text or "").strip()
            metadata["followup_prompt_sent_at"] = observed_at.isoformat()
            metadata["followup_prompt_count"] = int(metadata.get("followup_prompt_count") or 0) + 1
            state = state.model_copy(update={"metadata": metadata})
            self._states[session_id] = state
            return state

    def finalize_turn(
        self,
        session_id: str,
        *,
        expected_version: int | None = None,
        finalize_reason: str = "",
    ) -> FinalizedTurn | None:
        with self._lock:
            state = self._states.get(session_id)
            if state is None:
                return None
            if expected_version is not None and state.version != expected_version:
                return None
            self._states.pop(session_id, None)
            return FinalizedTurn(
                session_id=state.session_id,
                turn_id=state.turn_id,
                raw_user_turn_text=state.raw_user_turn_text,
                event_count=state.event_count,
                attachment_refs=list(state.attachment_refs),
                message_segments=[
                    str(event.text).strip()
                    for event in state.events
                    if event.event_type == "user_message" and str(event.text).strip()
                ],
                finalized_at=datetime.now(UTC),
                finalize_reason=finalize_reason,
                metadata=dict(state.metadata),
            )

    def discard_turn(self, session_id: str) -> None:
        with self._lock:
            self._states.pop(session_id, None)

    def decide_completion(
        self,
        *,
        state: LiveTurnState | None,
        quiet_window_ms: int,
        max_wait_ms: int,
        fragment_max_wait_ms: int,
        incomplete_extra_ms: int,
        attachment_extra_ms: int,
        typing_active: bool = False,
        llm_client: Any | None = None,
        recent_context: str = "",
        hot_context_summary: str = "",
        pending_task_summary: str = "",
        persona_name: str = "",
        now: datetime | None = None,
        use_llm_judge: bool = True,
        has_prior_context: bool = False,
    ) -> TurnCompletionDecision:
        if state is None:
            return TurnCompletionDecision(finalize=False, confidence=0.0, wait_ms=0, reason="no_live_turn", source="rule")

        observed_at = now or datetime.now(UTC)
        quiet_ms = max(50, int(quiet_window_ms or 0))
        max_wait_ms = max(quiet_ms, int(max_wait_ms or quiet_ms))
        fragment_max_wait_ms = max(max_wait_ms, int(fragment_max_wait_ms or max_wait_ms))
        incomplete_extra_ms = max(0, int(incomplete_extra_ms or 0))
        attachment_extra_ms = max(0, int(attachment_extra_ms or 0))

        age_ms = max(0, int((observed_at - state.first_event_at).total_seconds() * 1000))
        last_activity_at = self._last_activity_at(state)
        last_message_at = self._last_message_at(state)
        silence_ms = max(0, int((observed_at - last_activity_at).total_seconds() * 1000))
        since_last_message_ms = max(0, int((observed_at - last_message_at).total_seconds() * 1000))

        incomplete_signal = self._light_incomplete_signal(state.raw_user_turn_text)
        vague_starter_signal = self._light_vague_starter_signal(state.raw_user_turn_text)
        looks_incomplete = incomplete_signal >= 0.5
        looks_vague = vague_starter_signal >= 0.5

        effective_idle_timeout_ms = fragment_max_wait_ms if (looks_incomplete or looks_vague) else max_wait_ms
        target_quiet_ms = quiet_ms
        if looks_incomplete:
            target_quiet_ms += incomplete_extra_ms
        if state.attachment_refs:
            target_quiet_ms += attachment_extra_ms
        followup_prompt_sent = bool((state.metadata or {}).get("followup_prompt_sent"))
        idle_deadline_at = state.first_event_at + timedelta(milliseconds=effective_idle_timeout_ms)
        if followup_prompt_sent:
            followup_sent_at = self._followup_prompt_sent_at(state)
            if followup_sent_at is not None:
                idle_deadline_at = max(idle_deadline_at, followup_sent_at + timedelta(milliseconds=5000))
                effective_idle_timeout_ms = max(
                    effective_idle_timeout_ms,
                    int((idle_deadline_at - state.first_event_at).total_seconds() * 1000),
                )

        typing_grace_ms = max(target_quiet_ms, min(2500, quiet_ms + 1200))
        effective_typing_active = self._effective_typing_active(
            state=state,
            now=observed_at,
            typing_active=typing_active,
            since_last_message_ms=since_last_message_ms,
            typing_grace_ms=typing_grace_ms,
        )
        likely_contextual_short_turn = (
            state.event_count <= 2
            and len(str(state.raw_user_turn_text or "").strip()) <= 16
        )

        if effective_typing_active:
            remaining = min(effective_idle_timeout_ms - silence_ms, max(80, target_quiet_ms - silence_ms))
            return TurnCompletionDecision(
                finalize=False,
                confidence=0.2,
                wait_ms=max(0, remaining),
                reason="typing_active",
                source="rule",
            )

        if silence_ms >= target_quiet_ms:
            return TurnCompletionDecision(
                finalize=True,
                confidence=0.78,
                wait_ms=0,
                reason="quiet_window_elapsed",
                source="rule",
            )

        if silence_ms < target_quiet_ms:
            remaining = min(effective_idle_timeout_ms - silence_ms, target_quiet_ms - silence_ms)
            return TurnCompletionDecision(
                finalize=False,
                confidence=0.25,
                wait_ms=max(0, remaining),
                reason="quiet_window_not_elapsed",
                source="rule",
            )

        if use_llm_judge and llm_client is not None and hasattr(llm_client, "classify_turn_completion"):
            try:
                decision = llm_client.classify_turn_completion(
                    raw_user_turn_text=state.raw_user_turn_text,
                    recent_context=recent_context,
                    hot_context_summary=hot_context_summary,
                    pending_task_summary=pending_task_summary,
                    event_summaries=self._event_summaries(state),
                    attachment_refs=state.attachment_refs,
                    typing_active=effective_typing_active,
                    event_count=state.event_count,
                    turn_age_ms=age_ms,
                    silence_ms=silence_ms,
                    quiet_window_ms=target_quiet_ms,
                    idle_timeout_ms=effective_idle_timeout_ms,
                    persona_name=persona_name,
                    has_prior_context=has_prior_context,
                )
                remaining_idle_ms = max(0, int((idle_deadline_at - observed_at).total_seconds() * 1000))

                if (
                    decision.finalize
                    and (looks_incomplete or looks_vague)
                    and state.event_count <= 1
                    and silence_ms < effective_idle_timeout_ms
                    and decision.confidence < 0.9
                ):
                    return TurnCompletionDecision(
                        finalize=False,
                        confidence=decision.confidence,
                        wait_ms=max(0, min(remaining_idle_ms, max(1500, int(decision.wait_ms or 0) or target_quiet_ms))),
                        reason="fragment_wait_extended",
                        source="rule",
                    )

                if not decision.finalize:
                    normalized_wait = max(0, min(remaining_idle_ms, max(400, int(decision.wait_ms or 0) or target_quiet_ms)))
                    followup_text = str(decision.followup_text or "").strip()
                    idle_timeout_reached = remaining_idle_ms <= 0 or silence_ms >= effective_idle_timeout_ms
                    if idle_timeout_reached:
                        if has_prior_context and likely_contextual_short_turn:
                            return decision.model_copy(
                                update={
                                    "finalize": True,
                                    "confidence": max(0.62, decision.confidence),
                                    "wait_ms": 0,
                                    "reason": decision.reason or "contextual_short_turn_idle_timeout",
                                    "source": "rule",
                                    "ask_followup": False,
                                    "followup_text": "",
                                }
                            )
                        if not followup_prompt_sent:
                            if not followup_text:
                                followup_text = self._build_generic_followup_text(state, decision)
                            return decision.model_copy(
                                update={
                                    "wait_ms": 0,
                                    "ask_followup": True,
                                    "followup_text": followup_text,
                                }
                            )
                        return decision.model_copy(
                            update={
                                "finalize": True,
                                "confidence": max(0.55, decision.confidence),
                                "wait_ms": 0,
                                "reason": decision.reason or "idle_timeout_after_followup",
                                "source": "rule",
                                "ask_followup": False,
                                "followup_text": "",
                            }
                        )
                    ask_followup = (
                        decision.confidence >= 0.82
                        and silence_ms >= target_quiet_ms
                        and (remaining_idle_ms > 0 or silence_ms >= effective_idle_timeout_ms)
                        and not followup_prompt_sent
                    )
                    if ask_followup and not followup_text:
                        followup_text = self._build_generic_followup_text(state, decision)
                    return decision.model_copy(
                        update={
                            "wait_ms": normalized_wait,
                            "ask_followup": ask_followup,
                            "followup_text": followup_text if ask_followup else "",
                        }
                    )

                return decision.model_copy(update={"wait_ms": 0, "ask_followup": False})
            except Exception:
                pass

        if (looks_incomplete or looks_vague) and silence_ms < effective_idle_timeout_ms:
            return TurnCompletionDecision(
                finalize=False,
                confidence=0.4,
                wait_ms=max(0, min(effective_idle_timeout_ms - silence_ms, max(1500, target_quiet_ms))),
                reason="fragment_wait_extended",
                source="rule",
            )

        return TurnCompletionDecision(
            finalize=True,
            confidence=0.55,
            wait_ms=0,
            reason="quiet_window_elapsed_fallback",
            source="fallback",
        )

    @staticmethod
    def _build_generic_followup_text(state: LiveTurnState, decision: TurnCompletionDecision) -> str:
        understood_task = str(getattr(decision, "understood_task", "") or "").strip()
        if understood_task:
            return f"我先接住了。你把还没说完的细节补一下，我再一起处理：{understood_task}"
        event_texts = [
            str(event.text).strip()
            for event in state.events
            if event.event_type == "user_message" and str(event.text).strip()
        ]
        if len(event_texts) > 1:
            return "我先把这串需求接住了。你把还缺的那部分细节补一下，我再一起处理。"
        return "我先接住了。你把还缺的细节补一下，我再继续。"

    @staticmethod
    def _merge_turn_text(existing: str, incoming: str) -> str:
        current = str(existing or "").strip()
        new_text = str(incoming or "").strip()
        if not current:
            return new_text
        if not new_text or new_text == current:
            return current
        if new_text in current:
            return current
        return f"{current}\n{new_text}".strip()

    @staticmethod
    def _light_incomplete_signal(text: str) -> float:
        normalized = re.sub(r"\s+", "", str(text or "").strip()).lower()
        if not normalized:
            return 0.0

        score = 0.0
        if any(normalized.startswith(prefix) for prefix in _LIGHT_INCOMPLETE_CUES):
            score += 0.5
        if any(token in normalized for token in _LIGHT_INCOMPLETE_CUES):
            score += 0.25
        if len(normalized) <= 12:
            score += 0.15
        if normalized and normalized[-1] not in "".join(_COMPLETE_SUFFIXES):
            score += 0.2
        return min(score, 1.0)

    @staticmethod
    def _light_vague_starter_signal(text: str) -> float:
        normalized = str(text or "").strip().lower()
        if not normalized:
            return 0.0
        compact = re.sub(r"\s+", "", normalized)
        if any(phrase in compact for phrase in _LIGHT_VAGUE_STARTER_CUES):
            return 0.8
        return 0.0

    @staticmethod
    def _event_summaries(state: LiveTurnState) -> list[str]:
        summaries: list[str] = []
        for event in state.events[-6:]:
            text = event.text.strip()
            if text:
                summaries.append(text)
                continue
            if event.attachment_refs:
                summaries.append(f"attachment_event:{len(event.attachment_refs)}")
        return summaries

    @staticmethod
    def _is_typing_active(state: LiveTurnState, now: datetime) -> bool:
        if not state.typing_active:
            return False
        if state.typing_expires_at is None:
            return False
        return now < state.typing_expires_at

    @classmethod
    def _effective_typing_active(
        cls,
        *,
        state: LiveTurnState,
        now: datetime,
        typing_active: bool,
        since_last_message_ms: int,
        typing_grace_ms: int,
    ) -> bool:
        return bool(typing_active) or cls._is_typing_active(state, now)

    @staticmethod
    def _last_activity_at(state: LiveTurnState) -> datetime:
        if state.last_typing_at is not None and state.last_typing_at > state.last_event_at:
            return state.last_typing_at
        return state.last_event_at

    @staticmethod
    def _last_message_at(state: LiveTurnState) -> datetime:
        for event in reversed(state.events):
            if event.event_type == "user_message":
                return event.created_at
        return state.last_event_at

    @staticmethod
    def _followup_prompt_sent_at(state: LiveTurnState) -> datetime | None:
        raw_value = (state.metadata or {}).get("followup_prompt_sent_at")
        if not isinstance(raw_value, str) or not raw_value.strip():
            return None
        try:
            return datetime.fromisoformat(raw_value)
        except ValueError:
            return None
