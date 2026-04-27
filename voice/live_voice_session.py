from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from local_agent.protocol.channel_models import ChannelMessage, ChannelReply, ChannelUser
from local_agent.protocol.models import VoiceInputConfig
from local_agent.voice.output_service import VoiceOutputService, VoiceRenderResult
from local_agent.voice.wake_word_service import WakeWordMatch, WakeWordService


@dataclass(frozen=True)
class LiveVoiceTurnResult:
    accepted: bool
    wake_detected: bool = False
    ack_text: str = ""
    wake_match: WakeWordMatch | None = None
    reply: ChannelReply | None = None
    rendered_reply: VoiceRenderResult | None = None
    session_active: bool = False
    background_task_id: str | None = None


class LiveVoiceSessionService:
    def __init__(
        self,
        config: VoiceInputConfig,
        *,
        channel_router,
        voice_output: VoiceOutputService,
        task_service=None,
        channel_name: str = "live_voice",
        time_provider=None,
    ) -> None:
        self._config = config
        self._channel_router = channel_router
        self._voice_output = voice_output
        self._task_service = task_service
        self._channel_name = channel_name
        self._wake_word_service = WakeWordService(config)
        self._time = time_provider or time.time
        self._active_sessions: dict[str, float] = {}

    def handle_transcript(
        self,
        text: str,
        *,
        session_id: str = "live_voice_default",
        scope_root: str | None = None,
        runtime_settings: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        sender_id: str = "local_microphone",
        sender_name: str = "Microphone",
    ) -> LiveVoiceTurnResult:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return LiveVoiceTurnResult(accepted=False, session_active=self._is_active(session_id))

        wake_match = self._wake_word_service.match_text(normalized_text)
        active = self._is_active(session_id)
        if wake_match.matched:
            self._mark_active(session_id)
            ack_text = self._select_ack_phrase(session_id)
            remaining_text = wake_match.remaining_text.strip()
            if not remaining_text:
                return LiveVoiceTurnResult(
                    accepted=True,
                    wake_detected=True,
                    ack_text=ack_text,
                    wake_match=wake_match,
                    session_active=True,
                )
            turn = self._dispatch_or_submit(
                remaining_text,
                session_id=session_id,
                scope_root=scope_root,
                runtime_settings=runtime_settings,
                metadata={
                    **(metadata or {}),
                    "voice_entry": "wake_word",
                    "wake_word": wake_match.wake_word,
                },
                sender_id=sender_id,
                sender_name=sender_name,
                render_reply=False,
            )
            return LiveVoiceTurnResult(
                accepted=True,
                wake_detected=True,
                ack_text=ack_text,
                wake_match=wake_match,
                reply=turn.reply,
                rendered_reply=None,
                session_active=True,
                background_task_id=turn.background_task_id,
            )

        if not active:
            return LiveVoiceTurnResult(accepted=False, session_active=False)

        self._mark_active(session_id)
        turn = self._dispatch_or_submit(
            normalized_text,
            session_id=session_id,
            scope_root=scope_root,
            runtime_settings=runtime_settings,
            metadata={**(metadata or {}), "voice_entry": "active_session"},
            sender_id=sender_id,
            sender_name=sender_name,
            render_reply=True,
        )
        return LiveVoiceTurnResult(
            accepted=True,
            wake_detected=False,
            reply=turn.reply,
            rendered_reply=turn.rendered_reply,
            session_active=True,
            background_task_id=turn.background_task_id,
        )

    def _dispatch_or_submit(
        self,
        text: str,
        *,
        session_id: str,
        scope_root: str | None,
        runtime_settings: dict[str, Any] | None,
        metadata: dict[str, Any],
        sender_id: str,
        sender_name: str,
        render_reply: bool,
    ) -> LiveVoiceTurnResult:
        merged_runtime = dict(runtime_settings or {})
        voice_settings = merged_runtime.get("voice") if isinstance(merged_runtime.get("voice"), dict) else {}
        merged_runtime["voice"] = {
            **voice_settings,
            "enabled": False,
        }
        message = ChannelMessage(
            channel=self._channel_name,
            text=text,
            session_id=session_id,
            scope_root=scope_root,
            mode="auto",
            runtime_settings=merged_runtime,
            sender=ChannelUser(user_id=sender_id, display_name=sender_name),
            metadata=metadata,
        )
        if self._task_service is not None and self._channel_router.should_run_in_background(message):
            task = self._task_service.submit(
                user_text=text,
                session_id=session_id,
                scope_root=scope_root,
                runtime_settings=merged_runtime,
            )
            ack_text = self._select_background_ack_phrase(task.task_id)
            reply = ChannelReply(
                channel=self._channel_name,
                session_id=session_id,
                mode="agent",
                used_agent=True,
                response=ack_text,
                speech_text=ack_text,
                tts_dispatched=False,
                scope_root=scope_root,
                metadata={"background_task": True, "task_id": task.task_id},
            )
            return LiveVoiceTurnResult(
                accepted=True,
                reply=reply,
                rendered_reply=self._render_reply(reply) if render_reply else None,
                background_task_id=task.task_id,
            )
        reply = self._channel_router.dispatch(message)
        return LiveVoiceTurnResult(
            accepted=True,
            reply=reply,
            rendered_reply=self._render_reply(reply) if render_reply else None,
        )

    def _render_reply(self, reply: ChannelReply) -> VoiceRenderResult:
        speech_text = (reply.speech_text or reply.response).strip()
        return self._voice_output.synthesize_text(speech_text)

    def _select_ack_phrase(self, session_id: str) -> str:
        phrases = [phrase.strip() for phrase in self._config.wake_ack_phrases if phrase.strip()]
        return self._select_phrase(phrases, session_id)

    def _select_background_ack_phrase(self, task_id: str) -> str:
        phrases = [phrase.strip() for phrase in self._config.background_task_ack_phrases if phrase.strip()]
        return self._select_phrase(phrases, task_id)

    @staticmethod
    def _select_phrase(phrases: list[str], seed_text: str) -> str:
        if not phrases:
            return ""
        digest = hashlib.sha1(seed_text.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % len(phrases)
        return phrases[index]

    def _is_active(self, session_id: str) -> bool:
        expires_at = self._active_sessions.get(session_id)
        if expires_at is None:
            return False
        if expires_at <= self._time():
            self._active_sessions.pop(session_id, None)
            return False
        return True

    def _mark_active(self, session_id: str) -> None:
        ttl = max(0.5, float(self._config.active_session_seconds))
        self._active_sessions[session_id] = self._time() + ttl
