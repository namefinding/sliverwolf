from __future__ import annotations

from local_agent.app.channel_access import build_onebot_access_policy
from local_agent.app.chat_models import ChatTurnResult
from local_agent.app.chat_service import ChatService
from local_agent.protocol.channel_models import ChannelMessage, ChannelReply
from local_agent.protocol.models import OneBotConfig


class ChannelRouter:
    """Normalize channel input before it reaches the shared agent runtime."""

    def __init__(self, chat_service: ChatService, onebot_config: OneBotConfig | None = None) -> None:
        self.chat_service = chat_service
        self.onebot_config = onebot_config or OneBotConfig()

    def dispatch(self, message: ChannelMessage) -> ChannelReply:
        session_id = self._resolve_session_id(message)
        turn = self.chat_service.handle_message(
            text=message.text,
            session_id=session_id,
            mode=message.mode,
            scope_root=message.scope_root,
            runtime_settings=self._merge_runtime_settings(message),
            progress_callback=message.progress_callback,
        )
        return self._to_reply(message.channel, turn)

    def should_run_in_background(self, message: ChannelMessage) -> bool:
        session_id = self._resolve_session_id(message)
        return self.chat_service.should_run_in_background(
            text=message.text,
            session_id=session_id,
            mode=message.mode,
            scope_root=message.scope_root,
            runtime_settings=self._merge_runtime_settings(message),
        )

    @staticmethod
    def _to_reply(channel: str, turn: ChatTurnResult) -> ChannelReply:
        return ChannelReply(
            channel=channel,
            session_id=turn.session_id,
            mode=turn.mode,
            used_agent=turn.used_agent,
            response=turn.response,
            speech_text=turn.speech_text,
            tts_dispatched=turn.tts_dispatched,
            scope_root=turn.scope_root,
            overall_task_goal=turn.overall_task_goal,
            completed_outputs=turn.completed_outputs or [],
            metadata={
                **(turn.metadata or {}),
                "pending_task": turn.pending_task,
            },
        )

    @staticmethod
    def _resolve_session_id(message: ChannelMessage) -> str | None:
        if message.session_id:
            return message.session_id
        if message.sender is None:
            return None
        return f"{message.channel}_{message.sender.user_id}"

    def _merge_runtime_settings(self, message: ChannelMessage) -> dict:
        merged = dict(message.runtime_settings or {})
        message_metadata = message.metadata if isinstance(message.metadata, dict) else {}
        finalized_turn = message_metadata.get("finalized_turn")
        if isinstance(finalized_turn, dict) and finalized_turn:
            merged["finalized_turn"] = finalized_turn
        live_turn = message_metadata.get("live_turn")
        if isinstance(live_turn, dict) and live_turn:
            merged["live_turn"] = live_turn
        channel_settings = merged.get("channel") if isinstance(merged.get("channel"), dict) else {}
        merged["channel"] = {
            **channel_settings,
            "name": message.channel,
        }
        if message.channel == "onebot_v11":
            sender_id = None if message.sender is None else message.sender.user_id
            access_policy = build_onebot_access_policy(
                sender_id,
                self.onebot_config.full_access_user_ids,
                self.onebot_config.owner_user_ids,
                self.onebot_config.owner_display_name,
            )
            merged["access_policy"] = access_policy
            voice_settings = merged.get("voice") if isinstance(merged.get("voice"), dict) else {}
            merged["voice"] = {
                **voice_settings,
                "enabled": False,
            }
        return merged
