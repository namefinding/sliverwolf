from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from pathlib import Path

from local_agent.protocol.channel_models import ChannelReply
from local_agent.protocol.models import VoiceConfig
from local_agent.voice.gptsovits import GPTSoVITSAdapter


@dataclass(frozen=True)
class VoiceRenderResult:
    speech_text: str
    audio_path: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.audio_path) and not self.error


class VoiceOutputService:
    def __init__(self, adapter: GPTSoVITSAdapter | None) -> None:
        self._adapter = adapter

    @classmethod
    def from_config(
        cls,
        config: VoiceConfig,
        *,
        play_audio: bool,
        async_playback: bool,
        cleanup_audio: bool,
    ) -> VoiceOutputService:
        if not config.enabled:
            return cls(None)
        voice_config = config.model_copy(deep=True)
        voice_config.play_audio = play_audio
        voice_config.async_playback = async_playback
        voice_config.cleanup_audio = cleanup_audio
        return cls(GPTSoVITSAdapter(voice_config))

    @property
    def enabled(self) -> bool:
        return self._adapter is not None

    def synthesize_text(self, speech_text: str) -> VoiceRenderResult:
        if self._adapter is None:
            return VoiceRenderResult(speech_text=speech_text)

        normalized = speech_text.strip()
        if not normalized:
            return VoiceRenderResult(speech_text="")

        try:
            audio_path = self._adapter.synthesize_to_file(normalized)
            return VoiceRenderResult(speech_text=normalized, audio_path=audio_path)
        except Exception as exc:  # noqa: BLE001
            return VoiceRenderResult(speech_text=normalized, error=str(exc))

    def synthesize_reply(self, reply: ChannelReply) -> VoiceRenderResult:
        return self.synthesize_text(reply.speech_text)

    @staticmethod
    async def cleanup_temp_file_later(file_path: str, delay_seconds: float = 5.0) -> None:
        await asyncio.sleep(delay_seconds)
        with contextlib.suppress(OSError):
            Path(file_path).unlink(missing_ok=True)
