from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

from local_agent.app.onebot_models import OneBotAudioAttachment
from local_agent.protocol.models import VoiceInputConfig
from local_agent.voice.audio_preprocessor import AudioPreprocessor
from local_agent.voice.asr_service import ASRService, TranscriptionResult


@dataclass(frozen=True)
class MaterializedAudio:
    file_path: str
    temporary: bool = False


class VoiceInputService:
    def __init__(
        self,
        config: VoiceInputConfig,
        asr_service: ASRService | None,
        audio_preprocessor: AudioPreprocessor | None = None,
    ) -> None:
        self._config = config
        self._asr_service = asr_service
        self._audio_preprocessor = audio_preprocessor or AudioPreprocessor()

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled and self._config.process_onebot_voice and self._asr_service is not None)

    def transcribe_onebot_attachments(self, attachments: tuple[OneBotAudioAttachment, ...]) -> TranscriptionResult:
        if not self.enabled:
            return TranscriptionResult(error="Voice input service is disabled.", provider="disabled")

        errors: list[str] = []
        for attachment in attachments:
            materialized = self._materialize_attachment(attachment)
            if materialized is None:
                continue
            prepared = None
            try:
                prepared = self._audio_preprocessor.prepare_for_asr(materialized.file_path)
                result = self._asr_service.transcribe_file(prepared.file_path)
                if result.ok:
                    return result
                if result.error:
                    errors.append(result.error)
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
            finally:
                if prepared is not None and prepared.temporary and self._config.cleanup_temp:
                    Path(prepared.file_path).unlink(missing_ok=True)
                if materialized.temporary and self._config.cleanup_temp:
                    Path(materialized.file_path).unlink(missing_ok=True)

        return TranscriptionResult(
            error="; ".join(errors) if errors else "No usable audio attachment found.",
            provider=self._provider_name(),
        )

    def _materialize_attachment(self, attachment: OneBotAudioAttachment) -> MaterializedAudio | None:
        if attachment.local_path:
            local_path = self._wait_for_local_audio(attachment.local_path)
            if local_path is not None:
                return MaterializedAudio(file_path=str(local_path.resolve()))

        if attachment.remote_url:
            local_remote = self._wait_for_local_audio(attachment.remote_url)
            if local_remote is not None:
                return MaterializedAudio(file_path=str(local_remote.resolve()))
            return self._download_remote_audio(attachment.remote_url)
        return None

    def _download_remote_audio(self, remote_url: str) -> MaterializedAudio | None:
        parsed = urlparse(remote_url)
        suffix = Path(parsed.path).suffix or ".wav"
        temp_dir = Path(self._config.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        file_descriptor, temp_name = tempfile.mkstemp(prefix="onebot_audio_", suffix=suffix, dir=temp_dir)
        os.close(file_descriptor)
        temp_path = Path(temp_name)
        try:
            response = requests.get(remote_url, timeout=60)
            response.raise_for_status()
            temp_path.write_bytes(response.content)
        except Exception:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        return MaterializedAudio(file_path=str(temp_path.resolve()), temporary=True)

    def _wait_for_local_audio(self, raw_path: str) -> Path | None:
        candidate = Path(raw_path)
        deadline = time.time() + max(0.0, self._config.local_file_wait_seconds)
        interval = max(0.05, self._config.local_file_probe_interval_ms / 1000.0)

        while True:
            if candidate.is_file() and self._can_open_for_read(candidate):
                return candidate
            if time.time() >= deadline:
                return candidate if candidate.is_file() else None
            time.sleep(interval)

    @staticmethod
    def _can_open_for_read(path: Path) -> bool:
        try:
            with path.open("rb"):
                return True
        except OSError:
            return False

    def _provider_name(self) -> str:
        if self._asr_service is None:
            return "disabled"
        config = getattr(self._asr_service, "_config", None)
        provider = getattr(config, "provider", None)
        if isinstance(provider, str) and provider.strip():
            return provider.strip()
        return self._asr_service.__class__.__name__.lower()
