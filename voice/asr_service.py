from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from local_agent.protocol.models import ASRConfig


@dataclass(frozen=True)
class TranscriptionResult:
    text: str = ""
    error: str | None = None
    provider: str = ""
    source_path: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and not self.error


class ASRService:
    def __init__(self, config: ASRConfig) -> None:
        self._config = config
        self._faster_whisper_model = None

    @classmethod
    def from_config(cls, config: ASRConfig) -> ASRService | None:
        if not config.enabled:
            return None
        return cls(config)

    def transcribe_file(self, file_path: str) -> TranscriptionResult:
        normalized_path = str(Path(file_path).resolve())
        provider = self._config.provider.strip().lower()
        if provider == "faster_whisper":
            return self._transcribe_via_faster_whisper(normalized_path)
        if provider == "http":
            return self._transcribe_via_http(normalized_path)
        return self._transcribe_via_command(normalized_path)

    def _transcribe_via_faster_whisper(self, file_path: str) -> TranscriptionResult:
        try:
            model = self._get_faster_whisper_model()
        except Exception as exc:  # noqa: BLE001
            return TranscriptionResult(
                error=str(exc),
                provider="faster_whisper",
                source_path=file_path,
            )

        try:
            segments, _info = model.transcribe(
                file_path,
                language=self._normalize_language(),
                beam_size=self._config.beam_size,
                vad_filter=self._config.vad_filter,
                condition_on_previous_text=False,
                word_timestamps=False,
            )
            text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        except Exception as exc:  # noqa: BLE001
            return TranscriptionResult(
                error=str(exc),
                provider="faster_whisper",
                source_path=file_path,
            )

        if not text:
            return TranscriptionResult(
                error="faster-whisper returned empty text.",
                provider="faster_whisper",
                source_path=file_path,
            )
        return TranscriptionResult(text=text, provider="faster_whisper", source_path=file_path)

    def _transcribe_via_command(self, file_path: str) -> TranscriptionResult:
        if not self._config.command:
            return TranscriptionResult(
                error="ASR command provider is enabled but no command is configured.",
                provider="command",
                source_path=file_path,
            )

        command = [
            str(part).format(audio_path=file_path, language=self._config.language)
            for part in self._config.command
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=False,
                timeout=self._config.request_timeout_seconds,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            return TranscriptionResult(
                error=str(exc),
                provider="command",
                source_path=file_path,
            )

        stdout_text = self._decode_output(completed.stdout)
        stderr_text = self._decode_output(completed.stderr)

        if completed.returncode != 0:
            details = stderr_text.strip() or stdout_text.strip() or f"exit code {completed.returncode}"
            return TranscriptionResult(
                error=details,
                provider="command",
                source_path=file_path,
            )

        text = stdout_text.strip()
        if not text:
            return TranscriptionResult(
                error="ASR command returned empty text.",
                provider="command",
                source_path=file_path,
            )
        return TranscriptionResult(text=text, provider="command", source_path=file_path)

    def _transcribe_via_http(self, file_path: str) -> TranscriptionResult:
        if not self._config.endpoint:
            return TranscriptionResult(
                error="ASR http provider is enabled but no endpoint is configured.",
                provider="http",
                source_path=file_path,
            )

        try:
            with Path(file_path).open("rb") as handle:
                response = requests.post(
                    self._config.endpoint,
                    files={"file": (Path(file_path).name, handle, "audio/wav")},
                    data={"language": self._config.language},
                    timeout=self._config.request_timeout_seconds,
                )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return TranscriptionResult(
                error=str(exc),
                provider="http",
                source_path=file_path,
            )

        text = self._extract_http_text(response)
        if not text:
            return TranscriptionResult(
                error="ASR http provider returned empty text.",
                provider="http",
                source_path=file_path,
            )
        return TranscriptionResult(text=text, provider="http", source_path=file_path)

    @staticmethod
    def _extract_http_text(response: requests.Response) -> str:
        content_type = str(response.headers.get("content-type", "")).lower()
        if "application/json" in content_type:
            payload: Any = response.json()
            if isinstance(payload, dict):
                for key in ("text", "transcript", "content"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        text = response.text.strip()
        return text

    def _decode_output(self, content: bytes | str | None) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        try:
            return content.decode(self._config.output_encoding)
        except UnicodeDecodeError:
            return content.decode(errors="ignore")

    def _get_faster_whisper_model(self):
        if self._faster_whisper_model is not None:
            return self._faster_whisper_model

        from faster_whisper import WhisperModel

        download_root = Path(self._config.model_download_root)
        download_root.mkdir(parents=True, exist_ok=True)
        self._faster_whisper_model = WhisperModel(
            self._config.model,
            device=self._config.device,
            compute_type=self._config.compute_type,
            download_root=str(download_root),
        )
        return self._faster_whisper_model

    def _normalize_language(self) -> str | None:
        language = self._config.language.strip().lower()
        if not language or language in {"auto", "detect"}:
            return None
        return language
