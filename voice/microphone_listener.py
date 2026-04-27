from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from local_agent.protocol.models import VoiceInputConfig


@dataclass(frozen=True)
class AudioChunk:
    pcm: bytes
    sample_rate: int
    channels: int
    timestamp: float = field(default_factory=time.time)


class AudioSource(Protocol):
    def start(self, callback: Callable[[AudioChunk], None]) -> None: ...

    def stop(self) -> None: ...


class MicrophoneUnavailableError(RuntimeError):
    pass


class SoundDeviceAudioSource:
    def __init__(self, config: VoiceInputConfig) -> None:
        self._config = config
        self._stream = None

    def start(self, callback: Callable[[AudioChunk], None]) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MicrophoneUnavailableError("sounddevice is not installed.") from exc

        def _on_audio(indata, frames, _time_info, status) -> None:
            if status:
                return
            callback(
                AudioChunk(
                    pcm=bytes(indata.tobytes()),
                    sample_rate=self._config.microphone_sample_rate,
                    channels=self._config.microphone_channels,
                )
            )

        self._stream = sd.InputStream(
            samplerate=self._config.microphone_sample_rate,
            channels=self._config.microphone_channels,
            dtype="int16",
            blocksize=max(1, int(self._config.microphone_sample_rate * self._config.microphone_chunk_ms / 1000)),
            device=self._config.microphone_device,
            callback=_on_audio,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None


class MicrophoneListener:
    def __init__(
        self,
        config: VoiceInputConfig,
        *,
        on_chunk: Callable[[AudioChunk], None] | None = None,
        source_factory: Callable[[VoiceInputConfig], AudioSource] | None = None,
    ) -> None:
        self._config = config
        self._on_chunk = on_chunk
        self._source_factory = source_factory or SoundDeviceAudioSource
        self._source: AudioSource | None = None
        self._lock = threading.Lock()
        self._running = False

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled and self._config.microphone_enabled)

    @property
    def running(self) -> bool:
        return self._running

    def set_chunk_handler(self, handler: Callable[[AudioChunk], None] | None) -> None:
        self._on_chunk = handler

    def start(self) -> None:
        if not self.enabled:
            raise MicrophoneUnavailableError("Microphone listener is disabled.")
        if self._on_chunk is None:
            raise MicrophoneUnavailableError("No microphone chunk handler is configured.")
        with self._lock:
            if self._running:
                return
            source = self._source_factory(self._config)
            source.start(self._handle_chunk)
            self._source = source
            self._running = True

    def stop(self) -> None:
        with self._lock:
            if self._source is not None:
                self._source.stop()
                self._source = None
            self._running = False

    def _handle_chunk(self, chunk: AudioChunk) -> None:
        if self._on_chunk is None:
            return
        self._on_chunk(chunk)
