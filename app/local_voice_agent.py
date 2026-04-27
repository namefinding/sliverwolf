from __future__ import annotations

import math
import tempfile
import threading
import time
import wave
from collections import deque
from pathlib import Path

from local_agent.protocol.models import TaskStatus, VoiceInputConfig
from local_agent.voice.asr_service import ASRService
from local_agent.voice.live_voice_session import LiveVoiceSessionService, LiveVoiceTurnResult
from local_agent.voice.microphone_listener import AudioChunk, MicrophoneListener, MicrophoneUnavailableError
from local_agent.voice.output_service import VoiceOutputService


class LocalVoiceAgentService:
    _POSITIVE_RESULT_HINTS = (
        "要",
        "说吧",
        "讲吧",
        "念吧",
        "播报",
        "听结果",
        "讲结果",
        "说结果",
        "告诉我",
    )
    _NEGATIVE_RESULT_HINTS = (
        "不用",
        "先不用",
        "不用了",
        "别说",
        "不听",
        "先别",
        "不要",
        "稍后",
    )
    _READY_STATUSES = {
        TaskStatus.WAITING_FOR_CLARIFICATION,
        TaskStatus.WAITING_FOR_SELECTION,
        TaskStatus.WAITING_FOR_CONFIRMATION,
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }

    def __init__(
        self,
        config: VoiceInputConfig,
        *,
        channel_router,
        task_service,
        voice_output: VoiceOutputService,
        asr_service: ASRService | None,
        scope_root: str | None = None,
        runtime_settings: dict | None = None,
        session_id: str = "local_voice_assistant",
        time_provider=None,
        microphone_listener: MicrophoneListener | None = None,
    ) -> None:
        self._config = config
        self._channel_router = channel_router
        self._task_service = task_service
        self._voice_output = voice_output
        self._asr_service = asr_service
        self._scope_root = scope_root
        self._runtime_settings = dict(runtime_settings or {})
        self._session_id = session_id
        self._time = time_provider or time.time
        self._live_session = LiveVoiceSessionService(
            config,
            channel_router=channel_router,
            task_service=task_service,
            voice_output=voice_output,
            channel_name="live_voice",
            time_provider=self._time,
        )
        self._listener = microphone_listener or MicrophoneListener(
            config,
            on_chunk=self._on_audio_chunk,
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._running = False
        self._cooldown_until = 0.0
        self._capture_started_at = 0.0
        self._last_voice_at = 0.0
        self._capturing = False
        self._capture_chunks: list[bytes] = []
        self._sample_rate = config.microphone_sample_rate
        self._channels = config.microphone_channels
        self._pending_result_task_id: str | None = None
        self._queued_notification_task_ids: deque[str] = deque()
        self._notified_task_ids: set[str] = set()
        self._last_transcript = ""
        self._last_error = ""

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._poll_thread = threading.Thread(target=self._poll_background_tasks_loop, name="local-voice-notify", daemon=True)
        self._poll_thread.start()
        if self._config.microphone_enabled and self._asr_service is not None:
            try:
                self._listener.start()
            except MicrophoneUnavailableError as exc:
                self._last_error = str(exc)
        self._running = True

    def stop(self) -> None:
        self._stop_event.set()
        self._listener.stop()
        if self._poll_thread is not None and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
        self._poll_thread = None
        self._running = False

    def submit_transcript(self, text: str) -> LiveVoiceTurnResult:
        normalized = str(text or "").strip()
        if not normalized:
            return LiveVoiceTurnResult(accepted=False)
        self._last_transcript = normalized
        if self._handle_pending_result_choice(normalized):
            return LiveVoiceTurnResult(accepted=True, session_active=True)
        result = self._live_session.handle_transcript(
            normalized,
            session_id=self._session_id,
            scope_root=self._scope_root,
            runtime_settings=self._runtime_settings,
            metadata={"source": "local_voice"},
            sender_id="local_microphone",
            sender_name="Local Microphone",
        )
        if result.ack_text:
            self._speak_text(result.ack_text)
        if result.reply is not None and result.rendered_reply is None:
            self._speak_text((result.reply.speech_text or result.reply.response).strip())
        return result

    def poll_background_tasks(self) -> None:
        tasks = self._task_service.list_tasks(session_id=self._session_id)
        notification_text = None
        with self._lock:
            if self._pending_result_task_id is None:
                for task in reversed(tasks):
                    if (
                        task.status in self._READY_STATUSES
                        and task.needs_confirmation
                        and not task.acknowledged
                        and task.task_id not in self._notified_task_ids
                        and task.task_id not in self._queued_notification_task_ids
                    ):
                        self._queued_notification_task_ids.append(task.task_id)
            if self._pending_result_task_id is None and self._queued_notification_task_ids:
                task_id = self._queued_notification_task_ids.popleft()
                task = self._task_service.get_task(task_id)
                if task is None or task.acknowledged:
                    return
                self._pending_result_task_id = task.task_id
                self._notified_task_ids.add(task.task_id)
                notification_text = self._build_done_notification(task.task_id)
        if notification_text:
            self._speak_text(notification_text)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "running": self._running,
                "microphone_enabled": self._config.microphone_enabled,
                "microphone_running": self._listener.running,
                "pending_result_task_id": self._pending_result_task_id,
                "queued_notifications": list(self._queued_notification_task_ids),
                "last_transcript": self._last_transcript,
                "last_error": self._last_error,
            }

    def _poll_background_tasks_loop(self) -> None:
        interval = max(0.2, self._config.local_voice_poll_interval_ms / 1000.0)
        while not self._stop_event.wait(interval):
            self.poll_background_tasks()

    def _on_audio_chunk(self, chunk: AudioChunk) -> None:
        if self._asr_service is None:
            return
        now = self._time()
        with self._lock:
            if now < self._cooldown_until:
                return
        is_voice = self._is_voice_chunk(chunk.pcm)
        if not self._capturing:
            if not is_voice:
                return
            self._capturing = True
            self._capture_started_at = now
            self._last_voice_at = now
            self._sample_rate = chunk.sample_rate
            self._channels = chunk.channels
            self._capture_chunks = [chunk.pcm]
            return

        self._capture_chunks.append(chunk.pcm)
        if is_voice:
            self._last_voice_at = now

        elapsed_ms = (now - self._capture_started_at) * 1000.0
        silence_ms = (now - self._last_voice_at) * 1000.0
        if elapsed_ms >= self._config.utterance_max_seconds * 1000:
            self._finalize_capture()
            return
        if elapsed_ms >= self._config.utterance_min_ms and silence_ms >= self._config.utterance_silence_ms:
            self._finalize_capture()

    def _finalize_capture(self) -> None:
        pcm = b"".join(self._capture_chunks)
        self._capturing = False
        self._capture_chunks = []
        if not pcm:
            return
        transcript = self._transcribe_pcm(pcm)
        if transcript:
            self.submit_transcript(transcript)

    def _transcribe_pcm(self, pcm: bytes) -> str:
        temp_dir = Path(self._config.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="live_voice_", suffix=".wav", dir=temp_dir, delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            with wave.open(str(temp_path), "wb") as wav_file:
                wav_file.setnchannels(max(1, self._channels))
                wav_file.setsampwidth(2)
                wav_file.setframerate(max(8_000, self._sample_rate))
                wav_file.writeframes(pcm)
            result = self._asr_service.transcribe_file(str(temp_path))
            if result.ok:
                return result.text.strip()
            if result.error:
                self._last_error = result.error
            return ""
        finally:
            if self._config.cleanup_temp:
                temp_path.unlink(missing_ok=True)

    def _handle_pending_result_choice(self, text: str) -> bool:
        with self._lock:
            task_id = self._pending_result_task_id
        if task_id is None:
            return False
        normalized = self._normalize(text)
        if any(self._normalize(token) in normalized for token in self._POSITIVE_RESULT_HINTS):
            task = self._task_service.get_task(task_id)
            self._task_service.acknowledge_task(task_id)
            with self._lock:
                self._pending_result_task_id = None
            if task is not None:
                spoken = (task.speech_text or task.final_response or "任务已经完成了。").strip()
                self._speak_text(spoken)
            return True
        if any(self._normalize(token) in normalized for token in self._NEGATIVE_RESULT_HINTS):
            self._task_service.acknowledge_task(task_id)
            with self._lock:
                self._pending_result_task_id = None
            declined = self._select_declined_phrase(task_id)
            self._speak_text(declined)
            return True
        return False

    def _build_done_notification(self, task_id: str) -> str:
        phrases = [phrase.strip() for phrase in self._config.task_done_notify_phrases if phrase.strip()]
        if not phrases:
            return "你刚才交给我的任务做完了，要听结果吗？"
        return phrases[hash(task_id) % len(phrases)]

    def _select_declined_phrase(self, task_id: str) -> str:
        phrases = [phrase.strip() for phrase in self._config.task_result_declined_phrases if phrase.strip()]
        if not phrases:
            return "好，那我先不念。"
        return phrases[hash(f"declined:{task_id}") % len(phrases)]

    def _speak_text(self, text: str) -> None:
        normalized = str(text or "").strip()
        if not normalized:
            return
        self._voice_output.synthesize_text(normalized)
        cooldown = max(0.2, float(self._config.self_listen_cooldown_seconds))
        with self._lock:
            self._cooldown_until = self._time() + cooldown

    def _is_voice_chunk(self, pcm: bytes) -> bool:
        if not pcm:
            return False
        sample_count = len(pcm) // 2
        if sample_count <= 0:
            return False
        total = 0.0
        for index in range(0, len(pcm), 2):
            sample = int.from_bytes(pcm[index:index + 2], byteorder="little", signed=True)
            total += sample * sample
        rms = math.sqrt(total / sample_count)
        return rms >= max(50, self._config.audio_activity_threshold)

    @staticmethod
    def _normalize(text: str) -> str:
        return "".join(str(text or "").strip().lower().split())
