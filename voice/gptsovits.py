from __future__ import annotations

import subprocess
import shutil
import threading
import time
import uuid
import winsound
from pathlib import Path

import requests

from local_agent.protocol.models import VoiceConfig


class GPTSoVITSAdapter:
    def __init__(self, config: VoiceConfig) -> None:
        self.config = config
        self._server_process: subprocess.Popen[str] | None = None
        self._weights_ready = False
        self._prepared_ref_audio_path: str | None = None
        self._speak_lock = threading.Lock()

    def _docs_url(self) -> str:
        return f"{self.config.endpoint.rstrip('/')}/docs"

    def _api_url(self, path: str) -> str:
        return f"{self.config.endpoint.rstrip('/')}{path}"

    def _is_server_ready(self) -> bool:
        try:
            response = requests.get(self._docs_url(), timeout=3)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _wait_until_ready(self, timeout_seconds: int = 120) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if self._is_server_ready():
                return
            time.sleep(1)
        raise TimeoutError("GPT-SoVITS API did not become ready in time.")

    def _start_server(self) -> bool:
        if self._is_server_ready():
            return False
        if not self.config.auto_start_server:
            raise RuntimeError("GPT-SoVITS API is not running and auto_start_server is disabled.")

        python_path = self.config.runtime_python_path()
        script_path = self.config.api_script_path()
        root = Path(self.config.gptsovits_root)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        startupinfo = None
        creationflags = 0
        if hasattr(subprocess, "STARTUPINFO"):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        self._server_process = subprocess.Popen(
            [
                str(python_path),
                str(script_path),
                "-a",
                self.config.api_host,
                "-p",
                str(self.config.api_port),
            ],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            startupinfo=startupinfo,
            creationflags=creationflags,
        )
        self._wait_until_ready()
        self._weights_ready = False
        return True

    def _shutdown_server(self) -> None:
        try:
            requests.get(self._api_url("/control"), params={"command": "exit"}, timeout=10)
        except requests.RequestException:
            pass

        if self._server_process is not None:
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            finally:
                self._server_process = None
        self._weights_ready = False

    def _set_weights(self) -> None:
        if self._weights_ready:
            return
        for path, endpoint in (
            (self.config.gpt_weights_path, "/set_gpt_weights"),
            (self.config.sovits_weights_path, "/set_sovits_weights"),
        ):
            response = requests.get(self._api_url(endpoint), params={"weights_path": path}, timeout=60)
            response.raise_for_status()
        self._weights_ready = True

    def _prepare_ref_audio_path(self) -> str:
        if self._prepared_ref_audio_path is not None:
            return self._prepared_ref_audio_path

        source = Path(self.config.ref_audio_path)
        if not source.exists():
            fallback = self._locate_fallback_ref_audio(source)
            if fallback is not None:
                source = fallback
            else:
                raise FileNotFoundError(f"Reference audio not found: {self.config.ref_audio_path}")

        if all(ord(char) < 128 for char in str(source)):
            self._prepared_ref_audio_path = str(source)
            return self._prepared_ref_audio_path

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "silverwolf_reference.wav"
        shutil.copyfile(source, target)
        self._prepared_ref_audio_path = str(target)
        return self._prepared_ref_audio_path

    def _locate_fallback_ref_audio(self, original: Path) -> Path | None:
        candidates: list[Path] = []
        if original.parent.exists():
            candidates.extend(sorted(original.parent.glob("*.wav")))

        weights_dir = Path(self.config.gpt_weights_path).resolve().parent
        if weights_dir.exists() and weights_dir != original.parent:
            candidates.extend(sorted(weights_dir.glob("*.wav")))

        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_file():
                return candidate
        return None

    def _resolve_prompt_text(self) -> str:
        if self.config.prompt_text.strip():
            return self.config.prompt_text.strip()
        return Path(self.config.ref_audio_path).stem

    def _synthesize(self, text: str) -> bytes:
        payload = {
            "text": text,
            "text_lang": self.config.text_lang,
            "ref_audio_path": self._prepare_ref_audio_path(),
            "prompt_text": self._resolve_prompt_text(),
            "prompt_lang": self.config.prompt_lang,
            # The local GPT-SoVITS api_v2 instance on this machine accepts
            # `cut0` reliably; `cut5` was causing "请输入有效文本".
            "text_split_method": "cut0",
            "batch_size": 1,
            "media_type": "wav",
            "streaming_mode": False,
        }
        payload.update(self.config.extra_payload)
        response = requests.post(self._api_url("/tts"), json=payload, timeout=300)
        response.raise_for_status()
        return response.content

    def _write_temp_audio(self, wav_bytes: bytes) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_name = f"agent_tts_{uuid.uuid4().hex}.wav"
        temp_path = output_dir / temp_name
        temp_path.write_bytes(wav_bytes)
        return temp_path

    def _play_and_cleanup(self, wav_bytes: bytes) -> None:
        temp_path = self._write_temp_audio(wav_bytes)
        try:
            if self.config.play_audio:
                winsound.PlaySound(str(temp_path), winsound.SND_FILENAME)
        finally:
            if self.config.cleanup_audio and temp_path.exists():
                temp_path.unlink()

    def speak(self, text: str) -> bool:
        if not self.config.enabled:
            return False

        with self._speak_lock:
            started_here = self._start_server()
            try:
                self._set_weights()
                wav_bytes = self._synthesize(text)
                self._play_and_cleanup(wav_bytes)
                return True
            finally:
                if started_here and self.config.shutdown_when_idle:
                    self._shutdown_server()

    def dispatch(self, text: str) -> bool:
        if not self.config.enabled:
            return False
        if self.config.async_playback:
            worker = threading.Thread(target=self.speak, args=(text,), daemon=True)
            worker.start()
            return True
        return self.speak(text)

    def synthesize_to_file(self, text: str) -> str | None:
        if not self.config.enabled:
            return None

        with self._speak_lock:
            started_here = self._start_server()
            try:
                self._set_weights()
                wav_bytes = self._synthesize(text)
                return str(self._write_temp_audio(wav_bytes))
            finally:
                if started_here and self.config.shutdown_when_idle:
                    self._shutdown_server()
