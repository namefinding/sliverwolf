from __future__ import annotations

import io
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreparedAudio:
    file_path: str
    temporary: bool = False


class AudioPreprocessor:
    def __init__(self, *, silk_sample_rate: int = 24000) -> None:
        self._silk_sample_rate = silk_sample_rate

    def prepare_for_asr(self, file_path: str) -> PreparedAudio:
        source_path = Path(file_path).resolve()
        if self._is_silk_audio(source_path):
            return self._decode_silk_to_wav(source_path)
        return PreparedAudio(file_path=str(source_path), temporary=False)

    @staticmethod
    def _is_silk_audio(path: Path) -> bool:
        try:
            with path.open("rb") as handle:
                header = handle.read(16)
        except OSError:
            return False
        return b"#!SILK_V3" in header

    def _decode_silk_to_wav(self, source_path: Path) -> PreparedAudio:
        try:
            import pysilk
        except ImportError as exc:  # pragma: no cover - dependency presence tested indirectly
            raise RuntimeError("SILK audio decoding requires the 'silk-python' package.") from exc

        file_descriptor, temp_name = tempfile.mkstemp(prefix="onebot_silk_", suffix=".wav")
        os.close(file_descriptor)
        output_path = Path(temp_name)
        try:
            pcm_buffer = io.BytesIO()
            with source_path.open("rb") as source_handle:
                pysilk.decode(source_handle, pcm_buffer, self._silk_sample_rate)
            with wave.open(str(output_path), "wb") as wav_handle:
                wav_handle.setnchannels(1)
                wav_handle.setsampwidth(2)
                wav_handle.setframerate(self._silk_sample_rate)
                wav_handle.writeframes(pcm_buffer.getvalue())
        except Exception:
            output_path.unlink(missing_ok=True)
            raise

        return PreparedAudio(file_path=str(output_path.resolve()), temporary=True)
