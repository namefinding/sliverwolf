from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from local_agent.protocol.models import OutputKind, ToolManifest


_QMUSIC_EXE = r"C:\Program Files (x86)\Tencent\QQMusic\QQMusic.exe"
_QMUSIC_PROCESS = "QQMusic.exe"


class PlayMusicInput(BaseModel):
    query: str = Field(
        default="",
        alias="query",
        description="song name, or 'artist + song name'",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_query(cls, data: Any) -> Any:
        """LLM 有时会传 song 而不是 query，自动对齐."""
        if isinstance(data, dict):
            for alias in ("song", "name", "track", "music"):
                if alias in data and not data.get("query"):
                    data["query"] = str(data.pop(alias)).strip()
        return data

    @model_validator(mode="after")
    def validate_query(self) -> "PlayMusicInput":
        if not str(self.query or "").strip():
            raise ValueError("query must not be empty")
        self.query = str(self.query).strip()
        return self


class MusicControlInput(BaseModel):
    action: str = Field(description="play, pause, next, prev, volume_up, volume_down")

    @model_validator(mode="after")
    def validate_action(self) -> "MusicControlInput":
        valid = {"play", "pause", "next", "prev", "volume_up", "volume_down"}
        normalized = str(self.action or "").strip().lower()
        if normalized not in valid:
            raise ValueError(f"action must be one of: {', '.join(sorted(valid))}")
        self.action = normalized
        return self


class AppControlModule:
    def __init__(self, qmusic_exe: str | None = None) -> None:
        self.qmusic_exe = Path(qmusic_exe or _QMUSIC_EXE)

    def manifests(self) -> list[ToolManifest]:
        return [
            ToolManifest(
                tool_name="app.play_music",
                module="app_control",
                description="Search and play a song in QQ Music. Opens QQ Music if it is not already running.",
                side_effect=True,
                idempotent=False,
                requires_confirmation=True,
                timeout_ms=30_000,
                produces=[OutputKind.OBJECT_DETAILS, OutputKind.MESSAGE_SENT],
                input_schema=PlayMusicInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"playing": {"type": "boolean"}, "query": {"type": "string"}}},
            ),
            ToolManifest(
                tool_name="app.music_control",
                module="app_control",
                description="Control QQ Music playback: play, pause, next track, previous track, volume up, or volume down.",
                side_effect=True,
                idempotent=False,
                requires_confirmation=True,
                timeout_ms=15_000,
                produces=[OutputKind.OBJECT_DETAILS, OutputKind.MESSAGE_SENT],
                input_schema=MusicControlInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"action": {"type": "string"}, "success": {"type": "boolean"}}},
            ),
            ToolManifest(
                tool_name="app.open_qmusic",
                module="app_control",
                description="Launch QQ Music if it is not already running.",
                side_effect=True,
                idempotent=True,
                requires_confirmation=True,
                timeout_ms=20_000,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema={"type": "object", "properties": {}},
                output_schema={"type": "object", "properties": {"launched": {"type": "boolean"}}},
            ),
        ]

    def executor_map(self) -> dict[str, Any]:
        return {
            "app.play_music": self.play_music,
            "app.music_control": self.music_control,
            "app.open_qmusic": self.open_qmusic,
        }

    def play_music(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = PlayMusicInput.model_validate(arguments)
        self._ensure_qmusic_running()
        time.sleep(1.0)
        self._search_and_play(payload.query)
        return {"playing": True, "query": payload.query}

    def music_control(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = MusicControlInput.model_validate(arguments)
        self._ensure_qmusic_running()
        self._focus_qmusic()
        time.sleep(0.3)

        action = payload.action
        if action == "play" or action == "pause":
            self._pyautogui().press("space")
        elif action == "next":
            self._pyautogui().hotkey("ctrl", "right")
        elif action == "prev":
            self._pyautogui().hotkey("ctrl", "left")
        elif action == "volume_up":
            self._pyautogui().hotkey("ctrl", "up")
        elif action == "volume_down":
            self._pyautogui().hotkey("ctrl", "down")

        return {"action": action, "success": True}

    def open_qmusic(self, arguments: dict[str, Any]) -> dict[str, Any]:
        already_running = self._is_qmusic_running()
        if not already_running:
            self._launch_qmusic()
        return {"launched": not already_running}

    # ---------- internal helpers ----------

    @staticmethod
    def _pyautogui():
        import pyautogui
        pyautogui.FAILSAFE = True
        return pyautogui

    def _is_qmusic_running(self) -> bool:
        try:
            output = subprocess.run(
                ["tasklist", "/fi", f"imagename eq {_QMUSIC_PROCESS}"],
                capture_output=True,
                text=True,
                check=False,
            )
            return _QMUSIC_PROCESS.lower() in output.stdout.lower()
        except Exception:
            return False

    def _launch_qmusic(self) -> None:
        if not self.qmusic_exe.is_file():
            raise RuntimeError(f"QQ Music executable not found: {self.qmusic_exe}")
        subprocess.Popen([str(self.qmusic_exe)], shell=True)
        # Wait for app to be ready
        for _ in range(30):
            if self._is_qmusic_running():
                return
            time.sleep(1.0)
        raise RuntimeError("QQ Music did not start in time.")

    def _ensure_qmusic_running(self) -> None:
        """确保 QQ 音乐在运行并在前台。如果没开就启动。"""
        if not self._is_qmusic_running():
            self._launch_qmusic()
            # 等窗口完全就绪
            time.sleep(5.0)
        else:
            time.sleep(0.5)

    @classmethod
    def _search_and_play(cls, query: str) -> None:
        import pyautogui
        import pyperclip

        pyautogui.FAILSAFE = True

        # 在 QQ 音乐窗口左上角区域点击确保焦点
        pyautogui.click(400, 80)
        time.sleep(0.3)

        # Ctrl+F 打开搜索框（QQ 音乐默认焦点可能在搜索框，双保险）
        pyautogui.hotkey("ctrl", "f")
        time.sleep(0.5)

        # 用剪贴板粘贴中文歌名（比 pyautogui.write 更可靠）
        pyperclip.copy(query)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.4)

        # 回车搜索
        pyautogui.press("enter")
        time.sleep(3.0)

        # 回车播放第一首
        pyautogui.press("enter")
