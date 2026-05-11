from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone

from local_agent.protocol.models import OverseerConfig


class OverseerService:
    def __init__(
        self,
        config: OverseerConfig,
        llm_client,
        send_to_qq=None,
        trace_store=None,
    ) -> None:
        self._config = config
        self._llm = llm_client
        self._send_to_qq = send_to_qq
        self._trace_store = trace_store
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_intervention_at = 0.0
        self._consecutive_quiet = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="overseer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---------- internal ----------

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                screenshot = self._capture()
                if screenshot is None:
                    self._sleep(self._config.poll_interval_seconds)
                    continue

                result = self._analyze(screenshot)
                self._log_event("overseer_analysis", {
                    "activity": result.get("activity", ""),
                    "suggestion": result.get("suggestion", ""),
                    "should_intervene": result.get("should_intervene", False),
                })

                if result.get("should_intervene"):
                    suggestion = str(result.get("suggestion") or "").strip()
                    if suggestion:
                        self._intervene(suggestion)
                        self._consecutive_quiet = 0
                    else:
                        self._consecutive_quiet += 1
                else:
                    self._consecutive_quiet += 1
            except Exception as exc:
                self._log_event("overseer_error", {"error": str(exc)})

            interval = (
                self._config.poll_interval_seconds
                if self._consecutive_quiet < 3
                else self._config.quiet_cooldown_seconds
            )
            self._sleep(interval)

    def __init__(
        self,
        config: OverseerConfig,
        llm_client,
        send_to_qq=None,
        trace_store=None,
    ) -> None:
        self._config = config
        self._llm = llm_client
        self._send_to_qq = send_to_qq
        self._trace_store = trace_store
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_intervention_at = 0.0
        self._consecutive_quiet = 0
        self._last_clipboard = ""  # 追踪剪贴板变化
        self._last_selection = ""  # 追踪选中文本变化

    def _read_clipboard_safe(self) -> str:
        try:
            import pyperclip
            return str(pyperclip.paste() or "").strip()
        except Exception:
            return ""

    def _try_copy_selection(self) -> str | None:
        """尝试 Ctrl+C 复制当前选中文本，返回新选中的内容（如果有变化）。"""
        import pyautogui

        old_clipboard = self._read_clipboard_safe()
        try:
            pyautogui.hotkey("ctrl", "c")
        except Exception:
            return None
        time.sleep(0.15)
        new_text = self._read_clipboard_safe()

        # 恢复旧剪贴板（如果是我们改的）
        if new_text != old_clipboard and old_clipboard:
            try:
                import pyperclip
                pyperclip.copy(old_clipboard)
            except Exception:
                pass

        if new_text and new_text != old_clipboard and new_text != self._last_selection:
            self._last_selection = new_text
            return new_text
        return None

    def _capture(self) -> str | None:
        """采集当前上下文信息（全文本，不需要视觉模型）。"""
        parts: list[str] = []
        try:
            import ctypes

            # 1. 前台窗口标题
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            buf = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(hwnd, buf, 511)
            title = buf.value.strip()
            parts.append(f"前台窗口标题: {title}" if title else "前台窗口标题: (无法获取)")

            # 2. 用户框选的内容（尝试 Ctrl+C 读取）
            selection = self._try_copy_selection()
            if selection:
                parts.append(f"用户选中/高亮的文本: {selection[:500]}")

            # 3. 剪贴板内容（前 300 字符，和选中不同时才显示）
            clip = self._read_clipboard_safe()
            if clip and clip.strip() and clip != selection:
                parts.append(f"剪贴板: {clip.strip()[:300]}")

            # 4. 可见窗口列表
            visible_windows = []
            def enum_cb(h, _):
                if not user32.IsWindowVisible(h):
                    return True
                wbuf = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(h, wbuf, 255)
                wt = wbuf.value.strip()
                if wt and len(wt) > 2:
                    visible_windows.append(wt)
                return True
            user32.EnumWindows(
                ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_long, ctypes.c_long)(enum_cb),
                0,
            )
            if visible_windows:
                notable = sorted(set(visible_windows), key=len, reverse=True)[:15]
                parts.append(f"可见窗口: {', '.join(notable)}")
        except Exception:
            parts.append("(环境感知初始化失败)")

        context = "\n".join(parts)
        if not context.strip():
            return None
        return context

    def _analyze(self, context_text: str) -> dict:
        prompt = (
            "你正在后台感知用户的桌面环境，像一个沉默但关注一切的助手。\n"
            "根据以下上下文信息，简要回答（只返回 JSON）：\n"
            '{"activity": "用户正在做什么（一句话）", "suggestion": "可以主动帮忙的事（不需要就留空）", "should_intervene": true/false}\n\n'
            f"上下文:\n{context_text}\n\n"
            "只有真有价值时才 should_intervene=true。保持银狼的冷静简洁风格。只返回 JSON。"
        )
        try:
            raw = self._llm._chat(
                [
                    {"role": "system", "content": "你是银狼，冷静的桌面守望者。返回纯 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                model=self._llm.model,
            )
        except Exception:
            return {"activity": "", "suggestion": "", "should_intervene": False}
        return self._parse_analysis_json(raw)

    @staticmethod
    def _parse_analysis_json(raw: str) -> dict:
        if not raw or not raw.strip():
            return {"activity": "", "suggestion": "", "should_intervene": False}

        # 尝试提取 JSON
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(raw[start:end + 1])
                if isinstance(result, dict):
                    return {
                        "activity": str(result.get("activity", "")).strip(),
                        "suggestion": str(result.get("suggestion", "")).strip(),
                        "should_intervene": bool(
                            result.get("should_intervene", False)
                            or result.get("shouldIntervene", False)
                        ),
                    }
            except json.JSONDecodeError:
                pass
        return {"activity": "", "suggestion": "", "should_intervene": False}

    def _intervene(self, suggestion: str) -> None:
        now = time.time()
        if now - self._last_intervention_at < self._config.min_poll_interval_seconds:
            # 不要太频繁
            return
        self._last_intervention_at = now

        message = f"[👀] {suggestion}"
        if self._send_to_qq is not None:
            try:
                self._send_to_qq(
                    message=message,
                    session_id=self._config.qq_session_id,
                )
            except Exception as exc:
                self._log_event("overseer_send_error", {"error": str(exc)})

    def _log_event(self, event_type: str, payload: dict) -> None:
        if self._trace_store is not None and hasattr(self._trace_store, "append"):
            try:
                self._trace_store.append(
                    event_type,
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        **payload,
                    },
                )
            except Exception:
                pass

    @staticmethod
    def _sleep(seconds: float) -> None:
        time.sleep(max(0.5, seconds))
