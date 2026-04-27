from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, model_validator

from local_agent.protocol.models import OutputKind, ToolManifest
from local_agent.utils.workspace_path import WorkspacePathNormalizer

try:
    from PIL import ImageGrab
except Exception:  # noqa: BLE001
    ImageGrab = None


class ComputerBackend(Protocol):
    def screenshot(self, *, bbox: tuple[int, int, int, int] | None = None, all_screens: bool = True) -> Any:
        ...

    def position(self) -> tuple[int, int]:
        ...

    def move_to(self, x: int, y: int, *, duration: float = 0.0) -> None:
        ...

    def click(self, x: int | None = None, y: int | None = None, *, button: str = "left", clicks: int = 1) -> None:
        ...

    def drag_to(self, x: int, y: int, *, duration: float = 0.2, button: str = "left") -> None:
        ...

    def scroll(self, amount: int, *, x: int | None = None, y: int | None = None) -> None:
        ...

    def type_text(self, text: str, *, interval: float = 0.01) -> None:
        ...

    def press_key(self, key: str) -> None:
        ...

    def hotkey(self, keys: list[str]) -> None:
        ...

    def read_clipboard(self) -> str:
        ...

    def write_clipboard(self, text: str) -> None:
        ...


class PyAutoGuiComputerBackend:
    def __init__(self) -> None:
        self._pyautogui = None

    @property
    def pyautogui(self):
        if self._pyautogui is None:
            try:
                import pyautogui  # type: ignore
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Computer control requires pyautogui. Install dependencies with "
                    "`python -m pip install -e .` or `python -m pip install pyautogui pyperclip`."
                ) from exc
            pyautogui.FAILSAFE = True
            self._pyautogui = pyautogui
        return self._pyautogui

    def screenshot(self, *, bbox: tuple[int, int, int, int] | None = None, all_screens: bool = True):
        if ImageGrab is None:
            raise RuntimeError("Pillow ImageGrab is required for screenshots.")
        return ImageGrab.grab(bbox=bbox, all_screens=all_screens)

    def position(self) -> tuple[int, int]:
        pos = self.pyautogui.position()
        return int(pos.x), int(pos.y)

    def move_to(self, x: int, y: int, *, duration: float = 0.0) -> None:
        self.pyautogui.moveTo(x, y, duration=max(0.0, duration))

    def click(self, x: int | None = None, y: int | None = None, *, button: str = "left", clicks: int = 1) -> None:
        self.pyautogui.click(x=x, y=y, button=button, clicks=max(1, clicks))

    def drag_to(self, x: int, y: int, *, duration: float = 0.2, button: str = "left") -> None:
        self.pyautogui.dragTo(x, y, duration=max(0.0, duration), button=button)

    def scroll(self, amount: int, *, x: int | None = None, y: int | None = None) -> None:
        if x is not None and y is not None:
            self.move_to(x, y)
        self.pyautogui.scroll(amount)

    def type_text(self, text: str, *, interval: float = 0.01) -> None:
        self.pyautogui.write(text, interval=max(0.0, interval))

    def press_key(self, key: str) -> None:
        self.pyautogui.press(key)

    def hotkey(self, keys: list[str]) -> None:
        self.pyautogui.hotkey(*keys)

    def read_clipboard(self) -> str:
        try:
            import pyperclip  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Clipboard support requires pyperclip.") from exc
        value = pyperclip.paste()
        return "" if value is None else str(value)

    def write_clipboard(self, text: str) -> None:
        try:
            import pyperclip  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Clipboard support requires pyperclip.") from exc
        pyperclip.copy(text)


class ScreenshotInput(BaseModel):
    output_path: str | None = None
    include_base64: bool = False
    image_format: str = "PNG"
    all_screens: bool = True
    delay_ms: int = 0


class RegionInput(BaseModel):
    x: int
    y: int
    width: int
    height: int
    output_path: str | None = None
    include_base64: bool = False
    image_format: str = "PNG"
    delay_ms: int = 0

    @model_validator(mode="after")
    def validate_region(self) -> "RegionInput":
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        return self


class CoordinateInput(BaseModel):
    x: int
    y: int
    duration_ms: int = 0


class ClickInput(BaseModel):
    x: int | None = None
    y: int | None = None
    button: str = "left"
    clicks: int = 1

    @model_validator(mode="after")
    def validate_button(self) -> "ClickInput":
        if self.button not in {"left", "right", "middle"}:
            raise ValueError("button must be left, right, or middle")
        if self.clicks < 1 or self.clicks > 3:
            raise ValueError("clicks must be between 1 and 3")
        if (self.x is None) != (self.y is None):
            raise ValueError("x and y must be provided together")
        return self


class DragInput(BaseModel):
    x: int
    y: int
    duration_ms: int = 200
    button: str = "left"


class ScrollInput(BaseModel):
    amount: int = Field(description="Positive scrolls up; negative scrolls down.")
    x: int | None = None
    y: int | None = None

    @model_validator(mode="after")
    def validate_pointer(self) -> "ScrollInput":
        if (self.x is None) != (self.y is None):
            raise ValueError("x and y must be provided together")
        return self


class TypeInput(BaseModel):
    text: str
    interval_ms: int = 10
    paste_via_clipboard: bool = False


class KeyInput(BaseModel):
    key: str


class HotkeyInput(BaseModel):
    keys: list[str]

    @model_validator(mode="after")
    def validate_keys(self) -> "HotkeyInput":
        if not self.keys:
            raise ValueError("keys must not be empty")
        if len(self.keys) > 4:
            raise ValueError("hotkeys are limited to 4 keys")
        return self


class ClipboardWriteInput(BaseModel):
    text: str


class WaitInput(BaseModel):
    duration_ms: int = 1000


class ComputerUseModule:
    def __init__(
        self,
        workspace_root: str,
        backend: ComputerBackend | None = None,
        enabled: bool = True,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.path_normalizer = WorkspacePathNormalizer(str(self.workspace_root))
        self.backend = backend or PyAutoGuiComputerBackend()
        self.enabled = enabled

    def manifests(self) -> list[ToolManifest]:
        return [
            ToolManifest(
                tool_name="computer.screenshot",
                module="computer_use",
                description="Capture the current desktop screen. Optionally save the image inside the workspace or return base64 image data.",
                side_effect=False,
                idempotent=False,
                timeout_ms=15_000,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=ScreenshotInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"width": {"type": "integer"}, "height": {"type": "integer"}}},
            ),
            ToolManifest(
                tool_name="computer.capture_region",
                module="computer_use",
                description="Capture a rectangular desktop region by screen pixel coordinates.",
                side_effect=False,
                idempotent=False,
                timeout_ms=15_000,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=RegionInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"width": {"type": "integer"}, "height": {"type": "integer"}}},
            ),
            ToolManifest(
                tool_name="computer.cursor_position",
                module="computer_use",
                description="Return the current mouse cursor position in screen pixels.",
                side_effect=False,
                idempotent=False,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema={},
                output_schema={"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}},
            ),
            ToolManifest(
                tool_name="computer.mouse_move",
                module="computer_use",
                description="Move the mouse cursor to screen pixel coordinates.",
                side_effect=True,
                idempotent=False,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=CoordinateInput.model_json_schema(),
                output_schema={"type": "object"},
            ),
            ToolManifest(
                tool_name="computer.click",
                module="computer_use",
                description="Click at the current cursor position or at screen pixel coordinates. Supports left, right, middle, double, and triple clicks.",
                side_effect=True,
                idempotent=False,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=ClickInput.model_json_schema(),
                output_schema={"type": "object"},
            ),
            ToolManifest(
                tool_name="computer.drag_to",
                module="computer_use",
                description="Drag the mouse from its current position to target screen pixel coordinates.",
                side_effect=True,
                idempotent=False,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=DragInput.model_json_schema(),
                output_schema={"type": "object"},
            ),
            ToolManifest(
                tool_name="computer.scroll",
                module="computer_use",
                description="Scroll vertically at the current cursor position or at screen pixel coordinates.",
                side_effect=True,
                idempotent=False,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=ScrollInput.model_json_schema(),
                output_schema={"type": "object"},
            ),
            ToolManifest(
                tool_name="computer.type_text",
                module="computer_use",
                description="Type text into the active application. For non-ASCII or long text, paste_via_clipboard is usually more reliable.",
                side_effect=True,
                idempotent=False,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=TypeInput.model_json_schema(),
                output_schema={"type": "object"},
            ),
            ToolManifest(
                tool_name="computer.key",
                module="computer_use",
                description="Press a single keyboard key such as enter, escape, tab, backspace, or f5.",
                side_effect=True,
                idempotent=False,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=KeyInput.model_json_schema(),
                output_schema={"type": "object"},
            ),
            ToolManifest(
                tool_name="computer.hotkey",
                module="computer_use",
                description="Press a keyboard shortcut such as ctrl+c or alt+tab. Avoid destructive system shortcuts.",
                side_effect=True,
                idempotent=False,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=HotkeyInput.model_json_schema(),
                output_schema={"type": "object"},
            ),
            ToolManifest(
                tool_name="computer.read_clipboard",
                module="computer_use",
                description="Read plain text from the system clipboard.",
                side_effect=False,
                idempotent=False,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema={},
                output_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            ),
            ToolManifest(
                tool_name="computer.write_clipboard",
                module="computer_use",
                description="Write plain text to the system clipboard.",
                side_effect=True,
                idempotent=False,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=ClipboardWriteInput.model_json_schema(),
                output_schema={"type": "object"},
            ),
            ToolManifest(
                tool_name="computer.wait",
                module="computer_use",
                description="Wait for a short duration before taking another screenshot or action.",
                side_effect=False,
                idempotent=False,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=WaitInput.model_json_schema(),
                output_schema={"type": "object"},
            ),
        ]

    def executor_map(self) -> dict[str, Any]:
        return {
            "computer.screenshot": self.screenshot,
            "computer.capture_region": self.capture_region,
            "computer.cursor_position": self.cursor_position,
            "computer.mouse_move": self.mouse_move,
            "computer.click": self.click,
            "computer.drag_to": self.drag_to,
            "computer.scroll": self.scroll,
            "computer.type_text": self.type_text,
            "computer.key": self.key,
            "computer.hotkey": self.hotkey,
            "computer.read_clipboard": self.read_clipboard,
            "computer.write_clipboard": self.write_clipboard,
            "computer.wait": self.wait,
        }

    def screenshot(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_enabled()
        payload = ScreenshotInput.model_validate(arguments)
        self._sleep(payload.delay_ms)
        image = self.backend.screenshot(all_screens=payload.all_screens)
        return self._image_result(
            image,
            output_path=payload.output_path,
            include_base64=payload.include_base64,
            image_format=payload.image_format,
            capture_kind="screen",
        )

    def capture_region(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_enabled()
        payload = RegionInput.model_validate(arguments)
        self._sleep(payload.delay_ms)
        bbox = (payload.x, payload.y, payload.x + payload.width, payload.y + payload.height)
        image = self.backend.screenshot(bbox=bbox, all_screens=False)
        result = self._image_result(
            image,
            output_path=payload.output_path,
            include_base64=payload.include_base64,
            image_format=payload.image_format,
            capture_kind="region",
        )
        result["bbox"] = {"x": payload.x, "y": payload.y, "width": payload.width, "height": payload.height}
        return result

    def cursor_position(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_enabled()
        x, y = self.backend.position()
        return {"x": x, "y": y}

    def mouse_move(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_enabled()
        payload = CoordinateInput.model_validate(arguments)
        self.backend.move_to(payload.x, payload.y, duration=payload.duration_ms / 1000.0)
        return {"moved": True, "x": payload.x, "y": payload.y}

    def click(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_enabled()
        payload = ClickInput.model_validate(arguments)
        self.backend.click(payload.x, payload.y, button=payload.button, clicks=payload.clicks)
        return {"clicked": True, "x": payload.x, "y": payload.y, "button": payload.button, "clicks": payload.clicks}

    def drag_to(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_enabled()
        payload = DragInput.model_validate(arguments)
        self.backend.drag_to(payload.x, payload.y, duration=payload.duration_ms / 1000.0, button=payload.button)
        return {"dragged": True, "x": payload.x, "y": payload.y, "button": payload.button}

    def scroll(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_enabled()
        payload = ScrollInput.model_validate(arguments)
        self.backend.scroll(payload.amount, x=payload.x, y=payload.y)
        return {"scrolled": True, "amount": payload.amount, "x": payload.x, "y": payload.y}

    def type_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_enabled()
        payload = TypeInput.model_validate(arguments)
        if payload.paste_via_clipboard:
            original = self.backend.read_clipboard()
            try:
                self.backend.write_clipboard(payload.text)
                modifier = "command" if sys.platform == "darwin" else "ctrl"
                self.backend.hotkey([modifier, "v"])
            finally:
                self.backend.write_clipboard(original)
            return {"typed": True, "method": "clipboard_paste", "chars": len(payload.text)}
        self.backend.type_text(payload.text, interval=payload.interval_ms / 1000.0)
        return {"typed": True, "method": "keyboard", "chars": len(payload.text)}

    def key(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_enabled()
        payload = KeyInput.model_validate(arguments)
        self._reject_blocked_hotkey([payload.key])
        self.backend.press_key(payload.key)
        return {"pressed": True, "key": payload.key}

    def hotkey(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_enabled()
        payload = HotkeyInput.model_validate(arguments)
        normalized = [str(key).strip().lower() for key in payload.keys if str(key).strip()]
        self._reject_blocked_hotkey(normalized)
        self.backend.hotkey(normalized)
        return {"pressed": True, "keys": normalized}

    def read_clipboard(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_enabled()
        return {"text": self.backend.read_clipboard()}

    def write_clipboard(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_enabled()
        payload = ClipboardWriteInput.model_validate(arguments)
        self.backend.write_clipboard(payload.text)
        return {"written": True, "chars": len(payload.text)}

    def wait(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = WaitInput.model_validate(arguments)
        if payload.duration_ms < 0 or payload.duration_ms > 60_000:
            raise ValueError("duration_ms must be between 0 and 60000")
        self._sleep(payload.duration_ms)
        return {"waited_ms": payload.duration_ms}

    def _image_result(
        self,
        image,
        *,
        output_path: str | None,
        include_base64: bool,
        image_format: str,
        capture_kind: str,
    ) -> dict[str, Any]:
        fmt = self._normalize_image_format(image_format)
        result: dict[str, Any] = {
            "captured": True,
            "capture_kind": capture_kind,
            "width": int(image.width),
            "height": int(image.height),
            "format": fmt.lower(),
        }
        if output_path:
            target = self._resolve_output_path(output_path, fmt)
            image.save(target, format=fmt)
            result["path"] = str(target)
        if include_base64:
            buffer = io.BytesIO()
            image.save(buffer, format=fmt)
            result["base64"] = base64.b64encode(buffer.getvalue()).decode("ascii")
            result["mime_type"] = f"image/{fmt.lower()}"
        return result

    def _resolve_output_path(self, raw_path: str, image_format: str) -> Path:
        target = self.path_normalizer.resolve(raw_path)
        if not target.suffix:
            target = target.with_suffix("." + image_format.lower())
        self._ensure_workspace_path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _ensure_workspace_path(self, target: Path) -> None:
        try:
            target.resolve().relative_to(self.workspace_root)
        except ValueError as exc:
            raise PermissionError(f"Path is outside workspace: {target}") from exc

    def _ensure_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError("Computer Use is disabled.")

    @staticmethod
    def _sleep(delay_ms: int) -> None:
        if delay_ms < 0 or delay_ms > 60_000:
            raise ValueError("delay_ms must be between 0 and 60000")
        if delay_ms:
            time.sleep(delay_ms / 1000.0)

    @staticmethod
    def _normalize_image_format(raw_format: str) -> str:
        normalized = str(raw_format or "PNG").strip().upper()
        if normalized in {"JPG", "JPEG"}:
            return "JPEG"
        if normalized == "PNG":
            return "PNG"
        raise ValueError("image_format must be PNG or JPEG")

    @staticmethod
    def _reject_blocked_hotkey(keys: list[str]) -> None:
        normalized = {str(key).strip().lower() for key in keys}
        if not normalized:
            raise ValueError("key list must not be empty")
        dangerous = [
            {"alt", "f4"},
            {"ctrl", "alt", "delete"},
            {"win", "l"},
            {"cmd", "q"},
            {"command", "q"},
        ]
        if any(combo.issubset(normalized) for combo in dangerous):
            raise PermissionError(f"Blocked potentially disruptive hotkey: {'+'.join(keys)}")
