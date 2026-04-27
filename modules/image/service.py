from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from local_agent.protocol.models import OutputKind, ToolManifest
from local_agent.utils.workspace_path import WorkspacePathNormalizer

try:
    from PIL import Image, ImageGrab, ImageStat
except Exception:  # noqa: BLE001
    Image = None
    ImageGrab = None
    ImageStat = None


class ImageInspectInput(BaseModel):
    path: str
    include_ocr: bool = True
    ocr_max_chars: int = 4000


class ImageReadTextInput(BaseModel):
    paths: list[str]
    max_chars: int = 4000


class ImageDescribeInput(BaseModel):
    path: str
    prompt: str | None = None
    focus: str = Field(default="general")
    include_ocr: bool = True
    ocr_max_chars: int = 2000
    max_description_chars: int = 1200


class CaptureScreenInput(BaseModel):
    output_path: str
    delay_ms: int = 0
    all_screens: bool = True


class CaptureRegionInput(BaseModel):
    output_path: str
    x: int
    y: int
    width: int
    height: int
    delay_ms: int = 0


class ImageModule:
    def __init__(
        self,
        workspace_root: str,
        vision_describer: Callable[..., str] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.path_normalizer = WorkspacePathNormalizer(str(self.workspace_root))
        self.vision_describer = vision_describer

    def manifests(self) -> list[ToolManifest]:
        return [
            ToolManifest(
                tool_name="image.inspect",
                module="image",
                description="Inspect an image file, including size, format, average color, and optional OCR text when available.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=ImageInspectInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"path": {"type": "string"}, "width": {"type": "integer"}}},
            ),
            ToolManifest(
                tool_name="image.describe",
                module="image",
                description="Describe the semantic content of an image, optionally using a configured vision model and OCR as support.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.OBJECT_DETAILS],
                input_schema=ImageDescribeInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"path": {"type": "string"}, "summary": {"type": "string"}}},
            ),
            ToolManifest(
                tool_name="image.read_text",
                module="image",
                description="Extract readable text from one or more images using OCR.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.FILE_CONTENTS],
                input_schema=ImageReadTextInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"files": {"type": "array"}}},
            ),
            ToolManifest(
                tool_name="image.capture_screen",
                module="image",
                description="Capture the current screen to an image file inside the workspace. This can be combined with file.open_path to open a UI and then take a screenshot.",
                side_effect=True,
                idempotent=False,
                produces=[OutputKind.FILE_WRITTEN],
                input_schema=CaptureScreenInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
            ToolManifest(
                tool_name="image.capture_region",
                module="image",
                description="Capture a screen region to an image file inside the workspace.",
                side_effect=True,
                idempotent=False,
                produces=[OutputKind.FILE_WRITTEN],
                input_schema=CaptureRegionInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
        ]

    def executor_map(self) -> dict[str, Any]:
        return {
            "image.inspect": self.inspect,
            "image.describe": self.describe,
            "image.read_text": self.read_text,
            "image.capture_screen": self.capture_screen,
            "image.capture_region": self.capture_region,
        }

    def inspect(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = ImageInspectInput.model_validate(arguments)
        image_path = self._resolve_existing_image(payload.path)
        image = self._open_image(image_path)
        details = self._base_details(image_path, image)
        if payload.include_ocr:
            ocr = self._try_ocr_image(image_path, max_chars=payload.ocr_max_chars)
            details.update(
                {
                    "ocr_available": ocr["available"],
                    "ocr_backend": ocr["backend"],
                    "ocr_text": ocr["text"],
                    "ocr_error": ocr["error"],
                }
            )
        return details

    def read_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = ImageReadTextInput.model_validate(arguments)
        files: list[dict[str, Any]] = []
        for raw_path in payload.paths:
            image_path = self._resolve_existing_image(raw_path)
            ocr = self._ocr_image(image_path, max_chars=payload.max_chars)
            files.append(
                {
                    "path": str(image_path),
                    "content": ocr["text"],
                    "ocr_backend": ocr["backend"],
                    "mime_type": self._mime_type_for_path(image_path),
                }
            )
        return {"files": files}

    def describe(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = ImageDescribeInput.model_validate(arguments)
        image_path = self._resolve_existing_image(payload.path)
        image = self._open_image(image_path)
        details = self._base_details(image_path, image)
        focus = self._normalize_focus(payload.focus)
        ocr = None
        if payload.include_ocr:
            ocr = self._try_ocr_image(image_path, max_chars=payload.ocr_max_chars)
            details.update(
                {
                    "ocr_available": ocr["available"],
                    "ocr_backend": ocr["backend"],
                    "ocr_text": ocr["text"],
                    "ocr_error": ocr["error"],
                }
            )

        prompt = self._build_describe_prompt(
            image_name=image_path.name,
            focus=focus,
            user_prompt=payload.prompt,
            ocr_text="" if ocr is None else str(ocr.get("text", "") or ""),
        )
        summary = ""
        backend = "fallback"
        error = None
        if self.vision_describer is not None:
            try:
                summary = str(self.vision_describer(image_path=image_path, prompt=prompt)).strip()
                backend = "ollama_vision"
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
        if not summary:
            summary = self._fallback_description(details=details, focus=focus, ocr=ocr, vision_error=error)
            backend = "fallback"

        if len(summary) > payload.max_description_chars:
            summary = summary[: payload.max_description_chars].rstrip() + "…"

        details.update(
            {
                "summary": summary,
                "focus": focus,
                "description_backend": backend,
                "description_prompt": prompt,
            }
        )
        if error is not None:
            details["description_error"] = error
        return details

    def capture_screen(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = CaptureScreenInput.model_validate(arguments)
        target = self._resolve_output_path(payload.output_path)
        image = self._grab_screen(delay_ms=payload.delay_ms, all_screens=payload.all_screens)
        self._save_image(image, target)
        return {
            "path": str(target),
            "captured": True,
            "width": image.width,
            "height": image.height,
            "capture_kind": "screen",
        }

    def capture_region(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = CaptureRegionInput.model_validate(arguments)
        if payload.width <= 0 or payload.height <= 0:
            raise ValueError("width and height must be positive integers")
        target = self._resolve_output_path(payload.output_path)
        bbox = (payload.x, payload.y, payload.x + payload.width, payload.y + payload.height)
        image = self._grab_screen(delay_ms=payload.delay_ms, all_screens=False, bbox=bbox)
        self._save_image(image, target)
        return {
            "path": str(target),
            "captured": True,
            "width": image.width,
            "height": image.height,
            "capture_kind": "region",
            "bbox": {"x": payload.x, "y": payload.y, "width": payload.width, "height": payload.height},
        }

    def _resolve_existing_image(self, raw_path: str) -> Path:
        target = self.path_normalizer.resolve(raw_path)
        self._ensure_workspace_path(target)
        if not target.is_file():
            raise FileNotFoundError(f"Image path does not exist: {target}")
        return target

    def _resolve_output_path(self, raw_path: str) -> Path:
        target = self.path_normalizer.resolve(raw_path)
        if not target.suffix:
            target = target.with_suffix(".png")
        self._ensure_workspace_path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _ensure_workspace_path(self, target: Path) -> None:
        try:
            target.relative_to(self.workspace_root)
        except ValueError as exc:
            raise PermissionError(f"Path is outside workspace: {target}") from exc

    @staticmethod
    def _require_pillow() -> None:
        if Image is None or ImageGrab is None or ImageStat is None:
            raise RuntimeError("Pillow is required for image tools.")

    def _open_image(self, image_path: Path):
        self._require_pillow()
        return Image.open(image_path)

    def _base_details(self, image_path: Path, image) -> dict[str, Any]:
        rgb_image = image.convert("RGB")
        stat = ImageStat.Stat(rgb_image)
        average = tuple(int(round(channel)) for channel in stat.mean[:3])
        return {
            "path": str(image_path),
            "name": image_path.name,
            "format": image.format,
            "mode": image.mode,
            "width": int(image.width),
            "height": int(image.height),
            "has_alpha": "A" in image.mode,
            "average_color_rgb": list(average),
            "mime_type": self._mime_type_for_path(image_path),
        }

    @staticmethod
    def _normalize_focus(raw_focus: str) -> str:
        normalized = str(raw_focus or "general").strip().lower()
        if normalized not in {"general", "ui", "ocr"}:
            return "general"
        return normalized

    @staticmethod
    def _build_describe_prompt(
        *,
        image_name: str,
        focus: str,
        user_prompt: str | None,
        ocr_text: str,
    ) -> str:
        focus_prompt = {
            "general": (
                "请用中文简洁描述这张图片的主要内容、场景、主体和重要细节。"
                "如果看不清，请明确说不确定，不要编造。"
            ),
            "ui": (
                "请用中文分析这张界面或截图，概括这是哪类界面、主要区域、明显按钮、状态提示或报错信息。"
                "如果无法确认具体产品或页面，请明确说不确定。"
            ),
            "ocr": (
                "请用中文概括这张图片里最重要的文字信息，并说明这些文字大致属于什么内容。"
                "如果文字模糊或缺失，请直接说明。"
            ),
        }[focus]
        prompt = user_prompt.strip() if isinstance(user_prompt, str) and user_prompt.strip() else focus_prompt
        if ocr_text:
            prompt += f"\n\n补充 OCR 文本（可能不完整）:\n{ocr_text[:1200]}"
        prompt += f"\n\n图片文件名: {image_name}"
        return prompt

    @staticmethod
    def _fallback_description(
        *,
        details: dict[str, Any],
        focus: str,
        ocr: dict[str, Any] | None,
        vision_error: str | None,
    ) -> str:
        width = details.get("width")
        height = details.get("height")
        image_format = details.get("format") or "unknown"
        prefix = f"这是一个 {image_format} 图片，尺寸约 {width}x{height}。"
        if focus == "ui":
            prefix = f"这是一个 {image_format} 格式的截图或图片，尺寸约 {width}x{height}。"
        if ocr and str(ocr.get("text", "") or "").strip():
            text = str(ocr.get("text", "")).strip()
            return f"{prefix} 当前未启用视觉描述模型，我先根据 OCR 判断，图中可识别文字大致是：{text[:200]}"
        if vision_error:
            return f"{prefix} 当前视觉模型调用失败，所以暂时只能提供基础信息，暂时无法稳定判断更具体的图像语义内容。"
        return f"{prefix} 当前未配置视觉描述模型，所以暂时只能提供基础信息，还不能稳定判断更具体的图像语义内容。"

    def _grab_screen(self, *, delay_ms: int, all_screens: bool, bbox: tuple[int, int, int, int] | None = None):
        self._require_pillow()
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        if bbox is not None:
            return ImageGrab.grab(bbox=bbox, all_screens=all_screens)
        return ImageGrab.grab(all_screens=all_screens)

    @staticmethod
    def _save_image(image, target: Path) -> None:
        image.save(target)

    def _try_ocr_image(self, image_path: Path, *, max_chars: int) -> dict[str, Any]:
        try:
            return self._ocr_image(image_path, max_chars=max_chars)
        except Exception as exc:  # noqa: BLE001
            return {
                "available": False,
                "backend": None,
                "text": "",
                "error": str(exc),
            }

    def _ocr_image(self, image_path: Path, *, max_chars: int) -> dict[str, Any]:
        if importlib.util.find_spec("pytesseract") is not None:
            import pytesseract  # type: ignore

            text = pytesseract.image_to_string(self._open_image(image_path)).strip()
            return {
                "available": True,
                "backend": "pytesseract",
                "text": text[:max_chars],
                "error": None,
            }
        if sys.platform == "win32":
            text = self._ocr_with_windows_runtime(image_path).strip()
            return {
                "available": True,
                "backend": "windows_ocr",
                "text": text[:max_chars],
                "error": None,
            }
        raise RuntimeError("No OCR backend is available. Install pytesseract or use Windows OCR.")

    @staticmethod
    def _ocr_with_windows_runtime(image_path: Path) -> str:
        quoted_path = str(image_path).replace("'", "''")
        script = f"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Storage.FileAccessMode, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
$file = [Windows.Storage.StorageFile]::GetFileFromPathAsync('{quoted_path}').AsTask().GetAwaiter().GetResult()
$stream = $file.OpenAsync([Windows.Storage.FileAccessMode]::Read).AsTask().GetAwaiter().GetResult()
$decoder = [Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream).AsTask().GetAwaiter().GetResult()
$bitmap = $decoder.GetSoftwareBitmapAsync().AsTask().GetAwaiter().GetResult()
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) {{ throw 'Windows OCR engine unavailable.' }}
$result = $engine.RecognizeAsync($bitmap).AsTask().GetAwaiter().GetResult()
Write-Output ($result.Text)
"""
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()

    @staticmethod
    def _mime_type_for_path(image_path: Path) -> str:
        suffix = image_path.suffix.lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".bmp": "image/bmp",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "application/octet-stream")
