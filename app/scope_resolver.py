from __future__ import annotations

import re
from pathlib import Path


_ABSOLUTE_PATH_PATTERN = re.compile(r"[A-Za-z]:\\[^\s\"']+")
_PROJECT_HINTS = (
    "project",
    "repo",
    "repository",
    "workspace",
    "工程",
    "项目",
    "代码",
    "源码",
    "pythonproject",
    "local_agent",
    "src",
)
_DESKTOP_HINTS = ("桌面", "desktop")
_DOWNLOAD_HINTS = ("下载", "downloads", "download")
_DOCUMENT_HINTS = ("文档", "我的文档", "documents", "document", "docs")
_TESTING_HINTS = ("testing", "测试目录", "测试文件夹")


def infer_scope_root(
    user_text: str,
    *,
    configured_workspace: str | None = None,
    project_root: str | None = None,
    home_dir: str | Path | None = None,
    allow_default_fallback: bool = True,
) -> str | None:
    text = user_text.strip()
    lowered = text.lower()
    home = Path(home_dir) if home_dir is not None else Path.home()

    explicit_path = _extract_explicit_path(text)
    if explicit_path is not None:
        return str(explicit_path)

    desktop = home / "Desktop"
    downloads = home / "Downloads"
    documents = home / "Documents"
    testing = desktop / "testing"
    configured = Path(configured_workspace).resolve() if configured_workspace else None
    project = Path(project_root).resolve() if project_root else None

    if _contains_any(text, lowered, _DESKTOP_HINTS) and desktop.is_dir():
        return str(desktop)
    if _contains_any(text, lowered, _DOWNLOAD_HINTS) and downloads.is_dir():
        return str(downloads)
    if _contains_any(text, lowered, _DOCUMENT_HINTS) and documents.is_dir():
        return str(documents)
    if _contains_any(text, lowered, _TESTING_HINTS) and testing.is_dir():
        return str(testing)
    if _contains_any(text, lowered, _PROJECT_HINTS):
        if project and project.is_dir():
            return str(project)
        if configured and configured.is_dir():
            return str(configured)

    if not allow_default_fallback:
        return None

    if desktop.is_dir():
        return str(desktop)
    if configured and configured.is_dir():
        return str(configured)
    if project and project.is_dir():
        return str(project)
    return None


def _extract_explicit_path(text: str) -> Path | None:
    match = _ABSOLUTE_PATH_PATTERN.search(text)
    if not match:
        return None
    candidate = Path(match.group(0)).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if candidate.is_file():
        return candidate.resolve().parent
    return None


def _contains_any(text: str, lowered: str, terms: tuple[str, ...]) -> bool:
    return any(term in text or term in lowered for term in terms)
