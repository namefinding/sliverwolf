from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


TargetType = Literal["web", "local_file", "local_folder", "ambiguous", "unknown"]
TargetAction = Literal["open", "search", "research", "other"]

_OPEN_TERMS = (
    "open",
    "visit",
    "launch",
    "\u6253\u5f00",
    "\u8fdb\u5165",
    "\u53bb",
)

_SEARCH_TERMS = (
    "search",
    "find",
    "look up",
    "\u641c\u7d22",
    "\u67e5",
    "\u67e5\u627e",
    "\u67e5\u4e00\u4e0b",
    "\u4e0a\u7f51\u67e5",
)

_RESEARCH_TERMS = (
    "summarize",
    "summary",
    "research",
    "latest",
    "recent",
    "news",
    "\u603b\u7ed3",
    "\u6982\u62ec",
    "\u8c03\u7814",
    "\u6700\u65b0",
    "\u8fd1\u671f",
)

_WEB_DESCRIPTOR_TERMS = (
    "website",
    "webpage",
    "homepage",
    "official site",
    "site",
    "\u5b98\u7f51",
    "\u7f51\u7ad9",
    "\u7f51\u9875",
    "\u7ad9",
    "\u89c6\u9891\u7f51\u7ad9",
    "\u4ee3\u7801\u6258\u7ba1\u5e73\u53f0",
)

_LOCAL_HINT_TERMS = (
    "file",
    "folder",
    "directory",
    "path",
    "\u6587\u4ef6",
    "\u6587\u4ef6\u5939",
    "\u76ee\u5f55",
    "\u8def\u5f84",
)

_FILE_EXTENSIONS = (
    ".py",
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".xlsx",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".pdf",
    ".html",
    ".log",
)

_URL_PATTERN = re.compile(r"https?://[^\s]+", flags=re.IGNORECASE)
_DOMAIN_PATTERN = re.compile(
    r"\b(?:[a-z0-9-]+\.)+(?:com|cn|net|org|io|dev|ai|tv|gg)(?:/[^\s]*)?\b",
    flags=re.IGNORECASE,
)
_WINDOWS_PATH_PATTERN = re.compile(r"[A-Za-z]:\\[^\s]*|\\\\[^\\\s]+\\[^\s]+")
_FILE_LIKE_PATTERN = re.compile(
    r"\b[\w.\-/\\]+\.(?:py|txt|md|json|yaml|yml|csv|xlsx|doc|docx|ppt|pptx|pdf|html|log)\b",
    flags=re.IGNORECASE,
)

_SITE_ALIASES = {
    "bilibili": {
        "url": "https://www.bilibili.com",
        "aliases": ("bilibili", "b\u7ad9", "B\u7ad9"),
    },
    "github": {
        "url": "https://github.com",
        "aliases": ("github",),
    },
    "zhihu": {
        "url": "https://www.zhihu.com",
        "aliases": ("zhihu", "\u77e5\u4e4e"),
    },
}


@dataclass(frozen=True)
class TargetResolution:
    target_type: TargetType = "unknown"
    action: TargetAction = "other"
    raw_target: str = ""
    canonical_name: str | None = None
    resolved_target: str | None = None
    confidence: float = 0.0
    reason: str = ""


def resolve_target_reference(user_text: str) -> TargetResolution:
    normalized_text = " ".join(str(user_text).strip().split())
    if not normalized_text:
        return TargetResolution()

    lowered = normalized_text.lower()
    action = _infer_action(lowered)

    explicit_url = _extract_explicit_url(normalized_text)
    if explicit_url is not None:
        return TargetResolution(
            target_type="web",
            action=_normalize_action_for_web(action),
            raw_target=explicit_url,
            resolved_target=explicit_url,
            confidence=0.99,
            reason="explicit_url",
        )

    local_path = _extract_local_path(normalized_text)
    if local_path is not None:
        return TargetResolution(
            target_type="local_file" if _looks_like_file_reference(local_path) else "local_folder",
            action=action,
            raw_target=local_path,
            resolved_target=local_path,
            confidence=0.98,
            reason="explicit_local_path",
        )

    explicit_domain = _extract_domain(normalized_text)
    if explicit_domain is not None:
        return TargetResolution(
            target_type="web",
            action=_normalize_action_for_web(action),
            raw_target=explicit_domain,
            resolved_target=f"https://{explicit_domain}",
            confidence=0.97,
            reason="explicit_domain",
        )

    alias_match = _match_site_alias(lowered)
    if alias_match is not None:
        canonical_name, alias, url = alias_match
        return TargetResolution(
            target_type="web",
            action=_normalize_action_for_web(action),
            raw_target=alias,
            canonical_name=canonical_name,
            resolved_target=url,
            confidence=0.96,
            reason="known_site_alias",
        )

    if action in {"open", "search", "research"} and any(term in lowered for term in _WEB_DESCRIPTOR_TERMS):
        return TargetResolution(
            target_type="web",
            action="search" if action == "open" else action,
            raw_target=normalized_text,
            resolved_target=normalized_text,
            confidence=0.58,
            reason="generic_web_descriptor",
        )

    if action in {"open", "search"} and any(term in lowered for term in _LOCAL_HINT_TERMS):
        return TargetResolution(
            target_type="ambiguous",
            action=action,
            raw_target=normalized_text,
            resolved_target=normalized_text,
            confidence=0.32,
            reason="local_hint_without_path",
        )

    return TargetResolution(
        target_type="unknown",
        action=action,
        raw_target=normalized_text,
        resolved_target=normalized_text,
        confidence=0.0,
        reason="no_target_match",
    )


def _infer_action(lowered_text: str) -> TargetAction:
    if any(term in lowered_text for term in _RESEARCH_TERMS):
        return "research"
    if any(term in lowered_text for term in _SEARCH_TERMS):
        return "search"
    if any(term in lowered_text for term in _OPEN_TERMS):
        return "open"
    return "other"


def _normalize_action_for_web(action: TargetAction) -> TargetAction:
    if action == "other":
        return "open"
    return action


def _extract_explicit_url(text: str) -> str | None:
    match = _URL_PATTERN.search(text)
    if match is None:
        return None
    return match.group(0).rstrip(".,!?)]}")


def _extract_domain(text: str) -> str | None:
    match = _DOMAIN_PATTERN.search(text)
    if match is None:
        return None
    return match.group(0).rstrip(".,!?)]}")


def _extract_local_path(text: str) -> str | None:
    windows_match = _WINDOWS_PATH_PATTERN.search(text)
    if windows_match is not None:
        return windows_match.group(0)
    file_like_match = _FILE_LIKE_PATTERN.search(text)
    if file_like_match is not None:
        return file_like_match.group(0)
    return None


def _looks_like_file_reference(path_text: str) -> bool:
    lowered = path_text.lower()
    return any(lowered.endswith(extension) for extension in _FILE_EXTENSIONS)


def _match_site_alias(lowered_text: str) -> tuple[str, str, str] | None:
    for canonical_name, payload in _SITE_ALIASES.items():
        for alias in payload["aliases"]:
            if alias.lower() in lowered_text:
                return canonical_name, alias, payload["url"]
    return None
