from __future__ import annotations

import re
from pathlib import Path


class WorkspacePathNormalizer:
    def __init__(self, workspace_root: str) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_name = self.workspace_root.name.lower()
        self.parent_name = self.workspace_root.parent.name.lower()
        self._workspace_aliases = self._build_workspace_aliases()

    def normalize_reference(self, raw_path: str) -> str:
        candidate = (raw_path or "").strip().strip("\"'")
        if not candidate:
            return "."

        candidate = candidate.replace("\\", "/")
        candidate = re.sub(r"/+", "/", candidate).strip()
        candidate = candidate.lstrip("./")
        candidate = candidate.lstrip("/")

        lowered = candidate.lower()
        if lowered in {"", ".", "workspace", "current workspace", "current directory"}:
            return "."

        expanded = self._try_home_relative(candidate)
        if expanded is not None:
            return str(expanded)

        normalized_alias = self._strip_workspace_alias_prefix(candidate, lowered)
        if normalized_alias is not None:
            return normalized_alias

        absolute = self._try_absolute(candidate)
        if absolute is not None:
            return str(absolute)

        return candidate or "."

    def resolve(self, raw_path: str) -> Path:
        normalized = self.normalize_reference(raw_path)
        candidate = Path(normalized)
        if candidate.is_absolute():
            target = candidate.resolve()
        else:
            target = (self.workspace_root / candidate).resolve()
        return target

    def _try_absolute(self, candidate: str) -> Path | None:
        path = Path(candidate)
        if not path.is_absolute():
            return None
        return path.resolve()

    @staticmethod
    def _try_home_relative(candidate: str) -> Path | None:
        if candidate == "~" or candidate.startswith("~/"):
            return Path(candidate).expanduser().resolve()
        return None

    def _strip_workspace_alias_prefix(self, original: str, lowered: str) -> str | None:
        for alias in sorted(self._workspace_aliases, key=len, reverse=True):
            if lowered == alias:
                return "."
            prefix = f"{alias}/"
            if lowered.startswith(prefix):
                remainder = original[len(prefix) :].strip("/")
                return remainder or "."
        return None

    def _build_workspace_aliases(self) -> set[str]:
        aliases = {
            self.workspace_name,
            self.workspace_root.as_posix().lower().strip("/"),
            f"{self.parent_name}/{self.workspace_name}",
            f"./{self.workspace_name}",
            f"/{self.workspace_name}",
            "workspace",
            "current workspace",
            "current directory",
        }

        desktop_tokens = {"desktop", "桌面"}
        if self.parent_name in desktop_tokens:
            aliases.add(f"{self.parent_name}/{self.workspace_name}")
            aliases.add(f"/{self.parent_name}/{self.workspace_name}")
            aliases.add(f"./{self.parent_name}/{self.workspace_name}")
            aliases.add(f"{self.parent_name}\\{self.workspace_name}".replace("\\", "/"))

        chinese_workspace_aliases = {
            f"当前工作区/{self.workspace_name}",
            f"当前目录/{self.workspace_name}",
            f"工作区/{self.workspace_name}",
        }
        aliases.update(alias.lower().replace("\\", "/").strip("/") for alias in chinese_workspace_aliases)
        aliases.update(
            alias.lower().replace("\\", "/").strip("/")
            for alias in {
                f"桌面/{self.workspace_name}",
                f"/桌面/{self.workspace_name}",
                f"./桌面/{self.workspace_name}",
            }
        )
        return {alias.strip("/") for alias in aliases if alias}
