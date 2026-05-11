from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from local_agent.protocol.models import AppConfig


@dataclass
class HealthCheckResult:
    name: str
    status: str
    message: str
    critical: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "critical": self.critical,
        }


def run_startup_healthcheck(
    config: AppConfig,
    *,
    ollama_probe: Callable[[str], tuple[bool, str]] | None = None,
    voice_probe: Callable[[str], tuple[bool, str]] | None = None,
) -> dict[str, object]:
    ollama_probe = ollama_probe or _default_ollama_probe
    voice_probe = voice_probe or _default_voice_probe

    workspace = Path(config.agent.workspace_root).resolve()
    memory_db = Path(config.agent.memory_db_path)
    retrieval_db = Path(config.agent.retrieval_db_path)

    checks = [
        _check_workspace(workspace),
        _check_db_parent("memory_store", memory_db),
        _check_db_parent("retrieval_store", retrieval_db),
        _check_llm_provider(config, ollama_probe),
        _check_voice(config.voice.enabled, config.voice.endpoint, voice_probe),
    ]

    critical_failures = [check for check in checks if check.critical and check.status != "ok"]
    overall_status = "ok" if not critical_failures else "degraded"

    return {
        "overall_status": overall_status,
        "checks": [check.to_dict() for check in checks],
    }


def _check_workspace(workspace: Path) -> HealthCheckResult:
    if workspace.exists() and workspace.is_dir():
        return HealthCheckResult("workspace", "ok", f"Workspace is ready at {workspace}", critical=True)
    return HealthCheckResult("workspace", "error", f"Workspace is unavailable: {workspace}", critical=True)


def _check_db_parent(name: str, db_path: Path) -> HealthCheckResult:
    parent = db_path.resolve().parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return HealthCheckResult(name, "error", f"Cannot prepare directory {parent}: {exc}", critical=True)
    return HealthCheckResult(name, "ok", f"Storage directory is ready at {parent}", critical=True)


def _check_ollama(base_url: str, probe: Callable[[str], tuple[bool, str]]) -> HealthCheckResult:
    ok, message = probe(base_url)
    return HealthCheckResult("ollama", "ok" if ok else "error", message, critical=False)


def _check_llm_provider(config: AppConfig, ollama_probe: Callable[[str], tuple[bool, str]]) -> HealthCheckResult:
    provider = str(getattr(config.agent, "llm_provider", "") or "ollama").strip().lower()
    if provider in {"deepseek", "openai_compatible"}:
        key_env = str(getattr(config.agent, "api_key_env", "") or "DEEPSEEK_API_KEY").strip()
        base_url = str(getattr(config.agent, "api_base_url", "") or "").strip()
        if not key_env:
            return HealthCheckResult("llm_provider", "warning", "API key env var is not configured", critical=False)
        return HealthCheckResult(
            "llm_provider",
            "ok",
            f"Using {provider} at {base_url or 'default API base'}; API key is read from {key_env}",
            critical=False,
        )
    return _check_ollama(config.agent.ollama_base_url, ollama_probe)


def _check_voice(enabled: bool, endpoint: str, probe: Callable[[str], tuple[bool, str]]) -> HealthCheckResult:
    if not enabled:
        return HealthCheckResult("voice", "disabled", "Voice output is disabled in config", critical=False)
    ok, message = probe(endpoint)
    return HealthCheckResult("voice", "ok" if ok else "warning", message, critical=False)


def _default_ollama_probe(base_url: str) -> tuple[bool, str]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
        if response.ok:
            return True, "Ollama responded to /api/tags"
        return False, f"Ollama returned HTTP {response.status_code}"
    except requests.RequestException as exc:
        return False, f"Ollama probe failed: {exc}"


def _default_voice_probe(endpoint: str) -> tuple[bool, str]:
    try:
        response = requests.get(f"{endpoint.rstrip('/')}/docs", timeout=3)
        if response.ok:
            return True, "GPT-SoVITS docs endpoint is reachable"
        return False, f"GPT-SoVITS returned HTTP {response.status_code}"
    except requests.RequestException as exc:
        return False, f"GPT-SoVITS probe failed: {exc}"
