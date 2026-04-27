from __future__ import annotations

from pathlib import Path

import yaml

from local_agent.protocol.models import AppConfig


def load_config(config_path: str) -> AppConfig:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return AppConfig.model_validate(raw)
