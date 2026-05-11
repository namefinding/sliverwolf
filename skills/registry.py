"""Skill Registry — 扫 skills/ 目录，自动注册所有 skill."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from local_agent.modules.base import ToolRegistry
from local_agent.skills.base import Skill


def discover_skills(skills_dir: str | None = None) -> list[Skill]:
    """从 skills/ 目录自动发现并加载所有 skill。"""
    root = Path(skills_dir or __file__).parent if skills_dir is None else Path(skills_dir)
    skills: list[Skill] = []

    for py_file in sorted(root.glob("*.py")):
        name = py_file.stem
        if name in ("__init__", "base", "registry"):
            continue

        module_path = f"local_agent.skills.{name}"
        try:
            import importlib
            module = importlib.import_module(module_path)
            # 找模块中第一个 Skill 子类实例
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, Skill) and attr is not Skill:
                    skill = attr()
                    skills.append(skill)
                    break
        except Exception as exc:
            print(f"[skills] failed to load {name}: {exc}")

    return skills


def register_skills(registry: ToolRegistry, skills_dir: str | None = None) -> list[Skill]:
    """扫目录，把找到的 skill 注册到给定 registry。"""
    skills = discover_skills(skills_dir)
    for skill in skills:
        manifest = skill.manifest()
        registry.register(manifest, skill.execute)
        print(f"[skills] registered: {manifest.tool_name}")
    return skills
