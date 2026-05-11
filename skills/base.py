from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from local_agent.protocol.models import ToolManifest


class Skill(ABC):
    """一个 skill = 一个 self-contained 的功能包。

    主 agent 看到 manifest 决定要不要用这个 skill。
    skill 内部处理全部细节（多步确定性操作 + 可选内部 LLM 调用）。
    不需要主 agent 拆分步骤。
    """

    @abstractmethod
    def manifest(self) -> ToolManifest:
        """返回这个 skill 的工具声明。description 是 LLM 选择的关键依据。"""
        ...

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """执行 skill。arguments 是 LLM 传的参数。返回结果给主 agent。"""
        ...

    @property
    def tool_name(self) -> str:
        return self.manifest().tool_name
