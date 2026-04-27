from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import ValidationError

from local_agent.protocol.models import ToolCallRequest, ToolCallResult, ToolError, ToolManifest


@dataclass(slots=True)
class RegisteredTool:
    manifest: ToolManifest
    executor: Callable[[dict[str, Any]], dict[str, Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, manifest: ToolManifest, executor: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self._tools[manifest.tool_name] = RegisteredTool(manifest=manifest, executor=executor)

    def list_manifests(self) -> list[ToolManifest]:
        return [tool.manifest for tool in self._tools.values()]

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def get_manifest(self, tool_name: str) -> ToolManifest:
        return self._tools[tool_name].manifest

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        started = time.perf_counter()
        tool = self._tools.get(request.tool_name)
        if tool is None:
            return ToolCallResult(
                request_id=request.request_id,
                trace_id=request.trace_id,
                tool_name=request.tool_name,
                status="error",
                error=ToolError(code="tool_not_found", message=f"Unknown tool: {request.tool_name}"),
                metrics={"latency_ms": 0},
            )

        try:
            data = tool.executor(request.arguments)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return ToolCallResult(
                request_id=request.request_id,
                trace_id=request.trace_id,
                tool_name=request.tool_name,
                status="success",
                data=data,
                metrics={"latency_ms": latency_ms},
            )
        except ValidationError as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return ToolCallResult(
                request_id=request.request_id,
                trace_id=request.trace_id,
                tool_name=request.tool_name,
                status="error",
                error=ToolError(code="validation_error", message=str(exc)),
                metrics={"latency_ms": latency_ms},
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return ToolCallResult(
                request_id=request.request_id,
                trace_id=request.trace_id,
                tool_name=request.tool_name,
                status="error",
                error=ToolError(code=exc.__class__.__name__.lower(), message=str(exc)),
                metrics={"latency_ms": latency_ms},
            )

    @staticmethod
    def build_request(trace_id: str, session_id: str, tool_name: str, arguments: dict[str, Any]) -> ToolCallRequest:
        return ToolCallRequest(
            request_id=f"req_{uuid.uuid4().hex[:12]}",
            trace_id=trace_id,
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
        )
