from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
import json
from typing import Any, Callable

from pydantic import ValidationError

from local_agent.protocol.models import (
    ToolCallRequest,
    ToolCallResult,
    ToolError,
    ToolExecutionContext,
    ToolManifest,
    ToolPermissionDecision,
    ToolPermissionMode,
    ToolUseContext,
)
from local_agent.protocol.tool_outputs import build_evidence, build_observation, resolve_effective_outputs


@dataclass(slots=True)
class RegisteredTool:
    manifest: ToolManifest
    executor: Callable[..., dict[str, Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._aliases: dict[str, str] = {}

    def register(self, manifest: ToolManifest, executor: Callable[..., dict[str, Any]]) -> None:
        if manifest.tool_name in self._aliases:
            raise ValueError(f"Tool name conflicts with existing alias: {manifest.tool_name}")
        if not manifest.side_effect:
            manifest = manifest.model_copy(
                update={
                    "read_only": True,
                    "concurrency_safe": True,
                }
            )
        self._tools[manifest.tool_name] = RegisteredTool(manifest=manifest, executor=executor)
        for alias in manifest.aliases:
            if alias in self._tools:
                raise ValueError(f"Tool alias conflicts with existing tool: {alias}")
            owner = self._aliases.get(alias)
            if owner is not None and owner != manifest.tool_name:
                raise ValueError(f"Tool alias is already registered: {alias}")
            self._aliases[alias] = manifest.tool_name

    def list_manifests(self) -> list[ToolManifest]:
        return [tool.manifest for tool in self._tools.values()]

    def has_tool(self, tool_name: str) -> bool:
        return self._resolve_tool_name(tool_name) in self._tools

    def get_manifest(self, tool_name: str) -> ToolManifest:
        return self._tools[self._resolve_tool_name(tool_name)].manifest

    def execute(self, request: ToolCallRequest, context: ToolUseContext | None = None) -> ToolCallResult:
        started = time.perf_counter()
        canonical_tool_name = self._resolve_tool_name(request.tool_name)
        tool = self._tools.get(canonical_tool_name)
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
            permission = self.check_permission(tool.manifest, request.arguments, context)
            if permission.behavior == ToolPermissionMode.DENY:
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                return ToolCallResult(
                    request_id=request.request_id,
                    trace_id=request.trace_id,
                    tool_name=canonical_tool_name,
                    status="error",
                    error=ToolError(
                        code="permission_denied",
                        message=permission.reason or f"Permission denied for tool: {canonical_tool_name}",
                    ),
                    metrics={
                        "latency_ms": latency_ms,
                        "permission": permission.model_dump(mode="json"),
                    },
                )

            effective_arguments = permission.updated_arguments or request.arguments
            data = self._call_executor(tool.executor, effective_arguments, context)
            data, result_budget_metrics = self._apply_result_budget(tool.manifest, data)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            produced_outputs = resolve_effective_outputs(
                tool_name=canonical_tool_name,
                produced_outputs=list(tool.manifest.produces),
                data=data,
            )
            evidence = build_evidence(
                tool_name=canonical_tool_name,
                data=data,
                produced_outputs=produced_outputs,
            )
            observation = build_observation(
                request_id=request.request_id,
                tool_name=canonical_tool_name,
                status="success",
                data=data,
            )
            return ToolCallResult(
                request_id=request.request_id,
                trace_id=request.trace_id,
                tool_name=canonical_tool_name,
                status="success",
                data=data,
                produced_outputs=produced_outputs,
                evidence=evidence,
                metrics={
                    "latency_ms": latency_ms,
                    "permission": permission.model_dump(mode="json"),
                    "observation": observation,
                    **result_budget_metrics,
                },
            )
        except ValidationError as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return ToolCallResult(
                request_id=request.request_id,
                trace_id=request.trace_id,
                tool_name=canonical_tool_name,
                status="error",
                error=ToolError(code="validation_error", message=str(exc)),
                metrics={"latency_ms": latency_ms},
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return ToolCallResult(
                request_id=request.request_id,
                trace_id=request.trace_id,
                tool_name=canonical_tool_name,
                status="error",
                error=ToolError(code=exc.__class__.__name__.lower(), message=str(exc)),
                metrics={"latency_ms": latency_ms},
            )

    def _resolve_tool_name(self, tool_name: str) -> str:
        return self._aliases.get(tool_name, tool_name)

    def check_permission(
        self,
        manifest: ToolManifest,
        arguments: dict[str, Any],
        context: ToolUseContext | None = None,
    ) -> ToolPermissionDecision:
        if manifest.default_permission == ToolPermissionMode.DENY:
            return ToolPermissionDecision(
                behavior=ToolPermissionMode.DENY,
                reason=f"{manifest.tool_name} is denied by its default permission.",
            )

        if context is None:
            return ToolPermissionDecision(behavior=manifest.default_permission)

        access_policy = context.access_policy or {}
        denied_tools = {
            str(item).strip()
            for item in access_policy.get("denied_tools", [])
            if str(item).strip()
        }
        if manifest.tool_name in denied_tools or any(alias in denied_tools for alias in manifest.aliases):
            return ToolPermissionDecision(
                behavior=ToolPermissionMode.DENY,
                reason=f"{manifest.tool_name} is denied by access policy.",
            )

        allow_local_tools = bool(access_policy.get("allow_local_tools", True))
        if not allow_local_tools and manifest.module not in {"web", "memory", "qq", "system_utility"}:
            return ToolPermissionDecision(
                behavior=ToolPermissionMode.DENY,
                reason=f"{manifest.tool_name} requires local tool access.",
            )

        if manifest.destructive and not bool(access_policy.get("allow_destructive_tools", False)):
            return ToolPermissionDecision(
                behavior=ToolPermissionMode.ASK,
                reason=f"{manifest.tool_name} is destructive and should be confirmed.",
            )

        if manifest.requires_confirmation and not bool(access_policy.get("auto_confirm_tools", False)):
            return ToolPermissionDecision(
                behavior=ToolPermissionMode.ASK,
                reason=f"{manifest.tool_name} requires confirmation.",
            )

        if manifest.default_permission == ToolPermissionMode.ASK:
            return ToolPermissionDecision(
                behavior=ToolPermissionMode.ASK,
                reason=f"{manifest.tool_name} asks for confirmation by default.",
            )
        return ToolPermissionDecision(behavior=ToolPermissionMode.ALLOW)

    @staticmethod
    def _apply_result_budget(
        manifest: ToolManifest,
        data: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        limit = manifest.max_result_size_chars
        if limit is None or limit <= 0:
            return data, {}
        serialized = json.dumps(data, ensure_ascii=False, default=str)
        if len(serialized) <= limit:
            return data, {"result_size_chars": len(serialized)}
        preview_limit = max(0, limit - 200)
        preview = serialized[:preview_limit].rstrip()
        return (
            {
                "preview": preview,
                "truncated": True,
                "original_size_chars": len(serialized),
                "tool_name": manifest.tool_name,
            },
            {
                "result_size_chars": len(serialized),
                "result_truncated": True,
                "result_preview_chars": len(preview),
            },
        )

    @staticmethod
    def _call_executor(
        executor: Callable[..., dict[str, Any]],
        arguments: dict[str, Any],
        context: ToolUseContext | None,
    ) -> dict[str, Any]:
        if context is None:
            return executor(arguments)
        try:
            return executor(arguments, context=context)
        except TypeError as exc:
            message = str(exc)
            if "context" not in message and "positional" not in message and "keyword" not in message:
                raise
            return executor(arguments)

    @staticmethod
    def build_request(
        trace_id: str,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        execution_context: dict[str, Any] | ToolExecutionContext | None = None,
    ) -> ToolCallRequest:
        context = (
            execution_context
            if isinstance(execution_context, ToolExecutionContext)
            else ToolExecutionContext.model_validate(execution_context or {})
        )
        enriched_arguments = ToolRegistry._attach_execution_context(arguments, context)
        return ToolCallRequest(
            request_id=f"req_{uuid.uuid4().hex[:12]}",
            trace_id=trace_id,
            session_id=session_id,
            tool_name=tool_name,
            arguments=enriched_arguments,
            execution_context=context,
        )

    @staticmethod
    def _attach_execution_context(
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        enriched = dict(arguments or {})
        package = context.model_dump(mode="json")
        if package["execution_brief"] and not str(enriched.get("execution_brief", "") or "").strip():
            enriched["execution_brief"] = package["execution_brief"]
        if package["required_outputs"] and not enriched.get("required_outputs"):
            enriched["required_outputs"] = package["required_outputs"]

        grounded_inputs = dict(enriched.get("grounded_inputs") or {})
        for key, value in package["grounded_inputs"].items():
            grounded_inputs.setdefault(key, value)
        if grounded_inputs:
            enriched["grounded_inputs"] = grounded_inputs

        constraints = dict(enriched.get("constraints") or {})
        for key, value in package["constraints"].items():
            constraints.setdefault(key, value)
        if constraints:
            enriched["constraints"] = constraints
        return enriched
