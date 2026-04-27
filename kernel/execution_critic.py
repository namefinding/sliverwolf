from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

from local_agent.protocol.models import ExecutionReview, OutputKind


class ExecutionCritic:
    _SUMMARY_TERMS = re.compile(
        r"(总结|概括|讲了什么|写了什么|提炼|栏目|结构|重点|核心观点|内容|what\s+is\s+in|sections?|structure|key\s+points?|main\s+points?|summari[sz]e|summary|summarize)",
        flags=re.IGNORECASE,
    )
    _IMAGE_TEXT_TERMS = re.compile(
        r"(文字|文本|读字|识别文字|提取文字|ocr|read\s+text|text\s+in\s+the\s+image)",
        flags=re.IGNORECASE,
    )
    _IMAGE_HINT_TERMS = re.compile(r"(图片|截图|照片|图像|image|picture|photo|screenshot)", flags=re.IGNORECASE)
    _STRONG_COMPLETION_TERMS = re.compile(r"(主要|重点|图上写着|最新|最近两天|这篇|这份)", flags=re.IGNORECASE)
    _DATE_PATTERN = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")

    def review_completion(
        self,
        *,
        user_text: str,
        execution_summary: dict[str, Any],
        now: datetime | None = None,
    ) -> ExecutionReview:
        issues: list[str] = []
        missing_outputs: list[OutputKind] = []

        completed_outputs = self._normalize_outputs(execution_summary.get("completed_outputs"))
        successful_tools = self._successful_tools(execution_summary)
        task_kind = self._task_kind(execution_summary)
        knowledge_request = execution_summary.get("knowledge_request") or {}
        knowledge_type = str(knowledge_request.get("knowledge_type", "") or "").strip().lower()
        workflow_state = execution_summary.get("workflow_state") or {}
        contract_outputs = self._declared_required_outputs(execution_summary)
        has_contract = bool(contract_outputs)
        overall_task_goal = execution_summary.get("overall_task_goal") or {}
        completion_mode = str(overall_task_goal.get("completion_mode", "outputs") or "outputs").strip().lower()

        for output_kind in self._missing_contract_outputs(contract_outputs, completed_outputs):
            if output_kind not in missing_outputs:
                missing_outputs.append(output_kind)
        if missing_outputs:
            issues.append("workflow_required_outputs_missing")
        if (
            task_kind in {"summarize", "document_summary"}
            and OutputKind.FILE_CONTENTS in missing_outputs
            and "summary_requires_file_contents" not in issues
            and knowledge_type != "qq_history"
        ):
            issues.append("summary_requires_file_contents")

        if (
            not has_contract
            and knowledge_type != "qq_history"
            and self._requires_document_grounding(user_text, task_kind)
            and OutputKind.FILE_CONTENTS.value not in completed_outputs
        ):
            issues.append("summary_requires_file_contents")
            if OutputKind.FILE_CONTENTS not in missing_outputs:
                missing_outputs.append(OutputKind.FILE_CONTENTS)

        if self._requires_image_read_text(user_text, task_kind, workflow_state, contract_outputs, has_contract=has_contract):
            if "image.read_text" not in successful_tools:
                issues.append("image_text_request_requires_image_read_text")
                if OutputKind.FILE_CONTENTS not in missing_outputs:
                    missing_outputs.append(OutputKind.FILE_CONTENTS)

        if completion_mode == "outputs" and knowledge_request.get("time_sensitive"):
            web_sources = execution_summary.get("web_sources") or []
            if not self._has_web_source_evidence(web_sources):
                issues.append("time_sensitive_web_results_require_web_sources")
                missing_outputs.append(OutputKind.SEARCH_RESULTS)

        issues.extend(self._workflow_consistency_issues(execution_summary))
        for output_kind in self._workflow_missing_outputs(workflow_state):
            if output_kind not in missing_outputs:
                missing_outputs.append(output_kind)

        if not issues:
            return ExecutionReview(approved=True, summary="Execution completion checks passed.")

        return ExecutionReview(
            approved=False,
            issues=issues,
            summary="Execution critic rejected completion because required grounding evidence is missing.",
            force_partial=True,
            missing_outputs=missing_outputs,
        )

    def review_grounding(
        self,
        *,
        user_text: str,
        execution_summary: dict[str, Any],
        response_text: str = "",
        now: datetime | None = None,
    ) -> ExecutionReview:
        issues: list[str] = []
        missing_outputs: list[OutputKind] = []

        completed_outputs = self._normalize_outputs(execution_summary.get("completed_outputs"))
        successful_tools = self._successful_tools(execution_summary)
        task_kind = self._task_kind(execution_summary)
        knowledge_request = execution_summary.get("knowledge_request") or {}
        knowledge_type = str(knowledge_request.get("knowledge_type", "") or "").strip().lower()
        workflow_state = execution_summary.get("workflow_state") or {}
        text = f"{user_text}\n{response_text}".strip()
        contract_outputs = self._declared_required_outputs(execution_summary)
        has_contract = bool(contract_outputs)
        overall_task_goal = execution_summary.get("overall_task_goal") or {}
        completion_mode = str(overall_task_goal.get("completion_mode", "outputs") or "outputs").strip().lower()

        explicit_file_contents_request = False
        if has_contract:
            explicit_file_contents_request = OutputKind.FILE_CONTENTS.value in contract_outputs
        else:
            explicit_file_contents_request = knowledge_type != "qq_history" and (
                self._requires_document_grounding(
                user_text, task_kind
                ) or self._looks_like_image_text_request(user_text, task_kind)
            )
        requires_file_contents = self._workflow_requires_output(
            workflow_state, OutputKind.FILE_CONTENTS
        ) or explicit_file_contents_request

        requires_document_grounding = knowledge_type != "qq_history" and self._requires_document_grounding(text, task_kind)
        if has_contract and OutputKind.FILE_CONTENTS.value in contract_outputs:
            requires_document_grounding = (
                knowledge_type != "qq_history"
                and task_kind in {"summarize", "document_summary"}
            ) or requires_document_grounding
        if requires_file_contents and requires_document_grounding and OutputKind.FILE_CONTENTS.value not in completed_outputs:
            issues.append("ungrounded_summary_response")
            missing_outputs.append(OutputKind.FILE_CONTENTS)

        if (
            requires_file_contents
            and self._requires_image_read_text(user_text, task_kind, workflow_state, contract_outputs, has_contract=has_contract)
            and "image.read_text" not in successful_tools
        ):
            issues.append("ungrounded_image_text_response")
            if OutputKind.FILE_CONTENTS not in missing_outputs:
                missing_outputs.append(OutputKind.FILE_CONTENTS)

        if completion_mode == "outputs" and knowledge_request.get("time_sensitive"):
            web_sources = execution_summary.get("web_sources") or []
            if not self._has_web_source_evidence(web_sources):
                issues.append("ungrounded_time_sensitive_response")
                if OutputKind.SEARCH_RESULTS not in missing_outputs:
                    missing_outputs.append(OutputKind.SEARCH_RESULTS)
            elif self._response_mentions_stale_dates(response_text, now=now):
                issues.append("response_mentions_stale_dates_for_time_sensitive_request")
                if OutputKind.SEARCH_RESULTS not in missing_outputs:
                    missing_outputs.append(OutputKind.SEARCH_RESULTS)

        issues.extend(self._workflow_grounding_issues(workflow_state, response_text=response_text))

        if not issues:
            return ExecutionReview(approved=True, summary="Grounding review passed.")

        return ExecutionReview(
            approved=False,
            issues=issues,
            summary="Grounding review blocked an unsupported final response.",
            force_partial=True,
            missing_outputs=missing_outputs,
        )

    @staticmethod
    def _normalize_outputs(payload: Any) -> set[str]:
        if not isinstance(payload, (list, tuple, set)):
            return set()
        return {str(item).strip() for item in payload if str(item).strip()}

    @staticmethod
    def _workflow_requires_output(workflow_state: dict[str, Any], output_kind: OutputKind) -> bool:
        if not isinstance(workflow_state, dict):
            return False
        required = {
            str(item).strip()
            for item in workflow_state.get("required_outputs") or []
            if str(item).strip()
        }
        missing = {
            str(item).strip()
            for item in workflow_state.get("missing_outputs") or []
            if str(item).strip()
        }
        value = output_kind.value
        return value in required or value in missing

    @classmethod
    def _declared_required_outputs(cls, execution_summary: dict[str, Any]) -> set[str]:
        outputs: set[str] = set()
        workflow_state = execution_summary.get("workflow_state")
        if isinstance(workflow_state, dict):
            outputs.update(cls._normalize_outputs(workflow_state.get("required_outputs")))
            outputs.update(cls._normalize_outputs(workflow_state.get("missing_outputs")))
        overall_task_goal = execution_summary.get("overall_task_goal")
        if isinstance(overall_task_goal, dict):
            outputs.update(cls._normalize_outputs(overall_task_goal.get("required_outputs")))
        outputs.update(cls._normalize_outputs(execution_summary.get("missing_outputs")))
        task_kind = cls._task_kind(execution_summary)
        knowledge_request = execution_summary.get("knowledge_request") or {}
        knowledge_type = str(knowledge_request.get("knowledge_type", "") or "").strip().lower()
        if task_kind in {"summarize", "document_summary"} and knowledge_type != "qq_history":
            outputs.add(OutputKind.FILE_CONTENTS.value)
        return outputs

    @staticmethod
    def _missing_contract_outputs(contract_outputs: set[str], completed_outputs: set[str]) -> list[OutputKind]:
        mapping = {item.value: item for item in OutputKind}
        missing: list[OutputKind] = []
        for output in contract_outputs:
            output_kind = mapping.get(output)
            if output_kind is not None and output not in completed_outputs and output_kind not in missing:
                missing.append(output_kind)
        return missing

    @classmethod
    def _requires_image_read_text(
        cls,
        user_text: str,
        task_kind: str,
        workflow_state: dict[str, Any],
        contract_outputs: set[str],
        *,
        has_contract: bool,
    ) -> bool:
        if not has_contract:
            return cls._looks_like_image_text_request(user_text, task_kind)
        family = ""
        if isinstance(workflow_state, dict):
            family = str(workflow_state.get("workflow_family", "")).strip().lower()
        if OutputKind.FILE_CONTENTS.value not in contract_outputs:
            return False
        if task_kind == "inspect" or family in {"inspection", "inspect", "image_inspection"}:
            return True
        return False

    @staticmethod
    def _successful_tools(execution_summary: dict[str, Any]) -> set[str]:
        tools: set[str] = set()
        for item in execution_summary.get("successful_actions") or []:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool_name", "")).strip()
            if tool_name:
                tools.add(tool_name)
        return tools

    @staticmethod
    def _task_kind(execution_summary: dict[str, Any]) -> str:
        payload = execution_summary.get("task_classification") or {}
        return str(payload.get("task_kind", "")).strip().lower()

    @classmethod
    def _looks_like_image_text_request(cls, text: str, task_kind: str) -> bool:
        lowered = str(text or "")
        return (task_kind == "inspect" or cls._IMAGE_HINT_TERMS.search(lowered)) and bool(cls._IMAGE_TEXT_TERMS.search(lowered))

    @classmethod
    def _requires_document_grounding(cls, text: str, task_kind: str) -> bool:
        lowered = str(text or "")
        lowered = re.sub(r"\b[\w.-]+\.(?:txt|md|docx|pptx|xlsx|csv|json|log|pdf)\b", " ", lowered, flags=re.IGNORECASE)
        if cls._IMAGE_HINT_TERMS.search(lowered) and not cls._IMAGE_TEXT_TERMS.search(lowered):
            return False
        if task_kind in {"summarize", "document_summary"}:
            return True
        if task_kind == "delivery" and cls._SUMMARY_TERMS.search(lowered):
            return True
        return bool(cls._SUMMARY_TERMS.search(lowered))

    @classmethod
    def _has_recent_web_sources(cls, sources: list[dict[str, Any]], *, now: datetime | None = None) -> bool:
        if not sources:
            return False
        current = now or datetime.now(UTC)
        threshold = current - timedelta(days=2)
        for source in sources:
            if not isinstance(source, dict):
                continue
            parsed = cls._parse_datetime(source.get("published_at"))
            if parsed is not None and parsed >= threshold:
                return True
            if cls._looks_recent_from_search_snippet(source, now=current):
                return True
        return False

    @staticmethod
    def _has_web_source_evidence(sources: list[dict[str, Any]]) -> bool:
        for source in sources:
            if not isinstance(source, dict):
                continue
            if str(source.get("url", "") or "").strip() and any(
                str(source.get(key, "") or "").strip()
                for key in ("title", "snippet", "excerpt", "content")
            ):
                return True
        return False

    @staticmethod
    def _looks_recent_from_search_snippet(source: dict[str, Any], *, now: datetime | None = None) -> bool:
        text = " ".join(str(source.get(key, "") or "") for key in ("title", "snippet", "excerpt", "content"))
        if re.search(r"(\d+)\s*(分钟|小時|小时|hour|hours|minute|minutes)\s*(前|ago)", text, flags=re.IGNORECASE):
            return True
        day_match = re.search(r"(\d+)\s*(天|day|days)\s*(前|ago)", text, flags=re.IGNORECASE)
        if day_match:
            try:
                return int(day_match.group(1)) <= 2
            except ValueError:
                return False
        current = now or datetime.now(UTC)
        if re.search(rf"\b{current.year}\b|{current.year}\s*年", text):
            return True
        return False

    @classmethod
    def _response_mentions_stale_dates(cls, response_text: str, *, now: datetime | None = None) -> bool:
        if not response_text or not cls._STRONG_COMPLETION_TERMS.search(response_text):
            return False
        current = now or datetime.now(UTC)
        threshold_date = (current - timedelta(days=2)).date()
        for match in cls._DATE_PATTERN.finditer(response_text):
            try:
                mentioned = datetime(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                    tzinfo=UTC,
                ).date()
            except ValueError:
                continue
            if mentioned < threshold_date:
                return True
        return False

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    @staticmethod
    def _workflow_consistency_issues(execution_summary: dict[str, Any]) -> list[str]:
        workflow_state = execution_summary.get("workflow_state")
        if not isinstance(workflow_state, dict):
            return []
        issues: list[str] = []
        stage = str(workflow_state.get("workflow_stage", "")).strip().lower()
        next_actions = {
            str(item).strip().lower()
            for item in workflow_state.get("next_allowed_actions") or []
            if str(item).strip()
        }
        missing_outputs = {
            str(item).strip()
            for item in workflow_state.get("missing_outputs") or []
            if str(item).strip()
        }
        family = str(workflow_state.get("workflow_family", "")).strip().lower()
        primary_target_ref = str(workflow_state.get("primary_target_ref", "")).strip()

        if stage == "action_ready" and not next_actions:
            issues.append("workflow_action_ready_without_next_actions")
        if "message_sent" in missing_outputs and primary_target_ref and "deliver" not in next_actions:
            issues.append("workflow_missing_deliver_action")
        if "file_contents" in missing_outputs and primary_target_ref and not ({"read", "inspect"} & next_actions):
            issues.append("workflow_missing_read_action")
        if family == "web_lookup" and ({"search_results", "web_content"} & missing_outputs) and not ({"search", "fetch"} & next_actions):
            issues.append("workflow_missing_web_followup")
        if family in {"delivery", "document_summary", "local_lookup", "inspection"} and stage == "candidate_ready" and primary_target_ref and not next_actions:
            issues.append("workflow_candidate_ready_without_progress_action")
        return issues

    @staticmethod
    def _workflow_missing_outputs(workflow_state: dict[str, Any]) -> list[OutputKind]:
        if not isinstance(workflow_state, dict):
            return []
        mapping = {
            OutputKind.FILE_CONTENTS.value: OutputKind.FILE_CONTENTS,
            OutputKind.OBJECT_DETAILS.value: OutputKind.OBJECT_DETAILS,
            OutputKind.SEARCH_RESULTS.value: OutputKind.SEARCH_RESULTS,
            OutputKind.WEB_CONTENT.value: OutputKind.WEB_CONTENT,
            OutputKind.MESSAGE_SENT.value: OutputKind.MESSAGE_SENT,
        }
        resolved: list[OutputKind] = []
        for item in workflow_state.get("missing_outputs") or []:
            key = str(item).strip()
            output_kind = mapping.get(key)
            if output_kind is not None and output_kind not in resolved:
                resolved.append(output_kind)
        return resolved

    @staticmethod
    def _workflow_grounding_issues(workflow_state: dict[str, Any], *, response_text: str) -> list[str]:
        if not isinstance(workflow_state, dict):
            return []
        issues: list[str] = []
        stage = str(workflow_state.get("workflow_stage", "")).strip().lower()
        missing_outputs = {
            str(item).strip()
            for item in workflow_state.get("missing_outputs") or []
            if str(item).strip()
        }
        if stage == "candidate_ready" and response_text and missing_outputs & {
            OutputKind.FILE_CONTENTS.value,
            OutputKind.OBJECT_DETAILS.value,
            OutputKind.WEB_CONTENT.value,
        }:
            issues.append("workflow_stage_candidate_ready_cannot_support_grounded_response")
        if stage == "action_ready" and OutputKind.MESSAGE_SENT.value in missing_outputs and "已经发" in response_text:
            issues.append("workflow_stage_action_ready_cannot_claim_delivery_complete")
        return issues
