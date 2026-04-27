from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from local_agent.modules.web.site_adapters import build_site_search_directive
from local_agent.protocol.models import (
    DecisionType,
    DocumentDeliveryIntent,
    KnowledgeRequestIntent,
    OutputKind,
    RiskLevel,
    SiteSearchIntent,
    TaskGoal,
    ToolCallResult,
    ToolDecision,
)
from local_agent.utils.file_query_normalizer import FileQueryNormalizer
from local_agent.utils.target_resolver import TargetResolution, resolve_target_reference
from local_agent.utils.web_query_normalizer import WebQueryNormalizer


@dataclass
class WebQueryPlan:
    query: str
    query_terms: list[str]
    alias_terms: list[str]
    domains: list[str]
    preferred_domains: list[str]
    query_type: str
    action: str | None
    reason: str
    resolved_target: str | None = None
    output_file: str | None = None
    recency_days: int | None = None


class WebRetrievalStrategy:
    _SEARCH_TERMS = (
        "搜",
        "搜索",
        "搜一下",
        "搜搜看",
        "查",
        "查一下",
        "查查",
        "search",
        "look up",
        "find",
    )
    _RESEARCH_TERMS = (
        "是什么",
        "什么意思",
        "含义",
        "定义",
        "总结",
        "概括",
        "行情",
        "新闻",
        "最新",
        "今日",
        "今天",
        "recent",
        "latest",
        "news",
        "meaning",
        "definition",
    )
    _WRITE_TERMS = (
        "写入",
        "保存到",
        "保存进",
        "记到",
        "导出到",
        "write",
        "save",
        "export",
    )

    @classmethod
    def build_initial_lookup(
        cls,
        *,
        user_text: str,
        completed_outputs: list[OutputKind],
        delivery_intent: DocumentDeliveryIntent | None = None,
        knowledge_intent: KnowledgeRequestIntent | None = None,
        site_search_intent: SiteSearchIntent | None = None,
    ) -> ToolDecision | None:
        if OutputKind.SEARCH_RESULTS in completed_outputs or OutputKind.WEB_CONTENT in completed_outputs:
            return None

        resolution = resolve_target_reference(user_text)
        if cls._should_defer_to_local_target(user_text, resolution, delivery_intent, knowledge_intent):
            return None

        plan = cls._plan_query(user_text, resolution, delivery_intent, knowledge_intent, site_search_intent)
        if not plan.query or plan.action is None:
            return None
        weather_query = cls._is_weather_or_forecast_query(plan.query) or cls._is_weather_or_forecast_query(user_text)
        if weather_query and not cls._weather_query_has_location(plan.query):
            return ToolDecision(
                decision=DecisionType.CLARIFY,
                intent="weather_location_required",
                reason="Weather forecasts require a city or region before lookup.",
                selected_tool=None,
                arguments={},
                risk_level=RiskLevel.LOW,
                response_hint="你想查哪个城市或地区明天的天气？",
                overall_task_goal=None,
                expected_step_outputs=[],
            )

        goal_outputs = cls._goal_outputs_for_action("research", plan.output_file)
        if plan.action in {"research", "search"}:
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="research_web_topic",
                reason=plan.reason,
                selected_tool="web.research",
                arguments={
                    "query": plan.query,
                    "query_terms": plan.query_terms,
                    "alias_terms": plan.alias_terms,
                    "max_results": 5,
                    "max_pages": 1 if weather_query else 2,
                    "domains": plan.domains,
                    "preferred_domains": plan.preferred_domains,
                    "recency_days": plan.recency_days,
                    "prefer_browser": not weather_query,
                },
                risk_level=RiskLevel.LOW,
                overall_task_goal=TaskGoal(
                    summary=f"Research the web topic matching {plan.query}.",
                    required_outputs=goal_outputs,
                ),
                expected_step_outputs=[OutputKind.SEARCH_RESULTS, OutputKind.WEB_CONTENT],
            )
        return None

    @classmethod
    def build_empty_result_fallback(
        cls,
        *,
        user_text: str,
        last_decision: ToolDecision,
        last_result: ToolCallResult,
        delivery_intent: DocumentDeliveryIntent | None = None,
        knowledge_intent: KnowledgeRequestIntent | None = None,
        site_search_intent: SiteSearchIntent | None = None,
    ) -> ToolDecision | None:
        if last_decision.decision != DecisionType.TOOL_CALL or last_result.status != "success":
            return None

        selected_tool = last_decision.selected_tool or ""
        plan = cls._plan_query(
            user_text,
            resolve_target_reference(user_text),
            delivery_intent,
            knowledge_intent,
            site_search_intent,
        )
        if not plan.query:
            return None

        if selected_tool == "web.research" and not last_result.data.get("content") and not last_result.data.get("results"):
            return ToolDecision(
                decision=DecisionType.CLARIFY,
                intent="clarify_after_empty_web_research",
                reason="The web research step did not return usable results, so the user needs to narrow the topic.",
                response_hint="我这次上网检索还没拿到可靠结果。你可以告诉我更具体的关键词、公司名、股票代码，或者限定想看的范围。",
                overall_task_goal=last_decision.overall_task_goal,
            )

        return None

    @classmethod
    def _plan_query(
        cls,
        user_text: str,
        resolution: TargetResolution,
        delivery_intent: DocumentDeliveryIntent | None = None,
        knowledge_intent: KnowledgeRequestIntent | None = None,
        site_search_intent: SiteSearchIntent | None = None,
    ) -> WebQueryPlan:
        normalized = WebQueryNormalizer.normalize(user_text)
        action = cls._infer_action(
            user_text,
            normalized.query_type,
            resolution,
            delivery_intent,
            knowledge_intent,
            site_search_intent,
        )
        query = (normalized.search_query or resolution.resolved_target or resolution.raw_target).strip()
        domains: list[str] = []
        preferred_domains: list[str] = []

        if resolution.reason in {"explicit_url", "explicit_domain"} and resolution.resolved_target:
            query = resolution.resolved_target
        elif (
            resolution.reason == "known_site_alias"
            and resolution.resolved_target
            and cls._should_use_site_alias_as_query(normalized.search_query, resolution)
        ):
            query = resolution.resolved_target

        if site_search_intent is not None and site_search_intent.site and site_search_intent.query:
            directive = build_site_search_directive(
                site=site_search_intent.site,
                query=site_search_intent.query,
                content_type=site_search_intent.content_type,
            )
            if directive is not None:
                query = directive.query
                site_scope = str(getattr(site_search_intent, "site_scope", "preferred") or "preferred").strip().lower()
                if site_scope == "required":
                    domains = directive.domains
                else:
                    preferred_domains = directive.domains
        elif site_search_intent is not None and site_search_intent.query:
            query = site_search_intent.query.strip()

        recency_days = cls._recency_days_for_request(user_text, normalized.query_type, knowledge_intent)
        if recency_days is not None:
            query = cls._augment_time_sensitive_query(query)

        output_file = (delivery_intent.output_file if delivery_intent is not None else None) or cls._extract_output_file(user_text)
        if output_file is None and delivery_intent is not None and delivery_intent.save_output:
            output_file = cls._default_output_filename(delivery_intent)
        if output_file:
            action = "research"
        elif delivery_intent is not None and delivery_intent.wants_document:
            action = "research"

        if normalized.query_type == "live_topic":
            reason = "The request looks time-sensitive or market-oriented, so grounded web research should run first."
        elif normalized.query_type == "factual":
            reason = "The request looks like a factual definition or concept lookup, so grounded web research should run first."
        elif knowledge_intent is not None and knowledge_intent.needs_grounding:
            reason = "The request looks like an external knowledge question that should be grounded before answering."
        elif action == "search":
            reason = "The request explicitly asks for a web lookup, so start with bounded web research."
        else:
            reason = "The request looks like a web research task."

        return WebQueryPlan(
            query=query,
            query_terms=normalized.core_terms,
            alias_terms=normalized.alias_terms,
            domains=domains,
            preferred_domains=preferred_domains,
            query_type=normalized.query_type,
            action=action,
            reason=reason,
            resolved_target=resolution.resolved_target,
            output_file=output_file,
            recency_days=recency_days,
        )

    @classmethod
    def _infer_action(
        cls,
        user_text: str,
        query_type: str,
        resolution: TargetResolution,
        delivery_intent: DocumentDeliveryIntent | None = None,
        knowledge_intent: KnowledgeRequestIntent | None = None,
        site_search_intent: SiteSearchIntent | None = None,
    ) -> str | None:
        lowered = user_text.lower()
        if resolution.action == "research":
            return "research"
        if site_search_intent is not None and site_search_intent.site and site_search_intent.query:
            return "research"
        if cls._should_start_with_research(
            user_text=user_text,
            query_type=query_type,
            resolution=resolution,
            delivery_intent=delivery_intent,
            knowledge_intent=knowledge_intent,
        ):
            return "research"
        if resolution.action == "search":
            return "research"
        if any(term in lowered for term in cls._WRITE_TERMS):
            return "research"
        if any(term in lowered for term in cls._SEARCH_TERMS):
            return "research"
        if any(term in lowered for term in cls._RESEARCH_TERMS):
            return "research"
        return None

    @classmethod
    def _should_start_with_research(
        cls,
        *,
        user_text: str,
        query_type: str,
        resolution: TargetResolution,
        delivery_intent: DocumentDeliveryIntent | None,
        knowledge_intent: KnowledgeRequestIntent | None,
    ) -> bool:
        lowered = user_text.lower()
        if delivery_intent is not None and (delivery_intent.wants_document or delivery_intent.save_output):
            return True
        if resolution.target_type == "web_url":
            return True
        if knowledge_intent is None:
            return False
        if not knowledge_intent.needs_grounding:
            return False
        if cls._is_weather_or_forecast_query(user_text):
            return False
        if knowledge_intent.lookup_requested:
            return False
        if query_type == "live_topic":
            return False
        if any(term in lowered for term in cls._SEARCH_TERMS):
            return False
        return False

    @staticmethod
    def _should_use_site_alias_as_query(search_query: str, resolution: TargetResolution) -> bool:
        query = str(search_query or "").strip().lower()
        alias = str(resolution.raw_target or "").strip().lower()
        canonical = str(resolution.canonical_name or "").strip().lower()
        if not query:
            return True
        if query in {alias, canonical}:
            return True
        return any(term in query for term in ("官网", "官方网站", "homepage", "website", "official site"))

    @classmethod
    def _should_defer_to_local_target(
        cls,
        user_text: str,
        resolution: TargetResolution,
        delivery_intent: DocumentDeliveryIntent | None,
        knowledge_intent: KnowledgeRequestIntent | None,
    ) -> bool:
        local_output_requested = bool(delivery_intent is not None and delivery_intent.save_output)
        output_target = (
            (delivery_intent.output_file if delivery_intent is not None else None)
            or cls._extract_output_file(user_text)
            or ""
        ).strip().lower()
        resolved_target = str(resolution.resolved_target or resolution.raw_target or "").strip().lower()
        if knowledge_intent is not None and knowledge_intent.knowledge_type == "local_workspace":
            return True
        if resolution.target_type in {"local_file", "local_folder"}:
            if (
                output_target
                and resolved_target
                and output_target == resolved_target
                and not cls._looks_like_local_listing_request(user_text.lower())
                and (
                    any(term in user_text.lower() for term in cls._SEARCH_TERMS)
                    or any(term in user_text.lower() for term in cls._RESEARCH_TERMS)
                    or (knowledge_intent is not None and knowledge_intent.needs_grounding)
                )
            ):
                return False
            return True

        lowered = user_text.lower()
        if any(term in lowered for term in ("文件", "文件夹", "目录", "路径", "工作区", "folder", "directory", "path", "workspace")):
            return True

        file_query = FileQueryNormalizer.normalize(user_text)
        if file_query.file_type_hints and not local_output_requested:
            return True
        if file_query.query_type in {"name_like", "mixed"} and not local_output_requested:
            local_markers = {
                "readme",
                "log",
                "report",
                "startup",
                "folder",
                "directory",
                "markdown",
                "文档",
                "目录",
                "文件夹",
                "汇报",
                "日志",
                "架构",
            }
            if any(term.lower() in local_markers for term in file_query.core_terms):
                return True

        if delivery_intent is not None and delivery_intent.save_output and cls._looks_like_local_listing_request(lowered):
            return True
        return False

    @staticmethod
    def _looks_like_local_listing_request(lowered: str) -> bool:
        return any(
            phrase in lowered
            for phrase in (
                "list current",
                "current directory",
                "current workspace",
                "subfolder",
                "子文件夹",
                "一级目录",
                "当前目录",
                "当前工作区",
                "列出",
            )
        )

    @staticmethod
    def _default_output_filename(delivery_intent: DocumentDeliveryIntent) -> str:
        raw_title = delivery_intent.title or "agent-output"
        output_format = delivery_intent.output_format or "docx"
        safe_title = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff._-]+", "-", str(raw_title)).strip("-._") or "agent-output"
        return f"{safe_title}.{output_format}"

    @staticmethod
    def _extract_output_file(user_text: str) -> str | None:
        match = re.search(r"([A-Za-z0-9_.\\/-]+\.(txt|md|json|csv|yaml|yml))", user_text)
        return match.group(1) if match else None

    @staticmethod
    def _recency_days_for_request(
        user_text: str,
        query_type: str,
        knowledge_intent: KnowledgeRequestIntent | None,
    ) -> int | None:
        lowered = user_text.lower()
        if knowledge_intent is not None and knowledge_intent.time_sensitive:
            return 14
        if query_type == "live_topic":
            return 14
        if any(term in lowered for term in ("最新", "最近", "今日", "今天", "news", "latest", "recent")):
            return 14
        return None

    @staticmethod
    def _augment_time_sensitive_query(query: str) -> str:
        cleaned = query.strip()
        if not cleaned:
            return cleaned
        if WebRetrievalStrategy._is_weather_or_forecast_query(cleaned):
            return cleaned
        if any(term in cleaned.lower() for term in ("今日", "今天", "today")):
            return cleaned
        current_year = str(datetime.now().year)
        additions: list[str] = []
        if current_year not in cleaned:
            additions.append(current_year)
        if not any(term in cleaned.lower() for term in ("最新", "最近", "latest", "recent", "news")):
            additions.append("最新")
        if not additions:
            return cleaned
        return f"{cleaned} {' '.join(additions)}"

    @staticmethod
    def _is_weather_or_forecast_query(query: str) -> bool:
        lowered = str(query or "").lower()
        weather_terms = (
            "weather",
            "forecast",
            "temperature",
            "\u5929\u6c14",
            "\u6c14\u6e29",
            "\u6e29\u5ea6",
            "\u964d\u96e8",
            "\u4e0b\u96e8",
            "\u9884\u62a5",
            "\u5929\u6c23",
            "\u6c23\u6eab",
            "\u6eab\u5ea6",
        )
        relative_day_terms = (
            "tomorrow",
            "today",
            "tonight",
            "\u660e\u5929",
            "\u4eca\u665a",
            "\u4eca\u5929",
            "\u4eca\u65e5",
            "\u540e\u5929",
        )
        return any(term in lowered for term in weather_terms) or any(term in lowered for term in relative_day_terms)

    @staticmethod
    def _weather_query_has_location(query: str) -> bool:
        cleaned = str(query or "").lower()
        for term in (
            "weather",
            "forecast",
            "temperature",
            "tomorrow",
            "today",
            "tonight",
            "\u5929\u6c14",
            "\u6c14\u6e29",
            "\u6e29\u5ea6",
            "\u964d\u96e8",
            "\u4e0b\u96e8",
            "\u9884\u62a5",
            "\u5929\u6c23",
            "\u6c23\u6eab",
            "\u6eab\u5ea6",
            "\u660e\u5929",
            "\u4eca\u665a",
            "\u4eca\u5929",
            "\u4eca\u65e5",
            "\u540e\u5929",
            "\u7684",
            "\u5e2e\u6211",
            "\u8bf7",
            "\u67e5\u67e5",
            "\u67e5\u8be2",
            "\u67e5\u4e00\u4e0b",
        ):
            cleaned = cleaned.replace(term, " ")
        cleaned = re.sub(r"[\s,.!?;:()]+", "", cleaned)
        return bool(cleaned)

    @staticmethod
    def _goal_outputs_for_action(action: str, output_file: str | None) -> list[OutputKind]:
        outputs = [OutputKind.SEARCH_RESULTS]
        if action == "research":
            outputs.append(OutputKind.WEB_CONTENT)
        if output_file is not None:
            outputs.append(OutputKind.FILE_WRITTEN)
        return outputs
