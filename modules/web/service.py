from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from local_agent.modules.web.browser_opener import SystemBrowserOpener
from local_agent.modules.web.browser_runtime import SharedPlaywrightBrowser
from local_agent.modules.web.extract_basic import BasicHtmlExtractor
from local_agent.modules.web.extract_trafilatura import TrafilaturaExtractor
from local_agent.modules.web.fetch_browser import PlaywrightPageFetcher
from local_agent.modules.web.fetch_http import RequestsPageFetcher
from local_agent.modules.web.pipeline import DefaultWebResearchPipeline
from local_agent.modules.web.search_browser import PlaywrightSearchProvider
from local_agent.protocol.models import OutputKind, ToolManifest
from local_agent.protocol.models import WebConfig


class WebSearchInput(BaseModel):
    query: str
    max_results: int = 5
    domains: list[str] = Field(default_factory=list)
    preferred_domains: list[str] = Field(default_factory=list)
    recency_days: int | None = None
    language: str | None = None


class WebFetchInput(BaseModel):
    url: str
    max_chars: int = 8_000
    allow_insecure: bool = False
    prefer_browser: bool = False


class WebOpenPageInput(BaseModel):
    url: str


class WebResearchInput(BaseModel):
    query: str
    max_results: int = 5
    max_pages: int = 3
    max_chars: int = 4_000
    domains: list[str] = Field(default_factory=list)
    preferred_domains: list[str] = Field(default_factory=list)
    recency_days: int | None = None
    language: str | None = None
    allow_insecure: bool = False
    prefer_browser: bool = False


class WebModule:
    def __init__(self, pipeline=None, browser_opener=None, config: WebConfig | None = None) -> None:
        self.config = config or WebConfig()
        self.pipeline = pipeline or self._build_default_pipeline(self.config)
        self.browser_opener = browser_opener or SystemBrowserOpener()

    def manifests(self) -> list[ToolManifest]:
        return [
            ToolManifest(
                tool_name="web.search",
                module="web",
                description="Search the web for a query and return structured result candidates.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.SEARCH_RESULTS],
                input_schema=WebSearchInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"results": {"type": "array"}}},
            ),
            ToolManifest(
                tool_name="web.fetch",
                module="web",
                description="Fetch and extract readable content from a specific public webpage.",
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.WEB_CONTENT],
                input_schema=WebFetchInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"url": {"type": "string"}, "content": {"type": "string"}}},
            ),
            ToolManifest(
                tool_name="web.open_page",
                module="web",
                description="Open a public webpage in the local browser when the user explicitly wants to visit it.",
                side_effect=True,
                idempotent=False,
                produces=[OutputKind.PATH_OPENED],
                input_schema=WebOpenPageInput.model_json_schema(),
                output_schema={"type": "object", "properties": {"url": {"type": "string"}, "opened": {"type": "boolean"}}},
            ),
            ToolManifest(
                tool_name="web.research",
                module="web",
                description=(
                    "Search the web, fetch top sources, extract readable content, and return a grounded research bundle. "
                    "Prefer this when the user wants facts, latest information, or sourced summaries."
                ),
                side_effect=False,
                idempotent=True,
                produces=[OutputKind.SEARCH_RESULTS, OutputKind.WEB_CONTENT],
                input_schema=WebResearchInput.model_json_schema(),
                output_schema={
                    "type": "object",
                    "properties": {
                        "results": {"type": "array"},
                        "sources": {"type": "array"},
                        "content": {"type": "string"},
                    },
                },
            ),
        ]

    def executor_map(self) -> dict[str, Any]:
        return {
            "web.search": self.search,
            "web.fetch": self.fetch,
            "web.open_page": self.open_page,
            "web.research": self.research,
        }

    def search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = WebSearchInput.model_validate(arguments)
        results = self.pipeline.search(
            query=payload.query,
            max_results=payload.max_results,
            domains=payload.domains,
            preferred_domains=payload.preferred_domains,
            recency_days=payload.recency_days,
            language=payload.language,
        )
        return {
            "query": payload.query,
            "results": [item.model_dump(mode="json") for item in results],
        }

    def fetch(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = WebFetchInput.model_validate(arguments)
        return self.pipeline.fetch(
            url=payload.url,
            max_chars=payload.max_chars,
            allow_insecure=payload.allow_insecure,
            prefer_browser=payload.prefer_browser,
        )

    def open_page(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = WebOpenPageInput.model_validate(arguments)
        return self.browser_opener.open(payload.url)

    def research(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = WebResearchInput.model_validate(arguments)
        bundle = self.pipeline.research(
            query=payload.query,
            max_results=payload.max_results,
            max_pages=payload.max_pages,
            max_chars=payload.max_chars,
            domains=payload.domains,
            preferred_domains=payload.preferred_domains,
            recency_days=payload.recency_days,
            language=payload.language,
            allow_insecure=payload.allow_insecure,
            prefer_browser=payload.prefer_browser,
        )
        return bundle.model_dump(mode="json")

    @staticmethod
    def _build_default_pipeline(config: WebConfig) -> DefaultWebResearchPipeline:
        extractor = BasicHtmlExtractor()
        try:
            extractor = TrafilaturaExtractor()
        except RuntimeError:
            extractor = BasicHtmlExtractor()
        browser_runtime = SharedPlaywrightBrowser(
            browser_channel=config.browser_channel,
            headless=config.browser_headless,
        )
        return DefaultWebResearchPipeline(
            search_provider=WebModule._build_search_provider(config, runtime=browser_runtime),
            http_fetcher=RequestsPageFetcher(),
            browser_fetcher=PlaywrightPageFetcher(
                browser_channel=config.browser_channel,
                headless=config.browser_headless,
                timeout_seconds=config.browser_launch_timeout_seconds,
                runtime=browser_runtime,
            ),
            extractor=extractor,
            fallback_extractor=BasicHtmlExtractor(),
        )

    @staticmethod
    def _build_search_provider(config: WebConfig, runtime: SharedPlaywrightBrowser | None = None):
        provider_name = config.search_provider.strip().lower()
        if provider_name in {"browser", "auto"}:
            return PlaywrightSearchProvider(
                browser_channel=config.browser_channel,
                headless=config.browser_headless,
                search_engine=config.browser_search_engine,
                timeout_seconds=config.browser_launch_timeout_seconds,
                runtime=runtime,
            )
        raise RuntimeError(f"Unsupported web.search_provider: {config.search_provider}")
