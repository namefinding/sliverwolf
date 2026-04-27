from __future__ import annotations

from typing import Protocol

from local_agent.modules.web.models import ExtractedContent, FetchedPage, ResearchBundle, SearchQuery, SearchResultRecord


class SearchProvider(Protocol):
    name: str

    def search(self, query: SearchQuery) -> list[SearchResultRecord]:
        ...


class PageFetcher(Protocol):
    name: str

    def fetch(
        self,
        url: str,
        *,
        max_chars: int,
        timeout_seconds: int = 30,
        allow_insecure: bool = False,
    ) -> FetchedPage:
        ...


class ContentExtractor(Protocol):
    name: str

    def extract(self, page: FetchedPage, *, max_chars: int) -> ExtractedContent:
        ...


class ResearchPipeline(Protocol):
    def search(
        self,
        *,
        query: str,
        max_results: int,
        domains: list[str] | None = None,
        preferred_domains: list[str] | None = None,
        recency_days: int | None = None,
        language: str | None = None,
    ) -> list[SearchResultRecord]:
        ...

    def fetch(
        self,
        *,
        url: str,
        max_chars: int,
        allow_insecure: bool = False,
        prefer_browser: bool = False,
    ) -> dict:
        ...

    def research(
        self,
        *,
        query: str,
        max_results: int,
        max_pages: int,
        max_chars: int,
        domains: list[str] | None = None,
        preferred_domains: list[str] | None = None,
        recency_days: int | None = None,
        language: str | None = None,
        allow_insecure: bool = False,
        prefer_browser: bool = False,
    ) -> ResearchBundle:
        ...
