from __future__ import annotations

from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    query: str
    max_results: int = 5
    domains: list[str] = Field(default_factory=list)
    preferred_domains: list[str] = Field(default_factory=list)
    recency_days: int | None = None
    language: str | None = None


class SearchResultRecord(BaseModel):
    title: str
    url: str
    snippet: str = ""
    source_domain: str = ""
    published_at: str | None = None
    provider: str = ""
    rank: int = 0
    score: float | None = None


class FetchedPage(BaseModel):
    url: str
    final_url: str
    status_code: int
    content_type: str = ""
    html: str
    fetched_via: str = "http"
    title: str = ""
    warnings: list[str] = Field(default_factory=list)


class ExtractedContent(BaseModel):
    url: str
    title: str = ""
    content: str = ""
    excerpt: str = ""
    published_at: str | None = None
    extractor: str = "basic"
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)


class ResearchSource(BaseModel):
    title: str
    url: str
    source_domain: str = ""
    snippet: str = ""
    content: str = ""
    excerpt: str = ""
    provider: str = ""
    extractor: str = ""
    fetched_via: str = ""
    published_at: str | None = None
    status_code: int | None = None
    warnings: list[str] = Field(default_factory=list)


class ResearchBundle(BaseModel):
    query: str
    results: list[SearchResultRecord] = Field(default_factory=list)
    sources: list[ResearchSource] = Field(default_factory=list)
    content: str = ""
    warnings: list[str] = Field(default_factory=list)
