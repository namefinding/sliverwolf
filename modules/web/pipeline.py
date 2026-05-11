from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

from local_agent.modules.web.cache import InMemoryWebCache
from local_agent.modules.web.extract_basic import BasicHtmlExtractor
from local_agent.modules.web.models import ResearchBundle, ResearchSource, SearchQuery, SearchResultRecord
from local_agent.modules.web.providers import ContentExtractor, PageFetcher, SearchProvider


class DefaultWebResearchPipeline:
    HIGH_RISK_FETCH_DOMAINS = (
        "zhihu.com",
        "zhuanlan.zhihu.com",
        "baike.baidu.com",
    )
    ANTI_BOT_MARKERS = (
        "blocked_page",
        "40362",
        "access denied",
        "captcha",
        "verify you are human",
        "unusual traffic",
        "访问行为异常",
        "请求存在异常",
        "暂时限制",
    )

    def __init__(
        self,
        *,
        search_provider: SearchProvider,
        http_fetcher: PageFetcher,
        browser_fetcher: PageFetcher | None = None,
        extractor: ContentExtractor | None = None,
        fallback_extractor: ContentExtractor | None = None,
        cache: InMemoryWebCache | None = None,
    ) -> None:
        self.search_provider = search_provider
        self.http_fetcher = http_fetcher
        self.browser_fetcher = browser_fetcher
        self.extractor = extractor or BasicHtmlExtractor()
        self.fallback_extractor = fallback_extractor or BasicHtmlExtractor()
        self.cache = cache or InMemoryWebCache()

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
        normalized_query = SearchQuery(
            query=query,
            max_results=max_results,
            domains=domains or [],
            preferred_domains=preferred_domains or [],
            recency_days=recency_days,
            language=language,
        )
        cache_key = (
            f"search::{normalized_query.query}::{normalized_query.max_results}"
            f"::{','.join(normalized_query.domains)}::{','.join(normalized_query.preferred_domains)}"
            f"::{normalized_query.recency_days}::{normalized_query.language}"
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return [SearchResultRecord.model_validate(item) for item in cached]
        results = self.search_provider.search(normalized_query)
        self.cache.set(cache_key, [item.model_dump(mode="json") for item in results])
        return results

    def fetch(
        self,
        *,
        url: str,
        max_chars: int,
        allow_insecure: bool = False,
        prefer_browser: bool = False,
    ) -> dict:
        if self._is_high_risk_url(url):
            domain = urlparse(url).netloc
            return {
                "url": url,
                "final_url": url,
                "status_code": None,
                "content_type": "",
                "title": "",
                "content": "",
                "excerpt": "",
                "extractor": "none",
                "fetched_via": "skipped_high_risk_domain",
                "truncated": False,
                "warnings": [f"fetch_skipped_high_risk_domain:{domain}"],
            }
        cache_key = f"fetch::{url}::{max_chars}::{prefer_browser}::{allow_insecure}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        fetcher = self.browser_fetcher if prefer_browser and self.browser_fetcher is not None else self.http_fetcher
        page = fetcher.fetch(url, max_chars=max_chars, allow_insecure=allow_insecure)
        extracted = self._extract(page, max_chars=max_chars)
        self._raise_for_invalid_page(page, extracted.content or page.html)
        payload = {
            "url": url,
            "final_url": page.final_url,
            "status_code": page.status_code,
            "content_type": page.content_type,
            "title": extracted.title or page.title,
            "content": extracted.content,
            "excerpt": extracted.excerpt,
            "extractor": extracted.extractor,
            "fetched_via": page.fetched_via,
            "truncated": extracted.truncated,
            "warnings": [*page.warnings, *extracted.warnings],
        }
        self.cache.set(cache_key, payload)
        return payload

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
        warnings: list[str] = []
        try:
            results = self.search(
                query=query,
                max_results=max_results,
                domains=domains,
                preferred_domains=preferred_domains,
                recency_days=recency_days,
                language=language,
            )
        except Exception as exc:
            if self._looks_like_url(query):
                fetched = self.fetch(
                    url=query,
                    max_chars=max_chars,
                    allow_insecure=allow_insecure,
                    prefer_browser=prefer_browser,
                )
                source = ResearchSource(
                    title=str(fetched.get("title", "")),
                    url=str(fetched.get("final_url") or fetched.get("url") or query),
                    source_domain=urlparse(str(fetched.get("final_url") or query)).netloc,
                    content=str(fetched.get("content", "")),
                    excerpt=str(fetched.get("excerpt", "")),
                    provider="direct_fetch",
                    extractor=str(fetched.get("extractor", "")),
                    fetched_via=str(fetched.get("fetched_via", "")),
                    status_code=int(fetched.get("status_code", 0)),
                    warnings=[str(item) for item in fetched.get("warnings", []) if item],
                )
                return ResearchBundle(
                    query=query,
                    results=[],
                    sources=[source],
                    content=source.content,
                    warnings=["search_bypassed_for_direct_url"],
                )
            warnings.append(f"search_failed: {exc}")
            results = []

        unique_results = self._prioritize_fetchable_results(self._dedupe_results(results))
        sources: list[ResearchSource] = []
        capped_results = unique_results[:max_pages]
        if capped_results:
            max_workers = min(len(capped_results), 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_item = {
                    executor.submit(
                        self._build_research_source,
                        result=result,
                        max_chars=max_chars,
                        allow_insecure=allow_insecure,
                        prefer_browser=prefer_browser,
                    ): index
                    for index, result in enumerate(capped_results)
                }
                ordered_sources: dict[int, ResearchSource] = {}
                ordered_warnings: dict[int, str] = {}
                for future in as_completed(future_to_item):
                    index = future_to_item[future]
                    result = capped_results[index]
                    try:
                        ordered_sources[index] = future.result()
                    except Exception as exc:
                        warning = f"fetch_failed:{result.url}:{exc}"
                        ordered_warnings[index] = warning
                        if self._is_anti_bot_error(exc):
                            ordered_sources[index] = self._build_snippet_source(
                                result,
                                warnings=["fetch_skipped_or_blocked_by_anti_bot", warning],
                            )
                sources = [ordered_sources[index] for index in sorted(ordered_sources)]
                warnings.extend(ordered_warnings[index] for index in sorted(ordered_warnings))

        content = "\n\n".join(
            filter(
                None,
                [
                    self._format_source_summary(index + 1, source)
                    for index, source in enumerate(sources)
                ],
            )
        )[: max_chars * max(1, min(max_pages, 4))]
        return ResearchBundle(query=query, results=unique_results, sources=sources, content=content, warnings=warnings)

    def _build_research_source(
        self,
        *,
        result: SearchResultRecord,
        max_chars: int,
        allow_insecure: bool,
        prefer_browser: bool,
    ) -> ResearchSource:
        if self._is_high_risk_fetch_result(result):
            return self._build_snippet_source(result, warnings=["fetch_skipped_high_risk_domain"])
        fetched = self._fetch_research_page(
            url=result.url,
            max_chars=max_chars,
            allow_insecure=allow_insecure,
            prefer_browser=prefer_browser,
        )
        return ResearchSource(
            title=str(fetched.get("title") or result.title),
            url=str(fetched.get("final_url") or result.url),
            source_domain=result.source_domain,
            snippet=result.snippet,
            content=str(fetched.get("content", "")),
            excerpt=str(fetched.get("excerpt", "")),
            provider=result.provider,
            extractor=str(fetched.get("extractor", "")),
            fetched_via=str(fetched.get("fetched_via", "")),
            published_at=result.published_at,
            status_code=int(fetched.get("status_code", 0)),
            warnings=[str(item) for item in fetched.get("warnings", []) if item],
        )

    @staticmethod
    def _build_snippet_source(result: SearchResultRecord, *, warnings: list[str] | None = None) -> ResearchSource:
        snippet = str(result.snippet or "").strip()
        return ResearchSource(
            title=result.title,
            url=result.url,
            source_domain=result.source_domain,
            snippet=snippet,
            content=snippet,
            excerpt=snippet[:280],
            provider=result.provider,
            extractor="search_snippet",
            fetched_via="search_result",
            published_at=result.published_at,
            status_code=None,
            warnings=warnings or [],
        )

    def _fetch_research_page(
        self,
        *,
        url: str,
        max_chars: int,
        allow_insecure: bool,
        prefer_browser: bool,
    ) -> dict:
        try:
            return self.fetch(
                url=url,
                max_chars=max_chars,
                allow_insecure=allow_insecure,
                prefer_browser=False,
            )
        except Exception as http_exc:
            if self.browser_fetcher is None:
                raise
            fetched = self.fetch(
                url=url,
                max_chars=max_chars,
                allow_insecure=allow_insecure,
                prefer_browser=True,
            )
            warnings = [str(item) for item in fetched.get("warnings", []) if item]
            warning_text = f"http_fetch_failed:{http_exc}"
            if warning_text not in warnings:
                warnings.append(warning_text)
            fetched["warnings"] = warnings
            return fetched

    def _extract(self, page, *, max_chars: int):
        try:
            return self.extractor.extract(page, max_chars=max_chars)
        except Exception as primary_exc:
            extracted = self.fallback_extractor.extract(page, max_chars=max_chars)
            if f"extractor_fallback:{primary_exc}" not in extracted.warnings:
                extracted.warnings.append(f"extractor_fallback:{primary_exc}")
            return extracted

    @staticmethod
    def _raise_for_invalid_page(page, content: str) -> None:
        text = str(content or "").strip()
        lowered = text.lower()
        if not text:
            raise ValueError("empty_page_content")

        if "c837c3673f657290e6fa7f9625d58826" in text or '"code":40362' in lowered:
            raise ValueError("blocked_page: zhihu_40362")

        if lowered.startswith("{") and lowered.endswith("}"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                code = payload.get("code")
                message = payload.get("message")
                if code is not None and message:
                    raise ValueError(f"error_json_page: code={code} message={str(message)[:120]}")

        blocked_markers = (
            "当前请求存在异常",
            "暂时限制本次访问",
            "access denied",
            "captcha",
            "verify you are human",
            "unusual traffic",
        )
        if any(marker in lowered or marker in text for marker in blocked_markers):
            domain = urlparse(str(getattr(page, "final_url", "") or getattr(page, "url", ""))).netloc
            raise ValueError(f"blocked_page: {domain or 'unknown_domain'}")

    @classmethod
    def _prioritize_fetchable_results(cls, results: list[SearchResultRecord]) -> list[SearchResultRecord]:
        return sorted(
            results,
            key=lambda item: (
                1 if cls._is_high_risk_fetch_result(item) else 0,
                item.rank or 999,
            ),
        )

    @classmethod
    def _is_high_risk_fetch_result(cls, result: SearchResultRecord) -> bool:
        domain = (result.source_domain or urlparse(result.url).netloc or "").lower()
        return cls._is_high_risk_domain(domain)

    @classmethod
    def _is_high_risk_url(cls, url: str) -> bool:
        return cls._is_high_risk_domain(urlparse(str(url or "")).netloc.lower())

    @classmethod
    def _is_high_risk_domain(cls, domain: str) -> bool:
        return any(domain == risky or domain.endswith(f".{risky}") for risky in cls.HIGH_RISK_FETCH_DOMAINS)

    @classmethod
    def _is_anti_bot_error(cls, exc: Exception) -> bool:
        text = str(exc).lower()
        return any(marker.lower() in text for marker in cls.ANTI_BOT_MARKERS)

    @staticmethod
    def _dedupe_results(results: list[SearchResultRecord]) -> list[SearchResultRecord]:
        deduped: list[SearchResultRecord] = []
        seen: set[str] = set()
        for result in results:
            key = result.url.rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(result)
        return deduped

    @staticmethod
    def _format_source_summary(index: int, source: ResearchSource) -> str:
        head = (source.excerpt or source.content[:280])[:280]
        prefix = f"[{index}] {source.title or source.url}"
        if head:
            return f"{prefix}\n{head}"
        return prefix

    @staticmethod
    def _looks_like_url(text: str) -> bool:
        parsed = urlparse(text.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
