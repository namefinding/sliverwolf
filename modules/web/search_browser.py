from __future__ import annotations

import base64
from urllib.parse import parse_qs, quote_plus, urlparse

from local_agent.modules.web.browser_runtime import SharedPlaywrightBrowser
from local_agent.modules.web.models import SearchQuery, SearchResultRecord


class PlaywrightSearchProvider:
    name = "browser_search"

    def __init__(
        self,
        *,
        browser_channel: str = "msedge",
        headless: bool = False,
        search_engine: str = "bing",
        timeout_seconds: int = 30,
        runtime: SharedPlaywrightBrowser | None = None,
    ) -> None:
        self.browser_channel = browser_channel
        self.headless = headless
        self.search_engine = search_engine.strip().lower() or "bing"
        self.timeout_seconds = timeout_seconds
        self.runtime = runtime or SharedPlaywrightBrowser(browser_channel=browser_channel, headless=headless)

    def search(self, query: SearchQuery) -> list[SearchResultRecord]:
        def _run(page) -> list[SearchResultRecord]:
            timeout_ms = self.timeout_seconds * 1000
            page.set_default_timeout(timeout_ms)
            page.goto(self._engine_home_url(), wait_until="domcontentloaded", timeout=timeout_ms)
            self._submit_query(page, query.query)
            if not self._wait_for_results(page):
                page.goto(
                    self._search_url(query.query),
                    wait_until="domcontentloaded",
                    timeout=timeout_ms,
                )
                self._wait_for_results(page)
            return self._extract_results(page, provider=self.name)

        results = self.runtime.run_with_page(_run)

        if query.domains:
            allowed = {domain.strip().lower() for domain in query.domains if domain.strip()}
            results = [item for item in results if self._domain_allowed(item.source_domain, allowed)]
        if query.preferred_domains and results:
            preferred = {domain.strip().lower() for domain in query.preferred_domains if domain.strip()}
            results = sorted(
                results,
                key=lambda item: 0 if self._domain_allowed(item.source_domain, preferred) else 1,
            )
        return results[: query.max_results]

    def _engine_home_url(self) -> str:
        if self.search_engine == "google":
            return "https://www.google.com/"
        return "https://www.bing.com/"

    def _submit_query(self, page, query_text: str) -> None:
        if self.search_engine == "google":
            locator = page.locator('textarea[name="q"], input[name="q"]').first
        else:
            locator = page.locator('textarea[name="q"], input[name="q"], #sb_form_q').first
        locator.wait_for(timeout=self.timeout_seconds * 1000)
        locator.fill(query_text)
        locator.press("Enter")

    def _wait_for_results(self, page) -> bool:
        timeout_ms = self.timeout_seconds * 1000
        selectors = ["div.g"] if self.search_engine == "google" else ["li.b_algo", "#b_results"]
        for selector in selectors:
            try:
                page.locator(selector).first.wait_for(timeout=timeout_ms)
                return True
            except Exception:
                continue
        return self._has_result_candidates(page)

    def _extract_results(self, page, *, provider: str) -> list[SearchResultRecord]:
        if self.search_engine == "google":
            results = self._extract_google_results(page, provider=provider)
        else:
            results = self._extract_bing_results(page, provider=provider)
        if results:
            return results
        fallback = self._extract_visible_search_summary(page, provider=provider)
        return [fallback] if fallback is not None else []

    def _has_result_candidates(self, page) -> bool:
        if self.search_engine == "google":
            return page.locator("div.g").count() > 0
        return page.locator("li.b_algo").count() > 0

    def _search_url(self, query_text: str) -> str:
        encoded = quote_plus(query_text)
        if self.search_engine == "google":
            return f"https://www.google.com/search?q={encoded}"
        return f"https://www.bing.com/search?q={encoded}"

    @staticmethod
    def _domain_allowed(source_domain: str, allowed: set[str]) -> bool:
        normalized = source_domain.strip().lower()
        return any(normalized == domain or normalized.endswith(f".{domain}") for domain in allowed)

    @staticmethod
    def _extract_visible_search_summary(page, *, provider: str) -> SearchResultRecord | None:
        try:
            title = (page.title() or "").strip() or "Search result summary"
            body = (page.locator("body").first.inner_text(timeout=1500) or "").strip()
        except Exception:
            return None
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        snippet = " ".join(lines[:12]).strip()
        if not snippet:
            return None
        parsed = urlparse(page.url)
        return SearchResultRecord(
            title=title,
            url=page.url,
            snippet=snippet[:800],
            source_domain=parsed.netloc,
            provider=f"{provider}_visible_summary",
            rank=1,
        )

    @staticmethod
    def _extract_bing_results(page, *, provider: str) -> list[SearchResultRecord]:
        items = page.locator("li.b_algo")
        count = min(items.count(), 8)
        results: list[SearchResultRecord] = []
        for index in range(count):
            item = items.nth(index)
            link = item.locator("h2 a").first
            if link.count() == 0:
                continue
            title = (link.text_content() or "").strip()
            url = PlaywrightSearchProvider._normalize_bing_target_url((link.get_attribute("href") or "").strip())
            snippet = (item.locator("p").first.text_content() or "").strip()
            if not title or not url:
                continue
            parsed = urlparse(url)
            results.append(
                SearchResultRecord(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source_domain=parsed.netloc,
                    provider=provider,
                    rank=len(results) + 1,
                )
            )
        return results

    @staticmethod
    def _normalize_bing_target_url(raw_url: str) -> str:
        if not raw_url:
            return raw_url
        parsed = urlparse(raw_url)
        if parsed.netloc.lower() != "www.bing.com" or not parsed.path.startswith("/ck/"):
            return raw_url
        wrapped = parse_qs(parsed.query).get("u", [""])[0].strip()
        if not wrapped:
            return raw_url
        if wrapped.startswith("a1"):
            wrapped = wrapped[2:]
        padding = "=" * (-len(wrapped) % 4)
        try:
            decoded = base64.urlsafe_b64decode(wrapped + padding).decode("utf-8", errors="ignore").strip()
        except Exception:
            return raw_url
        if decoded.startswith("http://") or decoded.startswith("https://"):
            return decoded
        return raw_url

    @staticmethod
    def _extract_google_results(page, *, provider: str) -> list[SearchResultRecord]:
        items = page.locator("div.g")
        count = min(items.count(), 8)
        results: list[SearchResultRecord] = []
        for index in range(count):
            item = items.nth(index)
            link = item.locator("a").first
            title_node = item.locator("h3").first
            if link.count() == 0 or title_node.count() == 0:
                continue
            title = (title_node.text_content() or "").strip()
            url = (link.get_attribute("href") or "").strip()
            snippet = (item.locator("div.VwiC3b, span.aCOpRe").first.text_content() or "").strip()
            if not title or not url.startswith("http"):
                continue
            parsed = urlparse(url)
            results.append(
                SearchResultRecord(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source_domain=parsed.netloc,
                    provider=provider,
                    rank=len(results) + 1,
                )
            )
        return results
