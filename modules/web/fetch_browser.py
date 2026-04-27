from __future__ import annotations

from local_agent.modules.web.browser_runtime import SharedPlaywrightBrowser
from local_agent.modules.web.models import FetchedPage
from local_agent.modules.web.safety import validate_public_url


class PlaywrightPageFetcher:
    name = "browser"

    def __init__(
        self,
        *,
        browser_channel: str = "msedge",
        headless: bool = False,
        timeout_seconds: int = 30,
        runtime: SharedPlaywrightBrowser | None = None,
    ) -> None:
        self.browser_channel = browser_channel
        self.headless = headless
        self.timeout_seconds = timeout_seconds
        self.runtime = runtime or SharedPlaywrightBrowser(browser_channel=browser_channel, headless=headless)

    def fetch(
        self,
        url: str,
        *,
        max_chars: int,
        timeout_seconds: int = 30,
        allow_insecure: bool = False,
    ) -> FetchedPage:
        del allow_insecure
        validate_public_url(url)
        effective_timeout = timeout_seconds or self.timeout_seconds

        def _run(page) -> FetchedPage:
            timeout_ms = effective_timeout * 1000
            page.set_default_timeout(timeout_ms)
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            self._wait_for_readable_content(page, timeout_ms)
            html = page.content()[: max_chars * 4]
            return FetchedPage(
                url=url,
                final_url=page.url,
                status_code=200,
                content_type="text/html",
                html=html,
                fetched_via=self.name,
                title=page.title(),
            )

        return self.runtime.run_with_page(_run)

    @staticmethod
    def _wait_for_readable_content(page, timeout_ms: int) -> None:
        selectors = ("main", "article", "[role='main']", "body")
        for selector in selectors:
            try:
                page.locator(selector).first.wait_for(timeout=timeout_ms)
                return
            except Exception:
                continue
