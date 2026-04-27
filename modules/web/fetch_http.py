from __future__ import annotations

from html import unescape

import requests

from local_agent.modules.web.models import FetchedPage
from local_agent.modules.web.safety import validate_public_url


class RequestsPageFetcher:
    name = "http"

    def __init__(self, user_agent: str = "local-agent-kernel/0.2") -> None:
        self.user_agent = user_agent

    def fetch(
        self,
        url: str,
        *,
        max_chars: int,
        timeout_seconds: int = 12,
        allow_insecure: bool = False,
    ) -> FetchedPage:
        validate_public_url(url)
        warnings: list[str] = []
        verify = True
        try:
            response = self._request(url, timeout_seconds=timeout_seconds, verify=verify)
        except requests.exceptions.SSLError:
            if not allow_insecure:
                raise
            warnings.append("ssl_verification_disabled")
            verify = False
            response = self._request(url, timeout_seconds=timeout_seconds, verify=verify)

        html = response.text[: max_chars * 4]
        title = self._extract_title(html)
        return FetchedPage(
            url=url,
            final_url=str(response.url),
            status_code=response.status_code,
            content_type=response.headers.get("Content-Type", ""),
            html=html,
            fetched_via=self.name,
            title=title,
            warnings=warnings,
        )

    def _request(self, url: str, *, timeout_seconds: int, verify: bool) -> requests.Response:
        response = requests.get(
            url,
            timeout=timeout_seconds,
            verify=verify,
            headers={"User-Agent": self.user_agent},
        )
        response.raise_for_status()
        return response

    @staticmethod
    def _extract_title(html: str) -> str:
        lowered = html.lower()
        start = lowered.find("<title")
        if start < 0:
            return ""
        start = lowered.find(">", start)
        if start < 0:
            return ""
        end = lowered.find("</title>", start)
        if end < 0:
            return ""
        return unescape(html[start + 1 : end]).strip()
