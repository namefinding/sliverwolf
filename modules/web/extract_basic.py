from __future__ import annotations

import re
from html import unescape

from local_agent.modules.web.models import ExtractedContent, FetchedPage


class BasicHtmlExtractor:
    name = "basic"

    def extract(self, page: FetchedPage, *, max_chars: int) -> ExtractedContent:
        title = page.title or self._extract_title(page.html)
        text = re.sub(r"<script[\s\S]*?</script>", " ", page.html, flags=re.IGNORECASE)
        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<noscript[\s\S]*?</noscript>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<.*?>", " ", text, flags=re.DOTALL)
        text = re.sub(r"\s+", " ", unescape(text)).strip()
        truncated = len(text) > max_chars
        content = text[:max_chars]
        excerpt = content[:400]
        return ExtractedContent(
            url=page.final_url,
            title=title,
            content=content,
            excerpt=excerpt,
            extractor=self.name,
            truncated=truncated,
            warnings=list(page.warnings),
        )

    @staticmethod
    def _extract_title(html: str) -> str:
        match = re.search(r"<title[^>]*>(?P<title>.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        title = re.sub(r"\s+", " ", unescape(match.group("title"))).strip()
        return title
