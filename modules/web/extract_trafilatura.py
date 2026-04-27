from __future__ import annotations

from local_agent.modules.web.models import ExtractedContent, FetchedPage


class TrafilaturaExtractor:
    name = "trafilatura"

    def __init__(self) -> None:
        try:
            import trafilatura  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("trafilatura is not installed") from exc
        self._trafilatura = trafilatura

    def extract(self, page: FetchedPage, *, max_chars: int) -> ExtractedContent:
        extracted = self._trafilatura.extract(
            page.html,
            include_links=False,
            include_images=False,
            include_tables=True,
        )
        metadata = None
        try:
            metadata = self._trafilatura.extract_metadata(page.html)
        except Exception:  # pragma: no cover - best effort metadata
            metadata = None

        content = (extracted or "").strip()
        truncated = len(content) > max_chars
        normalized_content = content[:max_chars]
        title = getattr(metadata, "title", "") if metadata is not None else ""
        published_at = getattr(metadata, "date", None) if metadata is not None else None
        return ExtractedContent(
            url=page.final_url,
            title=title or page.title,
            content=normalized_content,
            excerpt=normalized_content[:400],
            published_at=published_at,
            extractor=self.name,
            truncated=truncated,
            warnings=list(page.warnings),
        )
