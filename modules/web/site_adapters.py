from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SiteSearchDirective:
    site: str
    query: str
    domains: list[str]
    content_type: str = "generic"
    search_url: str | None = None


@dataclass(frozen=True)
class SiteAdapter:
    site: str
    aliases: tuple[str, ...]
    domains: tuple[str, ...]
    search_url_template: str

    def build(self, query: str, content_type: str = "generic") -> SiteSearchDirective:
        normalized_query = " ".join(str(query).strip().split())
        if self.site == "bilibili" and content_type == "video" and "视频" not in normalized_query:
            normalized_query = f"{normalized_query} 视频".strip()
        if self.site == "github" and content_type == "repo" and "repo" not in normalized_query.lower():
            normalized_query = f"{normalized_query} repo".strip()
        search_url = self.search_url_template.format(query=normalized_query)
        return SiteSearchDirective(
            site=self.site,
            query=normalized_query,
            domains=list(self.domains),
            content_type=content_type,
            search_url=search_url,
        )


_SITE_ADAPTERS = {
    "zhihu": SiteAdapter(
        site="zhihu",
        aliases=("zhihu", "知乎"),
        domains=("zhihu.com",),
        search_url_template="https://www.zhihu.com/search?type=content&q={query}",
    ),
    "bilibili": SiteAdapter(
        site="bilibili",
        aliases=("bilibili", "B站", "b站"),
        domains=("bilibili.com",),
        search_url_template="https://search.bilibili.com/all?keyword={query}",
    ),
    "github": SiteAdapter(
        site="github",
        aliases=("github", "GitHub"),
        domains=("github.com",),
        search_url_template="https://github.com/search?q={query}",
    ),
}


def build_site_search_directive(site: str, query: str, content_type: str = "generic") -> SiteSearchDirective | None:
    adapter = _SITE_ADAPTERS.get(str(site).strip().lower())
    if adapter is None:
        return None
    return adapter.build(query=query, content_type=content_type)
