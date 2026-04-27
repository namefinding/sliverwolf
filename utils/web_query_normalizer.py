from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class NormalizedWebQuery:
    raw_text: str
    search_query: str
    normalized_text: str
    core_terms: list[str]
    alias_terms: list[str]
    query_type: str  # navigational | factual | live_topic | mixed


class WebQueryNormalizer:
    _LIGHT_STOP_TERMS = {"\u7684", "\u4e00\u4e0b", "\u4e00", "\u770b", "\u4e0b", "\u90a3\u4e2a", "\u8fd9\u4e2a"}
    _STOP_PHRASES = (
        "\u8054\u7f51",
        "\u4e0a\u7f51",
        "\u53bb\u7f51\u4e0a",
        "\u5728\u7f51\u4e0a",
        "\u5e2e\u6211",
        "\u8bf7\u5e2e\u6211",
        "\u8bf7",
        "\u9ebb\u70e6\u4f60",
        "\u5e2e\u5fd9",
        "\u641c\u4e00\u4e0b",
        "\u641c\u4e00\u641c",
        "\u641c\u641c\u770b",
        "\u67e5\u4e00\u4e0b",
        "\u67e5\u4e00\u67e5",
        "\u67e5\u67e5",
        "\u641c\u7d22",
        "\u68c0\u7d22",
        "\u627e\u4e00\u4e0b",
        "\u627e\u4e00\u627e",
        "\u627e\u627e",
        "\u76f4\u63a5\u544a\u8bc9\u6211",
        "\u544a\u8bc9\u6211",
        "\u7ed9\u6211\u8bb2\u8bb2",
        "look up",
        "search for",
        "search",
        "find out",
        "tell me",
        "please",
    )
    _GENERIC_SITE_TERMS = (
        "\u89c6\u9891\u7f51\u7ad9",
        "\u89c6\u9891\u5e73\u53f0",
        "\u4ee3\u7801\u6258\u7ba1\u7f51\u7ad9",
        "\u4ee3\u7801\u6258\u7ba1\u5e73\u53f0",
        "\u95ee\u7b54\u7f51\u7ad9",
        "\u793e\u4ea4\u7f51\u7ad9",
        "\u641c\u7d22\u5f15\u64ce",
        "\u8bba\u6587\u7f51\u7ad9",
    )
    _KNOWN_TERMS = (
        "\u4eca\u65e5\u80a1\u5e02\u884c\u60c5",
        "\u4eca\u5929\u80a1\u5e02\u884c\u60c5",
        "\u80a1\u5e02\u884c\u60c5",
        "\u80a1\u5e02",
        "\u80a1\u7968",
        "\u5927\u76d8",
        "\u7f8e\u80a1",
        "\u6e2f\u80a1",
        "\u6700\u8fd1",
        "\u8fd1\u671f",
        "\u6700\u65b0",
        "\u65b0\u52a8\u5411",
        "\u65b0\u8d8b\u52bf",
        "\u65b0\u8fdb\u5c55",
        "\u52a9\u624b\u9886\u57df",
        "ai\u52a9\u624b",
        "ai\u52a9\u624b\u9886\u57df",
        "\u4e0a\u8bc1",
        "llm",
        "\u5927\u8bed\u8a00\u6a21\u578b",
        "large language model",
        "\u4eba\u5de5\u667a\u80fd",
        "ai",
        "github",
        "bilibili",
        "b\u7ad9",
        "\u77e5\u4e4e",
        "\u5b98\u7f51",
        "\u65b0\u95fb",
        "\u542b\u4e49",
        "\u89c6\u9891\u7f51\u7ad9",
        "\u89c6\u9891\u5e73\u53f0",
        "\u4ee3\u7801\u6258\u7ba1\u5e73\u53f0",
    )
    _ALIAS_MAP = {
        "llm": ("large language model", "\u5927\u8bed\u8a00\u6a21\u578b"),
        "large language model": ("llm", "\u5927\u8bed\u8a00\u6a21\u578b"),
        "\u5927\u8bed\u8a00\u6a21\u578b": ("llm", "large language model"),
        "\u80a1\u5e02": ("\u80a1\u5e02\u884c\u60c5", "\u80a1\u7968\u884c\u60c5", "\u5927\u76d8"),
        "\u4eca\u65e5\u80a1\u5e02\u884c\u60c5": ("\u4eca\u5929\u80a1\u5e02\u884c\u60c5", "\u80a1\u5e02\u884c\u60c5", "\u5927\u76d8\u884c\u60c5"),
        "github": ("github \u5b98\u7f51",),
        "bilibili": ("b\u7ad9",),
        "b\u7ad9": ("bilibili",),
        "\u89c6\u9891\u7f51\u7ad9": ("\u89c6\u9891\u5e73\u53f0",),
        "\u4ee3\u7801\u6258\u7ba1\u5e73\u53f0": ("\u4ee3\u7801\u6258\u7ba1\u7f51\u7ad9",),
    }
    _LIVE_HINT_TERMS = (
        "\u4eca\u65e5",
        "\u4eca\u5929",
        "\u6700\u65b0",
        "\u6700\u8fd1",
        "\u8fd1\u671f",
        "\u884c\u60c5",
        "\u65b0\u95fb",
        "current",
        "latest",
        "today",
        "news",
    )
    _DEFINITION_HINT_TERMS = (
        "\u662f\u4ec0\u4e48",
        "\u4ec0\u4e48\u610f\u601d",
        "\u542b\u4e49",
        "\u5b9a\u4e49",
        "meaning",
        "definition",
        "stands for",
    )
    _NAVIGATION_HINT_TERMS = (
        "\u5b98\u7f51",
        "\u7f51\u7ad9",
        "\u7f51\u9875",
        "\u7ad9\u70b9",
        "homepage",
        "website",
        "official site",
    )
    _OUTPUT_FILE_PATTERN = re.compile(
        r"\s*(?:\u5e76|\u7136\u540e)?(?:\u5199\u5165|\u4fdd\u5b58\u5230|\u4fdd\u5b58\u8fdb|\u8bb0\u5230|\u5bfc\u51fa\u5230)\s+[A-Za-z0-9_.\\/-]+\.(?:txt|md|json|csv|yaml|yml)\s*$",
        flags=re.IGNORECASE,
    )

    @classmethod
    def normalize(cls, text: str) -> NormalizedWebQuery:
        raw_text = " ".join(str(text).strip().split())
        search_query = cls._extract_search_query(raw_text)
        cleaned = cls._clean_text(search_query)
        core_terms = cls._extract_core_terms(cleaned)
        alias_terms = cls._build_alias_terms(core_terms, cleaned)
        query_type = cls._infer_query_type(raw_text, cleaned)
        return NormalizedWebQuery(
            raw_text=raw_text,
            search_query=search_query or raw_text,
            normalized_text=cleaned,
            core_terms=core_terms,
            alias_terms=alias_terms,
            query_type=query_type,
        )

    @classmethod
    def _extract_search_query(cls, text: str) -> str:
        query = text.strip().rstrip("\u3002\uff01\uff1f!?")
        query = re.sub(r"^(?:\u8054\u7f51|\u4e0a\u7f51|\u53bb\u7f51\u4e0a|\u5728\u7f51\u4e0a)\s*", "", query)
        lowered = query.lower()

        prefix_patterns = (
            r"^(?:\u8bf7)?(?:\u5e2e\u6211|\u5e2e\u5fd9|\u9ebb\u70e6\u4f60)?(?:\u8054\u7f51|\u4e0a\u7f51|\u53bb\u7f51\u4e0a|\u5728\u7f51\u4e0a)?(?:\u641c\u4e00\u4e0b|\u641c\u4e00\u641c|\u641c\u641c\u770b|\u67e5\u4e00\u4e0b|\u67e5\u4e00\u67e5|\u67e5\u67e5|\u641c\u7d22|\u68c0\u7d22|\u627e\u4e00\u4e0b|\u627e\u4e00\u627e|\u627e\u627e)\s*",
            r"^(?:\u8bf7)?(?:\u5e2e\u6211|\u5e2e\u5fd9|\u9ebb\u70e6\u4f60)?(?:\u641c\u4e00\u4e0b|\u641c\u4e00\u641c|\u641c\u641c\u770b|\u67e5\u4e00\u4e0b|\u67e5\u4e00\u67e5|\u67e5\u67e5|\u641c\u7d22|\u68c0\u7d22|\u627e\u4e00\u4e0b|\u627e\u4e00\u627e|\u627e\u627e)\s*",
            r"^(?:\u76f4\u63a5)?(?:\u544a\u8bc9\u6211|\u7ed9\u6211\u8bb2\u8bb2)\s*",
            r"^(?:please\s+)?(?:search|look up|find out)\s+",
        )
        for pattern in prefix_patterns:
            query = re.sub(pattern, "", query, flags=re.IGNORECASE)

        query = re.sub(
            r"\s*[,\uff0c]?\s*(?:\u7ed9\u6211|\u5e2e\u6211)?(?:\u63d0\u70bc|\u6574\u7406|\u5217\u51fa|\u603b\u7ed3)?(?:\u4e00|\u4e24|\u4e8c|\u4e09|\u56db|\u4e94|\d+)?\u6761(?:\u8981\u70b9|\u91cd\u70b9|\u7ed3\u8bba|\u6458\u8981)\s*$",
            "",
            query,
        ).strip()
        query = re.sub(
            r"\s*[,\uff0c]?\s*(?:\u7ed9\u6211|\u5e2e\u6211)?(?:\u4e00|\u4e24|\u4e8c|\u4e09|\u56db|\u4e94|\d+)?(?:\u4e2a|\u6761)?(?:\u9009\u578b\u5efa\u8bae|\u5efa\u8bae)\s*$",
            "",
            query,
        ).strip()
        query = re.sub(r"\s*[,\uff0c]?\s*(?:\u987a\u4fbf|\u7b80\u8981|\u7b80\u5355|\u5927\u6982)?(?:\u8bf4\u4e0b|\u8bf4\u8bf4|\u8bb2\u8bb2)(?:\u539f\u56e0)?\s*$", "", query).strip()
        query = re.sub(r"(.+?)\u6709\u4ec0\u4e48\u7279\u70b9$", lambda match: f"{match.group(1)} \u7279\u70b9", query).strip()
        query = re.sub(r"\u600e\u4e48\u6837$", "", query).strip()
        query = re.sub(r"\u6709\u4ec0\u4e48(?=\u65b0\u52a8\u5411|\u65b0\u8d8b\u52bf|\u65b0\u8fdb\u5c55)", "", query).strip()
        query = cls._OUTPUT_FILE_PATTERN.sub("", query).strip()
        query = re.sub(
            r"\s*(?:\u5e76|\u7136\u540e)?(?:\u603b\u7ed3\u4e00\u4e0b|\u603b\u7ed3|\u6574\u7406\u4e00\u4e0b|\u6574\u7406|\u4ecb\u7ecd\u4e00\u4e0b|\u4ecb\u7ecd)\s*$",
            "",
            query,
        ).strip()
        query = re.sub(r"^(?:\u90a3\u4e2a|\u8fd9\u4e2a)\s*", "", query).strip()

        if re.search(r"(\u7684?\u662f\u4ec0\u4e48\u610f\u601d|\u7684?\u662f\u4ec0\u4e48|\u4ec0\u4e48\u610f\u601d|\u7684?\u542b\u4e49\u662f\u4ec0\u4e48)$", query):
            query = re.sub(r"(\u7684?\u662f\u4ec0\u4e48\u610f\u601d|\u7684?\u662f\u4ec0\u4e48|\u4ec0\u4e48\u610f\u601d|\u7684?\u542b\u4e49\u662f\u4ec0\u4e48)$", "", query).strip()
            if query:
                return f"{query} \u662f\u4ec0\u4e48"

        if re.search(r"(\u7684?\u542b\u4e49|\u5b9a\u4e49)$", query):
            query = re.sub(r"(\u7684?\u542b\u4e49|\u5b9a\u4e49)$", "", query).strip()
            if query:
                return f"{query} \u662f\u4ec0\u4e48"

        if re.search(r"(meaning|definition|stands for)$", lowered):
            query = re.sub(r"(meaning|definition|stands for)$", "", query, flags=re.IGNORECASE).strip()
            if query:
                return f"{query} meaning"

        generic_site_query = cls._extract_generic_site_query(query)
        if generic_site_query:
            return generic_site_query

        query = re.sub(r"^(?:\u5173\u4e8e)\s*", "", query)
        return query.strip(" \t,\uff0c\u3002\uff01\uff1f!?")

    @classmethod
    def _extract_generic_site_query(cls, query: str) -> str:
        compact = query.replace(" ", "")
        for term in cls._GENERIC_SITE_TERMS:
            if term in compact:
                return term
        return ""

    @classmethod
    def _clean_text(cls, text: str) -> str:
        cleaned = text.lower()
        for phrase in cls._STOP_PHRASES:
            cleaned = cleaned.replace(phrase.lower(), " ")
        cleaned = re.sub(r"[\"'“”‘’《》<>【】\[\]{}()，。！？?;:、/\\_\-]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def _extract_core_terms(cls, cleaned: str) -> list[str]:
        terms: list[str] = []
        for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", cleaned):
            if len(token) == 1 and re.fullmatch(r"[a-z]", token):
                continue
            if token in cls._KNOWN_TERMS:
                terms.append(token)
                continue
            if re.fullmatch(r"[a-z0-9]+", token):
                terms.append(token)
                continue
            terms.extend(piece for piece in cls._split_chinese_chunk(token) if piece not in cls._LIGHT_STOP_TERMS)
        return cls._dedupe(terms)

    @classmethod
    def _split_chinese_chunk(cls, chunk: str) -> list[str]:
        found = [term for term in cls._KNOWN_TERMS if term in chunk]
        if found:
            parts = list(found)
            remainder = chunk
            for term in sorted(found, key=len, reverse=True):
                remainder = remainder.replace(term, " ")
            for piece in re.findall(r"[\u4e00-\u9fff]{2,}", remainder):
                cleaned = piece.strip("\u7684\u4e86\u548c\u4e0e")
                if cleaned and cleaned not in cls._LIGHT_STOP_TERMS:
                    parts.append(cleaned)
            return cls._dedupe(parts)
        if len(chunk) <= 6:
            return [chunk]
        return [chunk]

    @classmethod
    def _build_alias_terms(cls, core_terms: list[str], cleaned: str) -> list[str]:
        aliases: list[str] = []
        for term in core_terms:
            aliases.extend(cls._ALIAS_MAP.get(term.lower(), ()))
        joined = "".join(core_terms)
        for key, values in cls._ALIAS_MAP.items():
            if key in cleaned or key in joined:
                aliases.extend(values)
        aliases.extend(core_terms)
        return cls._dedupe(aliases)

    @classmethod
    def _infer_query_type(cls, raw_text: str, cleaned: str) -> str:
        lowered = raw_text.lower()
        has_live = any(term.lower() in raw_text.lower() or term in cleaned for term in cls._LIVE_HINT_TERMS)
        has_definition = any(term.lower() in lowered or term in raw_text for term in cls._DEFINITION_HINT_TERMS)
        has_navigation = any(term.lower() in lowered or term in raw_text for term in cls._NAVIGATION_HINT_TERMS)
        if has_navigation and (has_live or has_definition):
            return "mixed"
        if has_live:
            return "live_topic"
        if has_definition:
            return "factual"
        if has_navigation:
            return "navigational"
        return "mixed"

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            normalized = item.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(item.strip())
        return ordered
