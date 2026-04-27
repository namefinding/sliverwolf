from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class NormalizedFileQuery:
    raw_text: str
    normalized_text: str
    core_terms: list[str]
    alias_terms: list[str]
    file_type_hints: list[str]
    query_type: str  # name_like | semantic_like | mixed


class FileQueryNormalizer:
    _TARGET_ACTION_PATTERN = re.compile(
        r"(?:加一条|新增|追加|插入|修改|改|替换|删除|删|写入|发给我|发我|传给我|传我|给我|打开|读一下|读一读|读|看一下|看看|看|总结一下|概括一下|总结|概括|提炼|提取|找一下|找找|找|搜索)"
    )
    _TARGET_OBJECT_HINTS = (
        "日志",
        "汇报",
        "模板",
        "文档",
        "文件",
        "资料",
        "记录",
        "报告",
        "表格",
        "截图",
        "图片",
        "照片",
        "方案",
        "计划",
        "结构",
        "栏目",
    )
    _NOISE_TERMS = {
        "日期就写",
        "期就写今",
        "就写今天",
        "写今天的",
        "日期就",
        "期就写",
        "就写今",
        "写今天",
        "今天的",
        "期就",
        "就写",
        "写今",
        "天的",
    }
    _INSTRUCTIONAL_HINTS = (
        "加一条",
        "新增",
        "追加",
        "插入",
        "修改",
        "替换",
        "删除",
        "写入",
        "今天",
        "日期",
        "时候",
        "聊天",
        "等待",
        "无聊感",
        "活人感",
    )
    _LIGHT_STOP_TERMS = {
        "的",
        "了",
        "吗",
        "呢",
        "吧",
        "呀",
        "啊",
        "一下",
        "一个",
        "找",
        "发",
    }

    _STOP_PHRASES = (
        "帮我",
        "请帮我",
        "总结一下",
        "概括一下",
        "讲了什么",
        "内容",
        "那个",
        "这个",
        "我的",
        "文件",
        "文档",
        "资料",
        "内容是",
        "内容里",
        "一下",
        "看一下",
        "找一下",
        "找找看",
        "帮我看看",
        "给我看看",
        "一下子",
        "the",
        "please",
        "tell me",
        "show me",
        "help me",
        "summarize",
        "summary",
        "content",
    )

    _SEMANTIC_HINT_TERMS = (
        "负责",
        "处理",
        "用来",
        "作用",
        "控制",
        "启动",
        "状态",
        "逻辑",
        "配置",
        "responsible",
        "used for",
        "what does",
        "related to",
    )

    _FILE_TYPE_ALIASES = {
        ".pptx": ("ppt", "pptx", "演示文稿", "幻灯片", "幻灯", "presentation"),
        ".docx": (".docx", "docx", "doc", "word", "word文档"),
        ".xlsx": ("xls", "xlsx", "excel", "表格", "工作簿", "spreadsheet"),
        ".md": ("md", "markdown"),
        ".txt": ("txt", "文本"),
        ".log": (".log", "log file", "日志文件", "日志格式"),
        ".pdf": (".pdf", "pdf"),
    }

    _KNOWN_TERMS = (
        "机器人自测",
        "自测模板",
        "报告模板",
        "模板",
        "栏目",
        "结构",
        "年份",
        "年终汇报",
        "年终总结",
        "年会总结",
        "组会汇报",
        "开发日志",
        "开发记录",
        "训练日志",
        "项目架构",
        "架构",
        "项目",
        "研究生",
        "组会",
        "年终",
        "汇报",
        "总结",
        "开发",
        "日志",
        "计划",
        "文档",
        "课件",
        "图片",
        "照片",
        "相机",
    )

    _ALIAS_MAP = {
        "年会总结": ("年终汇报", "年终总结"),
        "年终组会汇报": ("组会汇报", "年终汇报"),
        "组会年终汇报": ("组会汇报", "年终汇报"),
        "开发日志": ("开发记录", "训练日志"),
    }

    @classmethod
    def normalize(cls, text: str) -> NormalizedFileQuery:
        raw_text = text.strip()
        focus_text = cls._extract_target_span(raw_text) or raw_text
        cleaned = cls._clean_text(focus_text)
        file_type_hints = cls._infer_file_type_hints(raw_text, cleaned)
        core_terms = cls._filter_low_signal_terms(
            cls._extract_core_terms(cleaned, file_type_hints=file_type_hints)
        )
        alias_terms = cls._filter_low_signal_terms(
            cls._build_alias_terms(cleaned, core_terms, file_type_hints)
        )
        normalized_text = " ".join(core_terms).strip()
        query_type = cls._infer_query_type(raw_text, normalized_text, core_terms, file_type_hints)
        return NormalizedFileQuery(
            raw_text=raw_text,
            normalized_text=normalized_text,
            core_terms=core_terms,
            alias_terms=alias_terms,
            file_type_hints=file_type_hints,
            query_type=query_type,
        )

    @classmethod
    def _extract_target_span(cls, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return ""

        patterns = (
            re.compile(
                r"(?:在|把)(?P<target>[^，。！？\n]{2,40}?)(?:里面|里边|里|中)?"
                + cls._TARGET_ACTION_PATTERN.pattern
            ),
            re.compile(
                r"(?P<target>[^，。！？\n]{2,40}?(?:日志|汇报|模板|文档|文件|资料|记录|报告|表格|截图|图片|照片|方案|计划))"
                r"(?:的内容|这个文档|这个文件|这份文档|这份文件|这份|这个)?"
            ),
        )
        for pattern in patterns:
            match = pattern.search(stripped)
            if match is None:
                continue
            candidate = str(match.group("target") or "").strip(" ，。！？")
            candidate = re.sub(r"^(桌面上?的|桌面上|桌面的|本地的|那个|这个|这份|那份)", "", candidate)
            candidate = re.sub(r"(这个文档|这个文件|这份文档|这份文件|这份|这个)$", "", candidate)
            candidate = candidate.strip(" ，。！？")
            if len(candidate) >= 2:
                return candidate
        return ""

    @classmethod
    def _clean_text(cls, text: str) -> str:
        cleaned = text.strip().lower()
        for phrase in cls._STOP_PHRASES:
            cleaned = cleaned.replace(phrase.lower(), " ")
        cleaned = (
            cleaned.replace("pptx", " ppt ")
            .replace("docx", " doc ")
            .replace("xlsx", " excel ")
            .replace("markdown", " md ")
        )
        cleaned = re.sub(r"[\"'“”‘’《》<>【】\[\]{}()，。！？!?：:;、,/\\_\-]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def _infer_file_type_hints(cls, raw_text: str, cleaned: str) -> list[str]:
        lowered = f"{raw_text} {cleaned}".lower()
        hints: list[str] = []
        for ext, aliases in cls._FILE_TYPE_ALIASES.items():
            if any(cls._contains_file_type_alias(lowered, alias) for alias in aliases):
                hints.append(ext)
        return cls._dedupe(hints)

    @staticmethod
    def _contains_file_type_alias(lowered_text: str, alias: str) -> bool:
        normalized = alias.strip().lower()
        if not normalized:
            return False
        if normalized.startswith("."):
            return normalized in lowered_text
        if re.fullmatch(r"[a-z0-9]+", normalized):
            return re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", lowered_text) is not None
        return normalized in lowered_text

    @classmethod
    def strip_file_type_terms(cls, text: str, file_type_hints: list[str]) -> str:
        cleaned = text
        for ext in file_type_hints:
            for alias in cls._FILE_TYPE_ALIASES.get(ext, ()):
                normalized = alias.strip()
                if not normalized:
                    continue
                if normalized.startswith("."):
                    cleaned = re.sub(re.escape(normalized), " ", cleaned, flags=re.IGNORECASE)
                elif re.fullmatch(r"[A-Za-z0-9]+", normalized):
                    cleaned = re.sub(
                        rf"(?<![A-Za-z0-9]){re.escape(normalized)}(?![A-Za-z0-9])",
                        " ",
                        cleaned,
                        flags=re.IGNORECASE,
                    )
                else:
                    cleaned = re.sub(re.escape(normalized), " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def _extract_core_terms(cls, cleaned: str, *, file_type_hints: list[str] | None = None) -> list[str]:
        type_aliases = cls._file_type_alias_terms(file_type_hints or [])
        terms: list[str] = []
        for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", cleaned):
            if token.lower() in type_aliases:
                continue
            if re.fullmatch(r"[a-z0-9]+", token):
                if token not in {"ppt", "doc", "excel", "md", "txt", "pdf", "log"} and len(token) < 2:
                    continue
                terms.append(token)
                continue
            terms.extend(piece for piece in cls._split_chinese_chunk(token) if piece not in cls._LIGHT_STOP_TERMS)
        return cls._dedupe(terms)

    @classmethod
    def _split_chinese_chunk(cls, chunk: str) -> list[str]:
        found: list[str] = []
        for term in cls._KNOWN_TERMS:
            if term in chunk:
                found.append(term)
        if not found:
            if len(chunk) <= 4:
                found.append(chunk)
            else:
                for size in (4, 3, 2):
                    for index in range(0, len(chunk) - size + 1):
                        piece = chunk[index:index + size]
                        if piece not in found:
                            found.append(piece)
        return cls._dedupe(found)

    @classmethod
    def _build_alias_terms(
        cls,
        cleaned: str,
        core_terms: list[str],
        file_type_hints: list[str],
    ) -> list[str]:
        aliases: list[str] = []
        joined = "".join(core_terms)
        for key, values in cls._ALIAS_MAP.items():
            if key in cleaned or key in joined:
                aliases.extend(values)
        core_set = set(core_terms)
        if {"组会", "年终", "汇报"}.issubset(core_set):
            aliases.extend(["组会汇报", "年终汇报"])
        if {"年会", "总结"}.issubset(core_set):
            aliases.extend(["年终总结", "年终汇报"])
        aliases.extend(core_terms)
        return [alias for alias in cls._dedupe(aliases) if alias]

    @classmethod
    def _filter_low_signal_terms(cls, terms: list[str]) -> list[str]:
        anchors = [term for term in terms if cls._looks_like_target_anchor(term)]
        filtered: list[str] = []
        for term in terms:
            normalized = term.strip()
            if not normalized:
                continue
            if normalized in cls._NOISE_TERMS:
                continue
            if re.fullmatch(r"[\u4e00-\u9fff]", normalized):
                continue
            if anchors and cls._looks_like_instruction_fragment(normalized):
                continue
            filtered.append(normalized)
        return cls._dedupe(filtered)

    @classmethod
    def _looks_like_target_anchor(cls, term: str) -> bool:
        return any(hint in term for hint in cls._TARGET_OBJECT_HINTS)

    @classmethod
    def _looks_like_instruction_fragment(cls, term: str) -> bool:
        return any(hint in term for hint in cls._INSTRUCTIONAL_HINTS)

    @classmethod
    def _file_type_alias_terms(cls, file_type_hints: list[str]) -> set[str]:
        aliases: set[str] = set()
        for ext in file_type_hints:
            aliases.add(ext.lower().lstrip("."))
            for alias in cls._FILE_TYPE_ALIASES.get(ext, ()):
                normalized = alias.lower().lstrip(".")
                aliases.add(normalized)
        return aliases

    @classmethod
    def _infer_query_type(
        cls,
        raw_text: str,
        normalized_text: str,
        core_terms: list[str],
        file_type_hints: list[str],
    ) -> str:
        lowered = raw_text.lower()
        has_semantic = any(term in lowered for term in cls._SEMANTIC_HINT_TERMS)
        title_like = bool(file_type_hints) or len(core_terms) <= 6 or any(
            term in lowered for term in ("汇报", "日志", "架构", "计划", "模板", "栏目", "结构", "年份", "年", "表格", "excel")
        )
        if has_semantic and title_like:
            return "mixed"
        if has_semantic:
            return "semantic_like"
        if normalized_text:
            return "name_like" if title_like else "mixed"
        return "semantic_like"

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
