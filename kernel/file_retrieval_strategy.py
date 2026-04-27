from __future__ import annotations

import re
from dataclasses import dataclass

from local_agent.protocol.models import (
    CandidateState,
    DecisionType,
    OutputKind,
    RiskLevel,
    TaskGoal,
    ToolCallResult,
    ToolDecision,
)
from local_agent.utils.file_query_normalizer import FileQueryNormalizer


@dataclass
class FileQueryPlan:
    query: str
    query_terms: list[str]
    alias_terms: list[str]
    explicit_name: bool
    query_type: str  # name_like | semantic_like | mixed
    target_kind: str
    extensions: list[str]
    action: str | None
    reason: str


class FileRetrievalStrategy:
    _SEMANTIC_HINT_TERMS = (
        "responsible for",
        "used for",
        "related to",
        "what does",
        "for ",
        "负责",
        "处理",
        "用来",
        "作用",
        "配置",
        "启动",
        "状态",
        "逻辑",
    )

    _FOLDER_TERMS = ("folder", "directory", "文件夹", "目录")
    _READ_TERMS = (
        "summarize",
        "summary",
        "read",
        "content",
        "explain",
        "讲了什么",
        "总结",
        "概括",
        "内容",
        "读取",
        "栏目",
        "结构",
        "重点",
        "核心观点",
        "有哪些",
        "哪些部分",
    )
    _METADATA_TERMS = ("metadata", "details", "properties", "size", "modified", "信息", "属性", "大小", "修改时间")
    _PREVIEW_TERMS = ("preview", "peek", "head", "first lines", "前几行", "预览", "先看看", "开头")
    _OPEN_TERMS = ("open", "打开")
    _REVEAL_TERMS = ("reveal", "show in explorer", "locate in explorer", "在资源管理器", "在文件夹中显示", "定位到")
    _DELIVER_TERMS = (
        "发给我",
        "发我",
        "发过来",
        "传给我",
        "传我",
        "传过来",
        "傳給我",
        "傳我",
        "傳過來",
        "send me",
        "send it",
        "attach it",
        "upload it",
        "发给我",
        "发我",
        "发过来",
        "传给我",
        "传我",
        "传过来",
        "發給我",
        "發我",
        "發過來",
        "傳給我",
        "傳我",
        "傳過來",
    )
    _WRITE_TERMS = ("write", "save", "append", "export", "写入", "保存", "记到", "导出")
    _LIST_TERMS = ("list", "列出", "枚举", "有哪些", "一级子文件夹")
    _TITLEISH_TERMS = (
        "template",
        "模板",
        "report",
        "汇报",
        "报告",
        "log",
        "日志",
        "year",
        "年份",
        "栏目",
        "结构",
        "table",
        "excel",
        "spreadsheet",
        "表格",
        "自测",
        "screenshot",
        "截图",
        "聊天",
        "qq",
        "image",
        "picture",
        "photo",
        "单词",
        "英文",
        "手绘",
    )

    @classmethod
    def _candidate_state_is_salvageable(cls, candidate_state: CandidateState | None) -> bool:
        if candidate_state is None or not candidate_state.candidate_paths:
            return False
        confidence = float(candidate_state.confidence or 0.0)
        top_score = float(candidate_state.top_score or 0.0)
        score_gap = float(candidate_state.score_gap or 0.0)
        if top_score >= 0.42 and score_gap >= 0.1 and confidence >= 0.28:
            return True
        if len(candidate_state.candidate_paths) == 1 and top_score >= 0.35 and confidence >= 0.28:
            return True
        return False

    @classmethod
    def build_initial_lookup(
        cls,
        *,
        user_text: str,
        completed_outputs: list[OutputKind],
        candidate_state: CandidateState | None,
        overall_task_goal: TaskGoal | None,
    ) -> ToolDecision | None:
        if candidate_state is not None and candidate_state.candidate_paths:
            return None
        if OutputKind.OBJECT_CANDIDATES in completed_outputs:
            return None

        plan = cls._plan_query(user_text)
        if not plan.query or plan.action is None:
            return None

        # Keep this initial strategy conservative: only obvious title/name-like
        # requests should bypass the planner and go straight to name lookup.
        if plan.query_type not in {"name_like", "mixed"}:
            return None
        if not plan.explicit_name and not cls._looks_like_titleish_reference(user_text, plan.query):
            return None
        if cls._looks_like_image_request(user_text) and not plan.explicit_name:
            explicit_like_terms = cls._tokenize_query(plan.query)
            if not any(re.fullmatch(r"[a-z0-9_-]*\d{4,}[a-z0-9_-]*", term, flags=re.IGNORECASE) for term in explicit_like_terms):
                return None
        if cls._looks_like_exact_filename(plan.query):
            return None
        if plan.action not in {"read", "metadata", "preview", "open", "reveal", "deliver"}:
            return None

        goal = overall_task_goal or TaskGoal(
            summary=f"Find the target matching {plan.query} and continue with the requested action.",
            required_outputs=cls._required_outputs_for_action(plan.action),
        )

        return ToolDecision(
            decision=DecisionType.TOOL_CALL,
            intent="search_named_target",
            reason=plan.reason,
            selected_tool="file.search_by_name",
            arguments={
                "path": ".",
                "query": plan.query,
                "query_terms": plan.query_terms,
                "alias_terms": plan.alias_terms,
                "recursive": True,
                "scope_mode": cls._preferred_scope_mode(user_text),
                "target_kind": plan.target_kind,
                "extensions": plan.extensions,
                "include_dirs": True,
                "top_k": 8,
            },
            risk_level=RiskLevel.LOW,
            overall_task_goal=goal,
            expected_step_outputs=[OutputKind.OBJECT_CANDIDATES],
        )

    @classmethod
    def build_empty_result_fallback(
        cls,
        *,
        user_text: str,
        last_decision: ToolDecision,
        last_result: ToolCallResult,
        candidate_state: CandidateState | None,
        reliable_candidates: bool,
    ) -> ToolDecision | None:
        if last_decision.decision != DecisionType.TOOL_CALL or last_result.status != "success":
            return None

        tool_name = last_decision.selected_tool or ""
        plan = cls._plan_query(user_text)
        query = str(last_decision.arguments.get("query", plan.query)).strip()
        target_kind = str(last_decision.arguments.get("target_kind", plan.target_kind or "any"))
        path_scope = str(last_decision.arguments.get("path_scope", last_decision.arguments.get("path", ".")))
        scope_mode = str(last_decision.arguments.get("scope_mode", cls._preferred_scope_mode(user_text)))
        extensions = list(last_decision.arguments.get("extensions", plan.extensions))
        query_terms = list(last_decision.arguments.get("query_terms", plan.query_terms))
        alias_terms = list(last_decision.arguments.get("alias_terms", plan.alias_terms))

        if tool_name in {"file.search_by_name", "retrieval.search_local_objects"}:
            no_candidates = not last_result.data.get("candidates")
            low_confidence = candidate_state is not None and candidate_state.source_tool == tool_name and not reliable_candidates
            if not no_candidates and not low_confidence:
                return None

        if tool_name == "file.search_by_name":
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="fallback_semantic_lookup",
                reason="Name lookup was empty or weak, so broaden the search semantically inside the current scope.",
                selected_tool="retrieval.search_local_objects",
                arguments={
                    "query": query or plan.query,
                    "target_kind": target_kind,
                    "path_scope": path_scope,
                    "scope_mode": scope_mode,
                    "extensions": extensions,
                    "query_terms": query_terms,
                    "alias_terms": alias_terms,
                    "top_k": 8,
                },
                risk_level=last_decision.risk_level,
                overall_task_goal=last_decision.overall_task_goal,
                expected_step_outputs=[OutputKind.OBJECT_CANDIDATES],
            )

        if tool_name == "retrieval.search_local_objects":
            if candidate_state is not None and candidate_state.source_tool == "retrieval.search_local_objects":
                if cls._candidate_state_is_salvageable(candidate_state):
                    return None
            if plan.query_type in {"name_like", "mixed"} and (query or plan.query):
                return ToolDecision(
                    decision=DecisionType.TOOL_CALL,
                    intent="fallback_name_lookup",
                    reason="Semantic lookup was empty or weak, so try stronger title and file-name matching in the current scope.",
                    selected_tool="file.search_by_name",
                    arguments={
                        "path": path_scope,
                        "query": query or plan.query,
                        "query_terms": query_terms,
                        "alias_terms": alias_terms,
                        "recursive": True,
                        "scope_mode": scope_mode,
                        "target_kind": target_kind,
                        "extensions": extensions,
                        "include_dirs": True,
                        "top_k": 8,
                    },
                    risk_level=last_decision.risk_level,
                    overall_task_goal=last_decision.overall_task_goal,
                    expected_step_outputs=[OutputKind.OBJECT_CANDIDATES],
                )

            if target_kind == "folder":
                return ToolDecision(
                    decision=DecisionType.TOOL_CALL,
                    intent="fallback_list_directories",
                    reason="Folder lookup stayed weak, so enumerate directories in the current scope.",
                    selected_tool="file.list",
                    arguments={"path": path_scope, "recursive": True, "include_dirs": True},
                    risk_level=last_decision.risk_level,
                    overall_task_goal=last_decision.overall_task_goal,
                    expected_step_outputs=[OutputKind.DIRECTORY_ENTRIES],
                )

            query_terms = cls._tokenize_query(query or plan.query)
            query_terms = query_terms or plan.query_terms
            patterns = [f"*{ext}" if not ext.startswith("*") else ext for ext in extensions] if extensions else []
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="fallback_search_text",
                reason="Object lookup stayed weak, so search file contents inside the current scope.",
                selected_tool="file.search_text",
                arguments={
                    "path": path_scope,
                    "terms": query_terms or ([query or plan.query] if (query or plan.query) else []),
                    "recursive": True,
                    "patterns": patterns or ["*.py", "*.md", "*.txt", "*.json", "*.yaml", "*.yml", "*.docx", "*.csv", "*.pptx", "*.xlsx"],
                    "max_matches": 20,
                },
                risk_level=last_decision.risk_level,
                overall_task_goal=last_decision.overall_task_goal,
                expected_step_outputs=[OutputKind.SEARCH_MATCHES],
            )

        if tool_name == "file.search_text":
            matches = last_result.data.get("matches")
            low_confidence = candidate_state is not None and candidate_state.source_tool == "file.search_text" and not reliable_candidates
            if matches and not low_confidence:
                return None
            raw_patterns = list(last_decision.arguments.get("patterns", []))
            return ToolDecision(
                decision=DecisionType.TOOL_CALL,
                intent="fallback_list_files",
                reason="Text search still did not give reliable candidates, so enumerate matching files in the current scope.",
                selected_tool="file.list",
                arguments={
                    "path": path_scope,
                    "recursive": True,
                    "include_dirs": False,
                    "patterns": raw_patterns,
                },
                risk_level=last_decision.risk_level,
                overall_task_goal=last_decision.overall_task_goal,
                expected_step_outputs=[OutputKind.DIRECTORY_ENTRIES],
            )

        if tool_name == "file.list":
            entries = last_result.data.get("entries")
            if (
                entries
                and last_decision.overall_task_goal is not None
                and OutputKind.DIRECTORY_ENTRIES in last_decision.overall_task_goal.required_outputs
            ):
                return None
            low_confidence = candidate_state is not None and candidate_state.source_tool == "file.list" and not reliable_candidates
            if entries and not low_confidence:
                return None
            return ToolDecision(
                decision=DecisionType.CLARIFY,
                intent="clarify_after_empty_listing",
                reason="The current scope still does not contain a reliable match, so the user needs to provide more detail.",
                response_hint="我还没在当前范围里找到足够像的对象。你可以再告诉我更接近的名字、文件类型，或者大概放在哪个子目录里。",
                overall_task_goal=last_decision.overall_task_goal,
            )

        return None

    @staticmethod
    def _preferred_scope_mode(user_text: str) -> str:
        lowered = user_text.lower()
        if ("桌面" in user_text or "desktop" in lowered) and "testing" not in lowered and "测试" not in user_text:
            return "shallow_first"
        return "subtree"

    @classmethod
    def _plan_query(cls, user_text: str) -> FileQueryPlan:
        action = cls._infer_action(user_text)
        target_kind = cls._infer_target_kind(user_text)
        normalized = FileQueryNormalizer.normalize(user_text)
        explicit_query = cls._extract_explicit_name_query(user_text)
        contextual_query = normalized.normalized_text or cls._extract_contextual_query(user_text)
        if action == "deliver":
            raw_contextual_query = cls._extract_contextual_query(user_text)
            query = explicit_query or raw_contextual_query or contextual_query
            query_terms = cls._tokenize_query(query)
        else:
            query = explicit_query or contextual_query
            query_terms = normalized.core_terms or cls._tokenize_query(query)
        alias_terms = normalized.alias_terms
        extensions = cls._infer_extensions(user_text, query, target_kind, normalized.file_type_hints)
        query = FileQueryNormalizer.strip_file_type_terms(query, extensions) or query
        tokenized_query = cls._tokenize_query(query)
        if normalized.core_terms:
            query_terms = cls._dedupe_preserve_order(list(query_terms))
            if any("." in term or re.search(r"\d{4,}", term) for term in tokenized_query):
                query_terms = cls._dedupe_preserve_order([*tokenized_query, *query_terms])
        else:
            query_terms = cls._dedupe_preserve_order([*tokenized_query, *query_terms])

        if explicit_query:
            query_type = "name_like"
            reason = "The request references a specific title or file-name fragment, so name lookup should run first."
        elif normalized.query_type == "name_like":
            query_type = "name_like"
            reason = "The request looks title-like after normalization, so name lookup should run first."
        elif normalized.query_type == "mixed":
            query_type = "mixed"
            reason = "The normalized query contains both title-like and contextual signals, so name and semantic lookup should work together."
        elif query and cls._looks_like_contextual_description(query):
            query_type = "semantic_like"
            reason = "The request looks more like a purpose-based description, so semantic lookup should remain primary."
        elif query:
            query_type = "mixed"
            reason = "The request contains both title-like and contextual signals."
        else:
            query_type = "semantic_like"
            reason = "The request needs semantic retrieval."

        return FileQueryPlan(
            query=query,
            query_terms=query_terms,
            alias_terms=alias_terms,
            explicit_name=bool(explicit_query),
            query_type=query_type,
            target_kind=target_kind,
            extensions=extensions,
            action=action,
            reason=reason,
        )

    @classmethod
    def _infer_action(cls, user_text: str) -> str | None:
        lowered = user_text.lower()
        if any(term in lowered for term in cls._REVEAL_TERMS):
            return "reveal"
        if any(term in lowered for term in cls._METADATA_TERMS):
            return "metadata"
        if any(term in lowered for term in cls._PREVIEW_TERMS):
            return "preview"
        if any(term in lowered for term in cls._DELIVER_TERMS):
            return "deliver"
        if any(term in lowered for term in ("ocr", "text", "文字", "文本", "读字", "图上", "图里")) and any(
            term in lowered
            for term in ("image", "picture", "photo", "screenshot", "png", "jpg", "jpeg", "webp", "gif", "bmp", "图片", "照片", "截图", "图像")
        ):
            return "read"
        if any(term in lowered for term in cls._OPEN_TERMS):
            return "open"
        if any(term in lowered for term in cls._WRITE_TERMS):
            return "write"
        if any(term in lowered for term in cls._LIST_TERMS):
            return "list"
        if any(term in lowered for term in cls._READ_TERMS):
            return "read"
        if any(term in lowered for term in ("find", "search", "locate", "找", "查", "定位")):
            return "lookup"
        return None

    @classmethod
    def _infer_target_kind(cls, user_text: str) -> str:
        lowered = user_text.lower()
        if any(term in lowered for term in cls._FOLDER_TERMS):
            return "folder"
        return "file"

    @staticmethod
    def _looks_like_image_request(user_text: str) -> bool:
        lowered = user_text.lower()
        return any(
            term in lowered
            for term in ("image", "picture", "photo", "screenshot", "png", "jpg", "jpeg", "webp", "gif", "bmp", "图片", "照片", "截图", "图像")
        )

    @classmethod
    def _infer_extensions(
        cls,
        user_text: str,
        query: str,
        target_kind: str,
        file_type_hints: list[str] | None = None,
    ) -> list[str]:
        lowered = f"{user_text} {query}".lower()
        if target_kind == "folder":
            return []
        if file_type_hints:
            return file_type_hints
        if any(
            term in lowered
            for term in ("png", "jpg", "jpeg", "webp", "gif", "bmp", "image", "picture", "photo", "screenshot", "图片", "照片", "截图", "图像")
        ):
            return [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]
        if ".docx" in lowered or "word" in lowered:
            return [".docx"]
        if ".pptx" in lowered or "ppt" in lowered or "presentation" in lowered or "演示" in lowered:
            return [".pptx"]
        if ".xlsx" in lowered or ".csv" in lowered or "excel" in lowered or "spreadsheet" in lowered or "表格" in lowered:
            return [".xlsx", ".csv"]
        if ".md" in lowered or "markdown" in lowered:
            return [".md"]
        if ".pdf" in lowered or re.search(r"(?<![a-z0-9])pdf(?![a-z0-9])", lowered):
            return [".pdf"]
        if ".txt" in lowered:
            return [".txt"]
        if ".log" in lowered or "日志文件" in lowered or "日志格式" in lowered or "log file" in lowered:
            return [".log"]
        return []

    @classmethod
    def _extract_explicit_name_query(cls, user_text: str) -> str:
        patterns = (
            r"(?:file|folder|directory|document|doc|markdown)\s+named\s+([^.,!?]+)",
            r"(?:file|folder|directory|document|doc|markdown)\s+called\s+([^.,!?]+)",
            r"\bnamed\s+([^.,!?]+)",
            r"\bcalled\s+([^.,!?]+)",
            r"名为\s*([^，。！？?.]+)",
            r"叫\s*([^，。！？?.]+)",
            r"[\"“”《》]([^\"“”《》]+)[\"“”《》]",
            r"(?:^|[\s/\\\\])([A-Za-z0-9_\-\u4e00-\u9fff]+\.(?:docx|pptx|xlsx|md|txt|json|csv|yaml|yml|log|pdf))(?=$|[\s,.;!?，。！？])",
        )
        for pattern in patterns:
            match = re.search(pattern, user_text, flags=re.IGNORECASE)
            if match:
                query = match.group(1).strip(" .,!?:;\"'")
                if query:
                    return query
        return ""

    @classmethod
    def _extract_contextual_query(cls, user_text: str) -> str:
        return cls._strip_file_request_noise(user_text)
        cleaned = user_text.strip()
        noise_phrases = (
            "帮我",
            "请帮我",
            "总结一下",
            "概括一下",
            "看看",
            "告诉我",
            "讲了什么",
            "内容",
            "是什么",
            "在哪",
            "在哪里",
            "找一个",
            "你找一个",
            "读取",
            "打开",
            "桌面上的",
            "就在桌面上",
            "我的",
            "那个",
            "这个",
            "这份",
            "那份",
            "发给我",
            "发我",
            "传给我",
            "传我",
            "send me",
            "send it",
            "attach it",
            "upload it",
            "文件",
            "文档",
            "目录",
            "文件夹",
        )
        for fragment in noise_phrases:
            cleaned = re.sub(re.escape(fragment), " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[\s，。！？?.!?:;（）()\[\]{}]+", " ", cleaned)
        return cleaned.strip()

    @staticmethod
    def _strip_file_request_noise(user_text: str) -> str:
        cleaned = user_text.strip()
        for fragment in (
            "帮我",
            "请帮我",
            "请",
            "你",
            "把",
            "的",
            "你能把",
            "你帮我",
            "有没有",
            "有没",
            "一个",
            "一份",
            "看一下",
            "看看",
            "告诉我",
            "读取",
            "打开",
            "桌面",
            "桌面的",
            "桌面上的",
            "我桌面的",
            "下载",
            "下载里的",
            "下载中的",
            "我的",
            "那个",
            "这个",
            "这份",
            "那份",
            "发给我",
            "发我",
            "发过来",
            "传给我",
            "传我",
            "传过来",
            "傳給我",
            "傳我",
            "傳過來",
            "文件",
            "文档",
            "目录",
            "文件夹",
            "啊",
            "呀",
            "帮我",
            "请帮我",
            "请",
            "你",
            "把",
            "的",
            "能",
            "你能把",
            "你帮我",
            "有没有",
            "有沒有",
            "一个",
            "一個",
            "看一下",
            "看下",
            "看看",
            "告诉我",
            "读取",
            "打开",
            "桌面",
            "桌面的",
            "桌面上的",
            "我桌面的",
            "下载",
            "下载里的",
            "下载中的",
            "我的",
            "那个",
            "这个",
            "这份",
            "那份",
            "发给我",
            "发我",
            "发过来",
            "传给我",
            "传我",
            "传过来",
            "發給我",
            "發我",
            "發過來",
            "傳給我",
            "傳我",
            "傳過來",
            "send me",
            "send it",
            "attach it",
            "upload it",
            "文件",
            "文档",
            "目录",
            "文件夹",
            "吗",
            "嗎",
        ):
            cleaned = re.sub(re.escape(fragment), " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[\s，。！？、,.!?:;()（）\[\]{}\"']+", " ", cleaned)
        return cleaned.strip()

    @classmethod
    def _looks_like_contextual_description(cls, text: str) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in cls._SEMANTIC_HINT_TERMS)

    @staticmethod
    def _looks_like_titleish_reference(user_text: str, query: str) -> bool:
        lowered = f"{user_text} {query}".lower()
        title_markers = (
            "markdown",
            ".md",
            "document",
            "doc",
            ".docx",
            "ppt",
            ".pptx",
            "log",
            ".log",
            "文档",
            "日志",
            "汇报",
            "报告",
            "template",
            "模板",
            "year",
            "年份",
            "栏目",
            "结构",
            "表格",
            "excel",
            "自测",
            "screenshot",
            "截图",
            "聊天",
            "qq",
            "image",
            "picture",
            "photo",
            "单词",
            "英文",
            "手绘",
        )
        return any(marker in lowered for marker in title_markers)

    @staticmethod
    def _required_outputs_for_action(action: str) -> list[OutputKind]:
        if action == "read":
            return [OutputKind.OBJECT_CANDIDATES, OutputKind.FILE_CONTENTS]
        if action in {"metadata", "preview"}:
            return [OutputKind.OBJECT_CANDIDATES, OutputKind.OBJECT_DETAILS]
        if action in {"open", "reveal"}:
            return [OutputKind.OBJECT_CANDIDATES, OutputKind.PATH_OPENED]
        if action == "write":
            return [OutputKind.OBJECT_CANDIDATES, OutputKind.FILE_WRITTEN]
        if action == "deliver":
            return [OutputKind.OBJECT_CANDIDATES]
        return [OutputKind.OBJECT_CANDIDATES]

    @staticmethod
    def _tokenize_query(text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{1,8}", text.lower())
        seen: set[str] = set()
        ordered: list[str] = []
        for token in tokens:
            if token and token not in seen:
                seen.add(token)
                ordered.append(token)
        return ordered

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            normalized = str(value).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    @staticmethod
    def _looks_like_exact_filename(text: str) -> bool:
        return bool(
            re.fullmatch(
                r"[A-Za-z0-9_\-\u4e00-\u9fff]+\.(docx|pptx|xlsx|md|txt|json|csv|yaml|yml|log|pdf|py)",
                text.strip(),
                flags=re.IGNORECASE,
            )
        )
