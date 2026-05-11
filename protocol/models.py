from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ConfigDict, field_validator


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class DecisionType(str, Enum):
    RESPOND = "respond"
    TOOL_CALL = "tool_call"
    CLARIFY = "clarify"
    FINISH = "finish"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolInterruptBehavior(str, Enum):
    BLOCK = "block"
    CANCEL = "cancel"


class ToolPermissionMode(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class OutputKind(str, Enum):
    DIRECTORY_ENTRIES = "directory_entries"
    FILE_CONTENTS = "file_contents"
    SEARCH_MATCHES = "search_matches"
    OBJECT_CANDIDATES = "object_candidates"
    CONTACT_CANDIDATES = "contact_candidates"
    OBJECT_DETAILS = "object_details"
    FILE_WRITTEN = "file_written"
    PATH_OPENED = "path_opened"
    PATH_CREATED = "path_created"
    PATH_UPDATED = "path_updated"
    PATH_DELETED = "path_deleted"
    MEMORY_ITEMS = "memory_items"
    MEMORY_SAVED = "memory_saved"
    SEARCH_RESULTS = "search_results"
    WEB_CONTENT = "web_content"
    MESSAGE_SENT = "message_sent"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_CLARIFICATION = "waiting_for_clarification"
    WAITING_FOR_SELECTION = "waiting_for_selection"
    WAITING_FOR_CONFIRMATION = "waiting_for_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Message(BaseModel):
    role: Role
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolManifest(BaseModel):
    tool_name: str
    module: str
    description: str
    aliases: list[str] = Field(default_factory=list)
    search_hint: str = ""
    side_effect: bool = False
    idempotent: bool = True
    read_only: bool = False
    destructive: bool = False
    concurrency_safe: bool = False
    requires_confirmation: bool = False
    default_permission: ToolPermissionMode = ToolPermissionMode.ALLOW
    interrupt_behavior: ToolInterruptBehavior = ToolInterruptBehavior.BLOCK
    timeout_ms: int = 5_000
    max_result_size_chars: int | None = None
    produces: list[OutputKind] = Field(default_factory=list)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]

    def matches_name(self, name: str) -> bool:
        return self.tool_name == name or name in self.aliases

class ReminderRecord(BaseModel):
    reminder_id: str
    message: str
    when_text: str
    scheduled_for: datetime | None = None
    timezone: str | None = None
    status: str = "scheduled"   # scheduled / fired / cancelled / failed
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskGoal(BaseModel):
    summary: str = ""
    required_outputs: list[OutputKind] = Field(default_factory=list)
    completion_mode: str = "outputs"


class ToolExecutionContext(BaseModel):
    execution_brief: str = ""
    required_outputs: list[OutputKind] = Field(default_factory=list)
    grounded_inputs: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionContextFields(BaseModel):
    execution_brief: str = ""
    required_outputs: list[OutputKind] = Field(default_factory=list)
    grounded_inputs: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)


class ToolPermissionDecision(BaseModel):
    behavior: ToolPermissionMode = ToolPermissionMode.ALLOW
    reason: str = ""
    updated_arguments: dict[str, Any] | None = None


class ToolUseContext(BaseModel):
    trace_id: str = ""
    session_id: str = ""
    channel: str | None = None
    caller: str = "agent-kernel"
    workspace_root: str = "."
    permission_mode: str = "default"
    access_policy: dict[str, Any] = Field(default_factory=dict)
    runtime_settings: dict[str, Any] = Field(default_factory=dict)
    required_outputs: list[OutputKind] = Field(default_factory=list)
    completed_outputs: list[OutputKind] = Field(default_factory=list)
    execution_brief: str = ""
    grounded_inputs: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_execution_context(
        cls,
        *,
        trace_id: str,
        session_id: str,
        workspace_root: str,
        execution_context: ToolExecutionContext,
        channel: str | None = None,
        access_policy: dict[str, Any] | None = None,
        runtime_settings: dict[str, Any] | None = None,
        completed_outputs: list[OutputKind] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolUseContext":
        return cls(
            trace_id=trace_id,
            session_id=session_id,
            channel=channel,
            workspace_root=workspace_root,
            access_policy=dict(access_policy or {}),
            runtime_settings=dict(runtime_settings or {}),
            required_outputs=list(execution_context.required_outputs),
            completed_outputs=list(completed_outputs or []),
            execution_brief=execution_context.execution_brief,
            grounded_inputs=dict(execution_context.grounded_inputs),
            constraints=dict(execution_context.constraints),
            metadata=dict(metadata or {}),
        )

class ToolDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: DecisionType
    intent: str
    reason: str
    selected_tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW
    response_hint: str | None = None
    memory_write: str | None = None
    overall_task_goal: TaskGoal | None = None
    expected_step_outputs: list[OutputKind] = Field(default_factory=list)


class WorkflowNodeSpec(BaseModel):
    node_id: str
    tool: str | None = None
    intent: str = ""
    reason: str = ""
    requires: list[OutputKind] = Field(default_factory=list)
    produces: list[OutputKind] = Field(default_factory=list)

    @field_validator("requires", "produces", mode="before")
    @classmethod
    def _coerce_output_kinds(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        raw_items = value if isinstance(value, list) else [value]
        output_values = {item.value for item in OutputKind}
        normalized: list[str] = []
        for item in raw_items:
            raw = str(item.value if isinstance(item, OutputKind) else item).strip()
            if raw in output_values:
                normalized.append(raw)
        return normalized


class WorkflowSpec(BaseModel):
    workflow_name: str = "generic"
    goal: TaskGoal | None = None
    nodes: list[WorkflowNodeSpec] = Field(default_factory=list)


class DecisionReview(BaseModel):
    approved: bool
    issues: list[str] = Field(default_factory=list)
    summary: str = ""
    suggested_decision: ToolDecision | None = None


class ExecutionReview(BaseModel):
    approved: bool = True
    issues: list[str] = Field(default_factory=list)
    summary: str = ""
    force_partial: bool = False
    missing_outputs: list[OutputKind] = Field(default_factory=list)


class CompletionAssessment(BaseModel):
    done: bool
    reason: str = ""
    should_render_response: bool = True
    completed_outputs: list[OutputKind] = Field(default_factory=list)
    missing_outputs: list[OutputKind] = Field(default_factory=list)


class CandidateState(BaseModel):
    query: str = ""
    target_kind: str = "any"
    path_scope: str = "."
    query_terms: list[str] = Field(default_factory=list)
    candidate_paths: list[str] = Field(default_factory=list)
    candidate_names: list[str] = Field(default_factory=list)
    workflow_stage: str = "candidate_ready"
    source_tool: str = ""
    confidence: float = 0.0
    confidence_reason: str = ""
    top_score: float = 0.0
    second_score: float = 0.0
    score_gap: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowCandidate(BaseModel):
    candidate_id: str
    candidate_kind: str = "file"
    display_name: str
    path_or_ref: str
    subtitle: str = ""
    score: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    workflow_family: str = "generic"
    workflow_stage: str = "searching"
    required_outputs: list[OutputKind] = Field(default_factory=list)
    completed_outputs: list[OutputKind] = Field(default_factory=list)
    missing_outputs: list[OutputKind] = Field(default_factory=list)
    primary_target_kind: str = "unknown"
    primary_target_ref: str | None = None
    candidates: list[WorkflowCandidate] = Field(default_factory=list)
    next_allowed_actions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SelectionCandidate(BaseModel):
    candidate_id: str
    path: str
    name: str
    kind: str = "file"
    subtitle: str = ""


class PendingTask(BaseModel):
    task_id: str
    intent: str
    summary: str
    original_user_request: str
    state_kind: str = "clarification"
    clarification_prompt: str = ""
    selection_candidates: list[SelectionCandidate] = Field(default_factory=list)
    overall_task_goal: TaskGoal | None = None
    missing_slots: list[str] = Field(default_factory=list)
    collected_slots: dict[str, str] = Field(default_factory=dict)
    resume_hint: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ContextTaskRecord(BaseModel):
    task_id: str
    source_turn_id: str | None = None
    original_user_request: str
    summary: str = ""
    task_kind: str = "unknown"
    workflow_family: str = "generic"
    state_kind: str = "task_follow_up"
    selection_candidates: list[SelectionCandidate] = Field(default_factory=list)
    candidate_state: CandidateState | None = None
    workflow_state: WorkflowState | None = None
    overall_task_goal: TaskGoal | None = None
    completed_outputs: list[OutputKind] = Field(default_factory=list)
    missing_outputs: list[OutputKind] = Field(default_factory=list)
    collected_slots: dict[str, str] = Field(default_factory=dict)
    resume_hint: str = ""
    confidence: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_pending_task(self) -> PendingTask:
        return PendingTask(
            task_id=self.task_id,
            intent=self.task_kind or "context_follow_up",
            summary=self.summary or "Resume a recent contextual task.",
            original_user_request=self.original_user_request,
            state_kind=self.state_kind,
            selection_candidates=list(self.selection_candidates),
            overall_task_goal=self.overall_task_goal,
            missing_slots=["selected_candidate_path"] if self.selection_candidates else ["follow_up_instruction"],
            collected_slots=dict(self.collected_slots),
            resume_hint=self.resume_hint,
            created_at=self.created_at,
        )


class DocumentDeliveryIntent(BaseModel):
    wants_document: bool = False
    save_output: bool = False
    artifact_type: str | None = None
    output_format: str | None = None
    output_file: str | None = None
    title: str | None = None
    confidence: float = 0.0
    rationale: str = ""


class KnowledgeRequestIntent(BaseModel):
    needs_grounding: bool = False
    time_sensitive: bool = False
    lookup_requested: bool = False
    knowledge_type: str = "unknown"
    confidence: float = 0.0
    rationale: str = ""


class SiteSearchIntent(BaseModel):
    site: str | None = None
    query: str | None = None
    content_type: str = "generic"
    action: str = "search"
    site_scope: str = "none"
    open_first: bool = False
    confidence: float = 0.0
    rationale: str = ""


class InstructionIntent(BaseModel):
    is_instruction: bool = False
    scope: str = "none"  # none | turn | session | persistent
    kind: str = "none"  # none | naming | preference | workflow_method | tool_policy | correction | style | boundary
    apply_this_turn: bool = False
    persist_memory: bool = False
    normalized_instruction: str | None = None
    memory_text: str | None = None
    preferred_families: list[str] = Field(default_factory=list)
    blocked_families: list[str] = Field(default_factory=list)
    preferred_tools: list[str] = Field(default_factory=list)
    response_style: str | None = None
    confidence: float = 0.0
    rationale: str = ""


class MemoryCandidateIntent(BaseModel):
    is_memory_candidate: bool = False
    scope: str = "none"  # none | turn | session | persistent
    kind: str = "none"  # none | user_fact | naming | preference | workflow_method | tool_policy | correction | style | boundary
    apply_this_turn: bool = False
    persist_memory: bool = False
    should_write_memory: bool = False
    overwrite_existing: bool = False
    normalized_text: str | None = None
    memory_text: str | None = None
    memory_key: str | None = None
    canonical_value: dict[str, Any] = Field(default_factory=dict)
    preferred_families: list[str] = Field(default_factory=list)
    blocked_families: list[str] = Field(default_factory=list)
    preferred_tools: list[str] = Field(default_factory=list)
    response_style: str | None = None
    confidence: float = 0.0
    rationale: str = ""


class TaskSubtaskIntent(BaseModel):
    task_id: str = ""
    order: int = 0
    summary: str = ""
    task_text: str = ""
    kind: str = "generic"  # generic | web_lookup | document_edit | local_lookup | direct_answer | qq_history | system_utility | delivery
    status: str = "ready"  # ready | waiting_for_input | blocked | completed
    missing_slots: list[str] = Field(default_factory=list)
    slot_values: dict[str, str] = Field(default_factory=dict)
    rationale: str = ""


class TaskGraphIntent(BaseModel):
    is_multi_task: bool = False
    primary_task_text: str | None = None
    primary_task_id: str | None = None
    needs_clarification: bool = False
    followup_text: str | None = None
    subtasks: list[TaskSubtaskIntent] = Field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""


class ProxySendIntent(BaseModel):
    should_handle: bool = False
    recipient_query: str | None = None
    message_body: str | None = None
    intent_label: str = "send_message"
    confidence: float = 0.0
    rationale: str = ""


class FollowUpAssessment(BaseModel):
    action: str = "new_request"
    rationale: str = ""
    target_task_id: str | None = None
    slot_updates: dict[str, str] = Field(default_factory=dict)
    merged_user_request: str | None = None
    assistant_response: str | None = None


class LiveTurnEvent(BaseModel):
    event_type: str = "user_message"
    text: str = ""
    attachment_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LiveTurnState(BaseModel):
    session_id: str
    turn_id: str
    channel: str = "unknown"
    status: str = "collecting"   # collecting / finalized / submitted / discarded
    version: int = 0
    raw_user_turn_text: str = ""
    event_count: int = 0
    attachment_refs: list[str] = Field(default_factory=list)
    events: list[LiveTurnEvent] = Field(default_factory=list)
    first_event_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_event_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    typing_active: bool = False
    last_typing_at: datetime | None = None
    typing_expires_at: datetime | None = None
    finalized_at: datetime | None = None
    submitted_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class FinalizedTurn(BaseModel):
    session_id: str
    turn_id: str
    raw_user_turn_text: str
    event_count: int = 0
    attachment_refs: list[str] = Field(default_factory=list)
    message_segments: list[str] = Field(default_factory=list)
    finalized_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finalize_reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

class TurnResolution(BaseModel):
    session_id: str
    turn_id: str | None = None
    raw_user_turn_text: str
    turn_type: str = "fresh_turn"   # fresh_turn / followup / resume_pending / resume_prior_topic
    planner_visible_user_text: str
    recent_context: str = ""
    active_task_summary: str = ""
    pending_task_summary: str = ""
    retrieved_topic_summary: str = ""
    rationale: str = ""

class TurnCompletionDecision(BaseModel):
    finalize: bool = False
    confidence: float = 0.0
    wait_ms: int = 0
    reason: str = ""
    source: str = "rule"
    turn_kind: str = "uncertain"  # uncertain | execute_task | direct_reply | memory_update | instruction_update | chat
    ask_followup: bool = False
    followup_text: str = ""
    understood_task: str = ""
    should_ack_task: bool = False
    task_ack_text: str = ""


class ToolCallRequest(BaseModel):
    request_id: str
    trace_id: str
    session_id: str
    tool_name: str
    arguments: dict[str, Any]
    execution_context: ToolExecutionContext = Field(default_factory=ToolExecutionContext)
    caller: str = "agent-kernel"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolError(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ToolCallResult(BaseModel):
    request_id: str
    trace_id: str
    tool_name: str | None = None
    status: str
    data: dict[str, Any] = Field(default_factory=dict)
    produced_outputs: list[OutputKind] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    error: ToolError | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class MemoryRecord(BaseModel):
    memory_type: str
    scope: str
    content: str
    importance: float = 0.5
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TurnArtifacts(BaseModel):
    decision: ToolDecision | None = None
    tool_results: list[ToolCallResult] = Field(default_factory=list)
    final_response: str = ""
    speech_text: str = ""
    tts_dispatched: bool = False
    completed_outputs: list[OutputKind] = Field(default_factory=list)
    overall_task_goal: TaskGoal | None = None
    candidate_state: CandidateState | None = None
    workflow_state: WorkflowState | None = None
    pending_task: PendingTask | None = None
    execution_summary: dict[str, Any] | None = None
    debug_summary: str = ""
    trace_id: str = ""


class TaskProgressEvent(BaseModel):
    event_id: str
    task_id: str
    stage: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TaskRun(BaseModel):
    task_id: str
    session_id: str
    user_text: str
    scope_root: str | None = None
    runtime_settings: dict[str, Any] | None = None
    status: TaskStatus = TaskStatus.QUEUED
    mode: str = "agent"
    progress_message: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    response_ready_at: datetime | None = None
    elapsed_ms: int | None = None
    overall_task_goal: TaskGoal | None = None
    completed_outputs: list[OutputKind] = Field(default_factory=list)
    final_response: str = ""
    speech_text: str = ""
    tts_dispatched: bool = False
    needs_confirmation: bool = False
    acknowledged: bool = False
    error: str | None = None
    events: list[TaskProgressEvent] = Field(default_factory=list)
    pending_task: PendingTask | None = None


class AgentConfig(BaseModel):
    llm_provider: str = "ollama"
    model: str
    chat_model: str | None = None
    critic_model: str | None = None
    response_model: str | None = None
    vision_model: str | None = None
    api_base_url: str | None = None
    api_key_env: str = "DEEPSEEK_API_KEY"
    ollama_keep_alive: str | None = "15m"
    persona_name: str = "Local Butler"
    assistant_aliases: list[str] = Field(default_factory=list)
    persona_profile: str = ""
    chat_style_prompt: str = (
        "Keep the tone natural, warm, and clear. "
        "Sound like a real person talking in chat. "
        "Prefer colloquial Chinese over theatrical narration. "
        "Avoid bracketed stage directions, action descriptions, and overdone roleplay markers. "
        "In the chat box, prefer answers that cover the main point, the most relevant context, and the practical conclusion. "
        "Be more informative than a one-line reply, but avoid rambling, repetition, and over-explaining."
    )
    display_style_prompt: str = (
        "For screen display, write a fuller answer that covers the main result, the key supporting details, and the next useful takeaway. "
        "Prefer short paragraphs or light structure when it improves scanning. "
        "Keep enough information density to be useful, but do not become verbose or repetitive."
    )
    speech_style_prompt: str = (
        "Speech replies should stay brief, natural, and conversational Chinese. "
        "Sound light and human, like speaking directly to the user. "
        "Avoid bracketed stage directions, action descriptions, and dramatic narration. "
        "Focus on the conclusion or next step, avoid long paths and long lists, and do not restate the full display text."
    )
    fast_response_style_enabled: bool = True
    fast_response_model: str | None = "qwen2.5:1.5b"
    fast_response_timeout_seconds: int = 8
    live_turn_quiet_window_ms: int = 20000
    live_turn_incomplete_extra_ms: int = 450
    live_turn_attachment_extra_ms: int = 450
    live_turn_max_wait_ms: int = 30000
    live_turn_fragment_max_wait_ms: int = 40000
    live_turn_use_llm_judge: bool = True
    live_turn_typing_hold_ms: int = 10000
    tool_speech_enabled: bool = True
    speech_max_chars: int = 80
    retrieval_db_path: str = "data/local_retrieval.sqlite3"
    retrieval_embedding_enabled: bool = True
    retrieval_embedding_provider: str = "hash"
    retrieval_embedding_model: str | None = None
    retrieval_embedding_dimensions: int = 128
    retrieval_embedding_timeout_seconds: int = 30
    retrieval_embedding_batch_size: int = 64
    retrieval_reranker_provider: str = "heuristic"
    retrieval_reranker_model: str | None = None
    retrieval_reranker_timeout_seconds: int = 30
    retrieval_reranker_top_n: int = 5
    ollama_base_url: str = "http://127.0.0.1:11434"
    system_name: str = "Local Agent"
    workspace_root: str = "."
    memory_db_path: str = "data/agent_memory.sqlite3"
    trace_path: str = "data/traces.jsonl"
    max_steps: int = 6
    request_timeout_seconds: int = 120
    learning_reflection_enabled: bool = True


class VoiceConfig(BaseModel):
    enabled: bool = False
    async_playback: bool = True
    endpoint: str = "http://127.0.0.1:9880"
    gptsovits_root: str = "C:/GPT-SoVITS"
    api_host: str = "127.0.0.1"
    api_port: int = 9880
    api_script: str = "api_v2.py"
    runtime_python: str = "runtime/python.exe"
    gpt_weights_path: str = "C:/GPT-SoVITS/models/silverwolf/silverwolf_gpt.ckpt"
    sovits_weights_path: str = "C:/GPT-SoVITS/models/silverwolf/silverwolf_sovits.pth"
    ref_audio_path: str = "C:/GPT-SoVITS/models/silverwolf/该做的事都做完了么？好，别睡下了才想起来日常没做，拜拜。.wav"
    prompt_text: str = ""
    prompt_lang: str = "zh"
    text_lang: str = "zh"
    output_dir: str = "C:/GPT-SoVITS/output"
    auto_start_server: bool = True
    shutdown_when_idle: bool = True
    play_audio: bool = True
    cleanup_audio: bool = True
    extra_payload: dict[str, Any] = Field(default_factory=dict)

    def runtime_python_path(self) -> Path:
        root = Path(self.gptsovits_root)
        path = Path(self.runtime_python)
        return path if path.is_absolute() else root / path

    def api_script_path(self) -> Path:
        root = Path(self.gptsovits_root)
        path = Path(self.api_script)
        return path if path.is_absolute() else root / path


class ASRConfig(BaseModel):
    enabled: bool = False
    provider: str = "faster_whisper"
    model: str = "small"
    model_download_root: str = "data/models/faster-whisper"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 3
    vad_filter: bool = True
    command: list[str] = Field(default_factory=list)
    endpoint: str | None = None
    language: str = "zh"
    request_timeout_seconds: int = 120
    output_encoding: str = "utf-8"


class VoiceInputConfig(BaseModel):
    enabled: bool = True
    process_onebot_voice: bool = True
    wake_word_enabled: bool = False
    wake_word_mode: str = "asr"
    wake_word_fuzzy_threshold: float = 0.78
    background_task_ack_phrases: list[str] = Field(
        default_factory=lambda: ["收到，我放到后台处理。", "好，我先在后台弄，你可以继续说别的。"]
    )
    task_done_notify_phrases: list[str] = Field(
        default_factory=lambda: ["你刚才交给我的任务做完了，要听结果吗？", "刚才那个任务我处理好了，要我说结果吗？"]
    )
    task_result_declined_phrases: list[str] = Field(
        default_factory=lambda: ["好，那我先不念。", "行，结果我先给你留着。"]
    )
    wake_ack_phrases: list[str] = Field(default_factory=lambda: ["嗯？", "我在。", "说吧。"])
    microphone_enabled: bool = False
    microphone_device: str | None = None
    microphone_sample_rate: int = 16_000
    microphone_channels: int = 1
    microphone_chunk_ms: int = 320
    active_session_seconds: float = 10.0
    local_voice_poll_interval_ms: int = 700
    utterance_silence_ms: int = 650
    utterance_min_ms: int = 450
    utterance_max_seconds: float = 8.0
    audio_activity_threshold: int = 550
    self_listen_cooldown_seconds: float = 1.4
    wake_words: list[str] = Field(default_factory=lambda: ["嗨银狼"])
    vad_enabled: bool = False
    temp_dir: str = "data/audio_inputs"
    cleanup_temp: bool = True
    local_file_wait_seconds: float = 3.0
    local_file_probe_interval_ms: int = 200


class WebConfig(BaseModel):
    search_provider: str = "browser"
    browser_channel: str = "msedge"
    browser_headless: bool = False
    browser_search_engine: str = "bing"
    browser_launch_timeout_seconds: int = 30
    fetch_allow_insecure: bool = False
    prefer_browser_fetch: bool = False


class OneBotConfig(BaseModel):
    enabled: bool = False
    ws_url: str = "ws://127.0.0.1:3001"
    access_token: str | None = None
    send_replies: bool = True
    reconnect_delay_seconds: int = 5
    coalesce_window_ms: int = 900
    coalesce_short_message_extra_ms: int = 450
    coalesce_attachment_extra_ms: int = 500
    reply_delay_ms: int = 250
    reply_segment_max_chars: int = 110
    reply_segment_max_count: int = 3
    reply_segment_delay_ms: int = 650
    progress_update_enabled: bool = False
    progress_first_delay_ms: int = 2400
    progress_min_interval_ms: int = 2200
    progress_max_updates: int = 2
    recent_message_limit: int = 24
    startup_history_sync_enabled: bool = True
    startup_history_sync_group_count: int = 30
    startup_history_sync_friend_count: int = 20
    startup_history_sync_max_groups: int = 20
    startup_history_sync_max_friends: int = 20
    group_followup_window_ms: int = 20000
    group_context_review_interval_ms: int = 600000
    group_passive_min_interval_ms: int = 90000
    group_passive_batch_message_count: int = 5
    group_context_max_messages: int = 12
    full_access_user_ids: list[str] = Field(default_factory=list)
    owner_user_ids: list[str] = Field(default_factory=list)
    owner_display_name: str = "主人"


class OverseerConfig(BaseModel):
    enabled: bool = False
    poll_interval_seconds: int = 30
    min_poll_interval_seconds: int = 15
    quiet_cooldown_seconds: int = 60
    resize_width: int = 800
    jpeg_quality: int = 30
    qq_session_id: str = "onebot_private_2326478033"
    persona_name: str = "银狼"


class AppConfig(BaseModel):
    agent: AgentConfig
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    voice_input: VoiceInputConfig = Field(default_factory=VoiceInputConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    onebot: OneBotConfig = Field(default_factory=OneBotConfig)
    overseer: OverseerConfig = Field(default_factory=OverseerConfig)
