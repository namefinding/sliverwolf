from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field

from local_agent.protocol.models import (
    DocumentDeliveryIntent,
    InstructionIntent,
    KnowledgeRequestIntent,
    MemoryCandidateIntent,
    OutputKind,
    SiteSearchIntent,
    TaskGraphIntent,
    WorkflowSpec,
)



class TaskClassification(BaseModel):
    domain: str = "unknown"
    task_kind: str = "unknown"
    preferred_families: list[str] = Field(default_factory=list)

    run_mode: str = "immediate"  # immediate | scheduled
    scheduled_task_type: str | None = None  # notify | deferred_agent_task | None
    scheduled_task_payload_hint: dict[str, Any] = Field(default_factory=dict)

    confidence: float = 0.0
    rationale: str = ""


class TaskEnvelope(BaseModel):
    mode: str = "generic"
    conversation_mode: str = "new_request"
    primary_objective: str = ""
    needs_grounding: bool = False
    context_layers_used: list[str] = Field(default_factory=list)
    allowed_families: list[str] = Field(default_factory=list)
    blocked_families: list[str] = Field(default_factory=list)
    required_outputs: list[OutputKind] = Field(default_factory=list)
    should_reply_in_channel: bool = True
    recipient_mode: str = "current_user"
    style_intent: str | None = None
    persona_reference: str | None = None
    instruction_scope: str | None = None
    instruction_kind: str | None = None
    instruction_summary: str | None = None
    preferred_tools: list[str] = Field(default_factory=list)
    planning_focus_text: str | None = None
    execution_notes: list[str] = Field(default_factory=list)
    known_failure_avoidance: list[str] = Field(default_factory=list)
    tool_order_constraints: list[str] = Field(default_factory=list)
    response_strategy: str | None = None
    delegated_execution_brief: str | None = None
    workflow_spec: WorkflowSpec | None = None
    subtask_count: int = 0
    rationale: str = ""


class AnswerabilityAssessment(BaseModel):
    answerability: str = "verification_required"  # memory_or_local_answerable | local_tool_needed | verification_required
    preferred_family: str = "web_lookup"
    local_answer_kind: str = "none"  # none | date_time | arithmetic | memory_fact | direct_chat
    answer_basis: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""


class IntentBundle(BaseModel):
    document_delivery: DocumentDeliveryIntent = Field(default_factory=DocumentDeliveryIntent)
    knowledge_request: KnowledgeRequestIntent = Field(default_factory=KnowledgeRequestIntent)
    site_search: SiteSearchIntent = Field(default_factory=SiteSearchIntent)
    memory_candidate_intent: MemoryCandidateIntent = Field(default_factory=MemoryCandidateIntent)
    instruction_intent: InstructionIntent = Field(default_factory=InstructionIntent)
    task_graph: TaskGraphIntent = Field(default_factory=TaskGraphIntent)
    answerability: AnswerabilityAssessment = Field(default_factory=AnswerabilityAssessment)
    task_classification: TaskClassification | None = None
    task_envelope: TaskEnvelope = Field(default_factory=TaskEnvelope)


class LocalCollectionIntent(BaseModel):
    action: str
    destination: str
    source_scope: str = "."
    selection_query: str = ""
    category: str | None = None
    patterns: list[str] = Field(default_factory=list)
    extensions: list[str] = Field(default_factory=list)
    use_directory_listing: bool = False
    terminal_output: OutputKind = OutputKind.PATH_UPDATED
