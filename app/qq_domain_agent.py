from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from local_agent.app.onebot_contacts import OneBotContactMatch
from local_agent.app.onebot_send_proxy import (
    OneBotProxySendRequest,
    PendingProxySelection,
    is_assistant_recipient_query,
    resolve_proxy_selection,
)


@dataclass(frozen=True)
class QQProxySelectionDecision:
    action: str = "wait"
    selected: OneBotContactMatch | None = None
    confidence: float = 0.0
    rationale: str = ""


class QQDomainAgent:
    """LLM-backed QQ semantic layer.

    Deterministic code here is limited to exact identifiers such as candidate
    numbers, contact ids, and configured assistant aliases. Natural-language
    intent stays with the LLM methods.
    """

    def __init__(self, llm_client: Any, *, assistant_aliases: tuple[str, ...] = ()) -> None:
        self.llm_client = llm_client
        self.assistant_aliases = assistant_aliases

    def classify_proxy_send_request(
        self,
        *,
        user_text: str,
        recent_messages: list[dict[str, Any]],
        confidence_threshold: float = 0.82,
    ) -> OneBotProxySendRequest | None:
        if self.llm_client is None or not hasattr(self.llm_client, "classify_proxy_send_intent"):
            return None
        try:
            assessment = self.llm_client.classify_proxy_send_intent(
                user_text=user_text,
                recent_messages=recent_messages,
            )
        except Exception:
            return None

        if not bool(getattr(assessment, "should_handle", False)):
            return None
        recipient_query = str(getattr(assessment, "recipient_query", "") or "").strip()
        message_body = str(getattr(assessment, "message_body", "") or "").strip()
        if not recipient_query or not message_body:
            return None
        try:
            confidence = float(getattr(assessment, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < confidence_threshold:
            return None
        if is_assistant_recipient_query(recipient_query, self.assistant_aliases):
            return None

        intent_label = str(getattr(assessment, "intent_label", "send_message") or "send_message").strip().lower()
        if intent_label not in {"send_message", "notify", "remind", "relay"}:
            return None
        return OneBotProxySendRequest(
            recipient_query=recipient_query,
            message_body=message_body,
            intent_label=intent_label,
            source_text=str(user_text or "").strip(),
        )

    def classify_proxy_selection(
        self,
        *,
        pending: PendingProxySelection,
        user_text: str,
    ) -> QQProxySelectionDecision:
        selected = resolve_proxy_selection(pending, user_text)
        if selected is not None:
            return QQProxySelectionDecision(
                action="select",
                selected=selected,
                confidence=1.0,
                rationale="explicit_candidate_selection",
            )
        if self.llm_client is None or not hasattr(self.llm_client, "classify_proxy_selection_intent"):
            return QQProxySelectionDecision(action="wait", rationale="selection_agent_unavailable")
        try:
            payload = self.llm_client.classify_proxy_selection_intent(
                user_text=user_text,
                pending_request={
                    "recipient_query": pending.request.recipient_query,
                    "message_body": pending.request.message_body,
                    "intent_label": pending.request.intent_label,
                },
                candidates=[
                    {
                        "candidate_id": str(index),
                        "name": match.contact.name,
                        "target_id": match.contact.target_id,
                        "kind": match.contact.kind,
                    }
                    for index, match in enumerate(pending.candidates, start=1)
                ],
            )
        except Exception:
            return QQProxySelectionDecision(action="wait", rationale="selection_agent_error")

        action = str((payload or {}).get("action", "wait") or "wait").strip().lower()
        try:
            confidence = float((payload or {}).get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        rationale = str((payload or {}).get("rationale", "") or "").strip()
        if action == "cancel" and confidence >= 0.75:
            return QQProxySelectionDecision(action="cancel", confidence=confidence, rationale=rationale)
        if action == "select" and confidence >= 0.75:
            candidate_id = str((payload or {}).get("candidate_id", "") or "").strip()
            selected = resolve_proxy_selection(pending, candidate_id)
            if selected is not None:
                return QQProxySelectionDecision(
                    action="select",
                    selected=selected,
                    confidence=confidence,
                    rationale=rationale or "llm_candidate_selection",
                )
        return QQProxySelectionDecision(action="wait", confidence=confidence, rationale=rationale)
