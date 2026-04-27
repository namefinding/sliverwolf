from __future__ import annotations

from dataclasses import dataclass

from local_agent.protocol.models import AgentConfig, Message


@dataclass(frozen=True)
class ProxyMessageDraft:
    recipient_name: str
    original_request: str
    extracted_body: str
    intent_label: str = "send_message"


class MessageComposer:
    def __init__(self, llm_client, agent_config: AgentConfig) -> None:
        self._llm_client = llm_client
        self._agent_config = agent_config

    def compose_proxy_message(
        self,
        *,
        draft: ProxyMessageDraft,
        recent_messages: list[Message] | None = None,
    ) -> str:
        extracted = draft.extracted_body.strip()
        if not extracted:
            return extracted

        recent_messages = recent_messages or []
        conversation = "\n".join(
            f"{message.role.value}: {message.content}"
            for message in recent_messages[-8:]
        )
        persona_name = self._agent_config.persona_name
        persona_profile = self._agent_config.persona_profile
        chat_style = self._agent_config.chat_style_prompt

        system_prompt = (
            f"You are the message-composition layer for {self._agent_config.system_name}. "
            "The user wants you to help write a message that will be sent to another person in QQ chat. "
            "Return only the exact Chinese message that should be sent. "
            "Do not explain, do not use labels, and do not add quotation marks. "
            "Use the recent conversation as soft context for tone and intent, but keep the content grounded in the user's latest delegated-send request. "
            "Make the output natural, colloquial, and suitable for direct sending in a chat window. "
            "Write like a real person chatting, not like stage directions or narrated roleplay. "
            "Avoid bracketed actions, such as parenthetical gestures or mood tags. "
            "If the extracted body is already a direct message, polish lightly. "
            "If it is vague, such as '随便什么都行', infer a short but sensible message from the original request and recent context. "
            "Do not mention being an AI or a robot. "
            "Do not invent facts or commitments that the user did not imply. "
            f"Persona name: {persona_name}. Persona profile: {persona_profile}\n"
            f"Style reference: {chat_style}"
        )
        user_prompt = (
            f"Recent conversation with the user:\n{conversation or 'No prior context.'}\n\n"
            f"Recipient name:\n{draft.recipient_name}\n\n"
            f"Delegated-send intent:\n{draft.intent_label}\n\n"
            f"Original user request:\n{draft.original_request}\n\n"
            f"Extracted raw body:\n{draft.extracted_body}\n\n"
            "Return only the final message text to send."
        )

        try:
            rewritten = self._llm_client._chat(  # noqa: SLF001
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self._llm_client.chat_model,
            ).strip().strip("\"'“”")
        except Exception:
            return extracted
        return rewritten or extracted
