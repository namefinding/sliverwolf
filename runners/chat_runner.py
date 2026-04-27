from __future__ import annotations

from local_agent.app.chat_models import ChatTurnResult
from local_agent.protocol.models import Message, Role
from local_agent.runners.base import RunnerContext


class ChatTurnRunner:
    name = "chat"

    def run(self, context: RunnerContext) -> ChatTurnResult:
        session = context.session
        kernel = session.kernel
        context.service.prepare_turn_context(session, context.text)
        kernel.history.append(Message(role=Role.USER, content=context.text))
        try:
            response = kernel.llm_client.chat_reply(
                system_name=kernel.config.system_name,
                messages=context.service.prune_chat_history(session),
                persona_name=getattr(kernel.config, "persona_name", None),
                persona_profile=getattr(kernel.config, "persona_profile", None),
                chat_style_prompt=getattr(kernel.config, "chat_style_prompt", None),
            )
        except TypeError:
            response = kernel.llm_client.chat_reply(
                system_name=kernel.config.system_name,
                messages=context.service.prune_chat_history(session),
            )
        kernel.history.append(Message(role=Role.ASSISTANT, content=response))
        context.service.refresh_hot_context(session)

        tts_dispatched = context.service.dispatch_tts(session, response)
        session.touch("chat")
        return ChatTurnResult(
            session_id=session.session_id,
            mode="chat",
            response=response,
            speech_text=response,
            tts_dispatched=tts_dispatched,
            used_agent=False,
            scope_root=session.scope_root,
            metadata={},
        )
