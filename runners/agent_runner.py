from __future__ import annotations

from local_agent.app.chat_models import ChatTurnResult
from local_agent.runners.base import RunnerContext


class AgentTurnRunner:
    name = "agent"

    def run(self, context: RunnerContext) -> ChatTurnResult:
        print("[agent_runner:start]", {
            "session_id": context.session.session_id,
            "text": context.text,
        })

        context.service.prepare_turn_context(context.session, context.text)
        memory_update_ack = context.service.try_memory_update_ack(context.session, context.text)
        if memory_update_ack is not None:
            print("[agent_runner:memory_update_ack]", {
                "session_id": context.session.session_id,
                "response_preview": (memory_update_ack.response or "")[:80],
            })
            return memory_update_ack
        fast_local_result = context.service.try_fast_local_answer(context.session, context.text)
        if fast_local_result is not None:
            print("[agent_runner:fast_local_answer]", {
                "session_id": context.session.session_id,
                "response_preview": (fast_local_result.response or "")[:80],
            })
            return fast_local_result

        runtime_settings = context.session.runtime_settings or {}
        channel_settings = runtime_settings.get("channel") if isinstance(runtime_settings.get("channel"), dict) else {}
        channel_runtime = runtime_settings.get("channel_runtime") if isinstance(runtime_settings.get("channel_runtime"), dict) else None

        runtime_channel = str(channel_settings.get("name") or "").strip() or None
        runtime_session_id = str(context.session.session_id or "").strip() or None

        print("[agent_runner:kernel_call]", {
            "runtime_session_id": runtime_session_id,
            "runtime_channel": runtime_channel,
            "has_channel_runtime": isinstance(channel_runtime, dict),
        })

        try:
            artifacts = context.session.kernel.handle_user_input(
                context.text,
                progress_callback=context.progress_callback,
                runtime_session_id=runtime_session_id,
                runtime_channel=runtime_channel,
                runtime_channel_context=channel_runtime,
            )
            print("[agent_runner:kernel_returned]", {
                "trace_id": getattr(artifacts, "trace_id", ""),
                "response_preview": (artifacts.final_response or "")[:80],
            })
        except TypeError:
            # 兼容旧签名
            try:
                artifacts = context.session.kernel.handle_user_input(
                    context.text,
                    progress_callback=context.progress_callback,
                    runtime_session_id=runtime_session_id,
                    runtime_channel=runtime_channel,
                )
            except TypeError:
                try:
                    artifacts = context.session.kernel.handle_user_input(
                        context.text,
                        progress_callback=context.progress_callback,
                    )
                except TypeError:
                    artifacts = context.session.kernel.handle_user_input(context.text)
        except Exception as exc:
            print("[agent_runner:error]", repr(exc))
            raise

        context.service.session_store.set_pending_task(
            context.session.session_id,
            artifacts.pending_task,
            mode="agent",
        )
        context.session.pending_task = artifacts.pending_task
        context.service.refresh_hot_context(context.session)

        return ChatTurnResult(
            session_id=context.session.session_id,
            mode="agent",
            response=artifacts.final_response,
            speech_text=artifacts.speech_text or artifacts.final_response,
            tts_dispatched=artifacts.tts_dispatched,
            used_agent=True,
            scope_root=context.session.scope_root,
            overall_task_goal=(
                None
                if artifacts.overall_task_goal is None
                else artifacts.overall_task_goal.model_dump(mode="json")
            ),
            completed_outputs=[item.value for item in artifacts.completed_outputs],
            pending_task=(
                None
                if artifacts.pending_task is None
                else artifacts.pending_task.model_dump(mode="json")
            ),
            metadata={
                "trace_id": artifacts.trace_id,
                "execution_summary": artifacts.execution_summary,
            },
        )
