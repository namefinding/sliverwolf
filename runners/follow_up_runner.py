from __future__ import annotations

from local_agent.app.chat_models import ChatTurnResult
from local_agent.protocol.models import CandidateState, Message, Role
from local_agent.runners.base import RunnerContext


class FollowUpTurnRunner:
    name = "follow_up"

    def run(self, context: RunnerContext) -> ChatTurnResult:
        pending_task = context.session.pending_task
        assert pending_task is not None

        assessment = context.session.pending_follow_up_assessment
        context.session.pending_follow_up_assessment = None
        if assessment is None:
            assessment = context.service.classify_follow_up(
                context.session.kernel,
                pending_task,
                context.text,
            )
        if assessment.action == "cancel":
            context.service.session_store.set_pending_task(
                context.session.session_id,
                None,
                mode="chat",
            )
            context.session.pending_task = None
            context.session.active_context_task = None
            response = assessment.assistant_response
            if not response:
                try:
                    response = context.session.kernel.render_follow_up_cancel_response(
                        latest_user_text=context.text,
                        pending_task=pending_task,
                    )
                except Exception as exc:  # noqa: BLE001
                    response = context.session.kernel.llm_client.build_unavailable_response(exc)
            tts_dispatched = context.service.dispatch_tts(context.session, response)
            context.session.kernel.history.append(Message(role=Role.USER, content=context.text))
            context.session.kernel.history.append(Message(role=Role.ASSISTANT, content=response))
            context.service.refresh_hot_context(context.session)
            return ChatTurnResult(
                session_id=context.session.session_id,
                mode="chat",
                response=response,
                speech_text=response,
                tts_dispatched=tts_dispatched,
                used_agent=False,
                scope_root=context.session.scope_root,
                metadata={},
            )

        if assessment.action in {"resume", "resume_with_correction"}:
            updated_pending = pending_task.model_copy(deep=True)
            updated_pending.collected_slots.update(assessment.slot_updates)
            resumed_request = assessment.merged_user_request or context.service.merge_pending_request(
                updated_pending,
                context.text,
            )
            updated_pending.original_user_request = resumed_request
            if assessment.slot_updates:
                updated_pending.missing_slots = [
                    slot for slot in updated_pending.missing_slots if slot not in assessment.slot_updates
                ]
            if assessment.action == "resume_with_correction":
                updated_pending.summary = context.text.strip() or updated_pending.summary
            if not updated_pending.missing_slots:
                updated_pending.clarification_prompt = ""
            context.session.pending_task = updated_pending
            context.service.prepare_turn_context(context.session, resumed_request)
            active_context_task = context.session.active_context_task
            seed_candidate_state = None
            seed_workflow_state = None
            if active_context_task is not None:
                seed_candidate_state = active_context_task.candidate_state
                seed_workflow_state = active_context_task.workflow_state
            selected_path = updated_pending.collected_slots.get("selected_candidate_path")
            if isinstance(selected_path, str) and selected_path.strip():
                selected_name = str(
                    updated_pending.collected_slots.get("selected_candidate_name") or selected_path.split("\\")[-1]
                ).strip()
                seed_candidate_state = CandidateState(
                    query=selected_name,
                    target_kind="file",
                    path_scope=context.session.scope_root or ".",
                    query_terms=[],
                    candidate_paths=[selected_path],
                    candidate_names=[selected_name],
                    source_tool="user_selection",
                    confidence=1.0,
                    confidence_reason="user_selected_candidate",
                )
            try:
                artifacts = context.session.kernel.handle_user_input(
                    resumed_request,
                    progress_callback=context.progress_callback,
                    seed_candidate_state=seed_candidate_state,
                    seed_overall_task_goal=updated_pending.overall_task_goal,
                    seed_workflow_state=seed_workflow_state,
                )
            except TypeError:
                artifacts = context.session.kernel.handle_user_input(resumed_request)
            context.service.session_store.set_pending_task(
                context.session.session_id,
                artifacts.pending_task,
                mode="agent",
            )
            context.session.pending_task = artifacts.pending_task
            context.session.active_context_task = None
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

        context.session.pending_task = None
        context.session.active_context_task = None
        context.service.refresh_hot_context(context.session)
        return context.service.handle_message(
            text=context.text,
            session_id=context.session.session_id,
            mode="auto",
            scope_root=context.session.scope_root,
            progress_callback=context.progress_callback,
        )
