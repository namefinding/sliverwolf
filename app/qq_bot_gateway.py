from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import TimeoutError as FuturesTimeoutError
import contextlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import shutil
from threading import Lock
import time
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from local_agent.app.bootstrap import bootstrap_resident_app
from local_agent.app.channel_access import build_onebot_access_policy
from local_agent.app.message_composer import MessageComposer, ProxyMessageDraft
from local_agent.app.onebot_contacts import (
    OneBotContact,
    build_contacts,
    is_confident_unique_match,
    match_contacts,
)
from local_agent.app.onebot_delivery import (
    build_file_action,
    build_reply_action,
    build_voice_action,
    extract_delivery_file_paths,
)
from local_agent.app.onebot_models import OneBotAudioAttachment, OneBotImageAttachment, OneBotInboundMessage, OneBotTarget
from local_agent.app.onebot_parser import extract_onebot_message, extract_onebot_typing_notice, is_auth_failure
from local_agent.app.qq_domain_agent import QQDomainAgent
from local_agent.app.qq_runtime import QQRuntimeRegistry
from local_agent.app.onebot_send_proxy import (
    PendingProxySelection,
    OneBotProxySendRequest,
    build_proxy_selection_prompt,
    is_assistant_recipient_query,
)
from local_agent.modules.qq.history_store import QQHistoryStore
from local_agent.protocol.channel_models import ChannelMessage
from local_agent.llm.ollama_client import OllamaClient
from local_agent.voice.asr_service import ASRService
from local_agent.voice.input_service import VoiceInputService
from local_agent.voice.output_service import VoiceOutputService


class _QQProgressUpdateController:
    def __init__(self, gateway, *, websocket, target: OneBotTarget, session_id: str, user_text: str, loop) -> None:
        self._gateway = gateway
        self._websocket = websocket
        self._target = target
        self._session_id = session_id
        self._user_text = user_text
        self._loop = loop
        self._lock = Lock()
        self._done = False
        self._latest_token = 0
        self._latest_text: str | None = None
        self._sent_token = 0
        self._sent_count = 0
        self._last_sent_at = 0.0

    def callback(self, stage: str, message: str, payload: dict[str, Any] | None = None) -> None:
        text = self._gateway._translate_progress_update(  # noqa: SLF001
            stage=stage,
            message=message,
            payload=payload or {},
            user_text=self._user_text,
        )
        if not text:
            return
        with self._lock:
            if self._done:
                return
            self._latest_token += 1
            self._latest_text = text

    def finish(self) -> None:
        with self._lock:
            self._done = True

    async def run(self) -> None:
        if not bool(getattr(self._gateway.onebot_cfg, "progress_update_enabled", True)):  # noqa: SLF001
            return
        first_delay_ms = max(0, int(getattr(self._gateway.onebot_cfg, "progress_first_delay_ms", 2400) or 0))  # noqa: SLF001
        min_interval_ms = max(300, int(getattr(self._gateway.onebot_cfg, "progress_min_interval_ms", 2200) or 0))  # noqa: SLF001
        max_updates = max(1, int(getattr(self._gateway.onebot_cfg, "progress_max_updates", 2) or 2))  # noqa: SLF001
        if first_delay_ms > 0:
            await asyncio.sleep(first_delay_ms / 1000.0)
        while True:
            with self._lock:
                if self._done or self._sent_count >= max_updates:
                    return
                latest_token = self._latest_token
                latest_text = self._latest_text
                sent_token = self._sent_token
                sent_count = self._sent_count
                last_sent_at = self._last_sent_at
            if latest_text and latest_token > sent_token:
                elapsed_ms = (time.monotonic() - last_sent_at) * 1000.0 if sent_count > 0 else min_interval_ms
                if sent_count == 0 or elapsed_ms >= min_interval_ms:
                    await self._gateway._send_transient_text(  # noqa: SLF001
                        self._websocket,
                        self._target,
                        latest_text,
                        session_id=self._session_id,
                    )
                    with self._lock:
                        self._sent_token = latest_token
                        self._sent_count += 1
                        self._last_sent_at = time.monotonic()
            await asyncio.sleep(0.25)


class QQBotGatewayClient:
    _TASK_ACK_ACTION_PATTERN = re.compile(
        r"(查一下|搜一下|搜索|查询|总结|整理|写一|写个|写篇|生成|导出|保存|发送|发给|代发|提醒|定时|打开|修改|编辑|删除|读取|聊天记录|历史记录|附件|联网|浏览|检索|运行)",
        flags=re.IGNORECASE,
    )
    _TASK_ACK_CHAT_PATTERN = re.compile(
        r"(聊天|聊聊|找个话题|换个话题|闲聊|陪我聊|在吗|你有.*功能|你会什么|谁创造的你|谁开发的你|你是谁|介绍一下你自己)",
        flags=re.IGNORECASE,
    )
    _TASK_ACK_RESULT_PATTERN = re.compile(
        r"(已检查|已经检查|已查|已经查|查到|查好了|查完|查过|确认了|确认过|找到|找到了|没找到|没有|尚未|未曾|根据记录|结果是|显示|曾经|多次|最新)",
        flags=re.IGNORECASE,
    )
    _TASK_ACK_HISTORY_PATTERN = re.compile(
        r"(聊天记录|历史记录|之前|有找过|联系过|私聊|群聊|消息记录|翻一下|查一下.*记录)",
        flags=re.IGNORECASE,
    )

    def __init__(self, *, config_path: str) -> None:
        self.app = bootstrap_resident_app(
            config_path=config_path,
            session_ttl_minutes=60,
            max_sessions=100,
        )
        self.onebot_cfg = self.app.config.onebot
        self._voice_output = VoiceOutputService.from_config(
            self.app.config.voice,
            play_audio=False,
            async_playback=False,
            cleanup_audio=False,
        )
        self._voice_input = VoiceInputService(
            self.app.config.voice_input,
            ASRService.from_config(self.app.config.asr),
        )
        self._llm_client = OllamaClient(
            base_url=self.app.config.agent.ollama_base_url,
            model=self.app.config.agent.model,
            timeout_seconds=self.app.config.agent.request_timeout_seconds,
            chat_model=self.app.config.agent.chat_model,
            critic_model=self.app.config.agent.critic_model,
            response_model=self.app.config.agent.response_model,
            keep_alive=self.app.config.agent.ollama_keep_alive,
        )
        self._message_composer = MessageComposer(
            llm_client=self._llm_client,
            agent_config=self.app.config.agent,
        )
        self._send_lock = asyncio.Lock()
        self._action_waiters: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._contact_cache: tuple[datetime, tuple[OneBotContact, ...]] | None = None
        self._pending_proxy_selection: dict[str, PendingProxySelection] = {}
        self._recent_messages_lock = Lock()
        self._recent_messages: dict[str, list[dict[str, Any]]] = {}
        self._proxy_threads_lock = Lock()
        self._proxy_threads: dict[str, dict[str, Any]] = {}
        self._coalesce_lock = asyncio.Lock()
        self._pending_inbound_batches: dict[str, list[OneBotInboundMessage]] = {}
        self._pending_inbound_versions: dict[str, int] = {}
        self._runtime_id = f"onebot_runtime_{uuid4().hex[:12]}"
        self._connection_lock = Lock()
        self._active_loop = None
        self._active_websocket = None
        self._history_store = QQHistoryStore(str(self._qq_history_db_path(config_path, self.app.config.agent.memory_db_path)))
        QQRuntimeRegistry.register(self._runtime_id, self)

    async def run_forever(self) -> None:
        if not self.onebot_cfg.enabled:
            raise RuntimeError("onebot.enabled is false in config.yaml")

        while True:
            try:
                await self._run_once()
            except Exception as exc:  # noqa: BLE001
                print(f"[onebot] disconnected: {exc}")
            await asyncio.sleep(max(1, self.onebot_cfg.reconnect_delay_seconds))

    async def _run_once(self) -> None:
        headers: dict[str, str] = {}
        if self.onebot_cfg.access_token:
            headers["Authorization"] = f"Bearer {self.onebot_cfg.access_token}"

        print(f"[onebot] connecting to {self.onebot_cfg.ws_url}")
        async with connect(
            self.onebot_cfg.ws_url,
            additional_headers=headers or None,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        ) as websocket:
            print("[onebot] connected")
            self._set_active_connection(asyncio.get_running_loop(), websocket)
            active_tasks: set[asyncio.Task] = set()
            try:
                async for raw in websocket:
                    payload = self._parse_payload(raw)
                    if payload is None:
                        continue
                    if self._resolve_action_response(payload):
                        continue
                    if is_auth_failure(payload):
                        raise RuntimeError("OneBot token verification failed. Please check onebot.access_token in config.yaml.")
                    typing_notice = extract_onebot_typing_notice(payload)
                    if typing_notice is not None:
                        await self._handle_typing_notice(typing_notice)
                        continue
                    inbound = extract_onebot_message(payload)
                    if inbound is None:
                        continue
                    task = asyncio.create_task(self._handle_inbound(websocket, inbound))
                    active_tasks.add(task)
                    task.add_done_callback(active_tasks.discard)
            except ConnectionClosed:
                raise
            finally:
                self._clear_active_connection(websocket)
                for echo, future in list(self._action_waiters.items()):
                    if not future.done():
                        future.cancel()
                    self._action_waiters.pop(echo, None)
                for task in active_tasks:
                    task.cancel()
                for task in active_tasks:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task

    @staticmethod
    def _parse_payload(raw: Any) -> dict[str, Any] | None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        if not isinstance(raw, str):
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _resolve_action_response(self, payload: dict[str, Any]) -> bool:
        echo = payload.get("echo")
        if not isinstance(echo, str):
            return False
        future = self._action_waiters.pop(echo, None)
        if future is None:
            return False
        if not future.done():
            future.set_result(payload)
        return True

    async def _call_action(
        self,
        websocket,
        *,
        action: str,
        params: dict[str, Any] | None = None,
        timeout_seconds: float = 10.0,
    ) -> dict[str, Any]:
        echo = f"call_{uuid4().hex[:12]}"
        request_payload = {"action": action, "params": params or {}, "echo": echo}
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._action_waiters[echo] = future
        try:
            async with self._send_lock:
                await websocket.send(json.dumps(request_payload, ensure_ascii=False))
            return await asyncio.wait_for(future, timeout=timeout_seconds)
        finally:
            self._action_waiters.pop(echo, None)

    async def _handle_typing_notice(self, notice: dict[str, Any]) -> None:
        chat_service = getattr(getattr(self, "app", None), "chat_service", None)
        if chat_service is None or not hasattr(chat_service, "capture_live_typing"):
            return
        session_id = str(notice.get("session_id") or "").strip()
        if not session_id:
            return
        metadata = {
            "sender_id": notice.get("sender_id"),
            "message_type": notice.get("message_type"),
            "group_id": notice.get("group_id"),
            "platform": "onebot_v11",
            "raw_payload": notice.get("raw_payload"),
        }
        self._append_live_turn_trace(
            session_id,
            "live_turn_typing_notice",
            {
                "session_id": session_id,
                "active": bool(notice.get("active")),
                "sender_id": notice.get("sender_id"),
                "message_type": notice.get("message_type"),
                "group_id": notice.get("group_id"),
            },
        )
        await asyncio.to_thread(
            chat_service.capture_live_typing,
            session_id=session_id,
            channel="onebot_v11",
            active=bool(notice.get("active")),
            metadata=metadata,
            runtime_settings={"channel_runtime": {"runtime_id": self._runtime_id, "session_id": session_id}},
        )

    async def _handle_inbound(self, websocket, inbound: OneBotInboundMessage) -> None:
        try:
            resolved_inbound = await asyncio.to_thread(self._resolve_inbound, inbound)
            if resolved_inbound is None:
                return
            self._remember_recent_message(
                resolved_inbound.session_id,
                role="user",
                text=resolved_inbound.text,
                sender_id=resolved_inbound.sender_id,
                image_attachments=resolved_inbound.image_attachments,
            )
            self._record_inbound_history(resolved_inbound)
            self._sync_proxy_reply_context(resolved_inbound)

            if await self._maybe_handle_proxy_send(websocket, resolved_inbound):
                return

            merged_inbound = await self._coalesce_inbound(websocket, resolved_inbound)
            if merged_inbound is None:
                return

            loop = asyncio.get_running_loop()
            progress_controller = _QQProgressUpdateController(
                self,
                websocket=websocket,
                target=merged_inbound.target,
                session_id=merged_inbound.session_id,
                user_text=merged_inbound.text,
                loop=loop,
            )
            progress_task = asyncio.create_task(progress_controller.run())
            try:
                turn_payload = merged_inbound.metadata.get("finalized_turn") if isinstance(merged_inbound.metadata, dict) else None
                self._append_live_turn_trace(
                    merged_inbound.session_id,
                    "qq_dispatch_start",
                    {
                        "session_id": merged_inbound.session_id,
                        "text": merged_inbound.text,
                        "mode": merged_inbound.mode,
                        "turn_id": (
                            str(turn_payload.get("turn_id") or "").strip()
                            if isinstance(turn_payload, dict)
                            else None
                        ),
                        "has_finalized_turn": isinstance(turn_payload, dict),
                    },
                )
                reply = await asyncio.to_thread(
                    self._dispatch_message,
                    merged_inbound,
                    progress_controller.callback,
                )
                self._append_live_turn_trace(
                    merged_inbound.session_id,
                    "qq_dispatch_done",
                    {
                        "session_id": merged_inbound.session_id,
                        "text": merged_inbound.text,
                        "mode": getattr(reply, "mode", ""),
                        "used_agent": bool(getattr(reply, "used_agent", False)),
                        "has_response": bool(str(getattr(reply, "response", "")).strip()),
                        "trace_id": (
                            str(((getattr(reply, "metadata", None) or {}).get("trace_id") or "")).strip()
                            if isinstance(getattr(reply, "metadata", None), dict)
                            else ""
                        ),
                    },
                )
            finally:
                progress_controller.finish()
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task
            if not self.onebot_cfg.send_replies or not reply.response.strip():
                return

            await self._send_generated_reply(websocket, merged_inbound, reply)
        except Exception as exc:  # noqa: BLE001
            print(f"[onebot] inbound handling failed: {exc}")
            self._append_live_turn_trace(
                inbound.session_id,
                "onebot_inbound_error",
                {
                    "session_id": inbound.session_id,
                    "text": inbound.text,
                    "error": str(exc),
                },
            )
            try:
                await self._send_text_reply(
                    websocket,
                    inbound.target,
                    "这一步在 QQ 通道层卡住了，我已经把错误写进 trace，先别急着重发，我来查。",
                    session_id=inbound.session_id,
                )
            except Exception as send_exc:  # noqa: BLE001
                print(f"[onebot] failed to send fallback reply: {send_exc}")

    def _dispatch_message(
            self,
            inbound: OneBotInboundMessage,
            progress_callback=None,
    ):
        channel_runtime = self._build_channel_runtime(inbound)
        return self.app.channel_router.dispatch(
            ChannelMessage(
                channel="onebot_v11",
                text=inbound.text,
                session_id=inbound.session_id,
                scope_root=inbound.scope_root,
                mode=inbound.mode,
                runtime_settings={
                    "channel_runtime": channel_runtime,
                    "channel": {
                        "name": "onebot_v11",
                    },
                },
                progress_callback=progress_callback,
                sender={"user_id": inbound.sender_id},
                metadata=inbound.metadata,
            )
        )
    async def _send_generated_reply(self, websocket, inbound: OneBotInboundMessage, reply) -> None:
        delivery_paths = extract_delivery_file_paths(reply)
        voice_task = self._start_voice_reply_task(reply, delivery_paths=delivery_paths)

        reply_delay_seconds = max(0.0, float(getattr(self.onebot_cfg, "reply_delay_ms", 0) or 0) / 1000.0)
        if reply_delay_seconds > 0:
            await asyncio.sleep(reply_delay_seconds)

        reply_segments = self._build_reply_segments(reply.response)
        for index, segment in enumerate(reply_segments):
            if index > 0:
                pause_seconds = self._reply_segment_pause_seconds(segment)
                if pause_seconds > 0:
                    await asyncio.sleep(pause_seconds)
            await self._send_text_reply(
                websocket,
                inbound.target,
                segment,
                session_id=inbound.session_id,
            )

        voice_result = await self._await_voice_reply_task(voice_task)
        await self._send_reply_attachments(
            websocket,
            inbound,
            delivery_paths=delivery_paths,
            voice_result=voice_result,
        )

    def _start_voice_reply_task(self, reply, *, delivery_paths: list[str]) -> asyncio.Task | None:
        if delivery_paths:
            return None
        if not self._should_send_voice_reply(reply.response):
            return None
        return asyncio.create_task(asyncio.to_thread(self._voice_output.synthesize_reply, reply))

    async def _await_voice_reply_task(self, voice_task: asyncio.Task | None):
        if voice_task is None:
            return None
        voice_result = await voice_task
        if voice_result.error:
            print(f"[onebot] voice synthesis failed: {voice_result.error}")
        return voice_result

    async def _send_reply_attachments(
        self,
        websocket,
        inbound: OneBotInboundMessage,
        *,
        delivery_paths: list[str],
        voice_result,
    ) -> None:
        outbound_attachments = [self._build_file_attachment(file_path) for file_path in delivery_paths]
        if not delivery_paths and not (voice_result and voice_result.audio_path):
            return
        async with self._send_lock:
            for file_path in delivery_paths:
                await websocket.send(json.dumps(build_file_action(inbound.target, file_path), ensure_ascii=False))
            if voice_result and voice_result.audio_path:
                await websocket.send(json.dumps(build_voice_action(inbound.target, voice_result.audio_path), ensure_ascii=False))
                asyncio.create_task(self._voice_output.cleanup_temp_file_later(voice_result.audio_path))
        if voice_result and voice_result.audio_path:
            outbound_attachments.append(self._build_audio_attachment(voice_result.audio_path))
        attachment_message_type = "audio" if voice_result and voice_result.audio_path and not delivery_paths else "file"
        self._record_outbound_history(
            session_id=inbound.session_id,
            target=inbound.target,
            text="",
            message_type=attachment_message_type,
            attachments=outbound_attachments,
        )

    def get_current_context(self) -> dict[str, Any]:
        connected = False
        with self._connection_lock:
            connected = self._active_loop is not None and self._active_websocket is not None
        return {
            "runtime_id": self._runtime_id,
            "channel": "onebot_v11",
            "connected": connected,
        }

    def get_recent_messages(
        self,
        *,
        session_id: str | None = None,
        limit: int = 8,
        include_assistant: bool = True,
    ) -> list[dict[str, Any]]:
        normalized_session_id = str(session_id or "").strip()
        if not normalized_session_id:
            return []
        normalized_limit = max(1, int(limit or 1))
        with self._recent_messages_lock:
            messages = list(self._recent_messages.get(normalized_session_id, []))
        if not include_assistant:
            messages = [item for item in messages if item.get("role") != "assistant"]
        return messages[-normalized_limit:]

    def get_last_reply(
        self,
        *,
        session_id: str | None = None,
        contact_query: str | None = None,
    ) -> dict[str, Any] | None:
        store = getattr(self, "_history_store", None)
        if store is None:
            return None
        resolved_contact = self._resolve_history_contact(contact_query)
        return store.get_last_reply(
            session_id=self._history_session_filter(session_id, resolved_contact),
            contact_id=resolved_contact.get("contact_id"),
            contact_query=resolved_contact.get("fallback_query"),
        )

    def search_history(
        self,
        *,
        session_id: str | None = None,
        contact_query: str | None = None,
        query: str | None = None,
        limit: int = 5,
        reply_after_last_outbound: bool = False,
    ) -> list[dict[str, Any]]:
        store = getattr(self, "_history_store", None)
        if store is None:
            return []
        resolved_contact = self._resolve_history_contact(contact_query)
        return store.search_messages(
            session_id=self._history_session_filter(session_id, resolved_contact),
            contact_id=resolved_contact.get("contact_id"),
            contact_query=resolved_contact.get("fallback_query"),
            query=query,
            limit=limit,
            reply_after_last_outbound=reply_after_last_outbound,
        )

    def get_recent_attachments(
        self,
        *,
        session_id: str | None = None,
        contact_query: str | None = None,
        kind: str = "any",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        store = getattr(self, "_history_store", None)
        if store is None:
            return []
        resolved_contact = self._resolve_history_contact(contact_query)
        return store.get_recent_attachments(
            session_id=self._history_session_filter(session_id, resolved_contact),
            contact_id=resolved_contact.get("contact_id"),
            contact_query=resolved_contact.get("fallback_query"),
            kind=kind,
            limit=limit,
        )

    def search_contacts(
        self,
        query: str,
        *,
        target_kind: str = "any",
        limit: int = 5,
        exclude_sender: bool = False,
        sender_id: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_kind = (target_kind or "any").strip().lower()
        _, websocket = self._require_active_connection()

        async def _search() -> list[dict[str, Any]]:
            contacts = await self._load_contacts(websocket)
            if normalized_kind in {"friend", "group"}:
                contacts_pool = tuple(contact for contact in contacts if contact.kind == normalized_kind)
            else:
                contacts_pool = contacts
            if exclude_sender and sender_id:
                contacts_pool = self._exclude_sender_from_contacts(contacts_pool, sender_id)
            matches = match_contacts(query, contacts_pool, limit=limit)
            return [self._serialize_contact_match(match) for match in matches]

        return self._run_coroutine_sync(_search(), timeout_seconds=15.0)

    def send_text(
        self,
        message: str,
        *,
        target_kind: str = "current",
        target_id: int | None = None,
        current_target: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target = self._resolve_runtime_target(target_kind, target_id, current_target)
        _, websocket = self._require_active_connection()

        async def _send() -> dict[str, Any]:
            if target.message_type == "group":
                result = await self._call_action(
                    websocket,
                    action="send_group_msg",
                    params={"group_id": target.group_id, "message": message},
                )
            else:
                result = await self._call_action(
                    websocket,
                    action="send_private_msg",
                    params={"user_id": target.user_id, "message": message},
                )
            self._ensure_action_success(result, action_name="qq.send_text")
            self._record_outbound_history(
                session_id=self._session_id_for_target(target),
                target=target,
                text=message,
                message_type="text",
            )
            return {
                "sent": True,
                "message": message,
                "message_id": self._extract_message_id(result),
                "target": self._serialize_target(target),
                "result": result,
            }

        return self._run_coroutine_sync(_send(), timeout_seconds=15.0)

    def send_file(
        self,
        file_path: str,
        *,
        target_kind: str = "current",
        target_id: int | None = None,
        current_target: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target = self._resolve_runtime_target(target_kind, target_id, current_target)
        _, websocket = self._require_active_connection()

        async def _send() -> dict[str, Any]:
            action = build_file_action(target, file_path)
            result = await self._call_action(
                websocket,
                action=action["action"],
                params=action["params"],
            )
            self._ensure_action_success(result, action_name="qq.send_file")
            self._record_outbound_history(
                session_id=self._session_id_for_target(target),
                target=target,
                text="",
                message_type="file",
                attachments=[self._build_file_attachment(file_path)],
            )
            return {
                "sent": True,
                "path": file_path,
                "message_id": self._extract_message_id(result),
                "target": self._serialize_target(target),
                "result": result,
            }

        return self._run_coroutine_sync(_send(), timeout_seconds=20.0)

    def send_voice(
        self,
        *,
        speech_text: str | None = None,
        audio_path: str | None = None,
        target_kind: str = "current",
        target_id: int | None = None,
        current_target: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target = self._resolve_runtime_target(target_kind, target_id, current_target)
        _, websocket = self._require_active_connection()
        cleanup_path = None
        if audio_path is None:
            voice_result = self._voice_output.synthesize_text(speech_text or "")
            if voice_result.error or not voice_result.audio_path:
                raise RuntimeError(voice_result.error or "QQ voice synthesis failed.")
            audio_path = voice_result.audio_path
            cleanup_path = audio_path

        async def _send() -> dict[str, Any]:
            action = build_voice_action(target, audio_path)
            result = await self._call_action(
                websocket,
                action=action["action"],
                params=action["params"],
            )
            self._ensure_action_success(result, action_name="qq.send_voice")
            self._record_outbound_history(
                session_id=self._session_id_for_target(target),
                target=target,
                text=speech_text or "",
                message_type="audio",
                attachments=[self._build_audio_attachment(audio_path)],
            )
            return {
                "sent": True,
                "speech_text": speech_text or "",
                "audio_path": audio_path,
                "message_id": self._extract_message_id(result),
                "target": self._serialize_target(target),
                "result": result,
            }

        try:
            return self._run_coroutine_sync(_send(), timeout_seconds=20.0)
        finally:
            if cleanup_path:
                self._schedule_cleanup(cleanup_path)

    def _build_channel_runtime(self, inbound: OneBotInboundMessage) -> dict[str, Any]:
        raw_payload = inbound.metadata.get("raw_payload") if isinstance(inbound.metadata, dict) else None
        sender_name = ""
        if isinstance(raw_payload, dict):
            sender = raw_payload.get("sender")
            if isinstance(sender, dict):
                sender_name = str(sender.get("card") or sender.get("nickname") or "").strip()
        finalized_turn = inbound.metadata.get("finalized_turn") if isinstance(inbound.metadata, dict) else None
        finalized_segments = []
        if isinstance(finalized_turn, dict):
            finalized_segments = [
                str(item).strip()
                for item in (finalized_turn.get("message_segments") or [])
                if str(item).strip()
            ]
        recent_user_messages = []
        for item in self.get_recent_messages(session_id=inbound.session_id, limit=6, include_assistant=False):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            recent_user_messages.append(
                {
                    "text": text,
                    "created_at": str(item.get("created_at") or "").strip(),
                    "sender_id": str(item.get("sender_id") or "").strip(),
                }
            )
        return {
            "runtime_id": self._runtime_id,
            "session_id": inbound.session_id,
            "sender_id": inbound.sender_id,
            "sender_name": sender_name,
            "mode": inbound.mode,
            "current_target": self._serialize_target(inbound.target),
            "finalized_turn_segments": finalized_segments,
            "recent_user_messages": recent_user_messages,
        }

    @classmethod
    def _should_send_live_turn_task_ack(
        cls,
        *,
        raw_user_turn_text: str,
        decision,
    ) -> tuple[bool, str]:
        if not bool(getattr(decision, "should_ack_task", False)):
            return False, "disabled_by_decision"

        turn_kind = str(getattr(decision, "turn_kind", "") or "").strip().lower()
        if turn_kind and turn_kind != "execute_task":
            return False, f"non_execution_turn:{turn_kind}"

        task_ack_text = str(getattr(decision, "task_ack_text", "") or "").strip()
        if not task_ack_text:
            return False, "empty_task_ack_text"

        raw_text = str(raw_user_turn_text or "").strip()
        understood_task = str(getattr(decision, "understood_task", "") or "").strip()
        combined = "\n".join(part for part in (raw_text, understood_task) if part).lower()
        if not combined:
            return False, "empty_turn_context"

        has_action_signal = bool(cls._TASK_ACK_ACTION_PATTERN.search(combined))
        has_chat_signal = bool(cls._TASK_ACK_CHAT_PATTERN.search(combined))

        if has_chat_signal and not has_action_signal:
            return False, "light_chat_turn"
        if not has_action_signal and len(raw_text) <= 48:
            return False, "no_action_signal"
        return True, "task_execution_ack"

    @classmethod
    def _sanitize_live_turn_task_ack_text(
        cls,
        *,
        raw_user_turn_text: str,
        decision,
    ) -> tuple[str, str]:
        task_ack_text = str(getattr(decision, "task_ack_text", "") or "").strip()
        if not task_ack_text:
            return "", "empty_task_ack_text"

        raw_text = str(raw_user_turn_text or "").strip()
        understood_task = str(getattr(decision, "understood_task", "") or "").strip()
        combined = "\n".join(part for part in (raw_text, understood_task) if part)
        if not combined:
            return task_ack_text, "no_context"

        is_history_turn = bool(cls._TASK_ACK_HISTORY_PATTERN.search(combined))
        sounds_like_result = bool(cls._TASK_ACK_RESULT_PATTERN.search(task_ack_text))
        if is_history_turn and sounds_like_result:
            return "我先帮你翻一下这段聊天记录，马上回你。", "history_ack_result_sanitized"

        if sounds_like_result and bool(cls._TASK_ACK_ACTION_PATTERN.search(combined)):
            return "收到，我先帮你查一下，马上回你。", "result_like_ack_sanitized"

        return task_ack_text, "unchanged"

    async def _coalesce_inbound(self, websocket, inbound: OneBotInboundMessage) -> OneBotInboundMessage | None:
        window_ms = max(0, int(getattr(self.onebot_cfg, "coalesce_window_ms", 0) or 0))
        if window_ms <= 0:
            prepared = self._prepare_inbound_for_dispatch(inbound)
            if prepared is None:
                return None
            return await self._buffer_live_turn_and_maybe_finalize(websocket, prepared)

        session_id = inbound.session_id
        async with self._coalesce_lock:
            batch = self._pending_inbound_batches.setdefault(session_id, [])
            batch.append(inbound)
            version = self._pending_inbound_versions.get(session_id, 0) + 1
            self._pending_inbound_versions[session_id] = version

        await asyncio.sleep(self._coalesce_wait_seconds(inbound, base_window_ms=window_ms))

        async with self._coalesce_lock:
            if self._pending_inbound_versions.get(session_id) != version:
                return None
            batch = self._pending_inbound_batches.pop(session_id, [])
            self._pending_inbound_versions.pop(session_id, None)
        if not batch:
            return None
        prepared = self._prepare_inbound_for_dispatch(batch[0] if len(batch) == 1 else self._merge_inbound_batch(batch))
        if prepared is None:
            return None
        return await self._buffer_live_turn_and_maybe_finalize(websocket, prepared)

    @staticmethod
    def _merge_inbound_batch(batch: list[OneBotInboundMessage]) -> OneBotInboundMessage:
        latest = batch[-1]
        merged_text = "\n".join(item.text.strip() for item in batch if item.text.strip())
        merged_images: list[OneBotImageAttachment] = []
        for item in batch:
            merged_images.extend(item.image_attachments)
        merged_metadata = dict(latest.metadata)
        merged_metadata["coalesced_messages"] = [item.text for item in batch if item.text.strip()]
        merged_metadata["coalesced_count"] = len(batch)
        if merged_images:
            merged_metadata["coalesced_image_count"] = len(merged_images)
        return replace(latest, text=merged_text.strip(), metadata=merged_metadata, image_attachments=tuple(merged_images))

    def _prepare_inbound_for_dispatch(self, inbound: OneBotInboundMessage) -> OneBotInboundMessage | None:
        text = inbound.text.strip()
        current_images = tuple(
            attachment
            for attachment in inbound.image_attachments
            if attachment.local_path and Path(attachment.local_path).is_file()
        )
        if current_images and self._looks_like_image_inspection_request(text):
            return self._attach_image_context(inbound, current_images, source="current_message")
        if not current_images and self._looks_like_image_inspection_request(text):
            recent_images = self._recent_image_attachments(inbound.session_id)
            if recent_images:
                return self._attach_image_context(inbound, recent_images, source="recent_context")
        if not text:
            return None
        return inbound

    async def _buffer_live_turn_and_maybe_finalize(self, websocket, inbound: OneBotInboundMessage) -> OneBotInboundMessage | None:
        chat_service = getattr(getattr(self, "app", None), "chat_service", None)
        if chat_service is None or not hasattr(chat_service, "capture_live_turn_event"):
            return inbound

        attachment_refs = [
            str(item.get("local_path") or item.get("remote_url") or "")
            for item in self._build_inbound_attachments(inbound)
            if isinstance(item, dict) and str(item.get("local_path") or item.get("remote_url") or "").strip()
        ]
        state = await asyncio.to_thread(
            chat_service.capture_live_turn_event,
            session_id=inbound.session_id,
            text=inbound.text,
            channel="onebot_v11",
            attachment_refs=attachment_refs,
            metadata={
                "sender_id": inbound.sender_id,
                "mode": inbound.mode,
                "onebot_target": self._serialize_target(inbound.target),
                "onebot_metadata": dict(inbound.metadata),
                "onebot_image_attachments": [self._serialize_image_attachment(item) for item in inbound.image_attachments],
                "onebot_audio_attachments": self._build_inbound_attachments(replace(inbound, image_attachments=(), audio_attachments=inbound.audio_attachments)),
            },
        )
        self._append_live_turn_trace(
            inbound.session_id,
            "live_turn_buffered",
            {
                "session_id": inbound.session_id,
                "text": inbound.text,
                "raw_user_turn_text": state.raw_user_turn_text,
                "event_count": state.event_count,
                "version": state.version,
            },
        )
        observed_version = state.version

        while True:
            current_state = chat_service.get_live_turn_state(inbound.session_id)
            if current_state is None or current_state.version != observed_version:
                self._append_live_turn_trace(
                    inbound.session_id,
                    "live_turn_superseded",
                    {
                        "session_id": inbound.session_id,
                        "observed_version": observed_version,
                        "current_version": None if current_state is None else current_state.version,
                    },
                )
                return None
            decision = await asyncio.to_thread(
                chat_service.assess_live_turn,
                session_id=inbound.session_id,
                scope_root=inbound.scope_root,
                runtime_settings={"channel_runtime": self._build_channel_runtime(inbound)},
                typing_active=False,
            )
            self._append_live_turn_trace(
                inbound.session_id,
                "live_turn_assessment",
                {
                    "session_id": inbound.session_id,
                    "version": observed_version,
                    "raw_user_turn_text": current_state.raw_user_turn_text,
                    "event_count": current_state.event_count,
                    "typing_active": current_state.typing_active,
                    "typing_expires_at": (
                        current_state.typing_expires_at.isoformat()
                        if current_state.typing_expires_at is not None
                        else None
                    ),
                    "finalize": decision.finalize,
                    "wait_ms": decision.wait_ms,
                    "reason": decision.reason,
                    "source": decision.source,
                    "ask_followup": decision.ask_followup,
                    "followup_text": decision.followup_text,
                    "understood_task": decision.understood_task,
                    "should_ack_task": decision.should_ack_task,
                    "task_ack_text": decision.task_ack_text,
                },
            )
            if (
                not decision.finalize
                and hasattr(chat_service, "learn_live_turn_habit")
                and decision.reason in {"fragment_wait_extended", "typing_active"}
            ):
                await asyncio.to_thread(
                    chat_service.learn_live_turn_habit,
                    session_id=inbound.session_id,
                    raw_user_turn_text=current_state.raw_user_turn_text,
                    decision=decision,
                )
            if decision.ask_followup and str(decision.followup_text or "").strip():
                await self._send_text_reply(
                    websocket,
                    inbound.target,
                    str(decision.followup_text).strip(),
                    session_id=inbound.session_id,
                )
                if hasattr(chat_service, "mark_live_turn_followup_prompted"):
                    await asyncio.to_thread(
                        chat_service.mark_live_turn_followup_prompted,
                        inbound.session_id,
                        followup_text=str(decision.followup_text).strip(),
                    )
                self._append_live_turn_trace(
                    inbound.session_id,
                    "live_turn_followup_prompted",
                    {
                        "session_id": inbound.session_id,
                        "version": observed_version,
                        "raw_user_turn_text": current_state.raw_user_turn_text,
                        "followup_text": str(decision.followup_text).strip(),
                        "reason": decision.reason,
                        "confidence": decision.confidence,
                    },
                )
            if decision.finalize:
                finalized = await asyncio.to_thread(
                    chat_service.finalize_live_turn,
                    inbound.session_id,
                    expected_version=observed_version,
                    finalize_reason=decision.reason,
                )
                if finalized is None:
                    self._append_live_turn_trace(
                        inbound.session_id,
                        "live_turn_finalize_missed",
                        {
                            "session_id": inbound.session_id,
                            "expected_version": observed_version,
                        },
                    )
                    return None

                self._append_live_turn_trace(
                    inbound.session_id,
                    "live_turn_finalized",
                    {
                        "session_id": inbound.session_id,
                        "turn_id": finalized.turn_id,
                        "raw_user_turn_text": finalized.raw_user_turn_text,
                        "event_count": finalized.event_count,
                        "message_segments": list(finalized.message_segments),
                        "reason": decision.reason,
                        "understood_task": decision.understood_task,
                        "should_ack_task": decision.should_ack_task,
                        "task_ack_text": decision.task_ack_text,
                    },
                )

                task_ack_text, ack_text_reason = self._sanitize_live_turn_task_ack_text(
                    raw_user_turn_text=finalized.raw_user_turn_text,
                    decision=decision,
                )
                should_send_ack, ack_reason = self._should_send_live_turn_task_ack(
                    raw_user_turn_text=finalized.raw_user_turn_text,
                    decision=decision,
                )
                if should_send_ack:
                    try:
                        await self._send_transient_text(
                            websocket,
                            inbound.target,
                            task_ack_text,
                            session_id=inbound.session_id,
                        )
                        self._append_live_turn_trace(
                            inbound.session_id,
                            "live_turn_task_ack_sent",
                            {
                                "session_id": inbound.session_id,
                                "turn_id": finalized.turn_id,
                                "understood_task": decision.understood_task,
                                "task_ack_text": task_ack_text,
                                "ack_reason": ack_reason,
                                "ack_text_reason": ack_text_reason,
                            },
                        )
                    except Exception as exc:  # noqa: BLE001
                        self._append_live_turn_trace(
                            inbound.session_id,
                            "live_turn_task_ack_failed",
                            {
                                "session_id": inbound.session_id,
                                "turn_id": finalized.turn_id,
                                "understood_task": decision.understood_task,
                                "task_ack_text": task_ack_text,
                                "error": str(exc),
                                "ack_text_reason": ack_text_reason,
                            },
                        )
                elif decision.should_ack_task and task_ack_text:
                    self._append_live_turn_trace(
                        inbound.session_id,
                        "live_turn_task_ack_suppressed",
                        {
                            "session_id": inbound.session_id,
                            "turn_id": finalized.turn_id,
                            "understood_task": decision.understood_task,
                            "task_ack_text": task_ack_text,
                            "ack_reason": ack_reason,
                            "ack_text_reason": ack_text_reason,
                        },
                    )

                session_store = getattr(chat_service, "session_store", None)
                if session_store is not None and hasattr(session_store, "record_finalized_turn"):
                    await asyncio.to_thread(
                        session_store.record_finalized_turn,
                        inbound.session_id,
                        finalized.turn_id,
                        finalized.raw_user_turn_text,
                    )

                return self._build_inbound_from_live_turn(inbound, finalized, decision)
            wait_ms = max(50, int(decision.wait_ms or 0))
            await asyncio.sleep(wait_ms / 1000.0)
        return None

    def _append_live_turn_trace(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
        chat_service = getattr(getattr(self, "app", None), "chat_service", None)
        session_store = getattr(chat_service, "session_store", None)
        if session_store is None or not hasattr(session_store, "get"):
            return
        session = session_store.get(session_id)
        kernel = getattr(session, "kernel", None) if session is not None else None
        trace_store = getattr(kernel, "trace_store", None)
        if trace_store is None or not hasattr(trace_store, "append"):
            return
        try:
            trace_store.append(event_type, payload)
        except Exception:
            return

    def _build_inbound_from_live_turn(
            self,
            inbound: OneBotInboundMessage,
            finalized_turn,
            decision,
    ) -> OneBotInboundMessage:
        finalize_reason = str(getattr(decision, "reason", "") or "").strip()
        turn_kind = str(getattr(decision, "turn_kind", "") or "").strip()
        understood_task = str(getattr(decision, "understood_task", "") or "").strip()
        should_ack_task = bool(getattr(decision, "should_ack_task", False))
        forced_mode = self._forced_mode_for_live_turn(
            turn_kind=turn_kind,
            understood_task=understood_task,
            should_ack_task=should_ack_task,
        )
        merged_metadata = dict(inbound.metadata)
        merged_metadata["live_turn"] = {
            "event_count": finalized_turn.event_count,
            "turn_id": finalized_turn.turn_id,
            "finalize_reason": finalize_reason,
            "turn_kind": turn_kind,
            "understood_task": understood_task,
            "should_ack_task": should_ack_task,
            "forced_mode": forced_mode,
        }
        merged_metadata["finalized_turn"] = {
            "turn_id": finalized_turn.turn_id,
            "raw_user_turn_text": finalized_turn.raw_user_turn_text,
            "event_count": finalized_turn.event_count,
            "message_segments": list(finalized_turn.message_segments),
            "finalize_reason": finalize_reason,
            "turn_kind": turn_kind,
            "understood_task": understood_task,
        }

        image_attachments: list[OneBotImageAttachment] = []
        audio_attachments: list[OneBotAudioAttachment] = []
        # 你原来的附件恢复逻辑保留

        return replace(
            inbound,
            text=finalized_turn.raw_user_turn_text,
            mode=forced_mode or inbound.mode,
            metadata=merged_metadata,
            image_attachments=tuple(image_attachments or inbound.image_attachments),
            audio_attachments=tuple(audio_attachments or inbound.audio_attachments),
        )

    @staticmethod
    def _forced_mode_for_live_turn(
        *,
        turn_kind: str,
        understood_task: str,
        should_ack_task: bool,
    ) -> str | None:
        normalized_kind = str(turn_kind or "").strip().lower()
        if normalized_kind in {"execute_task", "memory_update", "instruction_update", "direct_reply"}:
            return "agent"
        if should_ack_task and str(understood_task or "").strip():
            return "agent"
        if str(understood_task or "").strip() and normalized_kind not in {"", "chat", "uncertain"}:
            return "agent"
        return None

    def _coalesce_wait_seconds(self, inbound: OneBotInboundMessage, *, base_window_ms: int) -> float:
        extra_ms = 0
        if self._looks_like_fragmented_turn(inbound):
            extra_ms += max(0, int(getattr(self.onebot_cfg, "coalesce_short_message_extra_ms", 0) or 0))
        if inbound.image_attachments or inbound.audio_attachments:
            extra_ms += max(0, int(getattr(self.onebot_cfg, "coalesce_attachment_extra_ms", 0) or 0))
        total_ms = max(0, int(base_window_ms)) + extra_ms
        return total_ms / 1000.0

    @staticmethod
    def _looks_like_fragmented_turn(inbound: OneBotInboundMessage) -> bool:
        text = inbound.text.strip()
        if not text:
            return bool(inbound.image_attachments or inbound.audio_attachments)
        compact_text = re.sub(r"\s+", "", text)
        if len(compact_text) <= 14:
            return True
        if compact_text[-1] not in "。！？!?~～…）)]】》\"'":
            return True
        fragment_markers = ("还有", "然后", "另外", "补充", "顺便", "以及", "再", "还有个")
        return any(marker in compact_text for marker in fragment_markers)

    def _build_reply_segments(self, text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []

        max_chars = max(40, int(getattr(self.onebot_cfg, "reply_segment_max_chars", 110) or 110))
        max_count = max(1, int(getattr(self.onebot_cfg, "reply_segment_max_count", 3) or 3))
        if max_count <= 1 or len(normalized) <= min(36, max_chars):
            return [normalized]

        blocks = [block.strip() for block in normalized.replace("\r\n", "\n").split("\n") if block.strip()]
        if 1 < len(blocks) <= max_count and all(len(block) <= max(max_chars, 140) for block in blocks):
            return blocks
        units: list[str] = []
        for block in blocks:
            if self._should_keep_reply_block_intact(block, max_chars=max_chars):
                units.append(block)
                continue
            sentences = [part.strip() for part in re.split(r"(?<=[。！？!?~～…])", block) if part.strip()]
            if not sentences:
                continue
            if len(sentences) == 1 and len(sentences[0]) > max_chars:
                units.extend(self._split_long_reply_unit(sentences[0], max_chars=max_chars))
                continue
            units.extend(sentences)

        if not units:
            return [normalized]

        segments: list[str] = []
        current = ""
        for unit in units:
            joiner = self._reply_joiner(current, unit)
            candidate = f"{current}{joiner}{unit}".strip() if current else unit
            if current and len(candidate) > max_chars:
                segments.append(current.strip())
                current = unit
                continue
            current = candidate
        if current.strip():
            segments.append(current.strip())

        collapsed = [segment for segment in segments if segment]
        if not collapsed:
            return [normalized]
        if len(collapsed) <= max_count:
            return collapsed
        head = collapsed[: max_count - 1]
        tail = "\n".join(collapsed[max_count - 1 :]).strip()
        return [*head, tail] if head else [tail]

    @staticmethod
    def _should_keep_reply_block_intact(block: str, *, max_chars: int) -> bool:
        stripped = block.strip()
        if not stripped:
            return True
        if len(stripped) <= max_chars // 2:
            return False
        if ":\\" in stripped or ":/" in stripped or stripped.startswith("C:\\"):
            return True
        if stripped.startswith(("```", "`", "-", "*", "1.", "2.", "3.", "4.")):
            return True
        return False

    @staticmethod
    def _split_long_reply_unit(unit: str, *, max_chars: int) -> list[str]:
        parts = [part.strip() for part in re.split(r"(?<=[，,；;：:])", unit) if part.strip()]
        if not parts:
            return [unit]
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{part}" if current else part
            if current and len(candidate) > max_chars:
                chunks.append(current.strip())
                current = part
                continue
            current = candidate
        if current.strip():
            chunks.append(current.strip())
        return chunks or [unit]

    @staticmethod
    def _reply_joiner(current: str, unit: str) -> str:
        if not current:
            return ""
        structured = unit.startswith(("```", "`", "-", "*", "1.", "2.", "3.", "4.")) or ":\\" in unit or ":/" in unit
        if structured or current.endswith(("：", ":", "\n")):
            return "\n"
        return ""

    def _reply_segment_pause_seconds(self, segment: str) -> float:
        base_ms = max(0, int(getattr(self.onebot_cfg, "reply_segment_delay_ms", 650) or 0))
        if base_ms <= 0:
            return 0.0
        length_bonus_ms = min(600, max(0, len(segment.strip()) - 18) * 8)
        return (base_ms + length_bonus_ms) / 1000.0

    def _attach_image_context(
        self,
        inbound: OneBotInboundMessage,
        attachments: tuple[OneBotImageAttachment, ...],
        *,
        source: str,
    ) -> OneBotInboundMessage:
        image_paths = [
            str(Path(attachment.local_path).resolve())
            for attachment in attachments
            if attachment.local_path and Path(attachment.local_path).is_file()
        ]
        if not image_paths:
            return inbound

        user_text = inbound.text.strip() or "What is in this image?"
        if self._looks_like_image_text_request(user_text):
            action_hint = "Please read/OCR the text in the image using these local image file paths."
        else:
            action_hint = "Please describe this image using these local image file paths."
        image_lines = "\n".join(f"Image path {index + 1}: {path}" for index, path in enumerate(image_paths[:3]))
        augmented_text = f"{user_text}\n\n[QQ image context]\n{action_hint}\n{image_lines}"

        metadata = dict(inbound.metadata)
        metadata["qq_image_context"] = {
            "source": source,
            "image_paths": image_paths[:3],
            "image_count": len(image_paths),
        }
        return replace(inbound, text=augmented_text, mode="agent", metadata=metadata, image_attachments=attachments)

    def _recent_image_attachments(self, session_id: str, *, limit: int = 3) -> tuple[OneBotImageAttachment, ...]:
        normalized_session_id = str(session_id).strip()
        if not normalized_session_id:
            return ()
        attachments: list[OneBotImageAttachment] = []
        with self._recent_messages_lock:
            history = list(self._recent_messages.get(normalized_session_id, []))
        for record in reversed(history):
            if record.get("role") != "user":
                continue
            for item in reversed(record.get("image_attachments") or []):
                if not isinstance(item, dict):
                    continue
                local_path = item.get("local_path")
                if not isinstance(local_path, str) or not Path(local_path).is_file():
                    continue
                attachments.append(
                    OneBotImageAttachment(
                        local_path=local_path,
                        remote_url=item.get("remote_url") if isinstance(item.get("remote_url"), str) else None,
                        file_id=item.get("file_id") if isinstance(item.get("file_id"), str) else None,
                        mime_type=item.get("mime_type") if isinstance(item.get("mime_type"), str) else None,
                        summary=item.get("summary") if isinstance(item.get("summary"), str) else None,
                        image_type=item.get("image_type") if isinstance(item.get("image_type"), str) else None,
                        is_emoji=bool(item.get("is_emoji", False)),
                    )
                )
                if len(attachments) >= limit:
                    return tuple(reversed(attachments))
        return tuple(reversed(attachments))

    @staticmethod
    def _looks_like_image_inspection_request(text: str) -> bool:
        normalized = str(text or "").strip().lower()
        if not normalized:
            return False
        image_terms = (
            "image",
            "picture",
            "photo",
            "screenshot",
            "pic",
            "img",
            "png",
            "jpg",
            "\u56fe",
            "\u56fe\u7247",
            "\u7167\u7247",
            "\u622a\u56fe",
            "\u8868\u60c5\u5305",
        )
        action_terms = (
            "what is",
            "what's",
            "what in",
            "describe",
            "caption",
            "look at",
            "analyze",
            "analyse",
            "inspect",
            "read",
            "ocr",
            "\u8fd9\u662f\u4ec0\u4e48",
            "\u662f\u4ec0\u4e48",
            "\u6709\u4ec0\u4e48",
            "\u5199\u4e86\u4ec0\u4e48",
            "\u770b\u770b",
            "\u770b\u4e00\u4e0b",
            "\u5e2e\u6211\u770b",
            "\u63cf\u8ff0",
            "\u8bc6\u522b",
            "\u8bfb\u4e00\u4e0b",
            "\u6587\u5b57",
            "\u5185\u5bb9",
            "\u610f\u601d",
        )
        deictic_terms = ("\u8fd9\u4e2a", "\u8fd9\u5f20", "\u521a\u624d", "\u4e0a\u9762", "\u91cc\u9762", "\u56fe\u91cc", "\u56fe\u4e0a", "this", "that")
        if any(term in normalized for term in image_terms) and any(term in normalized for term in action_terms):
            return True
        return any(term in normalized for term in deictic_terms) and any(term in normalized for term in action_terms)

    @staticmethod
    def _looks_like_image_text_request(text: str) -> bool:
        normalized = str(text or "").strip().lower()
        return any(
            term in normalized
            for term in (
                "ocr",
                "text",
                "read",
                "\u6587\u5b57",
                "\u5199\u4e86\u4ec0\u4e48",
                "\u8bfb\u4e00\u4e0b",
                "\u8bc6\u522b\u6587\u5b57",
                "\u63d0\u53d6\u6587\u5b57",
            )
        )

    def _set_active_connection(self, loop, websocket) -> None:
        with self._connection_lock:
            self._active_loop = loop
            self._active_websocket = websocket

    def _clear_active_connection(self, websocket) -> None:
        with self._connection_lock:
            if self._active_websocket is websocket:
                self._active_loop = None
                self._active_websocket = None

    def _require_active_connection(self):
        with self._connection_lock:
            loop = self._active_loop
            websocket = self._active_websocket
        if loop is None or websocket is None:
            raise RuntimeError("OneBot connection is not active.")
        return loop, websocket

    def _run_coroutine_sync(self, coroutine, *, timeout_seconds: float) -> Any:
        loop, _ = self._require_active_connection()
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise RuntimeError("OneBot action timed out.") from exc

    def _schedule_cleanup(self, file_path: str) -> None:
        loop, _ = self._require_active_connection()
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self._voice_output.cleanup_temp_file_later(file_path))
        )

    def _remember_recent_message(
        self,
        session_id: str,
        *,
        role: str,
        text: str,
        sender_id: str | None = None,
        image_attachments: tuple[OneBotImageAttachment, ...] = (),
    ) -> None:
        normalized_session_id = str(session_id).strip()
        normalized_text = str(text).strip()
        serialized_images = [self._serialize_image_attachment(attachment) for attachment in image_attachments]
        if not normalized_session_id or (not normalized_text and not serialized_images):
            return
        record = {
            "role": role,
            "text": normalized_text,
            "sender_id": sender_id,
            "created_at": datetime.now(UTC).isoformat(),
        }
        if serialized_images:
            record["image_attachments"] = serialized_images
        limit = max(4, int(getattr(self.onebot_cfg, "recent_message_limit", 24) or 24))
        with self._recent_messages_lock:
            history = self._recent_messages.setdefault(normalized_session_id, [])
            history.append(record)
            if len(history) > limit:
                del history[:-limit]

    def _record_inbound_history(self, inbound: OneBotInboundMessage) -> None:
        store = getattr(self, "_history_store", None)
        if store is None:
            return
        store.record_message(
            session_id=inbound.session_id,
            direction="inbound",
            message_type=self._message_type_from_inbound(inbound),
            text=inbound.text,
            sender_id=inbound.sender_id,
            contact_id=self._contact_id_for_target(inbound.target),
            contact_name=self._contact_name_for_session(inbound.session_id),
            attachments=self._build_inbound_attachments(inbound),
            metadata={"platform": "onebot_v11"},
            created_at=datetime.now(UTC).isoformat(),
        )

    def _record_outbound_history(
        self,
        *,
        session_id: str | None,
        target: OneBotTarget,
        text: str,
        message_type: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> None:
        store = getattr(self, "_history_store", None)
        normalized_session_id = str(session_id or "").strip()
        if store is None or not normalized_session_id:
            return
        store.record_message(
            session_id=normalized_session_id,
            direction="outbound",
            message_type=message_type,
            text=text,
            contact_id=self._contact_id_for_target(target),
            contact_name=self._contact_name_for_session(normalized_session_id),
            attachments=attachments or [],
            metadata={"platform": "onebot_v11"},
            created_at=datetime.now(UTC).isoformat(),
        )

    def _resolve_history_contact(self, contact_query: str | None) -> dict[str, str | None]:
        normalized_query = str(contact_query or "").strip()
        if not normalized_query:
            return {"contact_id": None, "fallback_query": None}
        if normalized_query.isdigit():
            return {"contact_id": normalized_query, "fallback_query": None}
        try:
            candidates = self.search_contacts(normalized_query, target_kind="any", limit=3)
        except Exception:  # noqa: BLE001
            return {"contact_id": None, "fallback_query": normalized_query}
        if candidates:
            top_candidate = candidates[0]
            target_id = top_candidate.get("target_id")
            if isinstance(target_id, int):
                return {"contact_id": str(target_id), "fallback_query": normalized_query}
            if isinstance(target_id, str) and target_id.strip():
                return {"contact_id": target_id.strip(), "fallback_query": normalized_query}
        return {"contact_id": None, "fallback_query": normalized_query}

    @staticmethod
    def _history_session_filter(session_id: str | None, resolved_contact: dict[str, str | None]) -> str | None:
        if resolved_contact.get("contact_id"):
            return None
        normalized_session_id = str(session_id or "").strip()
        return normalized_session_id or None

    def _remember_proxy_thread(
        self,
        *,
        source_session_id: str,
        contact: OneBotContact,
        outbound_message: str,
    ) -> None:
        target_session_id = self._target_session_id(contact)
        if not target_session_id:
            return
        if not hasattr(self, "_proxy_threads_lock"):
            self._proxy_threads_lock = Lock()
        if not hasattr(self, "_proxy_threads"):
            self._proxy_threads = {}
        with self._proxy_threads_lock:
            self._proxy_threads[target_session_id] = {
                "source_session_id": source_session_id,
                "contact_name": contact.name,
                "target_kind": contact.kind,
                "target_id": contact.target_id,
                "updated_at": datetime.now(UTC),
            }
        app = getattr(self, "app", None)
        chat_service = getattr(app, "chat_service", None)
        if chat_service is not None and hasattr(chat_service, "record_external_event"):
            chat_service.record_external_event(
                source_session_id,
                f'QQ代发已发送给{contact.kind}“{contact.name}”：{outbound_message}',
            )

        self._record_outbound_history(
            session_id=target_session_id,
            target=OneBotTarget(
                message_type="group" if contact.kind == "group" else "private",
                user_id=None if contact.kind == "group" else contact.target_id,
                group_id=contact.target_id if contact.kind == "group" else None,
            ),
            text=outbound_message,
            message_type="text",
        )

    def _sync_proxy_reply_context(self, inbound: OneBotInboundMessage) -> None:
        with self._proxy_threads_lock:
            linked = dict(self._proxy_threads.get(inbound.session_id, {}))
            if linked:
                linked["updated_at"] = datetime.now(UTC)
                self._proxy_threads[inbound.session_id] = linked
        if not linked:
            return

        source_session_id = str(linked.get("source_session_id", "")).strip()
        contact_name = str(linked.get("contact_name", "")).strip() or inbound.sender_id
        if not source_session_id:
            return

        note = self._build_proxy_reply_note(contact_name, inbound)
        if not note:
            return
        self.app.chat_service.record_external_event(source_session_id, note)

    @staticmethod
    def _target_session_id(contact: OneBotContact) -> str | None:
        if contact.kind == "group":
            return f"onebot_group_{contact.target_id}"
        if contact.kind == "friend":
            return f"onebot_private_{contact.target_id}"
        return None

    @staticmethod
    def _build_proxy_reply_note(contact_name: str, inbound: OneBotInboundMessage) -> str:
        parts: list[str] = []
        normalized_text = inbound.text.strip()
        if normalized_text:
            parts.append(f"文字：{normalized_text}")
        if inbound.image_attachments:
            image_count = len(inbound.image_attachments)
            parts.append(f"图片：{image_count}张")
        if inbound.audio_attachments and not normalized_text:
            parts.append(f"语音：{len(inbound.audio_attachments)}条")
        if not parts:
            return ""
        return f'QQ代发对象“{contact_name}”回复了你代发的消息。' + " ".join(parts)

    @staticmethod
    def _message_type_from_inbound(inbound: OneBotInboundMessage) -> str:
        if inbound.audio_attachments and not inbound.text.strip() and not inbound.image_attachments:
            return "audio"
        if inbound.image_attachments and not inbound.text.strip() and not inbound.audio_attachments:
            return "image"
        if inbound.image_attachments or inbound.audio_attachments:
            return "mixed"
        return "text"

    def _build_inbound_attachments(self, inbound: OneBotInboundMessage) -> list[dict[str, Any]]:
        attachments: list[dict[str, Any]] = []
        for attachment in inbound.image_attachments:
            attachments.append(
                {
                    "kind": "image",
                    "local_path": attachment.local_path,
                    "remote_url": attachment.remote_url,
                    "file_name": Path(str(attachment.local_path or attachment.file_id or "")).name or attachment.file_id,
                    "mime_type": attachment.mime_type,
                }
            )
        for attachment in inbound.audio_attachments:
            attachments.append(
                {
                    "kind": "audio",
                    "local_path": attachment.local_path,
                    "remote_url": attachment.remote_url,
                    "file_name": Path(str(attachment.local_path or "")).name or None,
                    "mime_type": attachment.mime_type,
                }
            )
        return attachments

    @staticmethod
    def _build_file_attachment(file_path: str) -> dict[str, Any]:
        target = Path(file_path)
        return {
            "kind": "file",
            "local_path": str(target),
            "remote_url": None,
            "file_name": target.name,
            "mime_type": None,
        }

    @staticmethod
    def _build_audio_attachment(audio_path: str) -> dict[str, Any]:
        target = Path(audio_path)
        return {
            "kind": "audio",
            "local_path": str(target),
            "remote_url": None,
            "file_name": target.name,
            "mime_type": None,
        }

    @staticmethod
    def _session_id_for_target(target: OneBotTarget) -> str | None:
        if target.message_type == "group" and isinstance(target.group_id, int):
            return f"onebot_group_{target.group_id}"
        if target.message_type == "private" and isinstance(target.user_id, int):
            return f"onebot_private_{target.user_id}"
        return None

    @staticmethod
    def _contact_id_for_target(target: OneBotTarget) -> str | None:
        if target.message_type == "group" and isinstance(target.group_id, int):
            return str(target.group_id)
        if target.message_type == "private" and isinstance(target.user_id, int):
            return str(target.user_id)
        return None

    def _contact_name_for_session(self, session_id: str | None) -> str | None:
        normalized_session_id = str(session_id or "").strip()
        if not normalized_session_id:
            return None
        proxy_threads = getattr(self, "_proxy_threads", {})
        proxy_lock = getattr(self, "_proxy_threads_lock", None)
        if proxy_lock is not None:
            with proxy_lock:
                linked = proxy_threads.get(normalized_session_id)
                if isinstance(linked, dict):
                    contact_name = str(linked.get("contact_name", "")).strip()
                    if contact_name:
                        return contact_name
        elif isinstance(proxy_threads, dict):
            linked = proxy_threads.get(normalized_session_id)
            if isinstance(linked, dict):
                contact_name = str(linked.get("contact_name", "")).strip()
                if contact_name:
                    return contact_name
        cached = self._contact_cache[1] if getattr(self, "_contact_cache", None) is not None else ()
        for contact in cached:
            if normalized_session_id == self._target_session_id(contact):
                return contact.name
        return None

    @staticmethod
    def _serialize_image_attachment(attachment: OneBotImageAttachment) -> dict[str, Any]:
        return {
            "local_path": attachment.local_path,
            "remote_url": attachment.remote_url,
            "file_id": attachment.file_id,
            "mime_type": attachment.mime_type,
            "summary": attachment.summary,
            "image_type": attachment.image_type,
            "is_emoji": attachment.is_emoji,
        }

    @staticmethod
    def _extract_message_id(result: dict[str, Any]) -> int | None:
        data = result.get("data")
        if isinstance(data, dict):
            message_id = data.get("message_id")
            if isinstance(message_id, int):
                return message_id
            if isinstance(message_id, str) and message_id.isdigit():
                return int(message_id)
        return None

    @staticmethod
    def _serialize_target(target: OneBotTarget) -> dict[str, Any]:
        return {
            "message_type": target.message_type,
            "user_id": target.user_id,
            "group_id": target.group_id,
        }

    @staticmethod
    def _serialize_contact_match(match) -> dict[str, Any]:
        return {
            "target_id": match.contact.target_id,
            "name": match.contact.name,
            "kind": match.contact.kind,
            "aliases": list(match.contact.aliases),
            "score": match.score,
            "reason": match.reason,
        }

    @staticmethod
    def _ensure_action_success(result: dict[str, Any], *, action_name: str) -> None:
        status_ok = str(result.get("status", "")).lower() == "ok"
        retcode = int(result.get("retcode", 0) or 0)
        if status_ok and retcode == 0:
            return
        failure_message = str(result.get("message") or result.get("wording") or result.get("msg") or "unknown error")
        raise RuntimeError(f"{action_name} failed: {failure_message}")

    @staticmethod
    def _resolve_runtime_target(
        target_kind: str,
        target_id: int | None,
        current_target: dict[str, Any] | None,
    ) -> OneBotTarget:
        normalized_kind = (target_kind or "current").strip().lower()
        if normalized_kind == "current":
            if not isinstance(current_target, dict):
                raise RuntimeError("Current QQ target is unavailable.")
            message_type = str(current_target.get("message_type", "")).strip().lower()
            if message_type == "group":
                group_id = current_target.get("group_id")
                if isinstance(group_id, int):
                    return OneBotTarget(message_type="group", group_id=group_id)
            elif message_type == "private":
                user_id = current_target.get("user_id")
                if isinstance(user_id, int):
                    return OneBotTarget(message_type="private", user_id=user_id)
            raise RuntimeError("Current QQ target is incomplete.")
        if normalized_kind == "group" and isinstance(target_id, int):
            return OneBotTarget(message_type="group", group_id=target_id)
        if normalized_kind == "friend" and isinstance(target_id, int):
            return OneBotTarget(message_type="private", user_id=target_id)
        raise RuntimeError("Explicit QQ target_kind/target_id is invalid.")

    def _resolve_inbound(self, inbound: OneBotInboundMessage) -> OneBotInboundMessage | None:
        if inbound.image_attachments:
            inbound = replace(inbound, image_attachments=self._resolve_image_attachments(inbound.image_attachments))
        if inbound.text.strip():
            return inbound
        if inbound.image_attachments and not inbound.audio_attachments:
            return inbound
        if not inbound.audio_attachments or not self._voice_input.enabled:
            return inbound if inbound.image_attachments else None

        transcription = self._voice_input.transcribe_onebot_attachments(inbound.audio_attachments)
        if not transcription.ok:
            if transcription.error:
                print(f"[onebot] voice transcription failed: {transcription.error}")
            return inbound if inbound.image_attachments else None

        metadata = dict(inbound.metadata)
        metadata["voice_input"] = {
            "provider": transcription.provider,
            "source_path": transcription.source_path,
            "transcribed_text": transcription.text,
        }
        return replace(inbound, text=transcription.text, metadata=metadata)

    def _resolve_image_attachments(
        self,
        attachments: tuple[OneBotImageAttachment, ...],
    ) -> tuple[OneBotImageAttachment, ...]:
        resolved: list[OneBotImageAttachment] = []
        for attachment in attachments:
            try:
                resolved.append(self._resolve_image_attachment(attachment))
            except Exception as exc:  # noqa: BLE001
                print(f"[onebot] image attachment resolution failed: {exc}")
                resolved.append(attachment)
        return tuple(resolved)

    def _resolve_image_attachment(self, attachment: OneBotImageAttachment) -> OneBotImageAttachment:
        if attachment.local_path:
            local_path = self._cache_local_image_attachment(attachment.local_path, attachment=attachment)
            if local_path:
                return replace(attachment, local_path=local_path)
        if attachment.remote_url:
            local_path = self._download_image_attachment(attachment)
            if local_path:
                return replace(attachment, local_path=local_path)
        return attachment

    def _cache_local_image_attachment(self, raw_path: str, *, attachment: OneBotImageAttachment) -> str | None:
        source = Path(raw_path).expanduser()
        if not source.is_absolute():
            source = source.resolve()
        if not source.is_file():
            return None
        workspace_root = self._workspace_root()
        try:
            source.relative_to(workspace_root)
            return str(source)
        except ValueError:
            pass

        cache_dir = self._onebot_image_cache_dir()
        digest_input = f"{source}|{source.stat().st_mtime_ns}|{source.stat().st_size}".encode("utf-8", errors="ignore")
        digest = hashlib.sha256(digest_input).hexdigest()[:20]
        suffix = source.suffix.lower() or self._extension_for_image_attachment(attachment)
        target = cache_dir / f"{digest}{suffix}"
        if not target.is_file():
            shutil.copyfile(source, target)
        return str(target)

    def _download_image_attachment(self, attachment: OneBotImageAttachment) -> str | None:
        remote_url = str(attachment.remote_url or "").strip()
        if not remote_url.startswith(("http://", "https://")):
            return None
        cache_dir = self._onebot_image_cache_dir()
        digest = hashlib.sha256(remote_url.encode("utf-8", errors="ignore")).hexdigest()[:24]
        suffix = self._extension_for_image_attachment(attachment)
        target = cache_dir / f"{digest}{suffix}"
        if target.is_file():
            return str(target)

        request = Request(remote_url, headers={"User-Agent": "local-agent-onebot-image-fetcher/1.0"})
        with urlopen(request, timeout=15) as response:
            data = response.read()
        if not data:
            return None
        target.write_bytes(data)
        return str(target)

    def _onebot_image_cache_dir(self) -> Path:
        cache_dir = self._workspace_root() / "data" / "onebot_images"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @staticmethod
    def _qq_history_db_path(config_path: str, memory_db_path: str) -> Path:
        config_root = Path(config_path).resolve().parent
        target = Path(memory_db_path)
        if not target.is_absolute():
            target = (config_root / target).resolve()
        if target.suffix:
            return target.with_name("qq_history.sqlite3")
        return target / "qq_history.sqlite3"

    def _workspace_root(self) -> Path:
        raw_root = getattr(getattr(self.app, "config", None), "agent", None)
        workspace_root = getattr(raw_root, "workspace_root", None)
        return Path(str(workspace_root or ".")).resolve()

    @staticmethod
    def _extension_for_image_attachment(attachment: OneBotImageAttachment) -> str:
        mime_type = str(attachment.mime_type or "").lower()
        if "png" in mime_type:
            return ".png"
        if "webp" in mime_type:
            return ".webp"
        if "gif" in mime_type:
            return ".gif"
        if "bmp" in mime_type:
            return ".bmp"
        if "jpeg" in mime_type or "jpg" in mime_type:
            return ".jpg"

        for candidate in (attachment.remote_url, attachment.file_id, attachment.local_path):
            parsed_path = urlparse(str(candidate or "")).path
            suffix = Path(parsed_path).suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
                return suffix
        return ".jpg"

    async def _maybe_handle_proxy_send(self, websocket, inbound: OneBotInboundMessage) -> bool:
        self._cleanup_pending_proxy_selections()
        access_policy = build_onebot_access_policy(
            inbound.sender_id,
            self.onebot_cfg.full_access_user_ids,
            self.onebot_cfg.owner_user_ids,
            self.onebot_cfg.owner_display_name,
        )
        pending = self._pending_proxy_selection.get(inbound.session_id)
        domain_agent = self._qq_domain_agent()
        send_request = self._decide_proxy_send_request(inbound, domain_agent=domain_agent)

        if pending is not None and pending.requester_id == inbound.sender_id:
            selection_decision = domain_agent.classify_proxy_selection(pending=pending, user_text=inbound.text)
            if selection_decision.action == "cancel":
                self._pending_proxy_selection.pop(inbound.session_id, None)
                await self._send_text_reply(websocket, inbound.target, "\u5df2\u53d6\u6d88\u8fd9\u6b21\u4ee3\u53d1\u3002", session_id=inbound.session_id)
                return True

            selected = selection_decision.selected
            if selected is not None:
                self._pending_proxy_selection.pop(inbound.session_id, None)
                confirmation = await self._send_proxy_message(websocket, selected.contact, pending.request, inbound.session_id)
                await self._send_text_reply(websocket, inbound.target, confirmation, session_id=inbound.session_id)
                return True

            if inbound.text.strip().isdigit() and send_request is None:
                prompt = build_proxy_selection_prompt(pending.request, list(pending.candidates))
                await self._send_text_reply(websocket, inbound.target, f"\u6ca1\u6709\u8fd9\u4e2a\u7f16\u53f7\uff0c\u4f60\u53ef\u4ee5\u91cd\u9009\uff1a\n{prompt}", session_id=inbound.session_id)
                return True

        if send_request is None:
            return False

        if not access_policy.get("allow_local_tools", False):
            await self._send_text_reply(websocket, inbound.target, "\u53ea\u6709\u7ba1\u7406\u5458\u53ef\u4ee5\u8ba9\u6211\u4ee3\u53d1 QQ \u6d88\u606f\u3002", session_id=inbound.session_id)
            return True

        contacts = await self._load_contacts(websocket)
        matches = match_contacts(
            send_request.recipient_query,
            self._exclude_sender_from_contacts(contacts, inbound.sender_id),
        )

        if not matches:
            await self._send_text_reply(
                websocket,
                inbound.target,
                f"\u6ca1\u627e\u5230\u300c{send_request.recipient_query}\u300d\u7684\u8054\u7cfb\u4eba\u6216\u7fa4\uff0c\u4f60\u53ef\u4ee5\u6362\u4e2a\u540d\u5b57\u8bd5\u8bd5\u3002",
                session_id=inbound.session_id,
            )
            return True

        if is_confident_unique_match(matches):
            confirmation = await self._send_proxy_message(websocket, matches[0].contact, send_request, inbound.session_id)
            await self._send_text_reply(websocket, inbound.target, confirmation, session_id=inbound.session_id)
            return True

        candidates = tuple(matches[:3])
        self._pending_proxy_selection[inbound.session_id] = PendingProxySelection(
            requester_id=inbound.sender_id,
            session_id=inbound.session_id,
            request=send_request,
            candidates=candidates,
            created_at=datetime.now(UTC),
        )
        prompt = build_proxy_selection_prompt(send_request, list(candidates))
        await self._send_text_reply(websocket, inbound.target, prompt, session_id=inbound.session_id)
        return True

    def _decide_proxy_send_request(
        self,
        inbound: OneBotInboundMessage,
        *,
        domain_agent: QQDomainAgent | None = None,
    ) -> OneBotProxySendRequest | None:
        domain_agent = domain_agent or self._qq_domain_agent()
        try:
            recent_messages = self.get_recent_messages(
                session_id=inbound.session_id,
                limit=6,
                include_assistant=True,
            )
            return domain_agent.classify_proxy_send_request(
                user_text=inbound.text,
                recent_messages=recent_messages,
                confidence_threshold=0.82,
            )
        except Exception:
            return None

    def _qq_domain_agent(self) -> QQDomainAgent:
        return QQDomainAgent(
            getattr(self, "_llm_client", None),
            assistant_aliases=self._assistant_recipient_aliases(),
        )

    def _assistant_recipient_aliases(self) -> tuple[str, ...]:
        aliases: list[str] = []
        app = getattr(self, "app", None)
        agent_config = getattr(getattr(app, "config", None), "agent", None)
        for candidate in (
            getattr(agent_config, "system_name", None),
            getattr(agent_config, "persona_name", None),
        ):
            if isinstance(candidate, str):
                stripped = candidate.strip()
                if stripped and stripped not in aliases:
                    aliases.append(stripped)
        configured_aliases = getattr(agent_config, "assistant_aliases", None)
        if isinstance(configured_aliases, list):
            for candidate in configured_aliases:
                if not isinstance(candidate, str):
                    continue
                stripped = candidate.strip()
                if stripped and stripped not in aliases:
                    aliases.append(stripped)
        return tuple(aliases)

    async def _load_contacts(self, websocket) -> tuple[OneBotContact, ...]:
        now = datetime.now().astimezone()
        if self._contact_cache is not None:
            cached_at, contacts = self._contact_cache
            if now - cached_at <= timedelta(minutes=2):
                return contacts

        friend_result = await self._call_action(websocket, action="get_friend_list", params={})
        group_result = await self._call_action(websocket, action="get_group_list", params={})
        friend_data = friend_result.get("data") if isinstance(friend_result.get("data"), list) else []
        group_data = group_result.get("data") if isinstance(group_result.get("data"), list) else []
        contacts = build_contacts(friend_data, group_data)
        self._contact_cache = (now, contacts)
        return contacts

    async def _send_proxy_message(
        self,
        websocket,
        contact: OneBotContact,
        request: OneBotProxySendRequest,
        source_session_id: str,
    ) -> str:
        outbound_message = await asyncio.to_thread(self._rewrite_proxy_message, contact, request, source_session_id)
        if contact.kind == "group":
            result = await self._call_action(
                websocket,
                action="send_group_msg",
                params={"group_id": contact.target_id, "message": outbound_message},
            )
        else:
            result = await self._call_action(
                websocket,
                action="send_private_msg",
                params={"user_id": contact.target_id, "message": outbound_message},
            )

        if str(result.get("status", "")).lower() == "ok" and int(result.get("retcode", 0) or 0) == 0:
            voice_status = await self._send_proxy_voice(websocket, contact, outbound_message)
            self._remember_proxy_thread(
                source_session_id=source_session_id,
                contact=contact,
                outbound_message=outbound_message,
            )
            target_label = "群" if contact.kind == "group" else "好友"
            if voice_status == "sent":
                return f"已代发给{target_label}“{contact.name}”：{outbound_message}（附语音）"
            if voice_status == "failed":
                return f"已代发给{target_label}“{contact.name}”：{outbound_message}（语音发送失败，仅文字已发）"
            return f"已代发给{target_label}“{contact.name}”：{outbound_message}"

        failure_message = str(result.get("message") or result.get("wording") or result.get("msg") or "未知错误")
        return f"代发失败：{failure_message}"

    async def _send_proxy_voice(self, websocket, contact: OneBotContact, message_body: str) -> str:
        if not self._voice_output.enabled:
            return "skipped"

        voice_result = await asyncio.to_thread(self._voice_output.synthesize_text, message_body)
        if voice_result.error or not voice_result.audio_path:
            if voice_result.error:
                print(f"[onebot] proxy voice synthesis failed: {voice_result.error}")
            return "failed"

        target = (
            OneBotTarget(message_type="group", group_id=contact.target_id)
            if contact.kind == "group"
            else OneBotTarget(message_type="private", user_id=contact.target_id)
        )
        async with self._send_lock:
            await websocket.send(json.dumps(build_voice_action(target, voice_result.audio_path), ensure_ascii=False))
        asyncio.create_task(self._voice_output.cleanup_temp_file_later(voice_result.audio_path))
        return "sent"

    def _rewrite_proxy_message(
        self,
        contact: OneBotContact,
        request: OneBotProxySendRequest,
        source_session_id: str,
    ) -> str:
        draft = ProxyMessageDraft(
            recipient_name=contact.name,
            original_request=request.source_text or request.message_body,
            extracted_body=request.message_body,
            intent_label=request.intent_label,
        )
        recent_messages = []
        current_session = self.app.chat_service.session_store.get(source_session_id)
        if current_session is not None:
            recent_messages = self.app.chat_service.prune_chat_history(current_session)
        return self._message_composer.compose_proxy_message(draft=draft, recent_messages=recent_messages)

    async def _send_transient_text(self, websocket, target: OneBotTarget, text: str, *, session_id: str | None = None) -> None:
        normalized = str(text or "").strip()
        if not normalized:
            return
        async with self._send_lock:
            await websocket.send(json.dumps(build_reply_action(target, normalized), ensure_ascii=False))

    async def _send_text_reply(self, websocket, target: OneBotTarget, text: str, *, session_id: str | None = None) -> None:
        async with self._send_lock:
            await websocket.send(json.dumps(build_reply_action(target, text), ensure_ascii=False))
        if session_id:
            self._remember_recent_message(session_id, role="assistant", text=text)
            self._record_outbound_history(
                session_id=session_id,
                target=target,
                text=text,
                message_type="text",
            )

    @staticmethod
    def _translate_progress_update(
        *,
        stage: str,
        message: str,
        payload: dict[str, Any],
        user_text: str,
    ) -> str | None:
        normalized_stage = str(stage or "").strip().lower()
        tool_name = str(payload.get("tool_name", "") or "").strip().lower()
        compact_user_text = re.sub(r"\s+", "", str(user_text or "").strip())
        if normalized_stage in {"received", "planning"}:
            if compact_user_text:
                return "我先把你刚刚这几句串一下，免得理解跑偏。"
            return "我先理一理你这次要我做的事。"
        if normalized_stage == "review":
            return "我再核对一下这一步，免得走偏。"
        if normalized_stage == "tool_start":
            if tool_name.startswith("qq."):
                return "我先翻一下这段聊天记录，看看能不能对上。"
            if tool_name in {"retrieval.search_local_objects", "file.search_by_name"}:
                return "我先帮你把目标内容定位出来。"
            if tool_name.startswith("file.") or tool_name.startswith("image."):
                return "我先翻一下你提到的文件和内容。"
            if tool_name.startswith("web.") or tool_name.startswith("browser."):
                return "我先去查一下相关信息。"
            return "我开始动手查这一步了。"
        if normalized_stage == "tool_success":
            if tool_name.startswith("qq."):
                return "我这边已经拿到一部分聊天记录了，再顺一下。"
            if tool_name.startswith("file.") or tool_name.startswith("retrieval."):
                return "我已经摸到相关内容了，再整理一下给你。"
            return "我这边已经拿到一部分结果了，再整理一下。"
        if normalized_stage == "tool_error":
            return "这一步有点卡住了，我换个办法继续试。"
        if normalized_stage in {"completion", "responding"}:
            return "我这边已经整理得差不多了，马上回你。"
        if normalized_stage == "waiting_for_selection":
            return "我先筛了几个最像的，等你点一个我就继续。"
        return None

    def _cleanup_pending_proxy_selections(self) -> None:
        expired_sessions = [
            session_id
            for session_id, pending in self._pending_proxy_selection.items()
            if pending.is_expired()
        ]
        for session_id in expired_sessions:
            self._pending_proxy_selection.pop(session_id, None)

    @staticmethod
    def _exclude_sender_from_contacts(
        contacts: tuple[OneBotContact, ...],
        sender_id: str,
    ) -> tuple[OneBotContact, ...]:
        normalized_sender = str(sender_id).strip()
        if not normalized_sender:
            return contacts
        return tuple(contact for contact in contacts if str(contact.target_id) != normalized_sender)

    @staticmethod
    def _should_send_voice_reply(response_text: str) -> bool:
        normalized = response_text.strip()
        if not normalized:
            return False
        if normalized.startswith("LLM 决策或自然语言生成失败"):
            return False
        if "validation error for ToolDecision" in normalized:
            return False
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local QQ bot gateway over OneBot WebSocket.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    args = parser.parse_args()

    client = QQBotGatewayClient(config_path=args.config)
    try:
        asyncio.run(client.run_forever())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
