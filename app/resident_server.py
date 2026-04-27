from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from local_agent.app.bootstrap import bootstrap_resident_app
from local_agent.app.chat_service import ChatService
from local_agent.app.channel_router import ChannelRouter
from local_agent.app.local_voice_agent import LocalVoiceAgentService
from local_agent.app.task_service import TaskService
from local_agent.protocol.channel_models import ChannelMessage
from local_agent.protocol.models import TaskRun
from local_agent.voice.asr_service import ASRService
from local_agent.voice.output_service import VoiceOutputService


class ResidentRequestHandler(BaseHTTPRequestHandler):
    server: "ResidentHTTPServer"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        if parsed.path in {"/", "/index.html"}:
            self._send_html(self.server.index_html)
            return

        if parsed.path.startswith("/static/"):
            asset_path = _resolve_static_asset(self.server.static_root, parsed.path)
            if asset_path is None or not asset_path.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            self._send_file(asset_path)
            return

        if parsed.path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "service": "local-agent-resident-server",
                    "sessions": self.server.chat_service.session_store.list_session_ids(),
                    "tasks": [task.task_id for task in self.server.task_service.list_tasks()],
                    "scope_options": self.server.scope_options,
                    "default_scope": self.server.default_scope,
                    "ui_defaults": self.server.ui_defaults,
                    "startup": self.server.startup_jobs.snapshot(),
                    "health_checks": self.server.health_report,
                    "local_voice": None if self.server.local_voice_service is None else self.server.local_voice_service.snapshot(),
                },
            )
            return

        if parsed.path == "/scopes":
            self._send_json(
                HTTPStatus.OK,
                {
                    "default_scope": self.server.default_scope,
                    "scope_options": self.server.scope_options,
                },
            )
            return

        if parsed.path == "/sessions":
            self._send_json(
                HTTPStatus.OK,
                {"sessions": self.server.chat_service.session_store.list_session_ids()},
            )
            return

        if parsed.path == "/tasks":
            query = parse_qs(parsed.query)
            session_id = query.get("session_id", [None])[0]
            self._send_json(
                HTTPStatus.OK,
                {"tasks": [self._serialize_task(task) for task in self.server.task_service.list_tasks(session_id=session_id)]},
            )
            return

        if parsed.path.startswith("/tasks/"):
            task_id = parsed.path.rsplit("/", maxsplit=1)[-1]
            task = self.server.task_service.get_task(task_id)
            if task is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "task not found"})
                return
            self._send_json(HTTPStatus.OK, self._serialize_task(task))
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            parsed = urlparse(self.path)

            if parsed.path == "/chat":
                self._handle_chat(payload)
                return

            if parsed.path == "/tasks":
                self._handle_task_create(payload)
                return

            if parsed.path.endswith("/acknowledge") and parsed.path.startswith("/tasks/"):
                task_id = parsed.path.split("/")[-2]
                task = self.server.task_service.acknowledge_task(task_id)
                if task is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "task not found"})
                    return
                self._send_json(HTTPStatus.OK, self._serialize_task(task))
                return

            if parsed.path.endswith("/cancel") and parsed.path.startswith("/tasks/"):
                task_id = parsed.path.split("/")[-2]
                task = self.server.task_service.cancel_task(task_id)
                if task is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "task not found"})
                    return
                self._send_json(HTTPStatus.OK, self._serialize_task(task))
                return

            if parsed.path.endswith("/select") and parsed.path.startswith("/tasks/"):
                task_id = parsed.path.split("/")[-2]
                task = self.server.task_service.select_candidate(
                    task_id,
                    candidate_id=payload.get("candidate_id"),
                    candidate_path=payload.get("candidate_path"),
                )
                if task is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "task or candidate not found"})
                    return
                self._send_json(HTTPStatus.OK, self._serialize_task(task))
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON body"})
        except Exception as exc:  # noqa: BLE001
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def do_DELETE(self) -> None:  # noqa: N802
        if not self.path.startswith("/sessions/"):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        session_id = self.path.rsplit("/", maxsplit=1)[-1]
        removed = self.server.chat_service.session_store.remove(session_id)
        if removed:
            self._send_json(HTTPStatus.OK, {"deleted": True, "session_id": session_id})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "session not found"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path) -> None:
        body = file_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_chat(self, payload: dict[str, Any]) -> None:
        text = str(payload.get("text", "")).strip()
        if not text:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "text is required"})
            return

        session_id = payload.get("session_id")
        scope_root = self._normalize_scope_root(payload.get("scope_root"))
        runtime_settings = self._normalize_runtime_settings(payload.get("runtime_settings"))
        mode = payload.get("mode", "auto")
        if mode not in {"auto", "chat", "agent"}:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "mode must be auto, chat, or agent"})
            return

        run_in_background_raw = payload.get("run_in_background")
        if run_in_background_raw is None:
            run_in_background = self.server.channel_router.should_run_in_background(
                ChannelMessage(
                    channel="resident_web",
                    text=text,
                    session_id=None if session_id is None else str(session_id),
                    scope_root=scope_root,
                    mode=mode,
                    runtime_settings=runtime_settings,
                )
            )
        else:
            run_in_background = bool(run_in_background_raw)
        if run_in_background:
            if session_id is None:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "session_id is required for background tasks"})
                return
            task = self.server.task_service.submit(
                user_text=text,
                session_id=str(session_id),
                scope_root=scope_root,
                runtime_settings=runtime_settings,
            )
            self._send_json(
                HTTPStatus.ACCEPTED,
                {
                    "session_id": session_id,
                    "mode": "agent",
                    "used_agent": True,
                    "response": "我已经把这件事放到后台去做了。你可以继续聊天，任务进展会出现在右侧任务面板里。",
                    "speech_text": "我先去后台处理，你可以继续和我聊天。",
                    "tts_dispatched": False,
                    "scope_root": scope_root or self.server.default_scope,
                    "background_task": True,
                    "task": self._serialize_task(task),
                },
            )
            return

        result = self.server.channel_router.dispatch(
            ChannelMessage(
                channel="resident_web",
                text=text,
                session_id=None if session_id is None else str(session_id),
                scope_root=scope_root,
                mode=mode,
                runtime_settings=runtime_settings,
            )
        )
        self._send_json(
            HTTPStatus.OK,
            {
                "session_id": result.session_id,
                "mode": result.mode,
                "used_agent": result.used_agent,
                "response": result.response,
                "speech_text": result.speech_text,
                "tts_dispatched": result.tts_dispatched,
                "scope_root": result.scope_root or self.server.default_scope,
                "overall_task_goal": result.overall_task_goal,
                "completed_outputs": result.completed_outputs or [],
            },
        )

    def _handle_task_create(self, payload: dict[str, Any]) -> None:
        text = str(payload.get("text", "")).strip()
        session_id = payload.get("session_id")
        scope_root = self._normalize_scope_root(payload.get("scope_root"))
        runtime_settings = self._normalize_runtime_settings(payload.get("runtime_settings"))
        if not text or not session_id:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "text and session_id are required"})
            return
        task = self.server.task_service.submit(
            user_text=text,
            session_id=str(session_id),
            scope_root=scope_root,
            runtime_settings=runtime_settings,
        )
        self._send_json(HTTPStatus.ACCEPTED, self._serialize_task(task))

    @staticmethod
    def _normalize_scope_root(raw_scope: Any) -> str | None:
        if raw_scope is None:
            return None
        scope_root = str(raw_scope).strip()
        if not scope_root:
            return None
        path = Path(scope_root).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"scope_root is not a valid directory: {scope_root}")
        return str(path)

    @staticmethod
    def _normalize_runtime_settings(raw_settings: Any) -> dict[str, Any] | None:
        if raw_settings is None:
            return None
        if not isinstance(raw_settings, dict):
            raise ValueError("runtime_settings must be an object")
        normalized: dict[str, Any] = {}
        agent_raw = raw_settings.get("agent")
        if isinstance(agent_raw, dict):
            allowed_agent_keys = {
                "persona_name",
                "persona_profile",
                "chat_style_prompt",
                "display_style_prompt",
                "speech_style_prompt",
                "tool_speech_enabled",
                "speech_max_chars",
            }
            normalized["agent"] = {key: value for key, value in agent_raw.items() if key in allowed_agent_keys}
        voice_raw = raw_settings.get("voice")
        if isinstance(voice_raw, dict):
            normalized["voice"] = {key: value for key, value in voice_raw.items() if key == "enabled"}
        return normalized or None

    @staticmethod
    def _serialize_task(task: TaskRun) -> dict[str, Any]:
        return task.model_dump(mode="json")


class ResidentHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        chat_service: ChatService,
        channel_router: ChannelRouter,
        task_service: TaskService,
        health_report: dict[str, object],
        startup_jobs,
        index_html: str,
        static_root: Path,
        scope_options: list[str],
        default_scope: str,
        ui_defaults: dict[str, object],
        local_voice_service: LocalVoiceAgentService | None = None,
    ) -> None:
        super().__init__(server_address, ResidentRequestHandler)
        self.chat_service = chat_service
        self.channel_router = channel_router
        self.task_service = task_service
        self.health_report = health_report
        self.startup_jobs = startup_jobs
        self.index_html = index_html
        self.static_root = static_root.resolve()
        self.scope_options = scope_options
        self.default_scope = default_scope
        self.ui_defaults = ui_defaults
        self.local_voice_service = local_voice_service


def _resolve_static_asset(static_root: Path, request_path: str) -> Path | None:
    if not request_path.startswith("/static/"):
        return None
    relative_path = unquote(request_path.removeprefix("/static/")).lstrip("/\\")
    if not relative_path:
        return None
    candidate = (static_root / relative_path).resolve()
    if not candidate.is_relative_to(static_root.resolve()):
        return None
    return candidate


def load_index_html() -> str:
    static_root = Path(__file__).with_name("static")
    return (static_root / "index.html").read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local agent resident chat server.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    parser.add_argument("--session-ttl-minutes", type=int, default=60, help="Inactive session TTL.")
    parser.add_argument("--max-sessions", type=int, default=100, help="Maximum number of cached sessions.")
    args = parser.parse_args()

    bootstrapped = bootstrap_resident_app(
        config_path=args.config,
        session_ttl_minutes=args.session_ttl_minutes,
        max_sessions=args.max_sessions,
    )
    local_voice_output = VoiceOutputService.from_config(
        bootstrapped.config.voice,
        play_audio=bootstrapped.config.voice.play_audio,
        async_playback=bootstrapped.config.voice.async_playback,
        cleanup_audio=bootstrapped.config.voice.cleanup_audio,
    )
    local_voice_service = LocalVoiceAgentService(
        bootstrapped.config.voice_input,
        channel_router=bootstrapped.channel_router,
        task_service=bootstrapped.task_service,
        voice_output=local_voice_output,
        asr_service=ASRService.from_config(bootstrapped.config.asr),
        scope_root=bootstrapped.config.agent.workspace_root,
        runtime_settings=None,
    )
    server = ResidentHTTPServer(
        (args.host, args.port),
        bootstrapped.chat_service,
        bootstrapped.channel_router,
        bootstrapped.task_service,
        bootstrapped.health_report,
        bootstrapped.startup_jobs,
        load_index_html(),
        Path(__file__).with_name("static"),
        bootstrapped.scope_options,
        bootstrapped.scope_options[0] if bootstrapped.scope_options else str(Path(args.config).resolve().parent),
        bootstrapped.ui_defaults,
        local_voice_service=local_voice_service,
    )
    print(f"Resident agent server listening on http://{args.host}:{args.port}")
    print("GET /  POST /chat  GET /health  GET /scopes  GET /sessions  GET /tasks  DELETE /sessions/{id}")
    local_voice_service.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        local_voice_service.stop()
        server.server_close()


if __name__ == "__main__":
    main()
