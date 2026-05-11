from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from local_agent.app.onebot_models import OneBotTarget
from local_agent.protocol.channel_models import ChannelReply


def build_reply_action(target: OneBotTarget, text: str) -> dict[str, Any]:
    if target.message_type == "group":
        return {
            "action": "send_group_msg",
            "params": {"group_id": target.group_id, "message": text},
            "echo": f"reply_{uuid4().hex[:12]}",
        }
    return {
        "action": "send_private_msg",
        "params": {"user_id": target.user_id, "message": text},
        "echo": f"reply_{uuid4().hex[:12]}",
    }


def build_file_action(target: OneBotTarget, file_path: str) -> dict[str, Any]:
    file_name = Path(file_path).name
    if target.message_type == "group":
        return {
            "action": "upload_group_file",
            "params": {"group_id": target.group_id, "file": file_path, "name": file_name},
            "echo": f"file_{uuid4().hex[:12]}",
        }
    return {
        "action": "upload_private_file",
        "params": {"user_id": target.user_id, "file": file_path, "name": file_name},
        "echo": f"file_{uuid4().hex[:12]}",
    }


def build_image_action(target: OneBotTarget, image_path: str) -> dict[str, Any]:
    """构造发送图片的 OneBot action。"""
    image_file = str(Path(image_path).resolve()).replace("\\", "/")
    message = [{"type": "image", "data": {"file": image_file}}]
    if target.message_type == "group":
        return {
            "action": "send_group_msg",
            "params": {"group_id": target.group_id, "message": message},
            "echo": f"image_{uuid4().hex[:12]}",
        }
    return {
        "action": "send_private_msg",
        "params": {"user_id": target.user_id, "message": message},
        "echo": f"image_{uuid4().hex[:12]}",
    }


def build_voice_action(target: OneBotTarget, audio_path: str) -> dict[str, Any]:
    audio_file = str(Path(audio_path).resolve()).replace("\\", "/")
    message = [{"type": "record", "data": {"file": audio_file}}]
    if target.message_type == "group":
        return {
            "action": "send_group_msg",
            "params": {"group_id": target.group_id, "message": message},
            "echo": f"voice_{uuid4().hex[:12]}",
        }
    return {
        "action": "send_private_msg",
        "params": {"user_id": target.user_id, "message": message},
        "echo": f"voice_{uuid4().hex[:12]}",
    }


def extract_delivery_file_paths(reply: ChannelReply, *, limit: int = 1) -> list[str]:
    metadata = reply.metadata or {}
    execution_summary = metadata.get("execution_summary")
    if not isinstance(execution_summary, dict):
        return []
    if _already_sent_by_agent(execution_summary):
        return []

    candidates: list[str] = []
    delivery_target_path = execution_summary.get("delivery_target_path")
    if isinstance(delivery_target_path, str) and delivery_target_path:
        candidates.append(delivery_target_path)

    written_files = execution_summary.get("written_files")
    if isinstance(written_files, list):
        for item in written_files:
            if isinstance(item, str) and item not in candidates:
                candidates.append(item)

    if _is_delivery_summary(execution_summary):
        candidate_paths = execution_summary.get("candidate_paths")
        if isinstance(candidate_paths, list):
            for item in candidate_paths:
                if isinstance(item, str) and item not in candidates:
                    candidates.append(item)

    resolved: list[str] = []
    for raw_path in candidates:
        if Path(raw_path).is_file() and raw_path not in resolved:
            resolved.append(raw_path)
        if len(resolved) >= limit:
            break
    return resolved


def _already_sent_by_agent(execution_summary: dict[str, Any]) -> bool:
    successful_actions = execution_summary.get("successful_actions")
    if not isinstance(successful_actions, list):
        return False
    for action in successful_actions:
        if isinstance(action, dict) and action.get("tool_name") == "qq.send_file":
            return True
    return False


def _is_delivery_summary(execution_summary: dict[str, Any]) -> bool:
    task_classification = execution_summary.get("task_classification")
    if isinstance(task_classification, dict):
        task_kind = str(task_classification.get("task_kind", "")).strip().lower()
        if task_kind == "delivery":
            return True

    overall_task_goal = execution_summary.get("overall_task_goal")
    if isinstance(overall_task_goal, dict):
        required_outputs = overall_task_goal.get("required_outputs")
        if isinstance(required_outputs, list) and "message_sent" in required_outputs:
            return True

    completed_outputs = execution_summary.get("completed_outputs")
    if isinstance(completed_outputs, list) and "message_sent" in completed_outputs:
        return True

    return False
