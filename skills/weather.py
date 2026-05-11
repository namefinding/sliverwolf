"""查天气的 skill —— 一个纯确定性操作 skill（不调 LLM）。

这是一个"内部不调 LLM"的 skill 示例。
主 agent 只需传 city，skill 内部完成请求+解析+返回。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import requests
from pydantic import BaseModel, Field

from local_agent.protocol.models import OutputKind, ToolManifest
from local_agent.skills.base import Skill


class WeatherInput(BaseModel):
    city: str = Field(default="Beijing", description="city name in English, e.g. Beijing, Tokyo, London")


class WeatherSkill(Skill):
    def manifest(self) -> ToolManifest:
        return ToolManifest(
            tool_name="skill.weather",
            module="skill",
            description=(
                "Get current weather for a city. No LLM calls needed internally — "
                "just fetches from public API and returns structured data."
            ),
            side_effect=False,
            idempotent=True,
            produces=[OutputKind.OBJECT_DETAILS],
            input_schema=WeatherInput.model_json_schema(),
            output_schema={"type": "object", "properties": {"city": {"type": "string"}, "temperature": {"type": "string"}, "description": {"type": "string"}}},
        )

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = WeatherInput.model_validate(arguments)
        city = payload.city.strip()

        try:
            # wttr.in 免费天气 API，不需要 key
            url = f"https://wttr.in/{city}?format=j1"
            resp = requests.get(url, timeout=10, headers={"User-Agent": "local-agent-weather/1.0"})
            resp.raise_for_status()
            data = resp.json()

            current = data.get("current_condition", [{}])[0]
            return {
                "city": city,
                "temperature": f"{current.get('temp_C', '?')}°C",
                "feels_like": f"{current.get('FeelsLikeC', '?')}°C",
                "humidity": f"{current.get('humidity', '?')}%",
                "description": current.get("weatherDesc", [{}])[0].get("value", "unknown"),
                "wind": f"{current.get('windspeedKmph', '?')} km/h",
                "updated_at": datetime.utcnow().isoformat(),
            }
        except Exception as exc:
            return {"city": city, "error": str(exc), "temperature": "?"}
