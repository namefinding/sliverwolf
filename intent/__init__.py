from __future__ import annotations

from local_agent.intent.models import IntentBundle, LocalCollectionIntent

__all__ = ["IntentBundle", "LocalCollectionIntent", "IntentService"]


def __getattr__(name: str):
    if name == "IntentService":
        from local_agent.intent.service import IntentService

        return IntentService
    raise AttributeError(f"module 'local_agent.intent' has no attribute {name!r}")
