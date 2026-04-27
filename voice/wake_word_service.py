from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from local_agent.protocol.models import VoiceInputConfig


@dataclass(frozen=True)
class WakeWordMatch:
    matched: bool
    wake_word: str | None = None
    score: float = 0.0
    remaining_text: str = ""
    match_type: str = "none"


class WakeWordService:
    _SEPARATOR_PATTERN = re.compile(r"^[\s,，。.!！？:：;；、~…-]+")
    _NORMALIZE_PATTERN = re.compile(r"[\s,，。.!！？:：;；、~…'\"`]+")

    def __init__(self, config: VoiceInputConfig) -> None:
        self._config = config

    @property
    def enabled(self) -> bool:
        return bool(self._config.wake_word_enabled and self._config.wake_words)

    def match_text(self, text: str) -> WakeWordMatch:
        raw_text = str(text or "").strip()
        if not raw_text or not self.enabled:
            return WakeWordMatch(matched=False, remaining_text=raw_text)

        exact_match = self._match_exact(raw_text)
        if exact_match.matched:
            return exact_match

        if self._config.wake_word_fuzzy_threshold <= 0:
            return WakeWordMatch(matched=False, remaining_text=raw_text)
        return self._match_fuzzy(raw_text)

    def _match_exact(self, raw_text: str) -> WakeWordMatch:
        normalized_text = self._normalize(raw_text)
        for wake_word in self._config.wake_words:
            normalized_wake_word = self._normalize(wake_word)
            if not normalized_wake_word:
                continue
            if normalized_text == normalized_wake_word:
                return WakeWordMatch(
                    matched=True,
                    wake_word=wake_word,
                    score=1.0,
                    remaining_text="",
                    match_type="exact",
                )
            if normalized_text.startswith(normalized_wake_word):
                remaining = self._strip_prefix(raw_text, wake_word)
                return WakeWordMatch(
                    matched=True,
                    wake_word=wake_word,
                    score=1.0,
                    remaining_text=remaining,
                    match_type="exact_prefix",
                )
        return WakeWordMatch(matched=False, remaining_text=raw_text)

    def _match_fuzzy(self, raw_text: str) -> WakeWordMatch:
        normalized_text = self._normalize(raw_text)
        threshold = max(0.0, min(1.0, self._config.wake_word_fuzzy_threshold))
        best_match = WakeWordMatch(matched=False, remaining_text=raw_text)
        for wake_word in self._config.wake_words:
            normalized_wake_word = self._normalize(wake_word)
            if not normalized_wake_word:
                continue
            candidate_windows = {
                normalized_text[: len(normalized_wake_word)],
                normalized_text[: len(normalized_wake_word) + 1],
                normalized_text[: len(normalized_wake_word) + 2],
            }
            score = max(
                SequenceMatcher(None, window, normalized_wake_word).ratio()
                for window in candidate_windows
                if window
            )
            if score < threshold or score <= best_match.score:
                continue
            remaining = raw_text
            if len(raw_text) > len(wake_word):
                remaining = raw_text[len(wake_word) :].lstrip()
                remaining = self._SEPARATOR_PATTERN.sub("", remaining)
            best_match = WakeWordMatch(
                matched=True,
                wake_word=wake_word,
                score=round(score, 3),
                remaining_text=remaining,
                match_type="fuzzy_prefix",
            )
        return best_match

    @classmethod
    def _normalize(cls, text: str) -> str:
        lowered = str(text or "").strip().lower()
        return cls._NORMALIZE_PATTERN.sub("", lowered)

    @classmethod
    def _strip_prefix(cls, raw_text: str, wake_word: str) -> str:
        stripped = raw_text.strip()
        if stripped.lower().startswith(wake_word.lower()):
            remainder = stripped[len(wake_word) :]
            return cls._SEPARATOR_PATTERN.sub("", remainder)
        return stripped
