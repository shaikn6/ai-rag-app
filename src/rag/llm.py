"""LLM interface + Anthropic Claude implementation + a fake for tests."""
from __future__ import annotations
import os
from typing import Protocol

MODEL = "claude-sonnet-4-6"


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class ClaudeLLM:
    """Real Claude via Anthropic SDK. Lazy import so tests/CI need no SDK key."""

    def __init__(self, model: str = MODEL):
        from anthropic import Anthropic
        self._client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._model = model

    def complete(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self._model, max_tokens=1024,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text


class FakeLLM:
    """Deterministic stub for tests — echoes which sources it was grounded on."""

    def complete(self, system: str, user: str) -> str:
        return f"[fake answer grounded on context]\n{user[:120]}"
