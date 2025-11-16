"""Helpers for capturing turn-level transcripts and querying an LLM."""

from __future__ import annotations

import asyncio
from typing import Optional

from openai import AsyncOpenAI

from pi_transcription.config import (
    ASSISTANT_MODEL,
    ASSISTANT_WEB_SEARCH_ENABLED,
    OPENAI_API_KEY,
)


class TurnTranscriptAggregator:
    """Collects finalized transcripts for the active turn."""

    def __init__(self, drain_timeout_seconds: float = 0.35):
        self._drain_timeout = drain_timeout_seconds
        self._lock = asyncio.Lock()
        self._segments: list[str] = []
        self._seen_items: set[str] = set()
        self._state: str = "idle"

    async def start_turn(self) -> None:
        """Begin capturing transcripts for a new turn."""

        async with self._lock:
            self._segments.clear()
            self._seen_items.clear()
            self._state = "active"

    async def append_transcript(self, item_id: Optional[str], text: str) -> None:
        """Store a completed transcript fragment for the current turn."""

        cleaned = text.strip()
        if not cleaned:
            return

        async with self._lock:
            if self._state == "idle":
                return
            if item_id and item_id in self._seen_items:
                return
            if item_id:
                self._seen_items.add(item_id)
            self._segments.append(cleaned)

    async def finalize_turn(self) -> Optional[str]:
        """Return the aggregated transcript once the turn is over."""

        async with self._lock:
            if self._state == "idle":
                return None
            self._state = "closing"

        if self._drain_timeout > 0:
            await asyncio.sleep(self._drain_timeout)

        async with self._lock:
            transcript = " ".join(self._segments).strip()
            self._segments.clear()
            self._seen_items.clear()
            self._state = "idle"
            return transcript or None


class LLMResponder:
    """Thin wrapper around the OpenAI Responses API."""

    def __init__(
        self,
        *,
        model: str = ASSISTANT_MODEL,
        enable_web_search: bool = ASSISTANT_WEB_SEARCH_ENABLED,
        client: Optional[AsyncOpenAI] = None,
    ):
        self._client = client or AsyncOpenAI(api_key=OPENAI_API_KEY)
        self._model = model
        self._enable_web_search = enable_web_search

    async def generate_reply(self, transcript: str) -> Optional[str]:
        """Send the transcript to the LLM and return the response text."""

        prompt = transcript.strip()
        if not prompt:
            return None

        payload = {
            "model": self._model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
        }
        if self._enable_web_search:
            payload["tools"] = [{"type": "web_search"}]

        response = await self._client.responses.create(**payload)
        return self._extract_primary_text(response)

    @staticmethod
    def _extract_primary_text(response) -> Optional[str]:
        """Pull assistant text blocks out of a Responses API payload."""

        if hasattr(response, "model_dump"):
            data = response.model_dump()
        else:  # pragma: no cover - fallback for unexpected response shape
            data = response

        output = data.get("output", []) if isinstance(data, dict) else []
        fragments: list[str] = []
        for block in output:
            for content in block.get("content", []):
                if content.get("type") == "output_text":
                    text = content.get("text", "").strip()
                    if text:
                        fragments.append(text)

        combined = "\n".join(fragments).strip()
        return combined or None


__all__ = ["LLMResponder", "TurnTranscriptAggregator"]
