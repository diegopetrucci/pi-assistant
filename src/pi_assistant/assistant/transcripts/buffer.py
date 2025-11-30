"""Utilities for collecting transcripts per assistant turn."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

from pi_assistant.cli.logging import LOGGER

TRANSCRIPT_SNIPPET_LIMIT = 80


class TurnTranscriptAggregator:
    """Collects finalized transcripts for the active turn."""

    def __init__(
        self,
        drain_timeout_seconds: float = 0.35,
        max_finalize_wait_seconds: float = 1.25,
    ):
        self._drain_timeout = max(drain_timeout_seconds, 0.0)
        self._max_finalize_wait = max(max_finalize_wait_seconds, self._drain_timeout)
        self._lock = asyncio.Lock()
        self._segments: list[str] = []
        self._seen_items: set[str] = set()
        self._state: str = "idle"
        self._trace_label = "TURN-TRACE"

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    async def start_turn(self) -> None:
        """Begin capturing transcripts for a new turn."""

        async with self._lock:
            self._segments.clear()
            self._seen_items.clear()
            self._state = "active"
            LOGGER.verbose(self._trace_label, f"{self._timestamp()} start turn")

    async def append_transcript(self, item_id: Optional[str], transcript: str) -> None:
        """Store a completed transcript fragment for the current turn."""

        cleaned = transcript.strip()
        if not cleaned:
            return

        async with self._lock:
            if self._state == "idle":
                LOGGER.verbose(
                    self._trace_label,
                    f"{self._timestamp()} append ignored (idle) item={item_id}",
                )
                return
            if item_id and item_id in self._seen_items:
                LOGGER.verbose(
                    self._trace_label,
                    f"{self._timestamp()} append ignored (duplicate) item={item_id}",
                )
                return
            if item_id:
                self._seen_items.add(item_id)
            self._segments.append(cleaned)
            LOGGER.verbose(
                self._trace_label,
                f"{self._timestamp()} append stored item={item_id} "
                f"segments={len(self._segments)} text={cleaned!r}",
            )

    async def finalize_turn(self) -> Optional[str]:
        """Return the aggregated transcript once the turn is over."""

        async with self._lock:
            if self._state == "idle":
                LOGGER.verbose(self._trace_label, f"{self._timestamp()} finalize skipped (idle)")
                return None
            self._state = "closing"
            pending_segments = len(self._segments)
            LOGGER.verbose(
                self._trace_label,
                f"{self._timestamp()} finalize start segments={pending_segments}",
            )

        wait_interval = self._drain_timeout if self._drain_timeout > 0 else 0.1
        total_wait = 0.0

        async def _maybe_finalize() -> tuple[bool, Optional[str]]:
            async with self._lock:
                ready = bool(self._segments) or total_wait >= self._max_finalize_wait
                if not ready:
                    return False, None
                transcript = " ".join(self._segments).strip()
                self._segments.clear()
                self._seen_items.clear()
                self._state = "idle"
                snippet = _shorten_transcript(transcript)
                reason = (
                    "timeout"
                    if (not transcript and total_wait >= self._max_finalize_wait)
                    else "complete"
                )
                LOGGER.verbose(
                    self._trace_label,
                    f"{self._timestamp()} finalize done segments_cleared={pending_segments} "
                    f"wait={total_wait:.3f}s mode={reason} transcript={snippet!r}",
                )
                transcript_value = transcript if transcript else None
                return True, transcript_value

        finalized, maybe_transcript = await _maybe_finalize()
        if finalized:
            return maybe_transcript

        while True:
            remaining = self._max_finalize_wait - total_wait
            if remaining <= 0:
                wait_duration = 0.0
            else:
                wait_duration = min(wait_interval, remaining)
            if wait_duration > 0:
                await asyncio.sleep(wait_duration)
                total_wait += wait_duration
            finalized, maybe_transcript = await _maybe_finalize()
            if finalized:
                return maybe_transcript

    async def clear_current_turn(self, reason: str = "") -> None:
        """Drop any buffered segments without ending the turn."""

        async with self._lock:
            segment_count = len(self._segments)
            self._segments.clear()
            self._seen_items.clear()
            suffix = f" reason={reason}" if reason else ""
            LOGGER.verbose(
                self._trace_label,
                f"{self._timestamp()} clear turn segments_dropped={segment_count}{suffix}",
            )


def _shorten_transcript(transcript: Optional[str]) -> Optional[str]:
    if not transcript:
        return None
    if len(transcript) <= TRANSCRIPT_SNIPPET_LIMIT:
        return transcript
    return transcript[:TRANSCRIPT_SNIPPET_LIMIT] + "…"


__all__ = ["TurnTranscriptAggregator"]
