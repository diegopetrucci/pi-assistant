"""Shared data structures for the CLI audio controller."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pi_assistant.assistant import LLMResponder, TurnTranscriptAggregator
from pi_assistant.audio import SpeechPlayer


@dataclass(slots=True)
class AudioControllerContext:
    transcript_buffer: TurnTranscriptAggregator
    assistant: LLMResponder
    speech_player: SpeechPlayer
    stop_signal: asyncio.Event
    speech_stopped_signal: asyncio.Event


__all__ = ["AudioControllerContext"]
