"""Event handling helpers for the CLI transcription app."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from typing import Any, Protocol

from pi_assistant.cli.logging_utils import (
    CONTROL_LOG_LABEL,
    ERROR_LOG_LABEL,
    LOGGER,
    TRANSCRIPT_LOG_LABEL,
    TURN_LOG_LABEL,
    VAD_LOG_LABEL,
)

STOP_COMMANDS = ("hey jarvis stop", "jarvis stop")


class _SpeechStopper(Protocol):
    async def stop(self) -> bool: ...


class _TranscriptBuffer(Protocol):
    async def append_transcript(self, item_id: str | None, transcript: str) -> None: ...

    async def clear_current_turn(self, reason: str) -> None: ...


class _EventStreamClient(Protocol):
    def receive_events(self) -> AsyncIterator[dict[str, Any]]: ...


def handle_transcription_event(event: dict) -> None:
    """Pretty-print OpenAI transcription events for debugging."""

    event_type = event.get("type")

    if event_type == "conversation.item.input_audio_transcription.delta":
        delta = event.get("delta", "")
        LOGGER.verbose("PARTIAL", delta, flush=True)

    elif event_type == "conversation.item.input_audio_transcription.completed":
        transcript = event.get("transcript", "")
        LOGGER.log(TRANSCRIPT_LOG_LABEL, transcript)

    elif event_type == "input_audio_buffer.committed":
        item_id = event.get("item_id", "")
        LOGGER.verbose(VAD_LOG_LABEL, f"Speech detected (item: {item_id})")

    elif event_type == "error":
        error = event.get("error", {})
        error_type = error.get("type", "unknown")
        error_message = error.get("message", "No message")
        error_code = error.get("code", "unknown")
        LOGGER.log(
            ERROR_LOG_LABEL,
            f"{error_type} ({error_code}): {error_message}",
            error=True,
        )

    elif event_type == "transcription_session.created":
        LOGGER.verbose("INFO", "Transcription session created")

    elif event_type == "transcription_session.updated":
        LOGGER.verbose("INFO", "Transcription session configuration updated")

    else:
        LOGGER.verbose("DEBUG", f"Received event: {event_type}")


def _normalize_command(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return " ".join(cleaned.split())


async def maybe_stop_playback(transcript: str, speech_player: _SpeechStopper) -> bool:
    """Stop assistant playback if a stop command is detected in the transcript."""

    normalized = _normalize_command(transcript)
    if not normalized:
        return False
    if any(cmd in normalized for cmd in STOP_COMMANDS):
        halted = await speech_player.stop()
        if halted:
            LOGGER.verbose(CONTROL_LOG_LABEL, "Stop command detected; halting assistant audio.")
        return True
    return False


async def receive_transcription_events(
    ws_client: _EventStreamClient,
    transcript_buffer: _TranscriptBuffer,
    speech_player: _SpeechStopper,
    *,
    stop_signal: asyncio.Event,
    speech_stopped_signal: asyncio.Event,
) -> None:
    """Continuously receive and handle transcription events from WebSocket."""

    LOGGER.verbose("INFO", "Starting event receiver...")
    event_count = 0

    try:
        async for event in ws_client.receive_events():
            event_count += 1
            event_type = event.get("type")
            handle_transcription_event(event)
            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "")
                item_id = event.get("item_id")
                if await maybe_stop_playback(transcript, speech_player):
                    await transcript_buffer.clear_current_turn("assistant stop command")
                    stop_signal.set()
                    continue
                await transcript_buffer.append_transcript(item_id, transcript)
            elif event_type == "input_audio_buffer.speech_stopped":
                item_id = event.get("item_id")
                LOGGER.log(
                    TURN_LOG_LABEL,
                    f"Server acknowledged speech stop (item={item_id}).",
                )
                speech_stopped_signal.set()

    except asyncio.CancelledError:
        LOGGER.verbose("INFO", f"Event receiver stopped ({event_count} events received)")
        raise
    except Exception as exc:
        LOGGER.log(ERROR_LOG_LABEL, f"Event receiver error: {exc}", error=True)
        raise
