"""Event handling helpers for the CLI transcription app."""

from __future__ import annotations

import asyncio
import re
import sys

from pi_assistant.assistant import TurnTranscriptAggregator
from pi_assistant.audio import SpeechPlayer
from pi_assistant.cli.logging_utils import (
    CONTROL_LOG_LABEL,
    ERROR_LOG_LABEL,
    TRANSCRIPT_LOG_LABEL,
    VAD_LOG_LABEL,
    verbose_print,
)
from pi_assistant.network import WebSocketClient

STOP_COMMANDS = ("hey jarvis stop", "jarvis stop")


def handle_transcription_event(event: dict) -> None:
    """Pretty-print OpenAI transcription events for debugging."""

    event_type = event.get("type")

    if event_type == "conversation.item.input_audio_transcription.delta":
        delta = event.get("delta", "")
        verbose_print(f"[PARTIAL] {delta}", end="", flush=True)

    elif event_type == "conversation.item.input_audio_transcription.completed":
        transcript = event.get("transcript", "")
        print(f"\n{TRANSCRIPT_LOG_LABEL} {transcript}")

    elif event_type == "input_audio_buffer.committed":
        item_id = event.get("item_id", "")
        verbose_print(f"{VAD_LOG_LABEL} Speech detected (item: {item_id})")

    elif event_type == "error":
        error = event.get("error", {})
        error_type = error.get("type", "unknown")
        error_message = error.get("message", "No message")
        error_code = error.get("code", "unknown")
        print(f"{ERROR_LOG_LABEL} {error_type} ({error_code}): {error_message}", file=sys.stderr)

    elif event_type == "transcription_session.created":
        verbose_print("[INFO] Transcription session created")

    elif event_type == "transcription_session.updated":
        verbose_print("[INFO] Transcription session configuration updated")

    else:
        verbose_print(f"[DEBUG] Received event: {event_type}")


def _normalize_command(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return " ".join(cleaned.split())


async def maybe_stop_playback(transcript: str, speech_player: SpeechPlayer) -> bool:
    """Stop assistant playback if a stop command is detected in the transcript."""

    normalized = _normalize_command(transcript)
    if not normalized:
        return False
    if any(cmd in normalized for cmd in STOP_COMMANDS):
        halted = await speech_player.stop()
        if halted:
            verbose_print(f"{CONTROL_LOG_LABEL} Stop command detected; halting assistant audio.")
        return True
    return False


async def receive_transcription_events(
    ws_client: WebSocketClient,
    transcript_buffer: TurnTranscriptAggregator,
    speech_player: SpeechPlayer,
    *,
    stop_signal: asyncio.Event,
    speech_stopped_signal: asyncio.Event,
) -> None:
    """Continuously receive and handle transcription events from WebSocket."""

    verbose_print("[INFO] Starting event receiver...")
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
                speech_stopped_signal.set()

    except asyncio.CancelledError:
        verbose_print(f"[INFO] Event receiver stopped ({event_count} events received)")
        raise
    except Exception as exc:
        print(f"{ERROR_LOG_LABEL} Event receiver error: {exc}", file=sys.stderr)
        raise
