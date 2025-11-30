"""Shared helpers for the CLI audio controller."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Optional

from pi_assistant.assistant import LLMReply, LLMResponder, TurnTranscriptAggregator
from pi_assistant.audio import SpeechPlayer
from pi_assistant.cli.logging_utils import (
    ASSISTANT_LOG_LABEL,
    CONTROL_LOG_LABEL,
    LOGGER,
    TURN_LOG_LABEL,
)
from pi_assistant.config import (
    ASSISTANT_TTS_SAMPLE_RATE,
    CONFIRMATION_CUE_ENABLED,
    CONFIRMATION_CUE_TEXT,
)
from pi_assistant.wake_word import StreamState

from .controller_components import SilenceTracker, StreamStateManager


@dataclass(slots=True)
class ResponseLifecycleHooks:
    on_transcript_ready: Optional[Callable[[], None]] = None
    on_reply_start: Optional[Callable[[], None]] = None
    on_reply_complete: Optional[Callable[[], None]] = None


def should_ignore_server_stop_event(
    state_manager: StreamStateManager,
    silence_tracker: SilenceTracker,
    min_silence_seconds: float,
) -> str | None:
    """Return a reason to ignore a server VAD stop, or ``None`` if it may proceed."""

    if state_manager.state != StreamState.STREAMING:
        return None
    if not silence_tracker.heard_speech:
        return None
    if not silence_tracker.has_observed_silence(min_silence_seconds):
        current = silence_tracker.silence_duration
        return f"{current:.2f}s silence < {min_silence_seconds:.2f}s minimum"
    return None


async def _finalize_transcript(
    transcript_buffer: TurnTranscriptAggregator,
    hooks: Optional[ResponseLifecycleHooks],
) -> Optional[str]:
    """Return the finalized transcript while honoring lifecycle hooks."""

    try:
        return await transcript_buffer.finalize_turn()
    finally:
        if hooks and hooks.on_transcript_ready:
            hooks.on_transcript_ready()


async def _request_assistant_reply(
    transcript: str,
    assistant: LLMResponder,
    hooks: Optional[ResponseLifecycleHooks],
) -> Optional[LLMReply]:
    """Fetch an assistant reply and handle lifecycle notifications."""

    reply_started = False
    if hooks and hooks.on_reply_start:
        hooks.on_reply_start()
        reply_started = True

    reply: Optional[LLMReply] = None
    try:
        reply = await assistant.generate_reply(transcript)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - network failure
        LOGGER.log(ASSISTANT_LOG_LABEL, f"Error requesting assistant reply: {exc}", error=True)
        return None
    finally:
        if reply_started and hooks and hooks.on_reply_complete:
            hooks.on_reply_complete()

    return reply


async def _play_assistant_audio(reply: LLMReply, speech_player: SpeechPlayer) -> None:
    if not reply.audio_bytes:
        return
    try:
        await speech_player.play(reply.audio_bytes, sample_rate=reply.audio_sample_rate)
    except Exception as exc:  # pragma: no cover - host audio failure
        LOGGER.log(ASSISTANT_LOG_LABEL, f"Error playing audio reply: {exc}", error=True)


def _maybe_schedule_confirmation_cue(
    assistant: LLMResponder,
    speech_player: SpeechPlayer,
) -> None:
    """Play a cached 'Got it' cue without blocking the main turn flow."""

    if not (assistant.tts_enabled and CONFIRMATION_CUE_ENABLED and CONFIRMATION_CUE_TEXT):
        return

    cached = assistant.peek_phrase_audio(CONFIRMATION_CUE_TEXT)
    if not cached:
        asyncio.create_task(assistant.warm_phrase_audio(CONFIRMATION_CUE_TEXT))
        return
    audio_bytes, sample_rate = cached
    task = asyncio.create_task(
        speech_player.play(audio_bytes, sample_rate=sample_rate or ASSISTANT_TTS_SAMPLE_RATE)
    )

    def _log_task_error(fut: asyncio.Task):
        try:
            fut.result()
        except Exception as exc:  # pragma: no cover - host audio failure
            LOGGER.verbose(CONTROL_LOG_LABEL, f"Confirmation cue failed: {exc}")

    task.add_done_callback(_log_task_error)


async def finalize_turn_and_respond(
    transcript_buffer: TurnTranscriptAggregator,
    assistant: LLMResponder,
    speech_player: SpeechPlayer,
    *,
    hooks: Optional[ResponseLifecycleHooks] = None,
) -> None:
    """Gather a completed turn transcript and fetch an assistant reply."""

    transcript = await _finalize_transcript(transcript_buffer, hooks)
    if not transcript:
        return

    _maybe_schedule_confirmation_cue(assistant, speech_player)

    LOGGER.log(
        TURN_LOG_LABEL,
        f"Transcript ready ({len(transcript)} chars); requesting assistant...",
    )
    LOGGER.verbose(TURN_LOG_LABEL, f"Sending transcript to assistant: {transcript}")

    reply = await _request_assistant_reply(transcript, assistant, hooks)
    if not reply:
        LOGGER.log(ASSISTANT_LOG_LABEL, "(empty response)")
        return

    if reply.text:
        LOGGER.log(ASSISTANT_LOG_LABEL, reply.text)
    else:
        LOGGER.log(ASSISTANT_LOG_LABEL, "(no text content)")

    await _play_assistant_audio(reply, speech_player)


def schedule_turn_response(
    transcript_buffer: TurnTranscriptAggregator,
    assistant: LLMResponder,
    speech_player: SpeechPlayer,
    *,
    hooks: Optional[ResponseLifecycleHooks] = None,
) -> asyncio.Task:
    """Fire-and-forget helper for assistant calls with error reporting."""

    task = asyncio.create_task(
        finalize_turn_and_respond(
            transcript_buffer,
            assistant,
            speech_player,
            hooks=hooks,
        )
    )

    def _log_task_error(fut: asyncio.Task):
        try:
            fut.result()
        except asyncio.CancelledError:
            LOGGER.verbose(TURN_LOG_LABEL, "Assistant reply task cancelled.")
        except Exception as exc:  # pragma: no cover - unexpected
            LOGGER.log(
                ASSISTANT_LOG_LABEL,
                f"Unexpected assistant error: {exc}",
                error=True,
            )

    task.add_done_callback(_log_task_error)
    return task


__all__ = [
    "ResponseLifecycleHooks",
    "finalize_turn_and_respond",
    "schedule_turn_response",
    "should_ignore_server_stop_event",
]
