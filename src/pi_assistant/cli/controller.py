"""Audio controller loop and assistant interaction helpers for the CLI."""

from __future__ import annotations

from typing import Optional

from pi_assistant.assistant import LLMResponder, TurnTranscriptAggregator
from pi_assistant.audio import SpeechPlayer
from pi_assistant.audio.resampler import LinearResampler
from pi_assistant.cli.controller_components import (
    AudioChunkPreparer,
    ResponseTaskManager,
    SilenceTracker,
    StreamStateManager,
)
from pi_assistant.cli.controller_helpers import (
    ResponseLifecycleHooks,
    schedule_turn_response,
    should_ignore_server_stop_event,
)
from pi_assistant.cli.logging_utils import (
    CONTROL_LOG_LABEL,
    ERROR_LOG_LABEL,
    LOGGER,
    TURN_LOG_LABEL,
    WAKE_LOG_LABEL,
    log_state_transition,
)
from pi_assistant.config import (
    AUTO_STOP_ENABLED,
    AUTO_STOP_MAX_SILENCE_SECONDS,
    AUTO_STOP_SILENCE_THRESHOLD,
    CHANNELS,
    SAMPLE_RATE,
    SERVER_STOP_MIN_SILENCE_SECONDS,
    SERVER_STOP_TIMEOUT_SECONDS,
    STREAM_SAMPLE_RATE,
)
from pi_assistant.network import WebSocketClient
from pi_assistant.wake_word import PreRollBuffer, StreamState, WakeWordEngine

from .controller_context import AudioControllerContext
from .controller_loop import _AudioControllerLoop


async def run_audio_controller(
    audio_capture,
    ws_client: WebSocketClient,
    *,
    context: Optional[AudioControllerContext] = None,
    **legacy_kwargs,
) -> None:
    """Multiplex microphone audio between the wake-word detector and the OpenAI stream."""

    if context and legacy_kwargs:
        raise TypeError("Provide either `context` or individual controller dependencies, not both.")
    if context is None:
        try:
            context = AudioControllerContext(**legacy_kwargs)
        except TypeError as exc:  # pragma: no cover - invalid wiring
            raise TypeError(f"Invalid audio controller configuration: {exc}") from exc
    runner = _AudioControllerLoop(audio_capture, ws_client, context)
    await runner.run()


__all__ = [
    "AUTO_STOP_ENABLED",
    "AUTO_STOP_MAX_SILENCE_SECONDS",
    "AUTO_STOP_SILENCE_THRESHOLD",
    "AudioChunkPreparer",
    "AudioControllerContext",
    "CHANNELS",
    "CONTROL_LOG_LABEL",
    "ERROR_LOG_LABEL",
    "LLMResponder",
    "LinearResampler",
    "TURN_LOG_LABEL",
    "WAKE_LOG_LABEL",
    "ResponseLifecycleHooks",
    "ResponseTaskManager",
    "SAMPLE_RATE",
    "STREAM_SAMPLE_RATE",
    "SERVER_STOP_TIMEOUT_SECONDS",
    "SERVER_STOP_MIN_SILENCE_SECONDS",
    "SilenceTracker",
    "StreamState",
    "StreamStateManager",
    "SpeechPlayer",
    "TurnTranscriptAggregator",
    "WakeWordEngine",
    "PreRollBuffer",
    "_AudioControllerLoop",
    "LOGGER",
    "log_state_transition",
    "run_audio_controller",
    "schedule_turn_response",
    "should_ignore_server_stop_event",
    "WebSocketClient",
]
