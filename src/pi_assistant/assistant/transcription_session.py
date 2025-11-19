"""Helpers for configuring and running a transcription session."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import Optional

from pi_assistant.assistant.llm import LLMResponder
from pi_assistant.assistant.transcript import TurnTranscriptAggregator
from pi_assistant.audio import AudioCapture, SpeechPlayer
from pi_assistant.cli.logging_utils import ASSISTANT_LOG_LABEL, TURN_LOG_LABEL, console_print
from pi_assistant.config import (
    ASSISTANT_MODEL,
    ASSISTANT_REASONING_EFFORT,
    ASSISTANT_TTS_RESPONSES_ENABLED,
    ASSISTANT_TTS_SAMPLE_RATE,
    ASSISTANT_WEB_SEARCH_ENABLED,
    CONFIRMATION_CUE_ENABLED,
    CONFIRMATION_CUE_TEXT,
    SIMULATED_QUERY_TEXT,
    reasoning_effort_choices_for_model,
)
from pi_assistant.network import WebSocketClient

ASSISTANT_AUDIO_MODE_CHOICES = ("responses", "local-tts")
DEFAULT_ASSISTANT_AUDIO_MODE = "responses" if ASSISTANT_TTS_RESPONSES_ENABLED else "local-tts"


@dataclass(frozen=True)
class TranscriptionRunConfig:
    """Resolved configuration for a transcription session run."""

    assistant_model: str
    reasoning_effort: Optional[str]
    assistant_audio_mode: str
    use_responses_audio: bool
    simulated_query: Optional[str]


class TranscriptionConfigValidator:
    """Validate CLI overrides and produce a session config."""

    def resolve(
        self,
        *,
        assistant_audio_mode: Optional[str],
        simulate_query: Optional[str],
        reasoning_effort: Optional[str],
        assistant_model: Optional[str],
    ) -> TranscriptionRunConfig:
        """Normalize inputs and raise if an invalid combination is requested."""

        audio_mode = (assistant_audio_mode or DEFAULT_ASSISTANT_AUDIO_MODE).strip()
        if audio_mode not in ASSISTANT_AUDIO_MODE_CHOICES:
            allowed = ", ".join(ASSISTANT_AUDIO_MODE_CHOICES)
            raise ValueError(f"Assistant audio mode '{audio_mode}' must be one of: {allowed}.")

        model_override = (assistant_model or ASSISTANT_MODEL).strip()
        reasoning_choices = reasoning_effort_choices_for_model(model_override)
        selected_reasoning = (reasoning_effort or ASSISTANT_REASONING_EFFORT or "").strip()
        selected_reasoning = selected_reasoning or None
        if selected_reasoning and selected_reasoning not in reasoning_choices:
            allowed = ", ".join(reasoning_choices)
            raise ValueError(
                f"Reasoning effort '{selected_reasoning}' is not supported by {model_override}. "
                f"Allowed values: {allowed}"
            )
        if selected_reasoning == "minimal" and ASSISTANT_WEB_SEARCH_ENABLED:
            raise ValueError(
                "Reasoning effort 'minimal' cannot be used while web search is enabled. "
                "Disable ASSISTANT_WEB_SEARCH_ENABLED or choose low/medium/high."
            )

        simulated_query_text = self._resolve_simulated_query(simulate_query)
        use_responses_audio = audio_mode == "responses"
        return TranscriptionRunConfig(
            assistant_model=model_override,
            reasoning_effort=selected_reasoning,
            assistant_audio_mode=audio_mode,
            use_responses_audio=use_responses_audio,
            simulated_query=simulated_query_text,
        )

    def _resolve_simulated_query(self, override: Optional[str]) -> Optional[str]:
        base = SIMULATED_QUERY_TEXT if override is None else override
        return (base or "").strip() or None


@dataclass
class TranscriptionComponents:
    """Container for runtime dependencies."""

    audio_capture: AudioCapture
    ws_client: WebSocketClient
    transcript_buffer: TurnTranscriptAggregator
    assistant: LLMResponder
    speech_player: SpeechPlayer


class TranscriptionComponentBuilder:
    """Factory for creating strongly-typed session components."""

    def __init__(self, config: TranscriptionRunConfig):
        self._config = config

    def build(self) -> TranscriptionComponents:
        audio_capture = AudioCapture()
        ws_client = WebSocketClient()
        transcript_buffer = TurnTranscriptAggregator()
        assistant = LLMResponder(
            model=self._config.assistant_model,
            use_responses_audio=self._config.use_responses_audio,
            reasoning_effort=self._config.reasoning_effort,
        )
        speech_player = SpeechPlayer(default_sample_rate=ASSISTANT_TTS_SAMPLE_RATE)
        return TranscriptionComponents(
            audio_capture=audio_capture,
            ws_client=ws_client,
            transcript_buffer=transcript_buffer,
            assistant=assistant,
            speech_player=speech_player,
        )


class TranscriptionTaskCoordinator:
    """Orchestrate the long-running tasks that make up a session."""

    def __init__(
        self,
        components: TranscriptionComponents,
        simulated_query: Optional[str],
    ):
        self._components = components
        self._simulated_query = simulated_query
        self._stop_signal = asyncio.Event()
        self._speech_stopped_signal = asyncio.Event()

    async def run(self) -> None:
        async with asyncio.TaskGroup() as task_group:
            task_group.create_task(self._run_audio_controller())
            task_group.create_task(self._run_event_receiver())
            if self._simulated_query:
                task_group.create_task(
                    run_simulated_query_once(
                        self._simulated_query,
                        self._components.assistant,
                        self._components.speech_player,
                    )
                )

    async def _run_audio_controller(self) -> None:
        from pi_assistant.cli.controller import run_audio_controller

        await run_audio_controller(
            self._components.audio_capture,
            self._components.ws_client,
            transcript_buffer=self._components.transcript_buffer,
            assistant=self._components.assistant,
            speech_player=self._components.speech_player,
            stop_signal=self._stop_signal,
            speech_stopped_signal=self._speech_stopped_signal,
        )

    async def _run_event_receiver(self) -> None:
        from pi_assistant.cli.events import receive_transcription_events

        await receive_transcription_events(
            self._components.ws_client,
            self._components.transcript_buffer,
            self._components.speech_player,
            stop_signal=self._stop_signal,
            speech_stopped_signal=self._speech_stopped_signal,
        )


class TranscriptionSession:
    """Context manager that wires configuration, components, and tasks together."""

    def __init__(
        self,
        config: TranscriptionRunConfig,
        components: TranscriptionComponents,
        *,
        task_coordinator_cls=TranscriptionTaskCoordinator,
    ):
        self._config = config
        self._components = components
        self._task_coordinator_cls = task_coordinator_cls
        self._responses_audio_enabled = False
        self._audio_started = False
        self._ws_connected = False
        self._cue_task: asyncio.Task | None = None

    async def __aenter__(self) -> TranscriptionSession:
        try:
            self._log_assistant_context()
            self._warm_confirmation_cue()
            await self._probe_responses_audio()
            await self._start_streams()
        except Exception:
            await self._teardown_partial()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        print("Cleaning up...")
        try:
            await self._teardown_partial()
        finally:
            print("✓ Shutdown complete\n")

    async def run(self) -> None:
        coordinator = self._task_coordinator_cls(self._components, self._config.simulated_query)
        await coordinator.run()

    def _log_assistant_context(self) -> None:
        assistant = self._components.assistant
        console_print(f"{ASSISTANT_LOG_LABEL} Using assistant model: {assistant.model_name}")
        enabled_tools = assistant.enabled_tools
        tools_summary = ", ".join(enabled_tools) if enabled_tools else "none"
        console_print(f"{ASSISTANT_LOG_LABEL} Tools enabled: {tools_summary}")
        reasoning_summary = self._config.reasoning_effort or "auto"
        console_print(f"{ASSISTANT_LOG_LABEL} Reasoning effort: {reasoning_summary}")
        location_summary = (assistant.location_name or "").strip() or "unspecified"
        console_print(f"{ASSISTANT_LOG_LABEL} Location context: {location_summary}")

    def _warm_confirmation_cue(self) -> None:
        assistant = self._components.assistant
        if not (assistant.tts_enabled and CONFIRMATION_CUE_ENABLED and CONFIRMATION_CUE_TEXT):
            return

        self._cue_task = asyncio.create_task(assistant.warm_phrase_audio(CONFIRMATION_CUE_TEXT))

        def _log_cue_error(fut: asyncio.Task) -> None:
            try:
                fut.result()
            except Exception:
                pass

        self._cue_task.add_done_callback(_log_cue_error)

    async def _probe_responses_audio(self) -> None:
        assistant = self._components.assistant
        if not (assistant.tts_enabled and self._config.use_responses_audio):
            self._announce_tts_mode(False)
            return

        try:
            self._responses_audio_enabled = await assistant.verify_responses_audio_support()
        except Exception as exc:  # pragma: no cover - network failure
            assistant.set_responses_audio_supported(False)
            print(
                f"{ASSISTANT_LOG_LABEL} Unable to verify Responses audio support: {exc}",
                file=sys.stderr,
            )
            self._responses_audio_enabled = False
        self._announce_tts_mode(self._responses_audio_enabled)

    def _announce_tts_mode(self, responses_audio_enabled: bool) -> None:
        assistant = self._components.assistant
        if responses_audio_enabled:
            print(f"{ASSISTANT_LOG_LABEL} Responses audio enabled; streaming assistant replies.")
        elif assistant.tts_enabled:
            if self._config.use_responses_audio:
                print(
                    f"{ASSISTANT_LOG_LABEL} Responses audio not available; using Audio API for TTS."
                )
            else:
                print(
                    f"{ASSISTANT_LOG_LABEL} Local TTS mode active; "
                    "synthesizing replies after receiving text."
                )

    async def _start_streams(self) -> None:
        loop = asyncio.get_running_loop()
        await self._components.ws_client.connect()
        self._ws_connected = True

        self._components.audio_capture.start_stream(loop)
        self._audio_started = True

        print("\n✓ System ready")
        print("Listening... (Press Ctrl+C to stop)\n")

    async def _teardown_partial(self) -> None:
        if self._audio_started:
            try:
                self._components.audio_capture.stop_stream()
            finally:
                self._audio_started = False

        if self._ws_connected:
            try:
                await self._components.ws_client.close()
            finally:
                self._ws_connected = False


async def run_simulated_query_once(
    query_text: str,
    assistant: LLMResponder,
    speech_player: SpeechPlayer,
) -> None:
    """Inject a one-off synthetic transcript to drive the assistant."""

    cleaned = query_text.strip()
    if not cleaned:
        return

    console_print(f"{TURN_LOG_LABEL} Injecting simulated query: {cleaned}")
    try:
        reply = await assistant.generate_reply(cleaned)
    except Exception as exc:  # pragma: no cover - network failure
        console_print(f"{ASSISTANT_LOG_LABEL} Simulated query failed: {exc}", file=sys.stderr)
        return

    if not reply:
        console_print(f"{ASSISTANT_LOG_LABEL} Simulated query returned no response.")
        return

    if reply.text:
        console_print(f"{ASSISTANT_LOG_LABEL} {reply.text}")
    else:
        console_print(f"{ASSISTANT_LOG_LABEL} (no text content)")

    if reply.audio_bytes:
        try:
            await speech_player.play(reply.audio_bytes, sample_rate=reply.audio_sample_rate)
        except Exception as exc:  # pragma: no cover - host audio failure
            console_print(
                f"{ASSISTANT_LOG_LABEL} Error playing simulated reply audio: {exc}",
                file=sys.stderr,
            )
