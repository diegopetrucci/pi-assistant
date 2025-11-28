"""Helpers for configuring and running a transcription session."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

from pi_assistant.assistant.llm import LLMResponder
from pi_assistant.assistant.session_services import (
    AssistantPrepService,
    AudioCaptureSessionService,
    DiagnosticsSessionService,
    SessionService,
    SessionSupervisor,
    WebSocketSessionService,
)
from pi_assistant.assistant.transcript import TurnTranscriptAggregator
from pi_assistant.assistant.transcription_task_coordinator import (
    TranscriptionTaskCoordinator,
)
from pi_assistant.audio import AudioCapture, SpeechPlayer
from pi_assistant.cli.logging_utils import ASSISTANT_LOG_LABEL, TURN_LOG_LABEL, console_print
from pi_assistant.config import (
    ASSISTANT_MODEL,
    ASSISTANT_REASONING_EFFORT,
    ASSISTANT_TTS_RESPONSES_ENABLED,
    ASSISTANT_TTS_SAMPLE_RATE,
    ASSISTANT_WEB_SEARCH_ENABLED,
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
            allowed = (
                ", ".join(reasoning_choices) if reasoning_choices else "none (reasoning disabled)"
            )
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
        self._supervisor: Optional[SessionSupervisor] = None

    async def __aenter__(self) -> TranscriptionSession:
        services = self._build_services()
        self._supervisor = SessionSupervisor(services)
        await self._supervisor.start_all()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        print("Cleaning up...")
        supervisor = self._supervisor
        if supervisor:
            await supervisor.stop_all()
        print("✓ Shutdown complete\n")

    async def run(self) -> None:
        coordinator = self._task_coordinator_cls(
            self._components,
            self._config.simulated_query,
            simulated_query_runner=_run_simulated_query_with_components,
        )
        await coordinator.run()

    def _build_services(self) -> list[SessionService]:
        assistant_service = AssistantPrepService(self._components.assistant, self._config)
        websocket_service = WebSocketSessionService(self._components.ws_client)
        audio_service = AudioCaptureSessionService(self._components.audio_capture)
        diagnostics_service = DiagnosticsSessionService(
            (assistant_service.ready, websocket_service.ready, audio_service.ready)
        )
        return [assistant_service, websocket_service, audio_service, diagnostics_service]


async def _run_simulated_query_with_components(
    query_text: str,
    components: TranscriptionComponents,
) -> None:
    await run_simulated_query_once(
        query_text,
        components.assistant,
        components.speech_player,
    )


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
