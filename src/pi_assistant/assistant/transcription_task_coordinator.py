"""Coordinate the long-running controller tasks for a session."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Optional

if TYPE_CHECKING:
    from pi_assistant.assistant.transcription_session import TranscriptionComponents


SimulatedQueryRunner = Callable[
    [str, "TranscriptionComponents"],
    Coroutine[Any, Any, None],
]


class TranscriptionTaskCoordinator:
    """Orchestrate the long-running tasks that make up a session."""

    def __init__(
        self,
        components: "TranscriptionComponents",
        simulated_query: Optional[str],
        *,
        simulated_query_runner: Optional[SimulatedQueryRunner] = None,
    ):
        self._components = components
        self._simulated_query = simulated_query
        self._stop_signal = asyncio.Event()
        self._speech_stopped_signal = asyncio.Event()
        self._simulated_query_runner = simulated_query_runner

    async def run(self) -> None:
        coroutines: list[Coroutine[Any, Any, None]] = [
            self._run_audio_controller(),
            self._run_event_receiver(),
        ]
        if self._simulated_query and self._simulated_query_runner:
            coroutines.append(self._simulated_query_runner(self._simulated_query, self._components))

        async with asyncio.TaskGroup() as task_group:
            for coroutine in coroutines:
                task_group.create_task(coroutine)

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
