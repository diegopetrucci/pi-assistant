"""Assistant warm-up and capability probing."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Optional

from pi_assistant.assistant.llm import LLMResponder
from pi_assistant.cli.logging_utils import ASSISTANT_LOG_LABEL, console_print
from pi_assistant.config import CONFIRMATION_CUE_ENABLED, CONFIRMATION_CUE_TEXT

from .session_service import BaseSessionService

if TYPE_CHECKING:
    from pi_assistant.assistant.transcription_session import TranscriptionRunConfig


class AssistantPrepService(BaseSessionService):
    """Warm the assistant and verify its runtime capabilities."""

    def __init__(self, assistant: LLMResponder, config: "TranscriptionRunConfig"):
        super().__init__("assistant")
        self._assistant = assistant
        self._config = config
        self._responses_audio_enabled = False
        self._cue_task: Optional[asyncio.Task] = None

    async def _start(self) -> None:
        self._log_assistant_context()
        self._warm_confirmation_cue()
        await self._probe_responses_audio()

    async def _stop(self) -> None:
        cue_task = self._cue_task
        self._cue_task = None
        if cue_task:
            if not cue_task.done():
                cue_task.cancel()
            try:
                await cue_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                print(
                    f"{ASSISTANT_LOG_LABEL} Confirmation cue cleanup failed: {exc}",
                    file=sys.stderr,
                )

    def _log_assistant_context(self) -> None:
        assistant = self._assistant
        console_print(f"{ASSISTANT_LOG_LABEL} Using assistant model: {assistant.model_name}")
        enabled_tools = assistant.enabled_tools
        tools_summary = ", ".join(enabled_tools) if enabled_tools else "none"
        console_print(f"{ASSISTANT_LOG_LABEL} Tools enabled: {tools_summary}")
        reasoning_summary = self._config.reasoning_effort or "auto"
        console_print(f"{ASSISTANT_LOG_LABEL} Reasoning effort: {reasoning_summary}")
        location_summary = (assistant.location_name or "").strip() or "unspecified"
        console_print(f"{ASSISTANT_LOG_LABEL} Location context: {location_summary}")

    def _warm_confirmation_cue(self) -> None:
        assistant = self._assistant
        if not (assistant.tts_enabled and CONFIRMATION_CUE_ENABLED and CONFIRMATION_CUE_TEXT):
            return

        self._cue_task = asyncio.create_task(assistant.warm_phrase_audio(CONFIRMATION_CUE_TEXT))

        def _log_cue_error(fut: asyncio.Task) -> None:
            try:
                fut.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                print(
                    f"{ASSISTANT_LOG_LABEL} Failed to warm confirmation cue: {exc}",
                    file=sys.stderr,
                )

        self._cue_task.add_done_callback(_log_cue_error)

    async def _probe_responses_audio(self) -> None:
        assistant = self._assistant
        if not (assistant.tts_enabled and self._config.use_responses_audio):
            self._announce_tts_mode(False)
            return

        try:
            self._responses_audio_enabled = await assistant.verify_responses_audio_support()
        except Exception as exc:
            assistant.set_responses_audio_supported(False)
            print(
                f"{ASSISTANT_LOG_LABEL} Unable to verify Responses audio support: {exc}",
                file=sys.stderr,
            )
            self._responses_audio_enabled = False
        self._announce_tts_mode(self._responses_audio_enabled)

    def _announce_tts_mode(self, responses_audio_enabled: bool) -> None:
        assistant = self._assistant
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
