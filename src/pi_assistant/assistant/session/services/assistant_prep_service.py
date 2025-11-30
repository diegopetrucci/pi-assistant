"""Assistant warm-up and capability probing."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional

from pi_assistant.assistant.llm import LLMResponder
from pi_assistant.cli.logging import ASSISTANT_LOG_LABEL, LOGGER
from pi_assistant.config import CONFIRMATION_CUE_ENABLED, CONFIRMATION_CUE_TEXT

from .session_service import BaseSessionService

if TYPE_CHECKING:
    from pi_assistant.assistant.transcription.session import TranscriptionRunConfig


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
        if not cue_task:
            return

        try:
            if not cue_task.done():
                cue_task.cancel()
            await cue_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            LOGGER.log(
                ASSISTANT_LOG_LABEL,
                f"Confirmation cue cleanup failed: {exc}",
                error=True,
                exc_info=exc,
            )
        finally:
            self._cue_task = None

    def _log_assistant_context(self) -> None:
        assistant = self._assistant
        LOGGER.log(ASSISTANT_LOG_LABEL, f"Using assistant model: {assistant.model_name}")
        enabled_tools = assistant.enabled_tools
        tools_summary = ", ".join(enabled_tools) if enabled_tools else "none"
        LOGGER.log(ASSISTANT_LOG_LABEL, f"Tools enabled: {tools_summary}")
        reasoning_summary = self._config.reasoning_effort or "auto"
        LOGGER.log(ASSISTANT_LOG_LABEL, f"Reasoning effort: {reasoning_summary}")
        location_summary = (assistant.location_name or "").strip() or "unspecified"
        LOGGER.log(ASSISTANT_LOG_LABEL, f"Location context: {location_summary}")

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
                LOGGER.log(
                    ASSISTANT_LOG_LABEL,
                    f"Failed to warm confirmation cue: {exc}",
                    error=True,
                    exc_info=exc,
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
            LOGGER.log(
                ASSISTANT_LOG_LABEL,
                f"Unable to verify Responses audio support: {exc}",
                error=True,
                exc_info=exc,
            )
            self._responses_audio_enabled = False
        self._announce_tts_mode(self._responses_audio_enabled)

    def _announce_tts_mode(self, responses_audio_enabled: bool) -> None:
        assistant = self._assistant
        if responses_audio_enabled:
            LOGGER.log(
                ASSISTANT_LOG_LABEL,
                "Responses audio enabled; streaming assistant replies.",
            )
        elif assistant.tts_enabled:
            if self._config.use_responses_audio:
                LOGGER.log(
                    ASSISTANT_LOG_LABEL,
                    "Responses audio not available; using Audio API for TTS.",
                )
            else:
                LOGGER.log(
                    ASSISTANT_LOG_LABEL,
                    "Local TTS mode active; synthesizing replies after receiving text.",
                )
