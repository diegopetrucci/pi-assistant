import asyncio
from typing import cast

import pytest

from pi_assistant.assistant.llm import LLMResponder
from pi_assistant.assistant.session_services.assistant_prep_service import (
    AssistantPrepService,
)
from pi_assistant.assistant.transcription_session import TranscriptionRunConfig
from pi_assistant.cli import logging_utils as cli_logging_utils


class _AssistantStub:
    def __init__(self):
        self.model_name = "demo-model"
        self.enabled_tools = ("weather", "search")
        self.location_name = "Garage"
        self.tts_enabled = True
        self.verify_result: bool | Exception = True
        self.verify_calls = 0
        self.warm_calls: list[str] = []
        self.supported_updates: list[bool] = []

    async def warm_phrase_audio(self, text: str):
        self.warm_calls.append(text)

    async def verify_responses_audio_support(self) -> bool:
        self.verify_calls += 1
        if isinstance(self.verify_result, Exception):
            raise self.verify_result
        return cast(bool, self.verify_result)

    def set_responses_audio_supported(self, enabled: bool) -> None:
        self.supported_updates.append(enabled)


def _make_config(*, use_responses_audio: bool, reasoning_effort: str | None = "low"):
    return TranscriptionRunConfig(
        assistant_model="demo-model",
        reasoning_effort=reasoning_effort,
        assistant_audio_mode="responses" if use_responses_audio else "local-tts",
        use_responses_audio=use_responses_audio,
        simulated_query=None,
    )


@pytest.mark.asyncio
async def test_assistant_prep_service_warms_and_announces_responses_audio(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assistant = _AssistantStub()
    config = _make_config(use_responses_audio=True, reasoning_effort="medium")

    logs: list[str] = []
    original_log = cli_logging_utils.LOGGER.log

    def capture_log(source: str, message: str, **kwargs) -> None:
        logs.append(message)
        original_log(source, message, **kwargs)

    monkeypatch.setattr(cli_logging_utils.LOGGER, "log", capture_log)
    monkeypatch.setattr(
        "pi_assistant.assistant.session_services.assistant_prep_service.CONFIRMATION_CUE_ENABLED",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        "pi_assistant.assistant.session_services.assistant_prep_service.CONFIRMATION_CUE_TEXT",
        "Ready when you are",
        raising=False,
    )

    service = AssistantPrepService(cast(LLMResponder, assistant), config)

    await service.start()
    await asyncio.sleep(0)

    combined_logs = "\n".join(logs)
    assert "Using assistant model: demo-model" in combined_logs
    assert "Tools enabled: weather, search" in combined_logs
    assert "Reasoning effort: medium" in combined_logs
    assert "Location context: Garage" in combined_logs

    assert assistant.warm_calls == ["Ready when you are"]
    assert assistant.verify_calls == 1

    captured = capsys.readouterr()
    assert "Responses audio enabled; streaming assistant replies." in captured.out

    await service.stop()


@pytest.mark.asyncio
async def test_assistant_prep_service_reports_responses_audio_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assistant = _AssistantStub()
    assistant.verify_result = RuntimeError("boom")
    config = _make_config(use_responses_audio=True)

    monkeypatch.setattr(
        "pi_assistant.assistant.session_services.assistant_prep_service.CONFIRMATION_CUE_ENABLED",
        False,
        raising=False,
    )
    service = AssistantPrepService(cast(LLMResponder, assistant), config)

    await service.start()
    await asyncio.sleep(0)

    captured = capsys.readouterr()
    assert "Unable to verify Responses audio support" in captured.err
    assert "Responses audio not available; using Audio API for TTS." in captured.out
    assert assistant.supported_updates == [False]

    await service.stop()


@pytest.mark.asyncio
async def test_assistant_prep_service_announces_local_tts_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assistant = _AssistantStub()
    config = _make_config(use_responses_audio=False, reasoning_effort=None)

    monkeypatch.setattr(
        "pi_assistant.assistant.session_services.assistant_prep_service.CONFIRMATION_CUE_ENABLED",
        False,
        raising=False,
    )

    service = AssistantPrepService(cast(LLMResponder, assistant), config)

    await service.start()
    await asyncio.sleep(0)

    captured = capsys.readouterr()
    assert "Local TTS mode active" in captured.out
    assert assistant.verify_calls == 0

    await service.stop()
