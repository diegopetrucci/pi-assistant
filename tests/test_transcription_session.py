import types
from typing import Any, Optional, cast

import pytest

import pi_assistant.assistant.transcription_session as ts
from pi_assistant.assistant.transcription_session import (
    DEFAULT_ASSISTANT_AUDIO_MODE,
    TranscriptionComponentBuilder,
    TranscriptionComponents,
    TranscriptionConfigValidator,
    TranscriptionRunConfig,
    TranscriptionSession,
    TranscriptionTaskCoordinator,
    run_simulated_query_once,
)
from pi_assistant.config import ASSISTANT_MODEL, ASSISTANT_REASONING_EFFORT


class _AudioCaptureStub:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.loop = None

    def start_stream(self, loop):
        self.started = True
        self.loop = loop

    def stop_stream(self):
        self.stopped = True


class _WebSocketStub:
    def __init__(self, *, fail_connect: bool = False):
        self.connected = False
        self.closed = False
        self.fail_connect = fail_connect

    async def connect(self):
        if self.fail_connect:
            raise RuntimeError("connect failed")
        self.connected = True

    async def close(self):
        self.closed = True

    def receive_events(self):
        async def _gen():
            yield {}

        return _gen()


class _AssistantStub:
    def __init__(self):
        self.tts_enabled = True
        self.model_name = "stub-model"
        self.enabled_tools = ("weather",)
        self.location_name = "Test Lab"
        self.verify_calls = 0
        self._responses_audio_result: bool | Exception = True
        self.set_supported: list[bool] = []
        self.warm_calls: list[str] = []
        self.generate_calls: list[str] = []
        self.reply = types.SimpleNamespace(text="ok", audio_bytes=b"bytes", audio_sample_rate=16000)

    async def verify_responses_audio_support(self) -> bool:
        self.verify_calls += 1
        if isinstance(self._responses_audio_result, Exception):
            raise self._responses_audio_result
        return self._responses_audio_result

    def set_responses_audio_supported(self, value: bool) -> None:
        self.set_supported.append(value)

    async def warm_phrase_audio(self, text: str):
        self.warm_calls.append(text)

    async def generate_reply(self, transcript: str):
        self.generate_calls.append(transcript)
        return self.reply


class _SpeechPlayerStub:
    def __init__(self):
        self.play_calls: list[tuple[bytes, Optional[int]]] = []
        self.stop_calls = 0

    async def play(self, audio_bytes: bytes, sample_rate: Optional[int] = None):
        self.play_calls.append((audio_bytes, sample_rate))

    async def stop(self) -> bool:
        self.stop_calls += 1
        return True


def _make_config(
    *,
    assistant_model: str = "gpt-5.1",
    reasoning_effort: Optional[str] = "low",
    assistant_audio_mode: str = "responses",
    use_responses_audio: bool = True,
    simulated_query: Optional[str] = "hello",
) -> TranscriptionRunConfig:
    return TranscriptionRunConfig(
        assistant_model=assistant_model,
        reasoning_effort=reasoning_effort,
        assistant_audio_mode=assistant_audio_mode,
        use_responses_audio=use_responses_audio,
        simulated_query=simulated_query,
    )


def _make_components(
    audio_capture,
    ws_client,
    transcript_buffer,
    assistant,
    speech_player,
) -> TranscriptionComponents:
    return TranscriptionComponents(
        audio_capture=cast(ts.AudioCapture, audio_capture),
        ws_client=cast(ts.WebSocketClient, ws_client),
        transcript_buffer=cast(ts.TurnTranscriptAggregator, transcript_buffer),
        assistant=cast(ts.LLMResponder, assistant),
        speech_player=cast(ts.SpeechPlayer, speech_player),
    )


def test_config_validator_resolves_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ts, "SIMULATED_QUERY_TEXT", "  default question  ", raising=False)
    validator = TranscriptionConfigValidator()

    config = validator.resolve(
        assistant_audio_mode=None,
        simulate_query=None,
        reasoning_effort=None,
        assistant_model=None,
    )

    assert config.assistant_audio_mode == DEFAULT_ASSISTANT_AUDIO_MODE
    assert config.use_responses_audio == (DEFAULT_ASSISTANT_AUDIO_MODE == "responses")
    assert config.assistant_model == ASSISTANT_MODEL
    expected_reasoning = (ASSISTANT_REASONING_EFFORT or "").strip() or None
    assert config.reasoning_effort == expected_reasoning
    assert config.simulated_query == "default question"


def test_config_validator_rejects_invalid_audio_mode() -> None:
    validator = TranscriptionConfigValidator()
    with pytest.raises(ValueError, match="Assistant audio mode 'invalid'"):
        validator.resolve(
            assistant_audio_mode="invalid",
            simulate_query=None,
            reasoning_effort=None,
            assistant_model=None,
        )


def test_config_validator_rejects_invalid_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ts, "reasoning_effort_choices_for_model", lambda *args, **kwargs: ("low",))
    validator = TranscriptionConfigValidator()

    with pytest.raises(ValueError, match="Reasoning effort 'high' is not supported"):
        validator.resolve(
            assistant_audio_mode="responses",
            simulate_query=None,
            reasoning_effort="high",
            assistant_model=None,
        )


def test_config_validator_rejects_minimal_when_web_search_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ts, "reasoning_effort_choices_for_model", lambda *args, **kwargs: ("minimal", "low")
    )
    monkeypatch.setattr(ts, "ASSISTANT_WEB_SEARCH_ENABLED", True, raising=False)
    validator = TranscriptionConfigValidator()

    with pytest.raises(ValueError, match="cannot be used while web search is enabled"):
        validator.resolve(
            assistant_audio_mode="responses",
            simulate_query=None,
            reasoning_effort="minimal",
            assistant_model=None,
        )


def test_component_builder_creates_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, Any] = {}

    class _Recorder:
        def __init__(self, name):
            self.name = name
            created[name] = created.get(name, 0) + 1

    class _Responder:
        def __init__(self, **kwargs):
            created["assistant_kwargs"] = kwargs

    monkeypatch.setattr(ts, "AudioCapture", lambda: _Recorder("audio"))
    monkeypatch.setattr(ts, "WebSocketClient", lambda: _Recorder("ws"))
    monkeypatch.setattr(ts, "TurnTranscriptAggregator", lambda: _Recorder("transcript"))
    monkeypatch.setattr(ts, "LLMResponder", _Responder)
    monkeypatch.setattr(ts, "SpeechPlayer", lambda **_: _Recorder("speech"))

    config = _make_config(reasoning_effort="medium", use_responses_audio=False)
    builder = TranscriptionComponentBuilder(config)
    components = builder.build()

    assert isinstance(components.audio_capture, _Recorder)
    assistant_kwargs = cast(dict[str, Any], created["assistant_kwargs"])
    assert assistant_kwargs["model"] == config.assistant_model
    assert assistant_kwargs["use_responses_audio"] is False
    assert assistant_kwargs["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_transcription_session_runs_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assistant = _AssistantStub()
    audio_capture = _AudioCaptureStub()
    ws_client = _WebSocketStub()
    speech_player = _SpeechPlayerStub()
    transcript_buffer = object()
    components = _make_components(
        audio_capture, ws_client, transcript_buffer, assistant, speech_player
    )
    config = _make_config(simulated_query=None)
    monkeypatch.setattr(ts, "CONFIRMATION_CUE_ENABLED", False, raising=False)

    class _CoordinatorStub(TranscriptionTaskCoordinator):
        def __init__(self, comps, simulated_query):
            self.comps = comps
            self.simulated_query = simulated_query
            self.run_calls = 0

        async def run(self):
            self.run_calls += 1

    logs: list[str] = []
    monkeypatch.setattr(ts, "console_print", lambda message, *args, **kwargs: logs.append(message))

    session = TranscriptionSession(config, components, task_coordinator_cls=_CoordinatorStub)
    async with session:
        await session.run()

    captured = capsys.readouterr()
    assert "System ready" in captured.out
    assert audio_capture.started and audio_capture.stopped
    assert ws_client.connected and ws_client.closed
    assert assistant.verify_calls == 1
    assert any("Reasoning effort" in entry for entry in logs)


@pytest.mark.asyncio
async def test_transcription_session_handles_connect_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assistant = _AssistantStub()
    audio_capture = _AudioCaptureStub()
    ws_client = _WebSocketStub(fail_connect=True)
    speech_player = _SpeechPlayerStub()
    components = _make_components(audio_capture, ws_client, object(), assistant, speech_player)
    config = _make_config()

    session = TranscriptionSession(config, components)
    with pytest.raises(RuntimeError, match="connect failed"):
        async with session:
            await session.run()

    assert audio_capture.started is False
    assert ws_client.closed is False


@pytest.mark.asyncio
async def test_transcription_session_logs_probe_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assistant = _AssistantStub()
    assistant._responses_audio_result = RuntimeError("probe failed")
    audio_capture = _AudioCaptureStub()
    ws_client = _WebSocketStub()
    speech_player = _SpeechPlayerStub()
    components = _make_components(audio_capture, ws_client, object(), assistant, speech_player)
    config = _make_config()

    class _CoordinatorStub(TranscriptionTaskCoordinator):
        def __init__(self, comps, simulated_query):
            self.run_calls = 0

        async def run(self):
            self.run_calls += 1

    session = TranscriptionSession(config, components, task_coordinator_cls=_CoordinatorStub)

    async with session:
        await session.run()

    stderr = capsys.readouterr().err
    assert "Unable to verify Responses audio support" in stderr
    assert assistant.set_supported == [False]


@pytest.mark.asyncio
async def test_task_coordinator_runs_all_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    audio_capture = object()
    ws_client = object()
    transcript_buffer = object()
    assistant = _AssistantStub()
    speech_player = _SpeechPlayerStub()
    components = _make_components(
        audio_capture, ws_client, transcript_buffer, assistant, speech_player
    )
    calls: dict[str, int] = {"audio": 0, "events": 0, "query": 0}

    async def fake_run_audio_controller(*args, **kwargs):
        calls["audio"] += 1
        kwargs["stop_signal"].set()
        kwargs["speech_stopped_signal"].set()

    async def fake_receive_events(*args, **kwargs):
        calls["events"] += 1
        await kwargs["stop_signal"].wait()

    async def fake_simulated_query(query_text, assistant_obj, speech_player_obj):
        assert query_text == "hello"
        calls["query"] += 1

    monkeypatch.setattr(
        "pi_assistant.cli.controller.run_audio_controller",
        fake_run_audio_controller,
    )
    monkeypatch.setattr(
        "pi_assistant.cli.events.receive_transcription_events",
        fake_receive_events,
    )
    monkeypatch.setattr(ts, "run_simulated_query_once", fake_simulated_query)

    coordinator = TranscriptionTaskCoordinator(components, "hello")
    await coordinator.run()

    assert calls == {"audio": 1, "events": 1, "query": 1}


@pytest.mark.asyncio
async def test_run_simulated_query_once_handles_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    assistant = _AssistantStub()
    speech_player = _SpeechPlayerStub()
    logs: list[str] = []
    monkeypatch.setattr(ts, "console_print", lambda message, *args, **kwargs: logs.append(message))

    await run_simulated_query_once(
        " hi ",
        cast(ts.LLMResponder, assistant),
        cast(ts.SpeechPlayer, speech_player),
    )

    assert assistant.generate_calls == ["hi"]
    assert speech_player.play_calls == [(b"bytes", 16000)]
    assert any("Injecting simulated query" in entry for entry in logs)


@pytest.mark.asyncio
async def test_run_simulated_query_once_noop_for_empty() -> None:
    assistant = _AssistantStub()
    speech_player = _SpeechPlayerStub()

    await run_simulated_query_once(
        "   ",
        cast(ts.LLMResponder, assistant),
        cast(ts.SpeechPlayer, speech_player),
    )

    assert assistant.generate_calls == []
