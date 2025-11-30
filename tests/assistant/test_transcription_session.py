import asyncio
import types
from typing import Any, Optional, cast

import pytest

import pi_assistant.assistant.transcription.session as ts
from pi_assistant.assistant.session.services import (
    AssistantPrepService,
    AudioCaptureSessionService,
    BaseSessionService,
    DiagnosticsSessionService,
    SessionSupervisor,
    WebSocketSessionService,
)
from pi_assistant.assistant.transcription.session import (
    DEFAULT_ASSISTANT_AUDIO_MODE,
    TranscriptionComponentBuilder,
    TranscriptionComponents,
    TranscriptionConfigValidator,
    TranscriptionRunConfig,
    TranscriptionSession,
    run_simulated_query_once,
)
from pi_assistant.assistant.transcription.task_coordinator import (
    TranscriptionTaskCoordinator,
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
        self.connect_calls = 0
        self.close_calls = 0

    async def connect(self):
        if self.fail_connect:
            raise RuntimeError("connect failed")
        self.connect_calls += 1
        self.connected = True

    async def close(self):
        self.close_calls += 1
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
        self.reply: types.SimpleNamespace | None = types.SimpleNamespace(
            text="ok", audio_bytes=b"bytes", audio_sample_rate=16000
        )

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


class _BaseServiceStub(BaseSessionService):
    def __init__(self):
        super().__init__("base-stub")
        self.start_calls = 0
        self.stop_calls = 0

    async def _start(self) -> None:
        self.start_calls += 1

    async def _stop(self) -> None:
        self.stop_calls += 1


class _ServiceStub:
    def __init__(
        self,
        name: str,
        *,
        fail_start: bool = False,
        log: Optional[list[str]] = None,
    ):
        self.name = name
        self.ready = asyncio.Event()
        self.fail_start = fail_start
        self.start_calls = 0
        self.stop_calls = 0
        self.events: list[str] = []
        self._log = log

    async def start(self) -> None:
        self.start_calls += 1
        self.events.append(f"start:{self.name}")
        if self._log is not None:
            self._log.append(f"start:{self.name}")
        if self.fail_start:
            raise RuntimeError(f"{self.name} boom")
        self.ready.set()

    async def stop(self) -> None:
        self.stop_calls += 1
        self.events.append(f"stop:{self.name}")
        if self._log is not None:
            self._log.append(f"stop:{self.name}")
        self.ready.clear()


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


def test_config_validator_rejects_reasoning_when_model_disables_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ts, "reasoning_effort_choices_for_model", lambda *args, **kwargs: ())
    validator = TranscriptionConfigValidator()

    with pytest.raises(ValueError, match="reasoning disabled"):
        validator.resolve(
            assistant_audio_mode="responses",
            simulate_query=None,
            reasoning_effort="low",
            assistant_model="gpt-4.1-2025-04-14",
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
        def __init__(self, comps, simulated_query, *, simulated_query_runner=None):
            super().__init__(
                cast(ts.TranscriptionComponents, comps),
                simulated_query,
                simulated_query_runner=simulated_query_runner,
            )
            self.run_calls = 0

        async def run(self):
            self.run_calls += 1

    session = TranscriptionSession(config, components, task_coordinator_cls=_CoordinatorStub)
    async with session:
        await session.run()

    captured = capsys.readouterr()
    assert "System ready" in captured.out
    assert "Reasoning effort" in captured.out
    assert audio_capture.started and audio_capture.stopped
    assert ws_client.connected and ws_client.closed
    assert assistant.verify_calls == 1


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
        def __init__(self, comps, simulated_query, *, simulated_query_runner=None):
            super().__init__(
                cast(ts.TranscriptionComponents, comps),
                simulated_query,
                simulated_query_runner=simulated_query_runner,
            )
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

    async def fake_simulated_query(query_text, components):
        assert query_text == "hello"
        assert components.assistant is assistant
        calls["query"] += 1

    monkeypatch.setattr(
        "pi_assistant.cli.controller.run_audio_controller",
        fake_run_audio_controller,
    )
    monkeypatch.setattr(
        "pi_assistant.cli.events.receive_transcription_events",
        fake_receive_events,
    )
    coordinator = TranscriptionTaskCoordinator(
        components,
        "hello",
        simulated_query_runner=fake_simulated_query,
    )
    await coordinator.run()

    assert calls == {"audio": 1, "events": 1, "query": 1}


@pytest.mark.asyncio
async def test_run_simulated_query_once_handles_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    assistant = _AssistantStub()
    speech_player = _SpeechPlayerStub()
    logs: list[str] = []
    monkeypatch.setattr(
        "pi_assistant.cli.logging.LOGGER.log",
        lambda _source, message, **__: logs.append(message),
    )

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


@pytest.mark.asyncio
async def test_run_simulated_query_once_logs_when_reply_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assistant = _AssistantStub()
    assistant.reply = None
    speech_player = _SpeechPlayerStub()

    await run_simulated_query_once(
        "hello",
        cast(ts.LLMResponder, assistant),
        cast(ts.SpeechPlayer, speech_player),
    )

    captured = capsys.readouterr()
    assert "Simulated query returned no response" in captured.out
    assert speech_player.play_calls == []


@pytest.mark.asyncio
async def test_run_simulated_query_once_logs_when_text_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assistant = _AssistantStub()
    assistant.reply = types.SimpleNamespace(text=None, audio_bytes=None, audio_sample_rate=None)
    speech_player = _SpeechPlayerStub()

    await run_simulated_query_once(
        "hello",
        cast(ts.LLMResponder, assistant),
        cast(ts.SpeechPlayer, speech_player),
    )

    captured = capsys.readouterr()
    assert "(no text content)" in captured.out


@pytest.mark.asyncio
async def test_run_simulated_query_helper_passes_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assistant = _AssistantStub()
    speech_player = _SpeechPlayerStub()
    components = _make_components(
        _AudioCaptureStub(), _WebSocketStub(), object(), assistant, speech_player
    )

    calls: list[str] = []

    async def fake_runner(query_text: str, assistant_obj, speech_player_obj):
        calls.append(query_text)
        assert assistant_obj is assistant
        assert speech_player_obj is speech_player

    monkeypatch.setattr(ts, "run_simulated_query_once", fake_runner)
    await ts._run_simulated_query_with_components("hello world", components)

    assert calls == ["hello world"]


@pytest.mark.asyncio
async def test_session_supervisor_starts_and_stops_services() -> None:
    log: list[str] = []
    services = [
        _ServiceStub("assistant", log=log),
        _ServiceStub("websocket", log=log),
        _ServiceStub("capture", log=log),
    ]

    supervisor = SessionSupervisor(services)
    await supervisor.start_all()
    await supervisor.stop_all()

    assert [svc.start_calls for svc in services] == [1, 1, 1]
    assert [svc.stop_calls for svc in services] == [1, 1, 1]
    assert log == [
        "start:assistant",
        "start:websocket",
        "start:capture",
        "stop:capture",
        "stop:websocket",
        "stop:assistant",
    ]


@pytest.mark.asyncio
async def test_session_supervisor_rolls_back_failed_start() -> None:
    log: list[str] = []
    services = [
        _ServiceStub("assistant", log=log),
        _ServiceStub("websocket", log=log, fail_start=True),
        _ServiceStub("capture", log=log),
    ]

    supervisor = SessionSupervisor(services)
    with pytest.raises(RuntimeError, match="websocket boom"):
        await supervisor.start_all()

    assert log == ["start:assistant", "start:websocket", "stop:assistant"]
    assert services[0].stop_calls == 1
    assert services[1].start_calls == 1
    assert services[1].stop_calls == 0


@pytest.mark.asyncio
async def test_diagnostics_service_waits_for_dependencies(
    capsys: pytest.CaptureFixture[str],
) -> None:
    dep_a = asyncio.Event()
    dep_b = asyncio.Event()
    service = DiagnosticsSessionService((dep_a, dep_b))

    start_task = asyncio.create_task(service.start())
    await asyncio.sleep(0)
    assert service.ready.is_set() is False

    dep_a.set()
    await asyncio.sleep(0)
    assert service.ready.is_set() is False

    dep_b.set()
    await start_task
    assert service.ready.is_set() is True

    output = capsys.readouterr().out
    assert "System ready" in output

    await service.stop()


@pytest.mark.asyncio
async def test_assistant_prep_service_cancels_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    assistant = _AssistantStub()
    config = _make_config()
    blocker = asyncio.Event()

    async def fake_warm(text: str):
        await blocker.wait()

    assistant.warm_phrase_audio = fake_warm
    monkeypatch.setattr(ts, "CONFIRMATION_CUE_ENABLED", True, raising=False)
    monkeypatch.setattr(ts, "CONFIRMATION_CUE_TEXT", "Test cue", raising=False)

    service = AssistantPrepService(cast(ts.LLMResponder, assistant), config)
    await service.start()
    cue_task = service._cue_task
    assert cue_task is not None
    assert cue_task.done() is False
    await service.stop()

    assert cue_task.cancelled() is True


@pytest.mark.asyncio
async def test_audio_capture_service_controls_stream() -> None:
    capture = _AudioCaptureStub()
    service = AudioCaptureSessionService(cast(ts.AudioCapture, capture))

    await service.start()
    assert capture.started is True

    await service.stop()
    assert capture.stopped is True


@pytest.mark.asyncio
async def test_websocket_session_service_manages_connection() -> None:
    ws_client = _WebSocketStub()
    service = WebSocketSessionService(cast(ts.WebSocketClient, ws_client))

    await service.start()
    assert ws_client.connected is True
    assert ws_client.connect_calls == 1
    assert service.ready.is_set()

    await service.start()
    assert ws_client.connect_calls == 1  # idempotent

    await service.stop()
    assert ws_client.close_calls == 1
    assert ws_client.closed is True
    assert service.ready.is_set() is False

    await service.stop()
    assert ws_client.close_calls == 1  # idempotent


@pytest.mark.asyncio
async def test_base_session_service_idempotent_start_stop() -> None:
    service = _BaseServiceStub()

    assert service.ready.is_set() is False
    await service.start()
    assert service.ready.is_set() is True

    await service.start()
    assert service.start_calls == 1

    await service.stop()
    assert service.ready.is_set() is False

    await service.stop()
    assert service.stop_calls == 1
