import asyncio
import os
import sys
import types
from collections.abc import Awaitable, Callable
from typing import Any, Optional, cast

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LOCATION_NAME", "Test City")

if "audioop" not in sys.modules:
    audioop_stub = types.ModuleType("audioop")

    def _ratecv(audio_bytes, width, channels, src_rate, dst_rate, state):  # noqa: PLR0913
        return audio_bytes, state

    cast(Any, audioop_stub).ratecv = _ratecv
    sys.modules["audioop"] = audioop_stub

from pi_assistant.cli.app import (
    SIMULATED_QUERY_FALLBACK,
    main,
    parse_args,
    run_transcription,
)
from pi_assistant.config import ASSISTANT_MODEL, ASSISTANT_REASONING_EFFORT


def _run_parse(monkeypatch: pytest.MonkeyPatch, argv: list[str]):
    monkeypatch.setattr(sys, "argv", argv)
    return parse_args()


def _run_coro_sync(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _run_parse(monkeypatch, ["pi-assistant"])

    assert args.mode == "run"
    assert args.verbose is False
    assert args.assistant_audio_mode is None
    assert args.simulate_query is None
    assert args.reasoning_effort is None
    assert args.assistant_model is None
    assert args.reset is False


def test_parse_args_invalid_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        _run_parse(monkeypatch, ["pi-assistant", "invalid-mode"])


def test_parse_args_test_audio_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _run_parse(monkeypatch, ["pi-assistant", "test-audio"])

    assert args.mode == "test-audio"


def test_parse_args_test_websocket_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _run_parse(monkeypatch, ["pi-assistant", "test-websocket"])

    assert args.mode == "test-websocket"


def test_parse_args_with_audio_mode_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _run_parse(monkeypatch, ["pi-assistant", "--assistant-audio-mode", "local-tts"])

    assert args.assistant_audio_mode == "local-tts"


def test_parse_args_simulate_query_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _run_parse(monkeypatch, ["pi-assistant", "--simulate-query"])

    assert args.simulate_query == SIMULATED_QUERY_FALLBACK


def test_parse_args_simulate_query_custom_text(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _run_parse(monkeypatch, ["pi-assistant", "--simulate-query", "Hello there"])

    assert args.simulate_query == "Hello there"


def test_parse_args_reasoning_effort_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _run_parse(monkeypatch, ["pi-assistant", "--reasoning-effort", "medium"])

    assert args.reasoning_effort == "medium"


def test_parse_args_assistant_model_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _run_parse(monkeypatch, ["pi-assistant", "--assistant-model", "5.1"])

    assert args.assistant_model == "gpt-5.1-2025-11-13"


def test_parse_args_assistant_model_nano_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _run_parse(monkeypatch, ["pi-assistant", "--assistant-model", "nano"])

    assert args.assistant_model == "gpt-5-nano-2025-08-07"


def test_parse_args_reset_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _run_parse(monkeypatch, ["pi-assistant", "--reset"])

    assert args.reset is True


class _StubAudioCapture:
    def __init__(self):
        self.loop = None
        self.started = False
        self.stopped = False

    def start_stream(self, loop):
        self.started = True
        self.loop = loop

    def stop_stream(self):
        self.stopped = True


class _StubWebSocketClient:
    def __init__(self, *, fail_connect: bool = False, raise_on_close: bool = False):
        self.connected = False
        self.closed = False
        self.fail_connect = fail_connect
        self.raise_on_close = raise_on_close

    async def connect(self):
        if self.fail_connect:
            raise RuntimeError("connect failed")
        self.connected = True

    async def close(self):
        self.closed = True
        if self.raise_on_close:
            raise RuntimeError("close failed")


class _StubAssistant:
    def __init__(
        self,
        *,
        tts_enabled: bool = True,
        supports_audio: bool = True,
        model_name: str = "test-model",
        tools: tuple[str, ...] = (),
        location_name: str = "Test City",
    ):
        self.tts_enabled = tts_enabled
        self._supports_audio = supports_audio
        self._model_name = model_name
        self._tools = tools
        self._location_name = location_name
        self.verify_calls = 0
        self.resp_audio_supported = None
        self.warm_calls: list[str] = []
        self.cached_cues: dict[str, tuple[bytes, int]] = {}
        self.generate_calls: list[str] = []

    async def verify_responses_audio_support(self) -> bool:
        self.verify_calls += 1
        if isinstance(self._supports_audio, Exception):
            raise self._supports_audio
        return self._supports_audio

    def set_responses_audio_supported(self, value: bool) -> None:
        self.resp_audio_supported = value

    async def warm_phrase_audio(self, text: str):
        self.warm_calls.append(text)
        return self.cached_cues.get(text)

    def peek_phrase_audio(self, text: str):
        return self.cached_cues.get(text)

    async def generate_reply(self, transcript: str):
        self.generate_calls.append(transcript)
        return types.SimpleNamespace(
            text=f"Echo: {transcript}",
            audio_bytes=None,
            audio_sample_rate=None,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def enabled_tools(self) -> tuple[str, ...]:
        return self._tools

    @property
    def location_name(self) -> str:
        return self._location_name


class _StubSpeechPlayer:
    def __init__(self, default_sample_rate: int):
        self.sample_rate = default_sample_rate
        self.play_calls: list[tuple[bytes, Optional[int]]] = []

    async def play(self, audio_bytes: bytes, sample_rate: Optional[int] = None):
        self.play_calls.append((audio_bytes, sample_rate))


def _patch_run_transcription_deps(  # noqa: PLR0913
    monkeypatch: pytest.MonkeyPatch,
    *,
    audio_capture: _StubAudioCapture | None = None,
    ws_client: _StubWebSocketClient | None = None,
    assistant: _StubAssistant | None = None,
    run_audio_fn: Callable[..., Awaitable[None]] | None = None,
    receive_fn: Callable[..., Awaitable[None]] | None = None,
) -> dict[str, object]:
    created: dict[str, object] = {"assistant_kwargs": None}
    audio_capture = audio_capture or _StubAudioCapture()
    ws_client = ws_client or _StubWebSocketClient()
    assistant = assistant or _StubAssistant()

    monkeypatch.setattr("pi_assistant.cli.app.AudioCapture", lambda: audio_capture)
    monkeypatch.setattr("pi_assistant.cli.app.WebSocketClient", lambda: ws_client)

    def _create_assistant(**kwargs):
        created["assistant_kwargs"] = kwargs
        return assistant

    monkeypatch.setattr("pi_assistant.cli.app.LLMResponder", _create_assistant)
    monkeypatch.setattr("pi_assistant.cli.app.TurnTranscriptAggregator", lambda: object())

    def _create_speech_player(**kwargs):
        player = _StubSpeechPlayer(**kwargs)
        created["speech_player"] = player
        return player

    monkeypatch.setattr("pi_assistant.cli.app.SpeechPlayer", _create_speech_player)

    async def default_run_audio_controller(*args, **kwargs):
        kwargs["stop_signal"].set()
        kwargs["speech_stopped_signal"].set()

    async def default_receive(*args, **kwargs):
        await kwargs["stop_signal"].wait()

    run_audio = run_audio_fn or default_run_audio_controller
    receive = receive_fn or default_receive

    monkeypatch.setattr("pi_assistant.cli.app.run_audio_controller", run_audio)
    monkeypatch.setattr("pi_assistant.cli.app.receive_transcription_events", receive)

    created["audio_capture"] = audio_capture
    created["ws_client"] = ws_client
    created["assistant"] = assistant
    created["speech_player"] = created.get("speech_player")
    created["run_audio"] = run_audio
    created["receive"] = receive
    return created


@pytest.mark.asyncio
async def test_run_transcription_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_audio_controller(*args, **kwargs):
        kwargs["stop_signal"].set()
        kwargs["speech_stopped_signal"].set()

    deps = _patch_run_transcription_deps(monkeypatch, run_audio_fn=fake_run_audio_controller)

    await run_transcription(assistant_audio_mode="responses")

    audio = cast(_StubAudioCapture, deps["audio_capture"])
    ws_client = cast(_StubWebSocketClient, deps["ws_client"])
    assistant = cast(_StubAssistant, deps["assistant"])
    assistant_kwargs = cast(Optional[dict[str, object]], deps["assistant_kwargs"])

    assert audio.started and audio.loop is asyncio.get_running_loop()
    assert audio.stopped
    assert ws_client.connected and ws_client.closed
    assert assistant.verify_calls == 1
    assert assistant_kwargs is not None
    use_responses_audio = assistant_kwargs.get("use_responses_audio")
    assert isinstance(use_responses_audio, bool)
    assert use_responses_audio is True
    assert assistant_kwargs.get("reasoning_effort") == ASSISTANT_REASONING_EFFORT
    assert assistant_kwargs.get("model") == ASSISTANT_MODEL


@pytest.mark.asyncio
async def test_run_transcription_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    class _InterruptWebSocket(_StubWebSocketClient):
        async def connect(self):
            raise KeyboardInterrupt

    ws_client = _InterruptWebSocket()
    deps = _patch_run_transcription_deps(monkeypatch, ws_client=ws_client)

    await run_transcription(assistant_audio_mode="responses")

    audio_capture = cast(_StubAudioCapture, deps["audio_capture"])
    ws_client = cast(_StubWebSocketClient, deps["ws_client"])

    assert audio_capture.stopped is True
    assert ws_client.closed is True


@pytest.mark.asyncio
async def test_run_transcription_propagates_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    ws_client = _StubWebSocketClient(fail_connect=True)
    deps = _patch_run_transcription_deps(monkeypatch, ws_client=ws_client)

    with pytest.raises(RuntimeError, match="connect failed"):
        await run_transcription(assistant_audio_mode="responses")

    audio_capture = cast(_StubAudioCapture, deps["audio_capture"])
    ws_client = cast(_StubWebSocketClient, deps["ws_client"])

    assert audio_capture.stopped is True
    assert ws_client.closed is True


@pytest.mark.asyncio
async def test_run_transcription_local_tts_mode_skips_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps = _patch_run_transcription_deps(monkeypatch)

    await run_transcription(assistant_audio_mode="local-tts")

    assistant = cast(_StubAssistant, deps["assistant"])
    assistant_kwargs = cast(Optional[dict[str, object]], deps["assistant_kwargs"])
    assert assistant.verify_calls == 0
    assert assistant_kwargs is not None
    use_responses_audio = assistant_kwargs.get("use_responses_audio")
    assert isinstance(use_responses_audio, bool)
    assert use_responses_audio is False


@pytest.mark.asyncio
async def test_run_transcription_simulated_query(monkeypatch: pytest.MonkeyPatch) -> None:
    deps = _patch_run_transcription_deps(monkeypatch)

    await run_transcription(
        assistant_audio_mode="responses",
        simulate_query="Testing 123",
    )

    assistant = cast(_StubAssistant, deps["assistant"])
    assert assistant.generate_calls == ["Testing 123"]


@pytest.mark.asyncio
async def test_run_transcription_applies_reasoning_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps = _patch_run_transcription_deps(monkeypatch)

    await run_transcription(assistant_audio_mode="responses", reasoning_effort="high")

    assistant_kwargs = cast(Optional[dict[str, object]], deps["assistant_kwargs"])
    assert assistant_kwargs is not None
    assert assistant_kwargs.get("reasoning_effort") == "high"


@pytest.mark.asyncio
async def test_run_transcription_logs_reasoning_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    logs: list[str] = []

    def fake_console_print(*args, **kwargs):
        logs.append(" ".join(str(arg) for arg in args))

    monkeypatch.setattr("pi_assistant.cli.app.console_print", fake_console_print)
    _patch_run_transcription_deps(monkeypatch)

    await run_transcription(assistant_audio_mode="responses", reasoning_effort="medium")

    assert any("Reasoning effort: medium" in entry for entry in logs)
    assert any("Location context: Test City" in entry for entry in logs)


@pytest.mark.asyncio
async def test_run_transcription_honors_assistant_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps = _patch_run_transcription_deps(monkeypatch)
    override_model = "gpt-5.1-2025-11-13"

    await run_transcription(
        assistant_audio_mode="responses",
        assistant_model=override_model,
        reasoning_effort="none",
    )

    assistant_kwargs = cast(Optional[dict[str, object]], deps["assistant_kwargs"])
    assert assistant_kwargs is not None
    assert assistant_kwargs.get("model") == override_model
    assert assistant_kwargs.get("reasoning_effort") == "none"


@pytest.mark.asyncio
async def test_run_transcription_rejects_invalid_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run_transcription_deps(monkeypatch)
    monkeypatch.setattr(
        "pi_assistant.cli.app.reasoning_effort_choices_for_model",
        lambda model: ("low",),
        raising=False,
    )

    with pytest.raises(ValueError, match="Reasoning effort 'high' is not supported"):
        await run_transcription(reasoning_effort="high")


@pytest.mark.asyncio
async def test_run_transcription_rejects_minimal_when_web_search_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_run_transcription_deps(monkeypatch)
    monkeypatch.setattr(
        "pi_assistant.cli.app.reasoning_effort_choices_for_model",
        lambda model: ("minimal", "low"),
        raising=False,
    )
    monkeypatch.setattr("pi_assistant.cli.app.ASSISTANT_WEB_SEARCH_ENABLED", True, raising=False)

    with pytest.raises(ValueError, match="cannot be used while web search is enabled"):
        await run_transcription(reasoning_effort="minimal")


def test_main_run_mode_invokes_transcription(monkeypatch: pytest.MonkeyPatch) -> None:
    args = types.SimpleNamespace(
        mode="run",
        verbose=True,
        assistant_audio_mode=None,
        simulate_query=None,
        reasoning_effort=None,
        assistant_model=None,
        reset=False,
    )
    monkeypatch.setattr("pi_assistant.cli.app.parse_args", lambda: args)

    calls: dict[str, object] = {}

    async def fake_run_transcription(
        *,
        assistant_audio_mode: Optional[str],
        simulate_query: Optional[str],
        reasoning_effort: Optional[str],
        assistant_model: Optional[str],
    ):
        calls["audio_mode"] = assistant_audio_mode
        calls["simulate_query"] = simulate_query
        calls["reasoning_effort"] = reasoning_effort
        calls["assistant_model"] = assistant_model

    monkeypatch.setattr("pi_assistant.cli.app.run_transcription", fake_run_transcription)
    monkeypatch.setattr(
        "pi_assistant.cli.app.set_verbose_logging",
        lambda verbose: calls.setdefault("verbose", verbose),
    )
    monkeypatch.setattr("pi_assistant.cli.app.asyncio.run", _run_coro_sync)

    main()

    assert calls["verbose"] is True
    assert calls["audio_mode"] is None
    assert calls["simulate_query"] is None
    assert calls["reasoning_effort"] is None
    assert calls["assistant_model"] is None


def test_main_reset_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    args = types.SimpleNamespace(
        mode="run",
        verbose=False,
        assistant_audio_mode=None,
        simulate_query=None,
        reasoning_effort=None,
        assistant_model=None,
        reset=True,
    )
    monkeypatch.setattr("pi_assistant.cli.app.parse_args", lambda: args)

    logs: list[str] = []

    def fake_console_print(message: str, *unused_args, **unused_kwargs):
        logs.append(message)

    monkeypatch.setattr("pi_assistant.cli.app.console_print", fake_console_print)

    cleared = {"ASSISTANT_MODEL", "LOCATION_NAME"}
    monkeypatch.setattr("pi_assistant.cli.app.reset_first_launch_choices", lambda: cleared)

    asyncio_called = {"value": False}

    def fake_asyncio_run(*_args, **_kwargs):
        asyncio_called["value"] = True

    monkeypatch.setattr("pi_assistant.cli.app.asyncio.run", fake_asyncio_run)

    main()

    assert any("Cleared saved selections" in entry for entry in logs)
    assert asyncio_called["value"] is False


def test_main_runs_audio_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    args = types.SimpleNamespace(
        mode="test-audio",
        verbose=False,
        assistant_audio_mode=None,
        simulate_query=None,
        reasoning_effort=None,
        assistant_model=None,
    )
    monkeypatch.setattr("pi_assistant.cli.app.parse_args", lambda: args)
    monkeypatch.setattr("pi_assistant.cli.app.asyncio.run", _run_coro_sync)

    calls: dict[str, bool] = {}

    async def fake_test_audio_capture():
        calls["audio"] = True

    monkeypatch.setattr("pi_assistant.cli.app.test_audio_capture", fake_test_audio_capture)
    monkeypatch.setattr("pi_assistant.cli.app.set_verbose_logging", lambda verbose: None)

    main()

    assert calls["audio"] is True


def test_main_runs_websocket_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    args = types.SimpleNamespace(
        mode="test-websocket",
        verbose=False,
        assistant_audio_mode=None,
        simulate_query=None,
        reasoning_effort=None,
        assistant_model=None,
    )
    monkeypatch.setattr("pi_assistant.cli.app.parse_args", lambda: args)
    monkeypatch.setattr("pi_assistant.cli.app.asyncio.run", _run_coro_sync)

    calls: dict[str, object] = {}

    async def fake_test_websocket_client(handler):
        calls["handler"] = handler

    monkeypatch.setattr("pi_assistant.cli.app.test_websocket_client", fake_test_websocket_client)
    monkeypatch.setattr("pi_assistant.cli.app.set_verbose_logging", lambda verbose: None)

    main()

    from pi_assistant.cli import app as cli_app

    assert calls["handler"] is cli_app.handle_transcription_event


def test_main_exits_on_unhandled_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    args = types.SimpleNamespace(
        mode="run",
        verbose=False,
        assistant_audio_mode=None,
        simulate_query=None,
        reasoning_effort=None,
        assistant_model=None,
    )
    monkeypatch.setattr("pi_assistant.cli.app.parse_args", lambda: args)
    monkeypatch.setattr("pi_assistant.cli.app.asyncio.run", _run_coro_sync)

    async def fake_run_transcription(
        *,
        assistant_audio_mode: Optional[str],
        simulate_query: Optional[str],
        reasoning_effort: Optional[str],
        assistant_model: Optional[str],
    ):
        raise RuntimeError("boom")

    monkeypatch.setattr("pi_assistant.cli.app.run_transcription", fake_run_transcription)
    monkeypatch.setattr("pi_assistant.cli.app.set_verbose_logging", lambda verbose: None)

    exit_calls: dict[str, int] = {}

    def fake_exit(code: int):
        exit_calls["code"] = code
        raise SystemExit(code)

    monkeypatch.setattr("pi_assistant.cli.app.sys.exit", fake_exit)

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert exit_calls["code"] == 1
    assert excinfo.value.code == 1


def test_main_passes_audio_mode_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    args = types.SimpleNamespace(
        mode="run",
        verbose=False,
        assistant_audio_mode="local-tts",
        simulate_query=None,
        reasoning_effort=None,
        assistant_model=None,
    )
    monkeypatch.setattr("pi_assistant.cli.app.parse_args", lambda: args)
    monkeypatch.setattr("pi_assistant.cli.app.asyncio.run", _run_coro_sync)

    calls: dict[str, object] = {}

    async def fake_run_transcription(
        *,
        assistant_audio_mode: Optional[str],
        simulate_query: Optional[str],
        reasoning_effort: Optional[str],
        assistant_model: Optional[str],
    ):
        calls["audio_mode"] = assistant_audio_mode
        calls["simulate_query"] = simulate_query
        calls["reasoning_effort"] = reasoning_effort
        calls["assistant_model"] = assistant_model

    monkeypatch.setattr("pi_assistant.cli.app.run_transcription", fake_run_transcription)
    monkeypatch.setattr("pi_assistant.cli.app.set_verbose_logging", lambda verbose: None)

    main()

    assert calls["audio_mode"] == "local-tts"
    assert calls["simulate_query"] is None
    assert calls["reasoning_effort"] is None
    assert calls["assistant_model"] is None


def test_main_passes_simulate_query_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    args = types.SimpleNamespace(
        mode="run",
        verbose=False,
        assistant_audio_mode=None,
        simulate_query="Hello!",
        reasoning_effort=None,
        assistant_model=None,
    )
    monkeypatch.setattr("pi_assistant.cli.app.parse_args", lambda: args)
    monkeypatch.setattr("pi_assistant.cli.app.asyncio.run", _run_coro_sync)

    calls: dict[str, object] = {}

    async def fake_run_transcription(
        *,
        assistant_audio_mode: Optional[str],
        simulate_query: Optional[str],
        reasoning_effort: Optional[str],
        assistant_model: Optional[str],
    ):
        calls["audio_mode"] = assistant_audio_mode
        calls["simulate_query"] = simulate_query
        calls["reasoning_effort"] = reasoning_effort
        calls["assistant_model"] = assistant_model

    monkeypatch.setattr("pi_assistant.cli.app.run_transcription", fake_run_transcription)
    monkeypatch.setattr("pi_assistant.cli.app.set_verbose_logging", lambda verbose: None)

    main()

    assert calls["audio_mode"] is None
    assert calls["simulate_query"] == "Hello!"
    assert calls["reasoning_effort"] is None
    assert calls["assistant_model"] is None


def test_main_passes_reasoning_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    args = types.SimpleNamespace(
        mode="run",
        verbose=False,
        assistant_audio_mode=None,
        simulate_query=None,
        reasoning_effort="medium",
        assistant_model=None,
    )
    monkeypatch.setattr("pi_assistant.cli.app.parse_args", lambda: args)
    monkeypatch.setattr("pi_assistant.cli.app.asyncio.run", _run_coro_sync)

    calls: dict[str, object] = {}

    async def fake_run_transcription(
        *,
        assistant_audio_mode: Optional[str],
        simulate_query: Optional[str],
        reasoning_effort: Optional[str],
        assistant_model: Optional[str],
    ):
        calls["audio_mode"] = assistant_audio_mode
        calls["simulate_query"] = simulate_query
        calls["reasoning_effort"] = reasoning_effort
        calls["assistant_model"] = assistant_model

    monkeypatch.setattr("pi_assistant.cli.app.run_transcription", fake_run_transcription)
    monkeypatch.setattr("pi_assistant.cli.app.set_verbose_logging", lambda verbose: None)

    main()

    assert calls["audio_mode"] is None
    assert calls["simulate_query"] is None
    assert calls["reasoning_effort"] == "medium"
    assert calls["assistant_model"] is None


def test_main_passes_assistant_model_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    override_model = "gpt-5.1-2025-11-13"
    args = types.SimpleNamespace(
        mode="run",
        verbose=False,
        assistant_audio_mode=None,
        simulate_query=None,
        reasoning_effort=None,
        assistant_model=override_model,
    )
    monkeypatch.setattr("pi_assistant.cli.app.parse_args", lambda: args)
    monkeypatch.setattr("pi_assistant.cli.app.asyncio.run", _run_coro_sync)

    calls: dict[str, object] = {}

    async def fake_run_transcription(
        *,
        assistant_audio_mode: Optional[str],
        simulate_query: Optional[str],
        reasoning_effort: Optional[str],
        assistant_model: Optional[str],
    ):
        calls["assistant_model"] = assistant_model

    monkeypatch.setattr("pi_assistant.cli.app.run_transcription", fake_run_transcription)
    monkeypatch.setattr("pi_assistant.cli.app.set_verbose_logging", lambda verbose: None)

    main()

    assert calls["assistant_model"] == override_model
