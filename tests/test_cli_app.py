import asyncio
import os
import sys
import types
from collections.abc import Awaitable, Callable

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LOCATION_NAME", "Test City")

if "audioop" not in sys.modules:
    stub = types.SimpleNamespace(
        ratecv=lambda audio_bytes, width, channels, src_rate, dst_rate, state: (audio_bytes, state),
    )
    sys.modules["audioop"] = stub  # pyright: ignore[reportArgumentType]

from pi_assistant.cli.app import main, parse_args, run_transcription


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
    assert args.force_always_on is None


def test_parse_args_with_force_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _run_parse(
        monkeypatch,
        ["pi-assistant", "run", "--force-always-on", "--verbose"],
    )

    assert args.mode == "run"
    assert args.verbose is True
    assert args.force_always_on is True


def test_parse_args_with_no_force_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _run_parse(
        monkeypatch,
        ["pi-assistant", "--no-force-always-on"],
    )

    assert args.mode == "run"
    assert args.force_always_on is False


def test_parse_args_invalid_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        _run_parse(monkeypatch, ["pi-assistant", "invalid-mode"])


def test_parse_args_conflicting_force_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        _run_parse(
            monkeypatch,
            ["pi-assistant", "--force-always-on", "--no-force-always-on"],
        )


def test_parse_args_test_audio_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _run_parse(monkeypatch, ["pi-assistant", "test-audio"])

    assert args.mode == "test-audio"
    assert args.force_always_on is None


def test_parse_args_test_websocket_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _run_parse(monkeypatch, ["pi-assistant", "test-websocket"])

    assert args.mode == "test-websocket"


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
    def __init__(self, *, tts_enabled: bool = True, supports_audio: bool = True):
        self.tts_enabled = tts_enabled
        self._supports_audio = supports_audio
        self.verify_calls = 0
        self.resp_audio_supported = None
        self.warm_calls: list[str] = []
        self.cached_cues: dict[str, tuple[bytes, int]] = {}

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


class _StubSpeechPlayer:
    def __init__(self, default_sample_rate: int):
        self.sample_rate = default_sample_rate


def _patch_run_transcription_deps(
    monkeypatch: pytest.MonkeyPatch,
    *,
    audio_capture: _StubAudioCapture | None = None,
    ws_client: _StubWebSocketClient | None = None,
    assistant: _StubAssistant | None = None,
    run_audio_fn: Callable[..., Awaitable[None]] | None = None,
    receive_fn: Callable[..., Awaitable[None]] | None = None,
) -> dict[str, object]:
    created = {}
    audio_capture = audio_capture or _StubAudioCapture()
    ws_client = ws_client or _StubWebSocketClient()
    assistant = assistant or _StubAssistant()

    monkeypatch.setattr("pi_assistant.cli.app.AudioCapture", lambda: audio_capture)
    monkeypatch.setattr("pi_assistant.cli.app.WebSocketClient", lambda: ws_client)
    monkeypatch.setattr("pi_assistant.cli.app.LLMResponder", lambda: assistant)
    monkeypatch.setattr("pi_assistant.cli.app.TurnTranscriptAggregator", lambda: object())
    monkeypatch.setattr(
        "pi_assistant.cli.app.SpeechPlayer", lambda **kwargs: _StubSpeechPlayer(**kwargs)
    )

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
    created["run_audio"] = run_audio
    created["receive"] = receive
    return created


@pytest.mark.asyncio
async def test_run_transcription_success(monkeypatch: pytest.MonkeyPatch) -> None:
    run_audio_calls = {}

    async def fake_run_audio_controller(*args, **kwargs):
        run_audio_calls["force_always_on"] = kwargs["force_always_on"]
        kwargs["stop_signal"].set()
        kwargs["speech_stopped_signal"].set()

    deps = _patch_run_transcription_deps(monkeypatch, run_audio_fn=fake_run_audio_controller)

    await run_transcription(force_always_on=True)

    audio = deps["audio_capture"]
    ws_client = deps["ws_client"]
    assistant = deps["assistant"]

    assert audio.started and audio.loop is asyncio.get_running_loop()  # pyright: ignore[reportAttributeAccessIssue]
    assert audio.stopped  # pyright: ignore[reportAttributeAccessIssue]
    assert ws_client.connected and ws_client.closed  # pyright: ignore[reportAttributeAccessIssue]
    assert run_audio_calls["force_always_on"] is True
    assert assistant.verify_calls == 1  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_run_transcription_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    class _InterruptWebSocket(_StubWebSocketClient):
        async def connect(self):
            raise KeyboardInterrupt

    ws_client = _InterruptWebSocket()
    deps = _patch_run_transcription_deps(monkeypatch, ws_client=ws_client)

    await run_transcription(force_always_on=False)

    assert deps["audio_capture"].stopped is True  # pyright: ignore[reportAttributeAccessIssue]
    assert deps["ws_client"].closed is True  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_run_transcription_propagates_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    ws_client = _StubWebSocketClient(fail_connect=True)
    deps = _patch_run_transcription_deps(monkeypatch, ws_client=ws_client)

    with pytest.raises(RuntimeError, match="connect failed"):
        await run_transcription()

    assert deps["audio_capture"].stopped is True  # pyright: ignore[reportAttributeAccessIssue]
    assert deps["ws_client"].closed is True  # pyright: ignore[reportAttributeAccessIssue]


def test_main_run_mode_uses_force_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    args = types.SimpleNamespace(mode="run", verbose=True, force_always_on=None)
    monkeypatch.setattr("pi_assistant.cli.app.parse_args", lambda: args)
    monkeypatch.setattr("pi_assistant.cli.app.FORCE_ALWAYS_ON", True, raising=False)

    calls: dict[str, object] = {}

    async def fake_run_transcription(*, force_always_on: bool):
        calls["force"] = force_always_on

    monkeypatch.setattr("pi_assistant.cli.app.run_transcription", fake_run_transcription)
    monkeypatch.setattr(
        "pi_assistant.cli.app.set_verbose_logging",
        lambda verbose: calls.setdefault("verbose", verbose),
    )
    monkeypatch.setattr("pi_assistant.cli.app.asyncio.run", _run_coro_sync)

    main()

    assert calls["force"] is True
    assert calls["verbose"] is True


def test_main_prefers_cli_force_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    args = types.SimpleNamespace(mode="run", verbose=False, force_always_on=False)
    monkeypatch.setattr("pi_assistant.cli.app.parse_args", lambda: args)
    monkeypatch.setattr("pi_assistant.cli.app.FORCE_ALWAYS_ON", True, raising=False)

    calls: dict[str, object] = {}

    async def fake_run_transcription(*, force_always_on: bool):
        calls["force"] = force_always_on

    monkeypatch.setattr("pi_assistant.cli.app.run_transcription", fake_run_transcription)
    monkeypatch.setattr("pi_assistant.cli.app.set_verbose_logging", lambda verbose: None)
    monkeypatch.setattr("pi_assistant.cli.app.asyncio.run", _run_coro_sync)

    main()

    assert calls["force"] is False


def test_main_runs_audio_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    args = types.SimpleNamespace(mode="test-audio", verbose=False, force_always_on=None)
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
    args = types.SimpleNamespace(mode="test-websocket", verbose=False, force_always_on=None)
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
    args = types.SimpleNamespace(mode="run", verbose=False, force_always_on=None)
    monkeypatch.setattr("pi_assistant.cli.app.parse_args", lambda: args)
    monkeypatch.setattr("pi_assistant.cli.app.asyncio.run", _run_coro_sync)

    async def fake_run_transcription(*, force_always_on: bool):
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
