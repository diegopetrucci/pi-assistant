import asyncio
import os
import sys
import types
from typing import Any, Optional, cast

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LOCATION_NAME", "Test City")

if "audioop" not in sys.modules:
    audioop_stub = types.ModuleType("audioop")

    def _ratecv(*ratecv_args):
        audio_bytes, _, _, _, _, state = ratecv_args
        return audio_bytes, state

    cast(Any, audioop_stub).ratecv = _ratecv
    sys.modules["audioop"] = audioop_stub

from pi_assistant.assistant import TranscriptionRunConfig
from pi_assistant.cli.app import (
    SIMULATED_QUERY_FALLBACK,
    main,
    parse_args,
    run_transcription,
)
from pi_assistant.exceptions import AssistantRestartRequired


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
    args = _run_parse(monkeypatch, ["pi-assistant", "--audio-mode", "local-tts"])

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
    args = _run_parse(monkeypatch, ["pi-assistant", "--model", "5.1"])

    assert args.assistant_model == "gpt-5.1-2025-11-13"


def test_parse_args_assistant_model_nano_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _run_parse(monkeypatch, ["pi-assistant", "--model", "nano"])

    assert args.assistant_model == "gpt-5-nano-2025-08-07"


def test_parse_args_assistant_model_41_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _run_parse(monkeypatch, ["pi-assistant", "--model", "4.1"])

    assert args.assistant_model == "gpt-4.1-2025-04-14"


def test_parse_args_reset_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _run_parse(monkeypatch, ["pi-assistant", "--reset"])

    assert args.reset is True


def test_parse_args_rejects_minimal_for_nano(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        _run_parse(
            monkeypatch,
            ["pi-assistant", "--model", "nano", "--reasoning-effort", "minimal"],
        )

    stderr = capsys.readouterr().err
    assert "not supported" in stderr
    assert "low, medium, high" in stderr


def test_parse_args_rejects_reasoning_for_models_without_support(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        _run_parse(
            monkeypatch,
            ["pi-assistant", "--model", "4.1", "--reasoning-effort", "low"],
        )

    stderr = capsys.readouterr().err.lower()
    assert "reasoning effort" in stderr
    assert "none (reasoning disabled)" in stderr


class _ValidatorStub:
    def __init__(self, config: TranscriptionRunConfig):
        self.config = config
        self.kwargs: dict[str, object] | None = None

    def resolve(self, **kwargs):
        self.kwargs = kwargs
        return self.config


class _BuilderStub:
    def __init__(self):
        self.config: TranscriptionRunConfig | None = None
        self.components = object()
        self.build_calls = 0

    def bind(self, config: TranscriptionRunConfig):
        self.config = config
        return self

    def build(self):
        self.build_calls += 1
        return self.components


class _SessionStub:
    def __init__(self):
        self.config: TranscriptionRunConfig | None = None
        self.components = None
        self.entered = False
        self.exited = False
        self.run_calls = 0
        self.to_raise: BaseException | None = None

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True

    async def run(self):
        self.run_calls += 1
        if self.to_raise:
            raise self.to_raise


def _setup_run_transcription(monkeypatch: pytest.MonkeyPatch):
    config = TranscriptionRunConfig(
        assistant_model="gpt-5.1",
        reasoning_effort="low",
        assistant_audio_mode="responses",
        use_responses_audio=True,
        simulated_query="hello",
    )
    validator = _ValidatorStub(config)
    builder = _BuilderStub()
    session = _SessionStub()

    monkeypatch.setattr("pi_assistant.cli.app.TranscriptionConfigValidator", lambda: validator)
    monkeypatch.setattr(
        "pi_assistant.cli.app.TranscriptionComponentBuilder", lambda cfg: builder.bind(cfg)
    )

    def _create_session(cfg, comps):
        session.config = cfg
        session.components = comps
        return session

    monkeypatch.setattr("pi_assistant.cli.app.TranscriptionSession", _create_session)
    return validator, builder, session


@pytest.mark.asyncio
async def test_run_transcription_success(monkeypatch: pytest.MonkeyPatch) -> None:
    validator, builder, session = _setup_run_transcription(monkeypatch)

    await run_transcription(
        assistant_audio_mode="responses",
        simulate_query=None,
        reasoning_effort=None,
        assistant_model=None,
    )

    assert validator.kwargs == {
        "assistant_audio_mode": "responses",
        "simulate_query": None,
        "reasoning_effort": None,
        "assistant_model": None,
    }
    assert builder.config is session.config
    assert builder.build_calls == 1
    assert session.components is builder.components
    assert session.entered is True and session.exited is True
    assert session.run_calls == 1


@pytest.mark.asyncio
async def test_run_transcription_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, _, session = _setup_run_transcription(monkeypatch)
    session.to_raise = KeyboardInterrupt()

    await run_transcription()

    captured = capsys.readouterr()
    assert "Shutdown requested" in captured.out
    assert session.run_calls == 1
    assert session.exited is True


@pytest.mark.asyncio
async def test_run_transcription_propagates_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, session = _setup_run_transcription(monkeypatch)
    session.to_raise = RuntimeError("connect failed")

    with pytest.raises(RuntimeError, match="connect failed"):
        await run_transcription()

    assert session.run_calls == 1


@pytest.mark.asyncio
async def test_run_transcription_requires_restart(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, _, session = _setup_run_transcription(monkeypatch)
    session.to_raise = AssistantRestartRequired("Restart me")

    with pytest.raises(SystemExit) as excinfo:
        await run_transcription()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Restart me" in captured.err


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
