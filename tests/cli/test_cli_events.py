import asyncio
from collections.abc import AsyncIterator

import pytest

from pi_assistant.cli import events
from pi_assistant.cli.logging import strip_ansi_sequences


def test_normalize_command_strips_non_alphanum():
    assert events._normalize_command("Hey, Jarvis! Stop?") == "hey jarvis stop"
    assert events._normalize_command("   123 --- ") == "123"
    assert events._normalize_command("") == ""


def test_handle_transcription_event_partial_and_vad(monkeypatch):
    calls = []

    def fake_verbose(source, message, **kwargs):
        calls.append((source, message, kwargs))

    monkeypatch.setattr(events.LOGGER, "verbose", fake_verbose)

    events.handle_transcription_event(
        {"type": "conversation.item.input_audio_transcription.delta", "delta": "hi"}
    )
    events.handle_transcription_event({"type": "input_audio_buffer.committed", "item_id": "xyz"})

    assert calls[0] == ("PARTIAL", "hi", {"flush": True})
    assert calls[1][0] == events.VAD_LOG_LABEL
    assert "Speech detected" in calls[1][1]


def test_handle_transcription_event_session_updates(monkeypatch):
    messages = []

    def fake_verbose(_source, message, **kwargs):
        messages.append(message)

    monkeypatch.setattr(events.LOGGER, "verbose", fake_verbose)

    events.handle_transcription_event({"type": "transcription_session.created"})
    events.handle_transcription_event({"type": "transcription_session.updated"})

    assert messages == [
        "Transcription session created",
        "Transcription session configuration updated",
    ]


def test_handle_transcription_event_error_branch(capsys):
    events.handle_transcription_event(
        {
            "type": "error",
            "error": {"type": "fatal", "message": "boom", "code": "E42"},
        }
    )

    captured = capsys.readouterr()
    clean_err = strip_ansi_sequences(captured.err)
    assert "[ERROR] fatal (E42): boom" in clean_err


@pytest.mark.asyncio
async def test_maybe_stop_playback_detects_stop_command(monkeypatch):
    halted = asyncio.Event()

    class DummySpeechPlayer:
        async def stop(self):
            halted.set()
            return True

    result = await events.maybe_stop_playback("Hey jarvis stop please", DummySpeechPlayer())

    assert result is True
    assert halted.is_set()


@pytest.mark.asyncio
async def test_maybe_stop_playback_ignores_other_text():
    class DummySpeechPlayer:
        async def stop(self):
            raise AssertionError("stop should not be called")

    result = await events.maybe_stop_playback("No command here", DummySpeechPlayer())

    assert result is False


@pytest.mark.asyncio
async def test_maybe_stop_playback_returns_false_for_empty_input():
    speech_player = DummySpeechPlayer(stop_result=True)

    result = await events.maybe_stop_playback("   ...   ", speech_player)

    assert result is False
    assert speech_player.stop_calls == 0


class DummyTranscriptBuffer:
    def __init__(self):
        self.appended = []
        self.cleared = []

    async def append_transcript(self, item_id: str | None, transcript: str) -> None:
        self.appended.append((item_id, transcript))

    async def clear_current_turn(self, reason: str) -> None:
        self.cleared.append(reason)


class DummySpeechPlayer:
    def __init__(self, stop_result=False):
        self.stop_calls = 0
        self._stop_result = stop_result

    async def stop(self) -> bool:
        self.stop_calls += 1
        return self._stop_result


class FakeWebSocketClient:
    def __init__(self, events: list[dict]):
        self._events = events

    async def receive_events(self) -> AsyncIterator[dict]:
        for event in self._events:
            yield event


class _RaisingEventIterator:
    def __init__(self, exc: BaseException):
        self._exc = exc

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise self._exc


class FaultyWebSocketClient:
    def __init__(self, exc: BaseException):
        self._exc = exc

    def receive_events(self) -> AsyncIterator[dict]:
        return _RaisingEventIterator(self._exc)


@pytest.mark.asyncio
async def test_receive_events_appends_transcripts_and_flags_speech_stop(monkeypatch):
    events_list = [
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "Hello",
            "item_id": "abc",
        },
        {"type": "input_audio_buffer.speech_stopped"},
    ]
    ws_client = FakeWebSocketClient(events_list)
    buffer = DummyTranscriptBuffer()
    speech_player = DummySpeechPlayer()
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()

    await events.receive_transcription_events(
        ws_client,
        buffer,
        speech_player,
        stop_signal=stop_signal,
        speech_stopped_signal=speech_stopped_signal,
    )

    assert buffer.appended == [("abc", "Hello")]
    assert speech_stopped_signal.is_set()
    assert not stop_signal.is_set()


@pytest.mark.asyncio
async def test_receive_events_handles_stop_command(monkeypatch):
    events_list = [
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "Jarvis stop please",
            "item_id": "xyz",
        }
    ]
    ws_client = FakeWebSocketClient(events_list)
    buffer = DummyTranscriptBuffer()
    speech_player = DummySpeechPlayer(stop_result=True)
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()

    await events.receive_transcription_events(
        ws_client,
        buffer,
        speech_player,
        stop_signal=stop_signal,
        speech_stopped_signal=speech_stopped_signal,
    )

    assert buffer.appended == []
    assert buffer.cleared == ["assistant stop command"]
    assert stop_signal.is_set()


@pytest.mark.asyncio
async def test_receive_events_logs_cancelled_error(monkeypatch):
    ws_client = FaultyWebSocketClient(asyncio.CancelledError())
    buffer = DummyTranscriptBuffer()
    speech_player = DummySpeechPlayer()
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()
    messages = []

    def fake_verbose(_source, message, **kwargs):
        messages.append(message)

    monkeypatch.setattr(events.LOGGER, "verbose", fake_verbose)

    with pytest.raises(asyncio.CancelledError):
        await events.receive_transcription_events(
            ws_client,
            buffer,
            speech_player,
            stop_signal=stop_signal,
            speech_stopped_signal=speech_stopped_signal,
        )

    assert messages[0] == "Starting event receiver..."
    assert messages[-1].startswith("Event receiver stopped")


@pytest.mark.asyncio
async def test_receive_events_logs_generic_exception(monkeypatch, capsys):
    ws_client = FaultyWebSocketClient(RuntimeError("boom"))
    buffer = DummyTranscriptBuffer()
    speech_player = DummySpeechPlayer()
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()

    monkeypatch.setattr(events.LOGGER, "verbose", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError):
        await events.receive_transcription_events(
            ws_client,
            buffer,
            speech_player,
            stop_signal=stop_signal,
            speech_stopped_signal=speech_stopped_signal,
        )

    stderr = capsys.readouterr().err
    clean_err = strip_ansi_sequences(stderr)
    assert "[ERROR] Event receiver error: boom" in clean_err
