import asyncio

import pytest

from pi_transcription.cli import events


def test_normalize_command_strips_non_alphanum():
    assert events._normalize_command("Hey, Jarvis! Stop?") == "hey jarvis stop"
    assert events._normalize_command("   123 --- ") == "123"
    assert events._normalize_command("") == ""


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


class DummyTranscriptBuffer:
    def __init__(self):
        self.appended = []
        self.cleared = []

    async def append_transcript(self, item_id, transcript):
        self.appended.append((item_id, transcript))

    async def clear_current_turn(self, reason):
        self.cleared.append(reason)


class DummySpeechPlayer:
    def __init__(self, stop_result=False):
        self.stop_calls = 0
        self._stop_result = stop_result

    async def stop(self):
        self.stop_calls += 1
        return self._stop_result


class FakeWebSocketClient:
    def __init__(self, events):
        self._events = events

    async def receive_events(self):
        for event in self._events:
            yield event


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
