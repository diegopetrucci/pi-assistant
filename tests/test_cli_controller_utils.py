import asyncio
import contextlib
import io
from typing import cast

import numpy as np
import pytest

from pi_assistant.assistant import LLMReply
from pi_assistant.audio.metrics import calculate_rms
from pi_assistant.cli import controller, controller_helpers

EXPECTED_RMS_MIN = 26000
EXPECTED_RMS_MAX = 28000


def test_calculate_rms_handles_empty_bytes():
    assert calculate_rms(b"") == 0.0
    assert calculate_rms(b"\x00\x00" * 0) == 0.0


def test_calculate_rms_returns_expected_value():
    samples = np.array([0, 32767, -32768], dtype=np.int16).tobytes()
    rms = calculate_rms(samples)

    assert EXPECTED_RMS_MIN < rms < EXPECTED_RMS_MAX


def _chunk_with_duration(seconds: float, amplitude: int = 0) -> bytes:
    frames = max(1, int(seconds * controller.SAMPLE_RATE))
    samples = np.full(frames * controller.CHANNELS, amplitude, dtype=np.int16)
    return samples.tobytes()


def test_audio_chunk_preparer_resamples_when_needed(monkeypatch):
    created = []

    class DummyResampler:
        def __init__(self, *args, **kwargs):
            self.reset_called = False
            created.append(self)

        def process(self, chunk):
            data = np.frombuffer(chunk, dtype=np.int16)
            return (data // 2).astype(np.int16)

        def reset(self):
            self.reset_called = True

    monkeypatch.setattr(controller, "LinearResampler", DummyResampler)
    preparer = controller.AudioChunkPreparer(
        48000,
        16000,
        resampler_factory=controller.LinearResampler,
    )

    assert preparer.is_resampling is True
    processed = preparer.prepare((np.ones(4, dtype=np.int16) * 4).tobytes())
    assert np.array_equal(np.frombuffer(processed, dtype=np.int16), np.ones(4, dtype=np.int16) * 2)

    preparer.reset()
    assert created[0].reset_called is True


def test_silence_tracker_requires_multiple_silence_chunks():
    tracker = controller.SilenceTracker(
        silence_threshold=1000,
        max_silence_seconds=0.01,
        sample_rate=controller.SAMPLE_RATE,
    )

    silence = _chunk_with_duration(0.005, amplitude=0)
    speech = _chunk_with_duration(0.02, amplitude=2000)

    assert tracker.observe(silence) is False  # no speech yet
    assert tracker.observe(speech) is False  # speech resets timer
    assert tracker.observe(silence) is False
    assert tracker.observe(silence) is True


def test_silence_tracker_triggers_on_single_long_chunk():
    tracker = controller.SilenceTracker(
        silence_threshold=1000,
        max_silence_seconds=0.01,
        sample_rate=controller.SAMPLE_RATE,
    )
    speech = _chunk_with_duration(0.02, amplitude=2000)
    long_silence = _chunk_with_duration(0.02, amplitude=0)

    tracker.observe(speech)
    assert tracker.observe(long_silence) is True


def test_silence_tracker_clear_silence_keeps_speech_flag():
    tracker = controller.SilenceTracker(
        silence_threshold=1000,
        max_silence_seconds=0.01,
        sample_rate=controller.SAMPLE_RATE,
    )
    speech = _chunk_with_duration(0.02, amplitude=2000)
    silence = _chunk_with_duration(0.005, amplitude=0)

    tracker.observe(speech)
    tracker.observe(silence)
    tracker.clear_silence()

    assert tracker.observe(silence) is False
    assert tracker.observe(silence) is True


def test_silence_tracker_exposes_silence_duration_and_flags():
    tracker = controller.SilenceTracker(
        silence_threshold=1000,
        max_silence_seconds=0.05,
        sample_rate=controller.SAMPLE_RATE,
    )
    short_silence = _chunk_with_duration(0.01, amplitude=0)
    long_silence = _chunk_with_duration(0.05, amplitude=0)
    speech = _chunk_with_duration(0.02, amplitude=2000)

    assert tracker.heard_speech is False
    assert tracker.silence_duration == 0.0
    assert tracker.has_observed_silence(0.01) is False

    tracker.observe(speech)
    assert tracker.heard_speech is True

    tracker.observe(short_silence)
    assert tracker.has_observed_silence(0.02) is False

    tracker.observe(long_silence)
    assert tracker.has_observed_silence(0.02) is True


def test_should_ignore_server_stop_event_logic():
    manager = controller.StreamStateManager()
    tracker = controller.SilenceTracker(
        silence_threshold=1000,
        max_silence_seconds=0.5,
        sample_rate=controller.SAMPLE_RATE,
    )

    assert controller.should_ignore_server_stop_event(manager, tracker, 0.25) is None

    manager.transition_to_streaming()
    reason = controller.should_ignore_server_stop_event(manager, tracker, 0.25)
    assert reason is None

    tracker.observe(_chunk_with_duration(0.02, amplitude=2000))
    tracker.observe(_chunk_with_duration(0.1, amplitude=0))

    reason = controller.should_ignore_server_stop_event(manager, tracker, 0.25)
    assert reason is not None
    assert "0.10s silence" in reason

    tracker.observe(_chunk_with_duration(0.2, amplitude=0))
    assert controller.should_ignore_server_stop_event(manager, tracker, 0.25) is None


def test_stream_state_manager_tracks_deferred_finalize():
    manager = controller.StreamStateManager()

    assert manager.state == controller.StreamState.LISTENING
    assert manager.transition_to_listening("noop") is None

    previous = manager.transition_to_streaming()
    assert previous == controller.StreamState.LISTENING
    assert manager.state == controller.StreamState.STREAMING

    manager.increment_retrigger_budget()
    assert manager.retrigger_budget == 1
    manager.reset_retrigger_budget()
    assert manager.retrigger_budget == 0

    previous = manager.transition_to_listening("silence", defer_finalize=True)
    assert previous == controller.StreamState.STREAMING
    assert manager.awaiting_server_stop is True
    assert manager.pending_finalize_reason == "silence"

    reason = manager.complete_deferred_finalize("fallback")
    assert reason == "silence"
    assert manager.awaiting_server_stop is False
    assert manager.pending_finalize_reason is None

    manager.suppress_next_server_stop_event()
    assert manager.consume_suppressed_stop_event() is True
    assert manager.consume_suppressed_stop_event() is False

    assert manager.awaiting_assistant_reply is False
    manager.mark_awaiting_assistant_reply()
    assert manager.awaiting_assistant_reply is True
    manager.clear_awaiting_assistant_reply()
    assert manager.awaiting_assistant_reply is False

    assert manager.finalizing_turn is False
    manager.mark_finalizing_turn()
    assert manager.finalizing_turn is True
    manager.clear_finalizing_turn()
    assert manager.finalizing_turn is False


@pytest.mark.asyncio
async def test_response_task_manager_schedules_and_cleanup():
    executed = []

    call_id = {"value": 0}

    async def quick_task(task_id: int):
        executed.append(task_id)

    def factory():
        call_id["value"] += 1
        task = asyncio.create_task(quick_task(call_id["value"]))
        return task

    manager = controller.ResponseTaskManager(task_factory=factory)
    manager.schedule("first")
    manager.schedule("second")

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert executed == [1, 2]


@pytest.mark.asyncio
async def test_response_task_manager_cancel_marks_tasks():
    event = asyncio.Event()
    tasks = []

    async def waiting():
        await event.wait()

    def factory():
        task = asyncio.create_task(waiting())
        tasks.append(task)
        return task

    manager = controller.ResponseTaskManager(task_factory=factory)
    manager.schedule("hold")
    manager.schedule("hold-again")
    manager.cancel("manual stop")

    await asyncio.sleep(0)

    assert all(task.cancelled() for task in tasks)


@pytest.mark.asyncio
async def test_response_task_manager_drain_cancels_pending_tasks():
    event = asyncio.Event()
    tasks = []

    def factory():
        task = asyncio.create_task(event.wait())
        tasks.append(task)
        return task

    manager = controller.ResponseTaskManager(task_factory=factory)
    manager.schedule("pending")
    await manager.drain()

    assert all(task.cancelled() for task in tasks)


class _DummyAssistant:
    def __init__(self):
        self.tts_enabled: bool = True
        self.peek_result: tuple[bytes, int | None] | None = None
        self.peek_calls: int = 0
        self.warm_calls: list[str] = []

    def peek_phrase_audio(self, text):
        self.peek_calls += 1
        return self.peek_result

    async def warm_phrase_audio(self, text):
        self.warm_calls.append(text)


class _DummySpeechPlayer:
    def __init__(self):
        self.play_calls: list[tuple[bytes, int]] = []

    async def play(self, audio_bytes, *, sample_rate):
        self.play_calls.append((audio_bytes, sample_rate))


class _DummyTranscriptBuffer:
    def __init__(self, result: str | None = None, to_raise: BaseException | None = None):
        self.result = result
        self.to_raise = to_raise
        self.finalize_calls = 0

    async def finalize_turn(self):
        self.finalize_calls += 1
        if self.to_raise:
            raise self.to_raise
        return self.result


class _DummyResponder:
    def __init__(self, reply=None, to_raise: BaseException | None = None):
        self.reply = reply
        self.to_raise = to_raise
        self.calls: list[str] = []

    async def generate_reply(self, transcript):
        self.calls.append(transcript)
        if self.to_raise:
            raise self.to_raise
        return self.reply


@pytest.mark.asyncio
async def test_maybe_schedule_confirmation_cue_warms_cache(monkeypatch):
    assistant = _DummyAssistant()
    assistant.peek_result = None
    player = _DummySpeechPlayer()

    monkeypatch.setattr(controller_helpers, "CONFIRMATION_CUE_ENABLED", True)
    monkeypatch.setattr(controller_helpers, "CONFIRMATION_CUE_TEXT", "Got it")

    controller_helpers._maybe_schedule_confirmation_cue(
        cast(controller.LLMResponder, assistant),
        cast(controller.SpeechPlayer, player),
    )
    await asyncio.sleep(0)

    assert assistant.warm_calls == ["Got it"]
    assert player.play_calls == []


@pytest.mark.asyncio
async def test_maybe_schedule_confirmation_cue_plays_cached_audio(monkeypatch):
    assistant = _DummyAssistant()
    assistant.peek_result = (b"123", None)
    player = _DummySpeechPlayer()

    monkeypatch.setattr(controller_helpers, "CONFIRMATION_CUE_ENABLED", True)
    monkeypatch.setattr(controller_helpers, "CONFIRMATION_CUE_TEXT", "Got it")
    monkeypatch.setattr(controller_helpers, "ASSISTANT_TTS_SAMPLE_RATE", 11025)

    controller_helpers._maybe_schedule_confirmation_cue(
        cast(controller.LLMResponder, assistant),
        cast(controller.SpeechPlayer, player),
    )
    await asyncio.sleep(0)

    assert assistant.warm_calls == []
    assert player.play_calls == [(b"123", 11025)]


@pytest.mark.asyncio
async def test_finalize_transcript_notifies_hooks():
    buffer = _DummyTranscriptBuffer(result="hello")
    calls: list[str] = []
    hooks = controller.ResponseLifecycleHooks(
        on_transcript_ready=lambda: calls.append("ready"),
    )

    transcript = await controller_helpers._finalize_transcript(
        cast(controller.TurnTranscriptAggregator, buffer),
        hooks,
    )

    assert transcript == "hello"
    assert calls == ["ready"]


@pytest.mark.asyncio
async def test_finalize_transcript_still_calls_hook_on_error():
    buffer = _DummyTranscriptBuffer(to_raise=RuntimeError("boom"))
    calls: list[str] = []
    hooks = controller.ResponseLifecycleHooks(
        on_transcript_ready=lambda: calls.append("ready"),
    )

    with pytest.raises(RuntimeError, match="boom"):
        await controller_helpers._finalize_transcript(
            cast(controller.TurnTranscriptAggregator, buffer),
            hooks,
        )

    assert calls == ["ready"]


@pytest.mark.asyncio
async def test_request_assistant_reply_runs_hooks():
    reply = LLMReply(text="ok", audio_bytes=None, audio_sample_rate=None)
    responder = _DummyResponder(reply=reply)
    calls: list[str] = []
    hooks = controller.ResponseLifecycleHooks(
        on_reply_start=lambda: calls.append("start"),
        on_reply_complete=lambda: calls.append("complete"),
    )

    result = await controller_helpers._request_assistant_reply(
        "hi",
        cast(controller.LLMResponder, responder),
        hooks,
    )

    assert result is reply
    assert calls == ["start", "complete"]


@pytest.mark.asyncio
async def test_request_assistant_reply_handles_errors(capsys):
    responder = _DummyResponder(to_raise=RuntimeError("network down"))
    calls: list[str] = []
    hooks = controller.ResponseLifecycleHooks(
        on_reply_start=lambda: calls.append("start"),
        on_reply_complete=lambda: calls.append("complete"),
    )

    result = await controller_helpers._request_assistant_reply(
        "hi",
        cast(controller.LLMResponder, responder),
        hooks,
    )

    captured = capsys.readouterr()
    assert "network down" in captured.err
    assert result is None
    assert calls == ["start", "complete"]


@pytest.mark.asyncio
async def test_request_assistant_reply_propagates_cancellation():
    responder = _DummyResponder(to_raise=asyncio.CancelledError())
    calls: list[str] = []
    hooks = controller.ResponseLifecycleHooks(
        on_reply_start=lambda: calls.append("start"),
        on_reply_complete=lambda: calls.append("complete"),
    )

    with pytest.raises(asyncio.CancelledError):
        await controller_helpers._request_assistant_reply(
            "hi",
            cast(controller.LLMResponder, responder),
            hooks,
        )

    assert calls == ["start", "complete"]


@pytest.mark.asyncio
async def test_play_assistant_audio_streams_bytes():
    reply = LLMReply(text="ok", audio_bytes=b"bytes", audio_sample_rate=8000)
    player = _DummySpeechPlayer()

    await controller_helpers._play_assistant_audio(
        cast(LLMReply, reply),
        cast(controller.SpeechPlayer, player),
    )

    assert player.play_calls == [(b"bytes", 8000)]


@pytest.mark.asyncio
async def test_play_assistant_audio_ignores_missing_audio():
    reply = LLMReply(text=None, audio_bytes=None, audio_sample_rate=8000)
    player = _DummySpeechPlayer()

    await controller_helpers._play_assistant_audio(
        cast(LLMReply, reply),
        cast(controller.SpeechPlayer, player),
    )

    assert player.play_calls == []


@pytest.mark.asyncio
async def test_finalize_turn_and_respond_skips_empty_transcript(monkeypatch):
    called = {"cue": 0, "request": 0, "play": 0}

    async def fake_finalize(tb, hooks):
        return None

    def fake_cue(*args, **kwargs):
        called["cue"] += 1

    async def fake_request(*args, **kwargs):
        called["request"] += 1

    async def fake_play(reply, player):
        called["play"] += 1

    monkeypatch.setattr(controller_helpers, "_finalize_transcript", fake_finalize)
    monkeypatch.setattr(controller_helpers, "_maybe_schedule_confirmation_cue", fake_cue)
    monkeypatch.setattr(controller_helpers, "_request_assistant_reply", fake_request)
    monkeypatch.setattr(controller_helpers, "_play_assistant_audio", fake_play)

    assistant = _DummyAssistant()
    player = _DummySpeechPlayer()

    await controller_helpers.finalize_turn_and_respond(
        transcript_buffer=cast(controller.TurnTranscriptAggregator, _DummyTranscriptBuffer()),
        assistant=cast(controller.LLMResponder, assistant),
        speech_player=cast(controller.SpeechPlayer, player),
    )

    assert called == {"cue": 0, "request": 0, "play": 0}


@pytest.mark.asyncio
async def test_finalize_turn_and_respond_happy_path(monkeypatch):
    reply = LLMReply(text="hello", audio_bytes=b"x", audio_sample_rate=123)
    calls: list[str] = []

    async def fake_finalize(tb, hooks):
        calls.append("finalize")
        return "transcript"

    def fake_cue(*args, **kwargs):
        calls.append("cue")

    async def fake_request(*args, **kwargs):
        calls.append("request")
        return reply

    async def fake_play(reply_obj, player):
        assert reply_obj is reply
        calls.append("play")

    monkeypatch.setattr(controller_helpers, "_finalize_transcript", fake_finalize)
    monkeypatch.setattr(controller_helpers, "_maybe_schedule_confirmation_cue", fake_cue)
    monkeypatch.setattr(controller_helpers, "_request_assistant_reply", fake_request)
    monkeypatch.setattr(controller_helpers, "_play_assistant_audio", fake_play)

    assistant = _DummyAssistant()
    player = _DummySpeechPlayer()

    await controller_helpers.finalize_turn_and_respond(
        transcript_buffer=cast(controller.TurnTranscriptAggregator, _DummyTranscriptBuffer()),
        assistant=cast(controller.LLMResponder, assistant),
        speech_player=cast(controller.SpeechPlayer, player),
    )

    assert calls == ["finalize", "cue", "request", "play"]


@pytest.mark.asyncio
async def test_schedule_turn_response_logs_cancellation(monkeypatch):
    event = asyncio.Event()

    async def pending(*args, **kwargs):
        await event.wait()

    monkeypatch.setattr(controller_helpers, "finalize_turn_and_respond", pending)

    logs: list[str] = []
    monkeypatch.setattr(
        controller_helpers.LOGGER,
        "verbose",
        lambda _source, message, **_: logs.append(message),
    )

    task = controller.schedule_turn_response(
        transcript_buffer=cast(controller.TurnTranscriptAggregator, _DummyTranscriptBuffer()),
        assistant=cast(controller.LLMResponder, _DummyAssistant()),
        speech_player=cast(controller.SpeechPlayer, _DummySpeechPlayer()),
    )

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert any("Assistant reply task cancelled" in msg for msg in logs)


@pytest.mark.asyncio
async def test_schedule_turn_response_reports_errors(monkeypatch):
    async def failing(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(controller_helpers, "finalize_turn_and_respond", failing)
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        task = controller.schedule_turn_response(
            transcript_buffer=cast(controller.TurnTranscriptAggregator, _DummyTranscriptBuffer()),
            assistant=cast(controller.LLMResponder, _DummyAssistant()),
            speech_player=cast(controller.SpeechPlayer, _DummySpeechPlayer()),
        )

        await asyncio.sleep(0)

        with pytest.raises(RuntimeError):
            await task

        await asyncio.sleep(0)

    assert "Unexpected assistant error" in buffer.getvalue()
