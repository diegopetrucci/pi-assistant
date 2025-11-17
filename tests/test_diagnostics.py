import asyncio
from collections.abc import AsyncIterator

import pytest

from pi_transcription import diagnostics as diagnostics_module


class _StubLoop:
    def __init__(self, timeline: list[float]):
        self._timeline = timeline
        self._index = 0

    def time(self) -> float:
        if self._index >= len(self._timeline):
            return self._timeline[-1]
        value = self._timeline[self._index]
        self._index += 1
        return value


class _StubAudioCapture:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks
        self.started_with = None
        self.stop_called = False
        self.chunk_calls = 0
        self.wait_timeouts: list[float] = []

    def start_stream(self, loop):
        self.started_with = loop

    def stop_stream(self):
        self.stop_called = True

    async def get_audio_chunk(self) -> bytes:
        self.chunk_calls += 1
        return self._chunks.pop(0)


@pytest.mark.asyncio
async def test_test_audio_capture_streams_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _StubAudioCapture([b"\x00\x01"])
    loop_stub = _StubLoop([0.0, 0.1, 5.5])

    monkeypatch.setattr("pi_transcription.diagnostics.AudioCapture", lambda: capture)
    monkeypatch.setattr("pi_transcription.diagnostics.asyncio.get_running_loop", lambda: loop_stub)

    async def fake_wait_for(coro, *, timeout):
        capture.wait_timeouts.append(timeout)
        return await coro

    monkeypatch.setattr("pi_transcription.diagnostics.asyncio.wait_for", fake_wait_for)

    await diagnostics_module.test_audio_capture()

    assert capture.started_with is loop_stub
    assert capture.chunk_calls == 1
    assert capture.wait_timeouts == [1.0]
    assert capture.stop_called is True


@pytest.mark.asyncio
async def test_test_audio_capture_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _StubAudioCapture([b"\x00\x01"])
    loop_stub = _StubLoop([0.0, 0.1])

    monkeypatch.setattr("pi_transcription.diagnostics.AudioCapture", lambda: capture)
    monkeypatch.setattr("pi_transcription.diagnostics.asyncio.get_running_loop", lambda: loop_stub)

    async def fake_wait_for(coro, *, timeout):
        coro.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr("pi_transcription.diagnostics.asyncio.wait_for", fake_wait_for)

    await diagnostics_module.test_audio_capture()

    assert capture.chunk_calls == 0
    assert capture.stop_called is True


class _StubWebSocketClient:
    def __init__(
        self,
        *,
        events: list[dict] | None = None,
        fail_connect: bool = False,
        timeout: bool = False,
    ):
        self._events = events or []
        self.fail_connect = fail_connect
        self.timeout = timeout
        self.connected = False
        self.closed = False

    async def connect(self):
        if self.fail_connect:
            raise RuntimeError("connect failed")
        self.connected = True

    async def receive_events(self) -> AsyncIterator[dict]:
        if self.timeout:
            raise asyncio.TimeoutError
        for event in self._events:
            yield event

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_test_websocket_client_streams_events(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubWebSocketClient(events=[{"type": "foo"}, {"type": "bar"}])
    monkeypatch.setattr("pi_transcription.diagnostics.WebSocketClient", lambda: stub)

    handled: list[dict] = []

    def handler(event: dict):
        handled.append(event)

    await diagnostics_module.test_websocket_client(event_handler=handler)

    assert stub.connected is True
    assert stub.closed is True
    assert handled == [{"type": "foo"}, {"type": "bar"}]


@pytest.mark.asyncio
async def test_test_websocket_client_closes_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubWebSocketClient(fail_connect=True)
    monkeypatch.setattr("pi_transcription.diagnostics.WebSocketClient", lambda: stub)

    with pytest.raises(RuntimeError, match="connect failed"):
        await diagnostics_module.test_websocket_client()

    assert stub.closed is True


@pytest.mark.asyncio
async def test_test_websocket_client_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubWebSocketClient(timeout=True)
    monkeypatch.setattr("pi_transcription.diagnostics.WebSocketClient", lambda: stub)

    await diagnostics_module.test_websocket_client()

    assert stub.closed is True
