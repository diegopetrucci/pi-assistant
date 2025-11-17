import asyncio
import base64
import json
import sys
import types
from types import SimpleNamespace

import pytest

# Stub out the CLI package to avoid circular imports when loading WebSocketClient.
if "pi_assistant.cli.logging_utils" not in sys.modules:
    cli_pkg = types.ModuleType("pi_assistant.cli")
    logging_utils = types.ModuleType("pi_assistant.cli.logging_utils")
    logging_utils.ERROR_LOG_LABEL = "[ERROR]"
    logging_utils.verbose_print = lambda *args, **kwargs: None
    cli_pkg.logging_utils = logging_utils
    sys.modules["pi_assistant.cli"] = cli_pkg
    sys.modules["pi_assistant.cli.logging_utils"] = logging_utils

from pi_assistant.network.websocket_client import WebSocketClient


class DummyWebSocket:
    def __init__(self, incoming=None):
        self._incoming = asyncio.Queue()
        for message in incoming or []:
            self._incoming.put_nowait(message)
        self.sent = []
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._incoming.empty():
            raise StopAsyncIteration
        return await self._incoming.get()

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_connect_waits_for_session(monkeypatch):
    handshake_events = [json.dumps({"type": "transcription_session.created"})]
    dummy_ws = DummyWebSocket(handshake_events)

    async def fake_connect(*args, **kwargs):
        return dummy_ws

    dummy_module = SimpleNamespace(
        connect=fake_connect, exceptions=SimpleNamespace(ConnectionClosed=Exception)
    )
    monkeypatch.setattr("pi_assistant.network.websocket_client.websockets", dummy_module)

    client = WebSocketClient()
    await client.connect()

    assert client.connected is True
    assert dummy_ws.sent  # session.config sent
    session_update = json.loads(dummy_ws.sent[0])
    assert session_update["type"] == "transcription_session.update"
    assert "session" in session_update


@pytest.mark.asyncio
async def test_connect_warns_when_session_timeout(monkeypatch, capsys):
    dummy_ws = DummyWebSocket()

    async def fake_connect(*args, **kwargs):
        return dummy_ws

    async def fake_wait_for(awaitable, timeout):
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise asyncio.TimeoutError

    dummy_module = SimpleNamespace(
        connect=fake_connect, exceptions=SimpleNamespace(ConnectionClosed=Exception)
    )
    monkeypatch.setattr("pi_assistant.network.websocket_client.websockets", dummy_module)
    monkeypatch.setattr("pi_assistant.network.websocket_client.asyncio.wait_for", fake_wait_for)

    client = WebSocketClient()
    await client.connect()

    out = capsys.readouterr().out
    assert "No session.created event received after 10s" in out

    assert client.connected is True
    assert dummy_ws.sent  # session config still sent


@pytest.mark.asyncio
async def test_connect_supports_multiple_attempts(monkeypatch):
    connections = [
        DummyWebSocket([json.dumps({"type": "transcription_session.created"})]),
        DummyWebSocket([json.dumps({"type": "transcription_session.created"})]),
    ]

    async def fake_connect(*args, **kwargs):
        return connections.pop(0)

    dummy_module = SimpleNamespace(
        connect=fake_connect, exceptions=SimpleNamespace(ConnectionClosed=Exception)
    )
    monkeypatch.setattr("pi_assistant.network.websocket_client.websockets", dummy_module)

    client = WebSocketClient()
    await client.connect()
    first_ws = client.websocket
    await client.close()

    await client.connect()

    assert client.websocket is not first_ws
    assert first_ws.closed is True
    assert client.connected is True


@pytest.mark.asyncio
async def test_send_audio_chunk_encodes_base64():
    client = WebSocketClient()
    dummy_ws = DummyWebSocket()
    client.websocket = dummy_ws
    client.connected = True

    await client.send_audio_chunk(b"\x00\x01\x02")

    payload = json.loads(dummy_ws.sent[0])
    decoded = base64.b64decode(payload["audio"])
    assert decoded == b"\x00\x01\x02"


@pytest.mark.asyncio
async def test_send_audio_chunk_requires_connection():
    client = WebSocketClient()

    with pytest.raises(RuntimeError, match="WebSocket not connected"):
        await client.send_audio_chunk(b"\x01\x02")


@pytest.mark.asyncio
async def test_receive_events_yields_until_closed():
    messages = [
        json.dumps({"type": "foo"}),
        json.dumps({"type": "bar"}),
    ]
    client = WebSocketClient()
    client.websocket = DummyWebSocket(messages)
    client.connected = True

    events = []
    async for event in client.receive_events():
        events.append(event)

    assert [e["type"] for e in events] == ["foo", "bar"]


@pytest.mark.asyncio
async def test_wait_for_session_created_malformed_json(capsys):
    client = WebSocketClient()
    client.websocket = DummyWebSocket(["{invalid json"])

    with pytest.raises(json.JSONDecodeError):
        await client.wait_for_session_created()

    err = capsys.readouterr().err
    assert "Error waiting for session" in err


@pytest.mark.asyncio
async def test_wait_for_session_created_error_event(monkeypatch, capsys):
    error_event = json.dumps({"type": "error", "error": {"message": "oops"}})
    created_event = json.dumps({"type": "transcription_session.created"})
    client = WebSocketClient()
    client.websocket = DummyWebSocket([error_event, created_event])

    result = await client.wait_for_session_created()

    assert result["type"] == "transcription_session.created"

    err = capsys.readouterr().err
    assert "Server error: oops" in err


@pytest.mark.asyncio
async def test_close_marks_disconnected():
    client = WebSocketClient()
    dummy_ws = DummyWebSocket()
    client.websocket = dummy_ws
    client.connected = True

    await client.close()

    assert dummy_ws.closed is True
    assert client.connected is False
