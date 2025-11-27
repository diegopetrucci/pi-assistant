import asyncio
import base64
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

# Stub out the CLI package to avoid circular imports when loading WebSocketClient.
if "pi_assistant.cli.logging_utils" not in sys.modules:

    class LoggingUtilsModule(ModuleType):
        ERROR_LOG_LABEL: str

        def __init__(self) -> None:
            super().__init__("pi_assistant.cli.logging_utils")
            self.ERROR_LOG_LABEL = "[ERROR]"

        @staticmethod
        def verbose_print(*args, **kwargs) -> None:
            return None

    class CliModule(ModuleType):
        logging_utils: LoggingUtilsModule

        def __init__(self, logging_utils_module: LoggingUtilsModule) -> None:
            super().__init__("pi_assistant.cli")
            self.logging_utils = logging_utils_module

    logging_utils = LoggingUtilsModule()
    cli_pkg = CliModule(logging_utils)
    sys.modules["pi_assistant.cli"] = cli_pkg
    sys.modules["pi_assistant.cli.logging_utils"] = logging_utils

from pi_assistant.network.websocket_client import WebSocketClient


class DummyWebSocket:
    def __init__(self, incoming: list[str] | None = None):
        self._incoming: asyncio.Queue[str] = asyncio.Queue()
        for message in incoming or []:
            self._incoming.put_nowait(message)
        self.sent: list[str] = []
        self.closed = False

    def __aiter__(self) -> "DummyWebSocket":
        return self

    async def __anext__(self) -> str:
        if self._incoming.empty():
            raise StopAsyncIteration
        return await self._incoming.get()

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def verbose_capture(monkeypatch):
    records: list[str] = []

    def _capture(*args, **kwargs):
        if args:
            records.append(args[0])
        else:
            records.append("")

    monkeypatch.setattr(
        "pi_assistant.network.websocket_client.verbose_print",
        _capture,
    )
    return records


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
    assert isinstance(client.websocket, DummyWebSocket)
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
async def test_send_audio_chunk_logs_summary(verbose_capture):
    client = WebSocketClient()
    client.websocket = DummyWebSocket()
    client.connected = True

    await client.send_audio_chunk(b"\x00\x01\x02\x03")

    assert any('{"type":"input_audio_buffer.append"' in entry for entry in verbose_capture)


@pytest.mark.asyncio
async def test_receive_events_logs_payload(verbose_capture):
    messages = [
        json.dumps({"type": "foo"}),
        json.dumps({"type": "bar"}),
    ]
    client = WebSocketClient()
    client.websocket = DummyWebSocket(messages)
    client.connected = True

    async for _ in client.receive_events():
        pass

    assert any('[WS←] {"type":"foo"}' in entry for entry in verbose_capture)


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
