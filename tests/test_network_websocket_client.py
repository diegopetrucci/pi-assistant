import asyncio
import base64
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

try:
    from pi_assistant.cli import logging_utils as cli_logging_utils
except ImportError:

    class _DummyLogger:
        def log(self, *args, **kwargs) -> None:
            pass

        def verbose(self, *args, **kwargs) -> None:
            pass

    class LoggingUtilsModule(ModuleType):
        ERROR_LOG_LABEL: str
        WS_LOG_LABEL: str
        LOGGER: _DummyLogger

        def __init__(self) -> None:
            super().__init__("pi_assistant.cli.logging_utils")
            self.ERROR_LOG_LABEL = "ERROR"
            self.WS_LOG_LABEL = "WS"
            self.LOGGER = _DummyLogger()

        @staticmethod
        def ws_log_label(direction=None) -> str:
            arrow = direction if direction in ("←", "→") else ""
            return f"WS{arrow}"

    class CliModule(ModuleType):
        logging_utils: LoggingUtilsModule

        def __init__(self, logging_utils_module: LoggingUtilsModule) -> None:
            super().__init__("pi_assistant.cli")
            self.logging_utils = logging_utils_module

    logging_utils = LoggingUtilsModule()
    cli_pkg = CliModule(logging_utils)
    sys.modules["pi_assistant.cli"] = cli_pkg
    sys.modules["pi_assistant.cli.logging_utils"] = logging_utils
    cli_logging_utils = logging_utils

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
    records: list[tuple[str, str]] = []

    def _capture(source, message, **kwargs):
        records.append((source, message))

    monkeypatch.setattr(
        "pi_assistant.network.websocket_client.LOGGER.verbose",
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
async def test_receive_events_skips_transcription_logs(verbose_capture):
    messages = [
        json.dumps(
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "delta": "hello",
            }
        ),
        json.dumps(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "hey raspy?",
            }
        ),
    ]
    client = WebSocketClient()
    client.websocket = DummyWebSocket(messages)
    client.connected = True

    async for _ in client.receive_events():
        pass

    assert verbose_capture == []


@pytest.mark.asyncio
async def test_receive_events_logs_other_payloads(verbose_capture):
    messages = [
        json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": "aGVsbG8=",
            }
        ),
    ]
    client = WebSocketClient()
    client.websocket = DummyWebSocket(messages)
    client.connected = True

    async for _ in client.receive_events():
        pass

    label = cli_logging_utils.ws_log_label("←")
    assert any(src == label and "audio_chars=8" in message for src, message in verbose_capture)


@pytest.mark.asyncio
async def test_send_audio_chunk_does_not_log(verbose_capture):
    client = WebSocketClient()
    client.websocket = DummyWebSocket()
    client.connected = True

    await client.send_audio_chunk(b"\x00\x01\x02\x03")

    assert not any(src == "WS→" for src, _ in verbose_capture)


@pytest.mark.asyncio
async def test_wait_for_session_created_malformed_json(capsys):
    client = WebSocketClient()
    client.websocket = DummyWebSocket(["{invalid json"])

    with pytest.raises(RuntimeError, match="transcription_session.created"):
        await client.wait_for_session_created()

    err = capsys.readouterr().err
    assert "Malformed session payload" in err


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


def test_summarize_payload_handles_invalid_string():
    client = WebSocketClient()

    summary = client._summarize_payload("not json")

    assert summary == "not json"


def test_summarize_payload_truncates_keys():
    client = WebSocketClient()
    payload: dict[str, Any] = {"type": "status"}
    payload.update({f"key{i}": i for i in range(7)})

    summary = client._summarize_payload(payload)

    assert summary is not None
    assert summary.startswith("type=status keys=")
    assert summary.endswith(",…")


def test_summarize_payload_for_lists_and_unknown_objects():
    client = WebSocketClient()

    list_summary = client._summarize_payload([1, 2, 3])
    assert list_summary is not None
    assert list_summary == "list(len=3)"

    class Custom:
        pass

    custom_summary = client._summarize_payload(Custom())
    assert custom_summary is not None
    assert custom_summary == "Custom"


def test_summarize_payload_without_type():
    client = WebSocketClient()

    summary = client._summarize_payload({"foo": 1, "bar": 2})

    assert summary is not None
    assert summary == "keys=bar, foo"


@pytest.mark.asyncio
async def test_connect_logs_error_when_connect_fails(monkeypatch, capsys):
    async def failing_connect(*args, **kwargs):
        raise RuntimeError("boom")

    dummy_module = SimpleNamespace(
        connect=failing_connect,
        exceptions=SimpleNamespace(ConnectionClosed=Exception),
    )
    monkeypatch.setattr("pi_assistant.network.websocket_client.websockets", dummy_module)

    client = WebSocketClient()
    with pytest.raises(RuntimeError, match="boom"):
        await client.connect()

    err = capsys.readouterr().err
    assert "Error connecting to OpenAI" in err


@pytest.mark.asyncio
async def test_wait_for_session_created_logs_full_event(verbose_capture):
    event = json.dumps({"foo": "bar"})
    client = WebSocketClient()
    client.websocket = DummyWebSocket([event])

    with pytest.raises(RuntimeError):
        await client.wait_for_session_created()

    assert any("Full event" in message for _, message in verbose_capture)


@pytest.mark.asyncio
async def test_wait_for_session_created_limits_decode_errors():
    client = WebSocketClient()
    client.websocket = DummyWebSocket(["{invalid json"] * 10)

    with pytest.raises(RuntimeError, match="Too many malformed"):
        await client.wait_for_session_created()


@pytest.mark.asyncio
async def test_wait_for_session_created_logs_exceptions(monkeypatch, capsys):
    class ExplodingWebSocket(DummyWebSocket):
        async def __anext__(self):
            raise RuntimeError("loop failure")

    client = WebSocketClient()
    client.websocket = ExplodingWebSocket()

    with pytest.raises(RuntimeError, match="loop failure"):
        await client.wait_for_session_created()

    err = capsys.readouterr().err
    assert "Error waiting for session" in err


@pytest.mark.asyncio
async def test_send_session_config_requires_connection(monkeypatch, capsys):
    client = WebSocketClient()

    with pytest.raises(RuntimeError, match="not connected"):
        await client.send_session_config()

    err = capsys.readouterr().err
    assert "Error sending session config" in err


@pytest.mark.asyncio
async def test_receive_events_handles_connection_closed(monkeypatch, capsys):
    class ConnectionClosed(Exception):
        pass

    class ClosingWebSocket(DummyWebSocket):
        async def __anext__(self):
            raise ConnectionClosed()

    monkeypatch.setattr(
        "pi_assistant.network.websocket_client.websockets",
        SimpleNamespace(exceptions=SimpleNamespace(ConnectionClosed=ConnectionClosed)),
    )
    client = WebSocketClient()
    client.websocket = ClosingWebSocket()
    client.connected = True

    events = []
    async for event in client.receive_events():
        events.append(event)

    assert events == []
    assert client.connected is False
    out = capsys.readouterr().out
    assert "WebSocket connection closed by server" in out


@pytest.mark.asyncio
async def test_receive_events_logs_json_errors(monkeypatch, capsys):
    class DummyConnectionClosed(Exception):
        pass

    monkeypatch.setattr(
        "pi_assistant.network.websocket_client.websockets",
        SimpleNamespace(exceptions=SimpleNamespace(ConnectionClosed=DummyConnectionClosed)),
    )
    client = WebSocketClient()
    client.websocket = DummyWebSocket(["{invalid"])
    client.connected = True

    with pytest.raises(json.JSONDecodeError):
        async for _ in client.receive_events():
            pass

    err = capsys.readouterr().err
    assert "Error receiving events" in err
