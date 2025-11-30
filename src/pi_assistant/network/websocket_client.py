"""
WebSocket client module for real-time speech-to-text transcription
Handles WebSocket connection to OpenAI Realtime API
"""

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import Any, Protocol, cast

import websockets

from pi_assistant.cli.logging_utils import ERROR_LOG_LABEL, LOGGER, ws_log_label
from pi_assistant.config import (
    OPENAI_REALTIME_ENDPOINT,
    SESSION_CONFIG,
    WEBSOCKET_HEADERS,
)

_TRANSCRIPTION_EVENT_TYPES = {
    "conversation.item.input_audio_transcription.delta",
    "conversation.item.input_audio_transcription.completed",
}
_MAX_SUMMARY_KEYS = 5
_MAX_CONSECUTIVE_SESSION_ERRORS = 10


class _WebSocketProtocol(Protocol):
    """Subset of the runtime WebSocket API used by the client."""

    def __aiter__(self) -> AsyncIterator[str]: ...

    async def send(self, payload: str) -> None: ...

    async def close(self) -> None: ...


class WebSocketClient:
    """Handles WebSocket connection to OpenAI Realtime API"""

    def __init__(self):
        self.websocket: _WebSocketProtocol | None = None
        self.connected = False

    def _log_ws_payload(self, direction: str, payload: Any) -> None:
        """Verbose helper to display websocket payloads."""

        summary = self._summarize_payload(payload)
        if summary is not None:
            LOGGER.verbose(ws_log_label(direction), summary)

    def _summarize_payload(self, payload: Any) -> str | None:
        """Return a concise description for websocket payload logging, or None to silence."""

        structured = payload
        if isinstance(payload, str):
            try:
                structured = json.loads(payload)
            except ValueError:
                return str(payload)

        def _summarize_keys(data: dict[str, Any]) -> str:
            keys = [key for key in data.keys() if key != "type"]
            if not keys:
                return "<none>"
            keys.sort()
            if len(keys) > _MAX_SUMMARY_KEYS:
                displayed = ", ".join(keys[:_MAX_SUMMARY_KEYS])
                return f"{displayed},…"
            return ", ".join(keys)

        summary: str
        if isinstance(structured, dict):
            payload_type = structured.get("type")
            if payload_type in _TRANSCRIPTION_EVENT_TYPES:
                # Conversation transcriptions may include raw user speech; suppress both directions.
                return None
            if payload_type == "input_audio_buffer.append":
                audio = structured.get("audio")
                length = len(audio) if isinstance(audio, str) else "?"
                summary = f"type={payload_type} audio_chars={length}"
                return summary
            elif payload_type:
                keys = _summarize_keys(structured)
                summary = f"type={payload_type} keys={keys}"
            else:
                keys = _summarize_keys(structured)
                summary = f"keys={keys}"
        elif isinstance(structured, list):
            summary = f"list(len={len(structured)})"
        else:
            summary = type(structured).__name__

        return summary

    async def connect(self) -> None:
        """
        Establish WebSocket connection to OpenAI Realtime API using API key
        """
        LOGGER.verbose(ws_log_label(), "Connecting to OpenAI Realtime API...")
        LOGGER.verbose(ws_log_label(), f"Endpoint: {OPENAI_REALTIME_ENDPOINT}")
        LOGGER.verbose(ws_log_label(), "Auth method: API key")

        try:
            # Connect with required headers
            self.websocket = cast(
                _WebSocketProtocol,
                await websockets.connect(
                    OPENAI_REALTIME_ENDPOINT, additional_headers=WEBSOCKET_HEADERS
                ),
            )
            self.connected = True
            LOGGER.verbose(ws_log_label(), "Connected to OpenAI Realtime API")

            # Wait for initial session.created event (with timeout)
            try:
                await asyncio.wait_for(self.wait_for_session_created(), timeout=10.0)
                LOGGER.verbose(ws_log_label(), "Initial session created")
            except asyncio.TimeoutError:
                LOGGER.log(
                    ws_log_label(),
                    "No session.created event received after 10s",
                )
                LOGGER.verbose(ws_log_label(), "Trying to send session.update anyway...")

            await self.send_session_config()

        except Exception as e:
            LOGGER.log(
                ERROR_LOG_LABEL,
                f"Error connecting to OpenAI: {e}",
                error=True,
            )
            raise

    def _require_websocket(self) -> _WebSocketProtocol:
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        return self.websocket

    async def wait_for_session_created(self) -> dict[str, Any]:
        """
        Wait for and capture the transcription_session.created event

        Returns:
            dict: The event from the server
        """
        LOGGER.verbose(ws_log_label(), "Waiting for transcription_session.created event...")
        websocket = self._require_websocket()
        consecutive_decode_errors = 0

        try:
            async for message in websocket:
                raw_message = (
                    message.decode("utf-8", errors="replace")
                    if isinstance(message, bytes)
                    else message
                )
                try:
                    event = cast(dict[str, Any], json.loads(raw_message))
                except json.JSONDecodeError as exc:
                    consecutive_decode_errors += 1
                    snippet = raw_message[:120]
                    message = f"Malformed session payload at pos {exc.pos}: {snippet!r}"
                    LOGGER.log(ERROR_LOG_LABEL, message, error=True)
                    if consecutive_decode_errors >= _MAX_CONSECUTIVE_SESSION_ERRORS:
                        raise RuntimeError(
                            "Too many malformed session payloads received from server"
                        ) from exc
                    continue
                consecutive_decode_errors = 0
                event_type = event.get("type")
                self._log_ws_payload("←", event)

                # Debug: print what we received
                LOGGER.verbose(ws_log_label("←"), f"Received event type: {event_type}")

                # Handle both transcription_session.created and error events
                if event_type == "transcription_session.created":
                    LOGGER.verbose(ws_log_label("←"), "Transcription session created")
                    return event
                elif event_type == "error":
                    error = event.get("error", {})
                    LOGGER.log(
                        ERROR_LOG_LABEL,
                        f"Server error: {error.get('message', 'Unknown error')}",
                        error=True,
                    )
                    # Don't raise - continue listening for more events
                else:
                    # Print the full event for debugging
                    LOGGER.verbose(ws_log_label("←"), f"Full event: {json.dumps(event, indent=2)}")

        except Exception as e:
            LOGGER.log(ERROR_LOG_LABEL, f"Error waiting for session: {e}", error=True)
            raise

        raise RuntimeError("Connection closed before receiving transcription_session.created")

    async def send_session_config(self) -> None:
        """
        Send transcription session configuration to OpenAI
        Configures audio format, model, VAD, and noise reduction
        """
        transcription = SESSION_CONFIG["input_audio_transcription"]
        vad = SESSION_CONFIG["turn_detection"]
        noise_reduction = SESSION_CONFIG["input_audio_noise_reduction"]

        LOGGER.verbose(ws_log_label(), "Sending session configuration...")
        LOGGER.verbose(ws_log_label(), f"Model: {transcription['model']}")
        LOGGER.verbose(ws_log_label(), f"VAD: {vad['type']}")
        LOGGER.verbose(ws_log_label(), f"Noise reduction: {noise_reduction['type']}")

        try:
            # Send the session update event with required session wrapper
            session_update = {"type": "transcription_session.update", "session": SESSION_CONFIG}
            websocket = self._require_websocket()
            await websocket.send(json.dumps(session_update))
            self._log_ws_payload("→", session_update)
            LOGGER.verbose(ws_log_label(), "Session configuration sent")

        except Exception as e:
            LOGGER.log(ERROR_LOG_LABEL, f"Error sending session config: {e}", error=True)
            raise

    async def send_audio_chunk(self, audio_bytes: bytes) -> None:
        """
        Send audio chunk to OpenAI Realtime API

        Args:
            audio_bytes: Raw PCM16 audio data as bytes
        """
        websocket = self._require_websocket()

        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Create audio append message
        message = {"type": "input_audio_buffer.append", "audio": audio_base64}

        # Send to OpenAI
        await websocket.send(json.dumps(message))

    async def receive_events(self) -> AsyncIterator[dict[str, Any]]:
        """
        Continuously receive and handle events from OpenAI

        Yields:
            dict: Parsed JSON event from server
        """
        websocket = self._require_websocket()

        try:
            async for message in websocket:
                # Parse JSON event
                event = cast(dict[str, Any], json.loads(message))
                self._log_ws_payload("←", event)
                yield event

        except websockets.exceptions.ConnectionClosed:
            LOGGER.log(ws_log_label(), "WebSocket connection closed by server")
            self.connected = False

        except Exception as e:
            LOGGER.log(ERROR_LOG_LABEL, f"Error receiving events: {e}", error=True)
            raise

    async def close(self) -> None:
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.connected = False
            LOGGER.verbose(ws_log_label(), "WebSocket connection closed")
