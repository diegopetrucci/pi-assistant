"""Manage the WebSocket connection lifecycle."""

from __future__ import annotations

from pi_assistant.network import WebSocketClient

from .session_service import BaseSessionService


class WebSocketSessionService(BaseSessionService):
    """Open and close the Realtime WebSocket connection."""

    def __init__(self, ws_client: WebSocketClient):
        super().__init__("websocket")
        self._ws_client = ws_client
        self._connected = False

    async def _start(self) -> None:
        await self._ws_client.connect()
        self._connected = True

    async def _stop(self) -> None:
        if not self._connected:
            return
        try:
            await self._ws_client.close()
        finally:
            self._connected = False
