"""Shared interfaces for run-loop services."""

from __future__ import annotations

import asyncio
from typing import Protocol


class SessionService(Protocol):
    """Minimal lifecycle interface."""

    name: str
    ready: asyncio.Event

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class BaseSessionService(SessionService):
    """Track readiness and ensure idempotent start/stop semantics."""

    def __init__(self, name: str):
        self.name = name
        self.ready = asyncio.Event()
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await self._start()
        self.ready.set()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        try:
            await self._stop()
        finally:
            self._started = False
            self.ready.clear()

    async def _start(self) -> None:
        raise NotImplementedError

    async def _stop(self) -> None:
        raise NotImplementedError
