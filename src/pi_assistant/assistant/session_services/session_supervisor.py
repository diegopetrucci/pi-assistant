"""Service supervisor handling start/stop ordering."""

from __future__ import annotations

import sys
from typing import Sequence

from pi_assistant.cli.logging_utils import ASSISTANT_LOG_LABEL

from .session_service import SessionService


class SessionSupervisor:
    """Start and stop a collection of services with rollback on failure."""

    def __init__(self, services: Sequence[SessionService]):
        self._services = list(services)
        self._started: list[SessionService] = []

    async def start_all(self) -> None:
        for service in self._services:
            try:
                await service.start()
            except Exception:
                await self._stop_started()
                raise
            self._started.append(service)

    async def stop_all(self) -> None:
        await self._stop_started()

    async def _stop_started(self) -> None:
        while self._started:
            service = self._started.pop()
            try:
                await service.stop()
            except Exception as exc:
                print(
                    f"{ASSISTANT_LOG_LABEL} Failed to stop service {service.name}: {exc}",
                    file=sys.stderr,
                )
