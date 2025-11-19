"""Manage background tasks that fetch assistant responses."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Set

from pi_assistant.cli.logging_utils import TURN_LOG_LABEL, verbose_print


class ResponseTaskManager:
    """Track assistant reply tasks so they can be cancelled as a group."""

    def __init__(self, *, task_factory: Callable[[], asyncio.Task]):
        self._task_factory = task_factory
        self._tasks: Set[asyncio.Task] = set()

    def schedule(self, reason: str) -> None:
        task = self._task_factory()
        self._tasks.add(task)

        def _discard_on_completion(fut: asyncio.Task, s=self._tasks):
            s.discard(fut)

        task.add_done_callback(_discard_on_completion)
        verbose_print(f"{TURN_LOG_LABEL} Scheduled assistant reply ({reason}).")

    def cancel(self, reason: str) -> None:
        if not self._tasks:
            return
        verbose_print(
            f"{TURN_LOG_LABEL} Canceling {len(self._tasks)} pending assistant reply task(s) "
            f"({reason})."
        )
        for pending in tuple(self._tasks):
            pending.cancel()

    async def drain(self) -> None:
        if not self._tasks:
            return
        for pending in tuple(self._tasks):
            pending.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
