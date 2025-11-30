import asyncio
from typing import Any

import pytest

from pi_assistant.cli import logging_utils
from pi_assistant.cli.controller_loop_server_stop import ServerStopTimeoutMixin


class _StateManagerStub:
    def __init__(self) -> None:
        self.awaiting_server_stop = True
        self.pending_finalize_reason: str | None = None
        self.suppressed = 0

    def suppress_next_server_stop_event(self) -> None:
        self.suppressed += 1

    def complete_deferred_finalize(self, fallback_reason: str) -> str:
        self.awaiting_server_stop = False
        return self.pending_finalize_reason or fallback_reason


class _ServerStopHarness(ServerStopTimeoutMixin):
    def __init__(self, timeout_seconds: float = 0.05) -> None:
        self._server_stop_timeout_seconds = timeout_seconds
        self._active_turn_id: str | None = "turn-1"
        self._server_stop_timeout_handle = None
        self._server_stop_timeout_task: asyncio.Task | None = None
        self._server_stop_timeout_turn_id: str | None = None
        self.state_manager = _StateManagerStub()
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.finalized: list[str] = []
        self.cleared: list[str] = []

    def _audit_log(self, action: str, **fields: Any) -> None:
        self.events.append((action, fields))

    def _finalize_turn(self, reason: str) -> None:
        self.finalized.append(reason)

    def _clear_server_stop_timeout(self, *, cause: str = "cleanup") -> None:
        self.cleared.append(cause)

    def _on_server_stop_timeout(self, turn_id: str) -> None:
        self.events.append(("timeout", {"turn": turn_id}))


def test_schedule_server_stop_timeout_skips_when_disabled() -> None:
    harness = _ServerStopHarness(timeout_seconds=0)
    harness._schedule_server_stop_timeout()

    assert harness._server_stop_timeout_handle is None


@pytest.mark.asyncio
async def test_handle_server_stop_timeout_ignores_mismatched_turn() -> None:
    harness = _ServerStopHarness()
    harness._server_stop_timeout_turn_id = "turn-expected"

    await harness._handle_server_stop_timeout("turn-other")

    assert harness.finalized == []


@pytest.mark.asyncio
async def test_handle_server_stop_timeout_ignores_when_not_awaiting() -> None:
    harness = _ServerStopHarness()
    harness._server_stop_timeout_turn_id = "turn-1"
    harness.state_manager.awaiting_server_stop = False

    await harness._handle_server_stop_timeout("turn-1")

    assert harness.finalized == []


@pytest.mark.asyncio
async def test_handle_server_stop_timeout_logs_on_turn_change(monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr(
        logging_utils.LOGGER,
        "verbose",
        lambda _source, message, **_: messages.append(message),
    )
    harness = _ServerStopHarness()
    harness._server_stop_timeout_turn_id = "turn-1"
    harness._active_turn_id = "turn-2"

    await harness._handle_server_stop_timeout("turn-1")

    assert any("ignored" in entry for entry in messages)


@pytest.mark.asyncio
async def test_await_server_stop_timeout_task_ignores_cancelled() -> None:
    harness = _ServerStopHarness()

    async def pending():
        await asyncio.sleep(1)

    task = asyncio.create_task(pending())
    harness._server_stop_timeout_task = task
    task.cancel()

    await harness._await_server_stop_timeout_task()

    assert harness._server_stop_timeout_task is None
