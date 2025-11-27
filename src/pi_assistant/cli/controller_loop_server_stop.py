"""Server stop timeout helpers shared by the audio controller loop."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any, Protocol, cast

from pi_assistant.cli.logging_utils import ERROR_LOG_LABEL, TURN_LOG_LABEL

if TYPE_CHECKING:
    from pi_assistant.cli.controller_components import StreamStateManager

    class _ServerStopLoop(Protocol):
        _server_stop_timeout_seconds: float
        _active_turn_id: str | None
        _server_stop_timeout_handle: asyncio.TimerHandle | None
        _server_stop_timeout_task: asyncio.Task[None] | None
        _server_stop_timeout_turn_id: str | None
        state_manager: StreamStateManager

        def _verbose_print(self, message: str) -> None: ...

        def _audit_log(self, action: str, **fields: Any) -> None: ...

        def _finalize_turn(self, reason: str) -> None: ...

        def _clear_server_stop_timeout(self, *, cause: str = ...) -> None: ...

        def _on_server_stop_timeout(self, turn_id: str) -> None: ...


class ServerStopTimeoutMixin:
    """Mixin that encapsulates server stop timeout orchestration."""

    _server_stop_timeout_handle: asyncio.TimerHandle | None
    _server_stop_timeout_task: asyncio.Task[None] | None
    _server_stop_timeout_turn_id: str | None

    def _schedule_server_stop_timeout(self) -> None:
        ctx = self._server_stop_context()
        if ctx._server_stop_timeout_seconds <= 0:
            return
        turn_id = ctx._active_turn_id
        if turn_id is None:
            ctx._verbose_print(
                f"{TURN_LOG_LABEL} Skipping server stop timeout; "
                "no active turn is associated with the pending stop."
            )
            return
        ctx._clear_server_stop_timeout(cause="reschedule")
        ctx._server_stop_timeout_turn_id = turn_id
        event_loop = asyncio.get_running_loop()
        self._server_stop_timeout_handle = event_loop.call_later(
            ctx._server_stop_timeout_seconds,
            ctx._on_server_stop_timeout,
            turn_id,
        )
        pending_reason = ctx.state_manager.pending_finalize_reason
        ctx._audit_log(
            "server_stop_timeout_scheduled",
            timeout_seconds=round(ctx._server_stop_timeout_seconds, 3),
            pending_reason=pending_reason or "pending_server_stop",
        )

    def _clear_server_stop_timeout(self, *, cause: str = "cleanup") -> None:
        handle = self._server_stop_timeout_handle
        if handle:
            handle.cancel()
        self._server_stop_timeout_handle = None
        ctx = self._server_stop_context()
        ctx._audit_log("server_stop_timeout_cleared", cause=cause)
        self._server_stop_timeout_turn_id = None

    def _on_server_stop_timeout(self, turn_id: str) -> None:
        # The timeout task is scheduled via call_later, so it may still fire after the
        # server acknowledges the stop. _handle_server_stop_timeout() rechecks controller
        # state to keep the sequence idempotent.
        self._server_stop_timeout_handle = None
        loop = asyncio.get_running_loop()
        task = loop.create_task(self._handle_server_stop_timeout(turn_id))
        self._server_stop_timeout_task = task

    async def _handle_server_stop_timeout(self, expected_turn_id: str) -> None:
        ctx = self._server_stop_context()
        try:
            if ctx._server_stop_timeout_turn_id != expected_turn_id:
                return
            if not ctx.state_manager.awaiting_server_stop:
                return
            active_turn = ctx._active_turn_id
            if active_turn and active_turn != expected_turn_id:
                ctx._verbose_print(
                    f"{TURN_LOG_LABEL} Timeout for {expected_turn_id} ignored; "
                    f"current turn is {active_turn}."
                )
                return
            ctx._audit_log(
                "server_stop_timeout_fired",
                timeout_seconds=round(ctx._server_stop_timeout_seconds, 3),
            )
            ctx.state_manager.suppress_next_server_stop_event()
            reason = ctx.state_manager.complete_deferred_finalize("server speech stop timeout")
            if reason:
                ctx._finalize_turn(reason)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(
                f"{ERROR_LOG_LABEL} Server stop timeout handler failed: {exc}",
                file=sys.stderr,
            )
        finally:
            if ctx._server_stop_timeout_turn_id == expected_turn_id:
                ctx._server_stop_timeout_turn_id = None

    async def _await_server_stop_timeout_task(self) -> None:
        task = self._server_stop_timeout_task
        if not task:
            return
        self._server_stop_timeout_task = None
        try:
            await task
        except asyncio.CancelledError:
            pass

    def _server_stop_context(self) -> "_ServerStopLoop":
        return cast("_ServerStopLoop", self)


__all__ = ["ServerStopTimeoutMixin"]
