"""Stream state management helpers."""

from __future__ import annotations

from typing import Optional

from pi_assistant.wake_word import StreamState


class StreamStateManager:
    """Encapsulate controller state transitions and bookkeeping."""

    def __init__(self):
        self.state = StreamState.LISTENING
        self.chunk_count = 0
        self.retrigger_budget = 0
        self.awaiting_server_stop = False
        self.awaiting_assistant_reply = False
        self.finalizing_turn = False
        self.pending_finalize_reason: Optional[str] = None
        self._suppress_next_server_stop_event = False

    def increment_chunk_count(self) -> int:
        self.chunk_count += 1
        return self.chunk_count

    def transition_to_streaming(self) -> Optional[StreamState]:
        if self.state == StreamState.STREAMING:
            return None
        previous = self.state
        self.state = StreamState.STREAMING
        self.awaiting_server_stop = False
        self.pending_finalize_reason = None
        self.reset_retrigger_budget()
        self.clear_awaiting_assistant_reply()
        self.clear_finalizing_turn()
        return previous

    def transition_to_listening(
        self,
        reason: str,
        *,
        defer_finalize: bool = False,
    ) -> Optional[StreamState]:
        if self.state != StreamState.STREAMING:
            return None
        previous = self.state
        self.state = StreamState.LISTENING
        self.reset_retrigger_budget()
        if defer_finalize:
            self.awaiting_server_stop = True
            self.pending_finalize_reason = reason
        else:
            self.awaiting_server_stop = False
            self.pending_finalize_reason = None
        return previous

    def reset_retrigger_budget(self) -> None:
        self.retrigger_budget = 0

    def increment_retrigger_budget(self) -> int:
        self.retrigger_budget += 1
        return self.retrigger_budget

    def mark_awaiting_assistant_reply(self) -> None:
        self.awaiting_assistant_reply = True

    def clear_awaiting_assistant_reply(self) -> None:
        self.awaiting_assistant_reply = False

    def mark_finalizing_turn(self) -> None:
        self.finalizing_turn = True

    def clear_finalizing_turn(self) -> None:
        self.finalizing_turn = False

    def complete_deferred_finalize(self, fallback_reason: str) -> Optional[str]:
        if not self.awaiting_server_stop:
            return None
        reason = self.pending_finalize_reason or fallback_reason
        self.awaiting_server_stop = False
        self.pending_finalize_reason = None
        return reason

    def suppress_next_server_stop_event(self) -> None:
        self._suppress_next_server_stop_event = True

    def consume_suppressed_stop_event(self) -> bool:
        if not self._suppress_next_server_stop_event:
            return False
        self._suppress_next_server_stop_event = False
        return True
