"""Event bus primitives for the CLI audio controller."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, DefaultDict, List, Optional, Type, TypeVar, cast

from pi_assistant.wake_word import StreamState, WakeWordDetection


@dataclass(slots=True)
class ControllerEvent:
    """Base class for audio controller events."""


@dataclass(slots=True)
class AudioChunkEvent(ControllerEvent):
    chunk: bytes
    index: int


@dataclass(slots=True)
class WakeWordTriggeredEvent(ControllerEvent):
    chunk: bytes
    chunk_index: int
    detection: WakeWordDetection
    state_at_detection: StreamState


@dataclass(slots=True)
class ManualStopEvent(ControllerEvent):
    reason: str = "manual stop command"


@dataclass(slots=True)
class ServerStopEvent(ControllerEvent):
    reason: str = "server speech stop event"


@dataclass(slots=True)
class SilenceDetectedEvent(ControllerEvent):
    chunk_index: int


@dataclass(slots=True)
class StateTransitionEvent(ControllerEvent):
    previous: StreamState
    current: StreamState
    reason: str
    trigger: str
    trigger_chunk_index: Optional[int] = None


@dataclass(slots=True)
class PrerollReadyEvent(ControllerEvent):
    payload: bytes
    chunk_index: Optional[int] = None


@dataclass(slots=True)
class StreamChunkEvent(ControllerEvent):
    payload: bytes
    chunk_index: int


@dataclass(slots=True)
class FinalizeTurnEvent(ControllerEvent):
    reason: str


@dataclass(slots=True)
class StateTransitionNotification:
    reason: str
    trigger: str
    priority: bool = False
    trigger_chunk_index: Optional[int] = None


E = TypeVar("E", bound=ControllerEvent)
EventHandler = Callable[[E], Awaitable[None]]


class ControllerEventBus:
    """Lightweight async event bus for coordinating controller actors."""

    def __init__(self) -> None:
        self._queue: deque[ControllerEvent | None] = deque()
        self._subscribers: DefaultDict[Type[ControllerEvent], List[EventHandler[Any]]] = (
            defaultdict(list)
        )
        self._closed = False
        self._condition = asyncio.Condition()

    def subscribe(self, event_type: Type[E], handler: EventHandler[E]) -> None:
        """Register ``handler`` to run whenever ``event_type`` is published."""

        handlers = self._subscribers[event_type]
        typed_handlers = cast(List[EventHandler[E]], handlers)
        typed_handlers.append(handler)

    async def publish(self, event: ControllerEvent, *, priority: bool = False) -> None:
        """Queue ``event`` for asynchronous delivery to subscribers."""

        if self._closed:
            return
        async with self._condition:
            if priority:
                self._queue.appendleft(event)
            else:
                self._queue.append(event)
            self._condition.notify()

    async def run(self) -> None:
        """Dispatch queued events until :meth:`shutdown` is invoked."""

        try:
            while True:
                async with self._condition:
                    while not self._queue:
                        await self._condition.wait()
                    event = self._queue.popleft()
                if event is None:
                    async with self._condition:
                        if self._queue:
                            continue
                    break
                handlers = list(self._subscribers.get(type(event), ()))
                for handler in handlers:
                    await handler(event)
        finally:
            self._closed = True

    async def shutdown(self) -> None:
        """Stop the dispatch loop once outstanding events have drained."""

        if self._closed:
            return
        async with self._condition:
            self._closed = True
            self._queue.append(None)
            self._condition.notify()


__all__ = [
    "AudioChunkEvent",
    "ControllerEvent",
    "ControllerEventBus",
    "FinalizeTurnEvent",
    "ManualStopEvent",
    "PrerollReadyEvent",
    "ServerStopEvent",
    "SilenceDetectedEvent",
    "StateTransitionEvent",
    "StateTransitionNotification",
    "StreamChunkEvent",
    "WakeWordTriggeredEvent",
]
