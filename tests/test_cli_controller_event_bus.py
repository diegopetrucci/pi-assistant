import asyncio
import logging

import pytest

from pi_assistant.cli.controller_events import AudioChunkEvent, ControllerEventBus

SECOND_CHUNK_INDEX = 2
PRIORITY_DELIVERY_THRESHOLD = 3
TOTAL_PRIORITY_EVENTS = 4
BACKPRESSURE_COMPLETION_INDEX = 3


@pytest.mark.asyncio
async def test_event_bus_delivers_events_in_order():
    bus = ControllerEventBus()
    delivered: list[int] = []
    done = asyncio.Event()

    async def _handler(event: AudioChunkEvent) -> None:
        delivered.append(event.index)
        if event.index == SECOND_CHUNK_INDEX:
            done.set()

    bus.subscribe(AudioChunkEvent, _handler)
    runner = asyncio.create_task(bus.run())

    try:
        await bus.publish(AudioChunkEvent(chunk=b"a", index=1))
        await bus.publish(AudioChunkEvent(chunk=b"b", index=SECOND_CHUNK_INDEX))
        await asyncio.wait_for(done.wait(), timeout=0.5)
    finally:
        await bus.shutdown()
        await runner

    assert delivered == [1, SECOND_CHUNK_INDEX]


@pytest.mark.asyncio
async def test_event_bus_prioritizes_high_priority_events():
    bus = ControllerEventBus()
    delivered: list[int] = []
    done = asyncio.Event()

    async def _handler(event: AudioChunkEvent) -> None:
        delivered.append(event.index)
        if len(delivered) == PRIORITY_DELIVERY_THRESHOLD:
            done.set()

    bus.subscribe(AudioChunkEvent, _handler)

    await bus.publish(AudioChunkEvent(chunk=b"a", index=1))
    await bus.publish(AudioChunkEvent(chunk=b"b", index=SECOND_CHUNK_INDEX))
    await bus.publish(
        AudioChunkEvent(chunk=b"priority", index=99),
        priority=True,
    )

    runner = asyncio.create_task(bus.run())

    try:
        await asyncio.wait_for(done.wait(), timeout=0.5)
    finally:
        await bus.shutdown()
        await runner

    assert delivered == [99, 1, SECOND_CHUNK_INDEX]


@pytest.mark.asyncio
async def test_event_bus_shutdown_prevents_further_publication():
    bus = ControllerEventBus()
    delivered: list[int] = []
    done = asyncio.Event()

    async def _handler(event: AudioChunkEvent) -> None:
        delivered.append(event.index)
        done.set()

    bus.subscribe(AudioChunkEvent, _handler)
    runner = asyncio.create_task(bus.run())

    try:
        await bus.publish(AudioChunkEvent(chunk=b"first", index=1))
        await asyncio.wait_for(done.wait(), timeout=0.5)
        await bus.shutdown()
        await bus.publish(AudioChunkEvent(chunk=b"second", index=SECOND_CHUNK_INDEX))
    finally:
        await runner

    assert delivered == [1]


@pytest.mark.asyncio
async def test_event_bus_logs_and_continues_after_handler_exception(
    caplog: pytest.LogCaptureFixture,
):
    bus = ControllerEventBus()
    delivered: list[int] = []
    done = asyncio.Event()

    async def _handler_raise(event: AudioChunkEvent) -> None:
        raise RuntimeError("boom")

    async def _handler_continue(event: AudioChunkEvent) -> None:
        delivered.append(event.index)
        done.set()

    bus.subscribe(AudioChunkEvent, _handler_raise)
    bus.subscribe(AudioChunkEvent, _handler_continue)
    runner = asyncio.create_task(bus.run())

    try:
        with caplog.at_level(logging.ERROR):
            await bus.publish(AudioChunkEvent(chunk=b"first", index=1))
            await asyncio.wait_for(done.wait(), timeout=0.5)
    finally:
        await bus.shutdown()
        await runner

    assert delivered == [1]
    assert any("ControllerEventBus handler" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_event_bus_supports_reentrant_publication():
    bus = ControllerEventBus()
    delivered: list[int] = []
    done = asyncio.Event()

    async def _handler(event: AudioChunkEvent) -> None:
        delivered.append(event.index)
        if event.index == 1:
            await bus.publish(AudioChunkEvent(chunk=b"follow_up", index=2))
        else:
            done.set()

    bus.subscribe(AudioChunkEvent, _handler)
    runner = asyncio.create_task(bus.run())

    try:
        await bus.publish(AudioChunkEvent(chunk=b"initial", index=1))
        await asyncio.wait_for(done.wait(), timeout=0.5)
    finally:
        await bus.shutdown()
        await runner

    assert delivered == [1, 2]


@pytest.mark.asyncio
async def test_event_bus_shutdown_during_inflight_handler():
    bus = ControllerEventBus()
    started = asyncio.Event()
    release_handler = asyncio.Event()
    finished = asyncio.Event()
    delivered: list[int] = []

    async def _slow_handler(event: AudioChunkEvent) -> None:
        started.set()
        await release_handler.wait()
        delivered.append(event.index)
        finished.set()

    bus.subscribe(AudioChunkEvent, _slow_handler)
    runner = asyncio.create_task(bus.run())

    await bus.publish(AudioChunkEvent(chunk=b"slow", index=1))
    await asyncio.wait_for(started.wait(), timeout=0.5)

    await bus.shutdown()
    assert not finished.is_set()

    release_handler.set()
    await asyncio.wait_for(finished.wait(), timeout=0.5)
    await runner

    assert delivered == [1]


@pytest.mark.asyncio
async def test_event_bus_handles_multiple_priority_events():
    bus = ControllerEventBus()
    delivered: list[int] = []
    done = asyncio.Event()

    async def _handler(event: AudioChunkEvent) -> None:
        delivered.append(event.index)
        if len(delivered) == TOTAL_PRIORITY_EVENTS:
            done.set()

    bus.subscribe(AudioChunkEvent, _handler)
    runner = asyncio.create_task(bus.run())

    try:
        await bus.publish(AudioChunkEvent(chunk=b"normal1", index=1))
        await bus.publish(AudioChunkEvent(chunk=b"normal2", index=2))
        await bus.publish(AudioChunkEvent(chunk=b"priority1", index=100), priority=True)
        await bus.publish(AudioChunkEvent(chunk=b"priority2", index=101), priority=True)
        await asyncio.wait_for(done.wait(), timeout=0.5)
    finally:
        await bus.shutdown()
        await runner

    assert delivered == [101, 100, 1, 2]


@pytest.mark.asyncio
async def test_event_bus_enforces_queue_backpressure():
    bus = ControllerEventBus(max_queue_size=1)
    delivered: list[int] = []
    done = asyncio.Event()

    async def _handler(event: AudioChunkEvent) -> None:
        delivered.append(event.index)
        if event.index == BACKPRESSURE_COMPLETION_INDEX:
            done.set()

    bus.subscribe(AudioChunkEvent, _handler)

    await bus.publish(AudioChunkEvent(chunk=b"first", index=1))
    second_publish = asyncio.create_task(
        bus.publish(AudioChunkEvent(chunk=b"second", index=2)),
    )
    await asyncio.sleep(0)
    assert not second_publish.done()

    runner = asyncio.create_task(bus.run())

    try:
        await asyncio.wait_for(second_publish, timeout=0.5)
        await bus.publish(AudioChunkEvent(chunk=b"third", index=BACKPRESSURE_COMPLETION_INDEX))
        await asyncio.wait_for(done.wait(), timeout=0.5)
    finally:
        await bus.shutdown()
        await runner

    assert delivered == [1, 2, BACKPRESSURE_COMPLETION_INDEX]
