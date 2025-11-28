import asyncio

import pytest

from pi_assistant.cli.controller_events import AudioChunkEvent, ControllerEventBus

SECOND_CHUNK_INDEX = 2
PRIORITY_DELIVERY_THRESHOLD = 3


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
