#!/usr/bin/env python3
# Copyright (c) 2026 cleveris research
# SPDX-License-Identifier: MIT
# Trademark: "LV Agent", "Lv Agent", "cleveris research" are trademarks of cleveris research





import asyncio

from agent_project.harness import events as ev
from agent_project.harness.stream import Delta, EventBus


def test_fanout_to_multiple_subscribers():
    bus = EventBus()
    a, b = [], []
    bus.subscribe(a.append)
    bus.subscribe(b.append)
    bus.emit_event(ev.TurnStarted(turn_index=0))
    assert len(a) == 1 and len(b) == 1


def test_predicate_filtering():
    bus = EventBus()
    events, deltas = [], []
    bus.subscribe_events(events.append)
    bus.subscribe_deltas(deltas.append)
    bus.emit_event(ev.TurnStarted(turn_index=1))
    bus.emit_delta("token", "hello")
    assert len(events) == 1 and isinstance(events[0], ev.TurnStarted)
    assert len(deltas) == 1 and deltas[0].text == "hello"


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    got = []
    unsub = bus.subscribe(got.append)
    bus.emit_delta("status", "one")
    unsub()
    bus.emit_delta("status", "two")
    assert [d.text for d in got] == ["one"]


def test_queue_subscriber_bounded_drops():
    bus = EventBus(queue_bound=1)
    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    bus.subscribe(q)
    bus.emit_delta("token", "a")
    bus.emit_delta("token", "b")   # queue full -> drop, counted
    assert q.qsize() == 1
    assert bus.total_drops == 1


def test_broken_subscriber_never_kills_emitter():
    bus = EventBus()

    def boom(_msg):
        raise RuntimeError("frontend exploded")

    good = []
    bus.subscribe(boom)
    bus.subscribe(good.append)
    bus.emit_delta("status", "still alive")
    assert len(good) == 1


def test_async_callable_subscriber():
    async def main():
        bus = EventBus()
        got = []

        async def sink(msg):
            got.append(msg)

        bus.subscribe(sink)
        bus.emit_delta("token", "async")
        await asyncio.sleep(0.05)
        return got

    assert len(asyncio.run(main())) == 1
