from __future__ import annotations

import asyncio

from inferfleet.bus import MemoryBus
from inferfleet.streams import JOBS


async def test_claim_is_fifo():
    bus = MemoryBus()
    for i in range(5):
        await bus.enqueue(JOBS, {"i": i})
    got = await bus.claim(JOBS, "w1", count=5, block_ms=10)
    assert [item["i"] for _, item in got] == [0, 1, 2, 3, 4]
    assert await bus.depth(JOBS) == 0


async def test_requeue_goes_to_front_and_tracks_attempts():
    bus = MemoryBus()
    await bus.enqueue(JOBS, {"i": 0})
    await bus.enqueue(JOBS, {"i": 1})
    (handle, item), *_ = await bus.claim(JOBS, "w1", count=1, block_ms=10)
    assert item["i"] == 0
    await bus.requeue(handle, item)
    got = await bus.claim(JOBS, "w2", count=2, block_ms=10)
    assert [h.attempt for h, _ in got] == [1, 0]
    assert [i["i"] for _, i in got] == [0, 1]


async def test_claim_blocks_until_first_item():
    bus = MemoryBus()
    loop = asyncio.get_running_loop()
    loop.call_later(0.05, lambda: asyncio.ensure_future(bus.enqueue(JOBS, {"i": 9})))
    got = await bus.claim(JOBS, "w1", count=4, block_ms=500)
    assert len(got) == 1 and got[0][1]["i"] == 9


async def test_subscription_buffers_events_published_before_iteration():
    bus = MemoryBus()
    sub = await bus.subscribe("ch:1")
    await bus.publish("ch:1", {"n": 1})
    await bus.publish("ch:1", {"n": 2})
    iterator = sub.__aiter__()
    first = await iterator.__anext__()
    second = await iterator.__anext__()
    assert [first["n"], second["n"]] == [1, 2]
    await sub.aclose()


async def test_fanout_reaches_all_subscribers():
    bus = MemoryBus()
    a = await bus.subscribe("ch:2")
    b = await bus.subscribe("ch:2")
    await bus.publish("ch:2", {"hello": True})
    assert (await a.__aiter__().__anext__())["hello"] is True
    assert (await b.__aiter__().__anext__())["hello"] is True
    await a.aclose()
    await b.aclose()


async def test_kv_expiry():
    bus = MemoryBus()
    await bus.kv_set("k", "v", ttl_s=10)
    assert await bus.kv_get("k") == "v"
    bus._kv["k"] = (bus._kv["k"][0], 0)  # force expiry
    assert await bus.kv_get("k") is None


async def test_worker_registry_ttl():
    bus = MemoryBus()
    await bus.heartbeat("w1", {"inflight": 2})
    workers = await bus.workers()
    assert workers and workers[0]["inflight"] == 2
    bus._workers["w1"] = ({"inflight": 0}, 0)  # force stale
    assert await bus.workers() == []


async def test_window_counter_increments_until_ttl_reset():
    bus = MemoryBus()
    assert await bus.incr_window("rl:x", ttl_s=60) == 1
    assert await bus.incr_window("rl:x", ttl_s=60) == 2
