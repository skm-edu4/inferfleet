from __future__ import annotations

import asyncio
import collections
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from .config import settings


@dataclass
class Handle:
    stream: str
    key: Any
    attempt: int = 0


class Subscription:
    """Buffered fan-out subscription. Events published after subscribe() are queued."""

    def __init__(self, owner: "Bus", channel: str, queue: Optional[asyncio.Queue] = None):
        self._owner = owner
        self._channel = channel
        self._queue = queue or asyncio.Queue()
        self._closed = False

    def __aiter__(self) -> AsyncIterator[dict]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[dict]:
        while not self._closed:
            event = await self._queue.get()
            if event is _STOP:
                break
            yield event

    async def _push(self, event: dict) -> None:
        await self._queue.put(event)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._owner._unsubscribe(self._channel, self)
        try:
            self._queue.put_nowait(_STOP)
        except asyncio.QueueFull:  # pragma: no cover
            pass


class _StopSentinel:
    def __repr__(self) -> str:  # pragma: no cover
        return "<stop>"


_STOP = _StopSentinel()


class Bus:
    """Abstract control/data plane: job queues, pub/sub, KV, worker registry."""

    # -- jobs ---------------------------------------------------------------
    async def enqueue(self, stream: str, item: dict, *, delay_s: float = 0.0) -> None:
        raise NotImplementedError

    async def claim(
        self, stream: str, consumer: str, count: int, block_ms: int
    ) -> list[tuple[Handle, dict]]:
        raise NotImplementedError

    async def ack(self, handle: Handle) -> None:
        raise NotImplementedError

    async def requeue(self, handle: Handle, item: dict) -> None:
        raise NotImplementedError

    async def reclaim_stale(
        self, stream: str, consumer: str, min_idle_ms: int, count: int
    ) -> list[tuple[Handle, dict]]:
        return []

    async def depth(self, stream: str) -> int:
        return 0

    # -- pub/sub ------------------------------------------------------------
    async def publish(self, channel: str, event: dict) -> None:
        raise NotImplementedError

    async def subscribe(self, channel: str) -> Subscription:
        raise NotImplementedError

    async def _unsubscribe(self, channel: str, sub: Subscription) -> None:
        pass

    # -- kv -----------------------------------------------------------------
    async def incr_window(self, key: str, ttl_s: int) -> int:
        raise NotImplementedError

    async def kv_get(self, key: str) -> Optional[str]:
        return None

    async def kv_set(self, key: str, value: str, ttl_s: int) -> None:
        pass

    # -- registry -----------------------------------------------------------
    async def heartbeat(self, worker_id: str, meta: dict) -> None:
        raise NotImplementedError

    async def workers(self) -> list[dict]:
        return []

    # -- lifecycle ----------------------------------------------------------
    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass


class MemoryBus(Bus):
    """Single-process data plane: deque queues + asyncio pub/sub."""

    WORKER_TTL_S = 8.0

    def __init__(self) -> None:
        self._streams: dict[str, collections.deque] = {}
        self._cond = asyncio.Condition()
        self._subs: dict[str, set[Subscription]] = {}
        self._window: dict[str, tuple[int, float]] = {}
        self._kv: dict[str, tuple[str, float]] = {}
        self._workers: dict[str, tuple[dict, float]] = {}

    async def enqueue(self, stream: str, item: dict, *, delay_s: float = 0.0) -> None:
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        async with self._cond:
            self._streams.setdefault(stream, collections.deque()).append(
                dict(item, _attempt=item.get("attempt", 0))
            )
            self._cond.notify_all()

    async def claim(
        self, stream: str, consumer: str, count: int, block_ms: int
    ) -> list[tuple[Handle, dict]]:
        started = time.monotonic()
        first_at: Optional[float] = None
        out: list[tuple[Handle, dict]] = []
        window = block_ms / 1000
        async with self._cond:
            queue = self._streams.setdefault(stream, collections.deque())
            while len(out) < count:
                if queue:
                    item = queue.popleft()
                    attempt = int(item.pop("_attempt", 0))
                    if first_at is None:
                        first_at = time.monotonic()
                    out.append((Handle(stream=stream, key=stream, attempt=attempt), item))
                    continue
                now = time.monotonic()
                limit = (
                    started + window
                    if first_at is None
                    else first_at + window  # dynamic batch window opens on first arrival
                )
                remaining = limit - now
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
        return out

    async def ack(self, handle: Handle) -> None:
        return None

    async def requeue(self, handle: Handle, item: dict) -> None:
        async with self._cond:
            queue = self._streams.setdefault(handle.stream, collections.deque())
            queue.appendleft(dict(item, _attempt=handle.attempt + 1))
            self._cond.notify_all()

    async def depth(self, stream: str) -> int:
        return len(self._streams.get(stream, ()))

    async def publish(self, channel: str, event: dict) -> None:
        for sub in list(self._subs.get(channel, ())):
            await sub._push(event)

    async def subscribe(self, channel: str) -> Subscription:
        sub = Subscription(self, channel)
        self._subs.setdefault(channel, set()).add(sub)
        return sub

    async def _unsubscribe(self, channel: str, sub: Subscription) -> None:
        self._subs.get(channel, set()).discard(sub)

    async def incr_window(self, key: str, ttl_s: int) -> int:
        now = time.monotonic()
        stored = self._window.get(key)
        if stored is None or stored[1] <= now:
            count, expires = 1, now + ttl_s
        else:
            count, expires = stored[0] + 1, stored[1]
        self._window[key] = (count, expires)
        # lazy prune
        if len(self._window) > 10_000:
            for k, (_, exp) in list(self._window.items()):
                if exp <= now:
                    self._window.pop(k, None)
        return count

    async def kv_get(self, key: str) -> Optional[str]:
        stored = self._kv.get(key)
        if stored is None:
            return None
        value, expires = stored
        if expires <= time.monotonic():
            self._kv.pop(key, None)
            return None
        return value

    async def kv_set(self, key: str, value: str, ttl_s: int) -> None:
        self._kv[key] = (value, time.monotonic() + ttl_s)

    async def heartbeat(self, worker_id: str, meta: dict) -> None:
        self._workers[worker_id] = (meta, time.monotonic())

    async def workers(self) -> list[dict]:
        now = time.monotonic()
        stale = [w for w, (_, ts) in self._workers.items() if now - ts > self.WORKER_TTL_S]
        for w in stale:
            self._workers.pop(w, None)
        return [meta for meta, _ in self._workers.values()]


class RedisBus(Bus):
    """Production data plane: Redis Streams (jobs) + Pub/Sub (token fan-out)."""

    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis

        self._redis = aioredis.from_url(url, decode_responses=True)
        self._pubsubs: dict[str, Any] = {}
        self._groups: set[str] = set()
        self.WORKER_TTL_S = 10

    async def _ensure_group(self, stream: str) -> None:
        if stream in self._groups:
            return
        try:
            await self._redis.xgroup_create(stream, "workers", id="0", mkstream=True)
        except Exception as exc:  # BUSYGROUP == already exists
            if "BUSYGROUP" not in str(exc):
                raise
        self._groups.add(stream)

    async def enqueue(self, stream: str, item: dict, *, delay_s: float = 0.0) -> None:
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        await self._ensure_group(stream)
        await self._redis.xadd(stream, {"data": json.dumps(item)}, maxlen=100_000)

    async def claim(
        self, stream: str, consumer: str, count: int, block_ms: int
    ) -> list[tuple[Handle, dict]]:
        await self._ensure_group(stream)
        rows = await self._redis.xreadgroup(
            "workers",
            consumer,
            {stream: ">"},
            count=count,
            block=block_ms,
        )
        out: list[tuple[Handle, dict]] = []
        for _stream_name, messages in rows or []:
            for msg_id, payload in messages:
                item = json.loads(payload["data"])
                attempt = int(item.pop("attempt", 0))
                out.append((Handle(stream=stream, key=msg_id, attempt=attempt), item))
        return out

    async def ack(self, handle: Handle) -> None:
        await self._redis.xack(handle.stream, "workers", handle.key)

    async def requeue(self, handle: Handle, item: dict) -> None:
        await self.ack(handle)
        await self.enqueue(handle.stream, dict(item, attempt=handle.attempt + 1))

    async def reclaim_stale(
        self, stream: str, consumer: str, min_idle_ms: int, count: int
    ) -> list[tuple[Handle, dict]]:
        await self._ensure_group(stream)
        try:
            result = await self._redis.xautoclaim(
                stream, "workers", consumer, min_idle_ms, start_id="0-0", count=count
            )
        except Exception:
            return []
        messages = result[1] if isinstance(result, (list, tuple)) and len(result) >= 2 else []
        out: list[tuple[Handle, dict]] = []
        for msg_id, payload in messages or []:
            try:
                item = json.loads(payload["data"])
            except (KeyError, ValueError):
                continue
            attempt = int(item.pop("attempt", 0))
            out.append((Handle(stream=stream, key=msg_id, attempt=attempt), item))
        return out

    async def depth(self, stream: str) -> int:
        try:
            return int(await self._redis.xlen(stream) or 0)
        except Exception:  # pragma: no cover
            return 0

    async def publish(self, channel: str, event: dict) -> None:
        await self._redis.publish(channel, json.dumps(event))

    async def subscribe(self, channel: str) -> Subscription:

        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel)
        queue: asyncio.Queue = asyncio.Queue()
        sub = Subscription(self, channel, queue)

        async def _pump() -> None:
            try:
                while True:
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=1.0
                    )
                    if message is None:
                        if sub._closed:
                            break
                        continue
                    if message.get("type") != "message":
                        continue
                    try:
                        await queue.put(json.loads(message["data"]))
                    except ValueError:
                        continue
            finally:  # pragma: no cover
                try:
                    await pubsub.unsubscribe(channel)
                    await pubsub.aclose()
                except Exception:
                    pass

        sub._task = asyncio.create_task(_pump())  # type: ignore[attr-defined]
        return sub

    async def _unsubscribe(self, channel: str, sub: Subscription) -> None:
        task = getattr(sub, "_task", None)
        if task is not None:
            await asyncio.sleep(0)
            task.cancel()

    async def incr_window(self, key: str, ttl_s: int) -> int:
        count = await self._redis.incr(key)
        if count == 1:
            await self._redis.expire(key, ttl_s)
        return int(count)

    async def kv_get(self, key: str) -> Optional[str]:
        return await self._redis.get(key)

    async def kv_set(self, key: str, value: str, ttl_s: int) -> None:
        await self._redis.set(key, value, ex=ttl_s)

    async def heartbeat(self, worker_id: str, meta: dict) -> None:
        payload = json.dumps(dict(meta, ts=time.time()))
        await self._redis.hset("inferfleet:workers", worker_id, payload)
        await self._redis.expire("inferfleet:workers", 30)

    async def workers(self) -> list[dict]:
        raw = await self._redis.hgetall("inferfleet:workers") or {}
        now = time.time()
        out = []
        for worker_id, payload in raw.items():
            try:
                meta = json.loads(payload)
            except ValueError:
                continue
            if now - float(meta.get("ts", 0)) > self.WORKER_TTL_S:
                await self._redis.hdel("inferfleet:workers", worker_id)
                continue
            out.append(meta)
        return out

    async def close(self) -> None:
        try:
            await self._redis.aclose()
        except Exception:  # pragma: no cover
            pass


def make_bus() -> Bus:
    if settings.is_memory:
        return MemoryBus()
    return RedisBus(settings.redis_url)
