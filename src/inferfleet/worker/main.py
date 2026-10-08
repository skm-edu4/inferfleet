from __future__ import annotations

import asyncio
import os
import random
import signal
import time
import uuid

from .. import metrics
from ..auth import est_tokens
from ..billing import result_event
from ..bus import Bus, Handle
from ..config import settings
from ..streams import JOBS, RESULTS, generation_channel
from .engine import CRASH_MARKER, FAIL_MARKER, plan

MAX_ATTEMPTS = 3


class WorkerCrash(Exception):
    """Simulated GPU node death mid-generation."""


def _channel(request_id: str) -> str:
    return generation_channel(request_id)


async def _publish(bus: Bus, request_id: str, event: dict) -> None:
    await bus.publish(_channel(request_id), event)


async def _finish_failed(
    bus: Bus,
    handle: Handle,
    item: dict,
    *,
    code: str,
    message: str,
    tokens_out: int = 0,
    latency_ms: float = 0.0,
) -> None:
    await _publish(
        bus,
        item["request_id"],
        {"type": "error", "code": code, "message": message, "seq": -1},
    )
    await bus.enqueue(
        RESULTS,
        result_event(
            request_id=item["request_id"],
            user_id=item["user_id"],
            key_id=item["key_id"],
            prompt=item["prompt"],
            completion="",
            model=item["model"],
            tokens_in=item.get("tokens_in", est_tokens(item["prompt"])),
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cached=False,
            status="failed",
            reserved=item.get("reserved", 0),
        ),
    )
    await bus.ack(handle)
    metrics.JOBS.labels(status="failed").inc()


async def run_batch(
    bus: Bus, batch: list[tuple[Handle, dict]], step_ms: float
) -> list[tuple[Handle, dict]]:
    """One GPU forward-pass loop: every request in the batch advances in lockstep.

    Returns the handles that must be requeued (simulated crash).
    """
    if not batch:
        return []
    metrics.BATCH_SIZE.observe(len(batch))
    metrics.GPU_UTILIZATION.set(len(batch) / max(1, settings.batch_size))
    metrics.WORKER_INFLIGHT.set(len(batch))

    started = time.monotonic()
    active: dict[str, dict] = {}
    seq: dict[str, int] = {}
    iterators: dict[str, list[str]] = {}
    crashed: list[tuple[Handle, dict]] = []

    try:
        for handle, item in batch:
            request_id = item["request_id"]
            prompt = item["prompt"]
            if FAIL_MARKER in prompt:
                await _finish_failed(
                    bus,
                    handle,
                    item,
                    code="engine_error",
                    message="Inference engine returned an error (fault injection).",
                )
                continue
            if handle.attempt > 0:
                await _publish(bus, request_id, {"type": "restart", "seq": -1})
                seq[request_id] = -1
            active[request_id] = item
            iterators[request_id] = plan(prompt, item.get("max_tokens", 128))
            seq.setdefault(request_id, -1)

        handles = {item["request_id"]: handle for handle, item in batch if item["request_id"] in active}

        steps = max((len(v) for v in iterators.values()), default=0)
        producing = set(iterators)
        for step in range(steps):
            finished = {rid for rid in producing if step >= len(iterators[rid])}
            producing -= finished
            if not producing:
                break
            for request_id in producing:
                item = active[request_id]
                tokens = iterators[request_id]
                if CRASH_MARKER in item["prompt"] and step >= 3:
                    raise WorkerCrash(request_id)
                seq[request_id] += 1
                await _publish(
                    bus,
                    request_id,
                    {
                        "type": "token",
                        "seq": seq[request_id],
                        "text": ("" if step == 0 else " ") + tokens[step],
                    },
                )
                metrics.TOKENS.labels(direction="out").inc()
            await asyncio.sleep(step_ms / 1000 * random.uniform(0.9, 1.1))

        latency_ms = (time.monotonic() - started) * 1000
        for request_id, item in list(active.items()):
            completion = " ".join(iterators.get(request_id, []))
            tokens_out = len(completion.split())
            await _publish(
                bus,
                request_id,
                {
                    "type": "done",
                    "seq": seq.get(request_id, -1) + 1,
                    "completion": completion,
                    "tokens_out": tokens_out,
                    "latency_ms": round(latency_ms, 2),
                },
            )
            await bus.enqueue(
                RESULTS,
                result_event(
                    request_id=request_id,
                    user_id=item["user_id"],
                    key_id=item["key_id"],
                    prompt=item["prompt"],
                    completion=completion,
                    model=item["model"],
                    tokens_in=item.get("tokens_in", est_tokens(item["prompt"])),
                    tokens_out=tokens_out,
                    latency_ms=latency_ms,
                    cached=False,
                    status="ok",
                    reserved=item.get("reserved", 0),
                ),
            )
            await bus.ack(handles[request_id])
            metrics.JOBS.labels(status="ok").inc()
        active.clear()
    except WorkerCrash:
        # The node "died": anything unfinished goes back on the queue (at-least-once).
        for handle, item in batch:
            request_id = item["request_id"]
            if request_id in active:
                if handle.attempt + 1 >= MAX_ATTEMPTS:
                    await _finish_failed(
                        bus,
                        handle,
                        item,
                        code="worker_crash",
                        message="Inference node failed repeatedly; request abandoned.",
                        tokens_out=seq.get(request_id, -1) + 1,
                    )
                    active.pop(request_id, None)
                else:
                    crashed.append((handle, item))
    finally:
        metrics.GPU_UTILIZATION.set(0)
        metrics.WORKER_INFLIGHT.set(0)
    return crashed


async def worker_loop(bus: Bus, stop: asyncio.Event) -> None:
    worker_id = os.getenv("IF_WORKER_ID") or f"gpu-{uuid.uuid4().hex[:8]}"
    inflight: list[tuple[Handle, dict]] = []
    step_ms = float(os.getenv("IF_STEP_MS", "18"))

    async def heartbeat() -> None:
        while not stop.is_set():
            await bus.heartbeat(
                worker_id,
                {
                    "inflight": len(inflight),
                    "model": settings.model_name,
                    "batch_size": settings.batch_size,
                },
            )
            try:
                await asyncio.wait_for(stop.wait(), timeout=2)
            except asyncio.TimeoutError:
                continue

    async def janitor() -> None:
        """Reclaim jobs orphaned by a dead consumer (Redis PEL sweep)."""
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=10)
                return
            except asyncio.TimeoutError:
                pass
            stale = await bus.reclaim_stale(JOBS, worker_id, min_idle_ms=60_000, count=8)
            for handle, item in stale:
                if handle.attempt + 1 >= MAX_ATTEMPTS:
                    await _finish_failed(
                        bus, handle, item, code="worker_crash", message="Job abandoned after retries."
                    )
                else:
                    await bus.requeue(handle, item)

    hb_task = asyncio.create_task(heartbeat())
    jan_task = asyncio.create_task(janitor())
    print(f"[worker {worker_id}] ready (batch={settings.batch_size}, wait={settings.batch_wait_ms}ms)")

    try:
        while not stop.is_set():
            batch = await bus.claim(
                JOBS, worker_id, settings.batch_size, settings.batch_wait_ms
            )
            if not batch:
                continue
            inflight = batch
            try:
                requeue = await run_batch(bus, batch, step_ms)
                for handle, item in requeue:
                    await bus.requeue(handle, item)
            except Exception:  # unexpected engine fault: fail the batch, keep the worker alive
                for handle, item in batch:
                    await _finish_failed(
                        bus, handle, item, code="internal_error", message="Unexpected worker error."
                    )
            finally:
                inflight = []
    except asyncio.CancelledError:
        raise
    finally:
        stop.set()
        hb_task.cancel()
        jan_task.cancel()
        for handle, item in inflight:
            await bus.requeue(handle, item)
        await bus.heartbeat(worker_id, {"inflight": 0, "dead": True})
        print(f"[worker {worker_id}] stopped")


def install_stop(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):  # pragma: no cover
            pass
