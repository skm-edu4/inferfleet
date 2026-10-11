from __future__ import annotations

import asyncio
import os
import random
import signal
import time
import uuid
from typing import Optional

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
    bus: Bus,
    batch: list[tuple[Handle, dict]],
    step_ms: float,
    *,
    worker_id: Optional[str] = None,
    capacity: Optional[int] = None,
    continuous: bool = False,
    stop: Optional[asyncio.Event] = None,
    inflight: Optional[list[tuple[Handle, dict]]] = None,
) -> list[tuple[Handle, dict]]:
    """Iteration-level scheduler for one GPU ("continuous batching").

    Every loop iteration runs three phases:

    1. ADMIT  — fill free batch slots from the queue with a non-blocking
                claim, so waiting jobs join a batch that is already decoding
                (vLLM-style admission). Disabled in static mode and once
                shutdown has begun (drain, don't admit).
    2. DECODE — advance every active request by exactly one token: one mock
                GPU forward pass costing ``step_ms`` of simulated kernel time.
    3. RETIRE — requests whose token plan is exhausted publish ``done``,
                enqueue their billing settlement, ack, and free their slot.

    ``inflight`` (owned by the caller) is kept in sync with the active set so
    heartbeats report live occupancy and shutdown can requeue whatever a
    cancelled run never finished.

    Returns the handles that must be requeued (simulated crash).
    """
    capacity = capacity or max(len(batch), settings.batch_size)
    tracked = inflight if inflight is not None else []

    active: dict[str, dict] = {}
    handles: dict[str, Handle] = {}
    iterators: dict[str, list[str]] = {}
    seq: dict[str, int] = {}
    produced: dict[str, int] = {}
    admitted_at: dict[str, float] = {}
    crashed: list[tuple[Handle, dict]] = []

    def _untrack(handle: Handle, item: dict) -> None:
        try:
            tracked.remove((handle, item))
        except ValueError:
            pass

    def _discard(rid: str, handle: Handle, item: dict) -> None:
        for store in (active, handles, iterators, seq, produced, admitted_at):
            store.pop(rid, None)
        _untrack(handle, item)

    async def _admit(handle: Handle, item: dict, *, warm: bool) -> None:
        rid = item["request_id"]
        prompt = item["prompt"]
        if FAIL_MARKER in prompt:
            await _finish_failed(
                bus,
                handle,
                item,
                code="engine_error",
                message="Inference engine returned an error (fault injection).",
            )
            return
        metrics.ADMISSIONS.labels(kind="warm" if warm else "cold").inc()
        enqueued_at = item.get("enqueued_at")
        if enqueued_at:
            metrics.ADMISSION_WAIT.observe(max(0.0, time.time() - float(enqueued_at)))
        if handle.attempt > 0:
            await _publish(bus, rid, {"type": "restart", "seq": -1})
            seq[rid] = -1
        active[rid] = item
        handles[rid] = handle
        iterators[rid] = plan(prompt, item.get("max_tokens", 128))
        seq.setdefault(rid, -1)
        produced[rid] = 0
        admitted_at[rid] = time.monotonic()
        tracked.append((handle, item))

    async def _retire(rid: str) -> None:
        # Order matters: publish → settle → ack → untrack. If we die anywhere
        # before the untrack, at-least-once redelivery replays the job (the
        # gateway drops duplicate seqs, history settlement is PK-idempotent).
        item, handle = active[rid], handles[rid]
        completion = " ".join(iterators[rid])
        tokens_out = len(iterators[rid])  # published tokens (the "..." truncation
        # marker is layout, not a generated token — billing must not charge it)
        latency_ms = (time.monotonic() - admitted_at[rid]) * 1000
        await _publish(
            bus,
            rid,
            {
                "type": "done",
                "seq": seq[rid] + 1,
                "completion": completion,
                "tokens_out": tokens_out,
                "latency_ms": round(latency_ms, 2),
            },
        )
        await bus.enqueue(
            RESULTS,
            result_event(
                request_id=rid,
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
        await bus.ack(handle)
        metrics.JOBS.labels(status="ok").inc()
        _discard(rid, handle, item)

    try:
        # ---- initial admission: the jobs already claimed by the worker loop ----
        for handle, item in batch:
            await _admit(handle, item, warm=False)

        while active:
            # ---- ADMIT: free slots are refilled between decode steps ----------
            if (
                continuous
                and worker_id is not None
                and len(active) < capacity
                and not (stop is not None and stop.is_set())
            ):
                slots = capacity - len(active)
                for handle, item in await bus.claim(JOBS, worker_id, slots, 0):
                    await _admit(handle, item, warm=True)

            # ---- DECODE: one token for every active request --------------------
            producing = [rid for rid in iterators if produced[rid] < len(iterators[rid])]
            if not producing:
                for rid in list(active):  # empty token plans: retire, don't spin
                    await _retire(rid)
                break

            for rid in producing:
                item = active[rid]
                tokens = iterators[rid]
                idx = produced[rid]
                if CRASH_MARKER in item["prompt"] and idx >= 3:
                    raise WorkerCrash(rid)
                seq[rid] += 1
                await _publish(
                    bus,
                    rid,
                    {
                        "type": "token",
                        "seq": seq[rid],
                        "text": ("" if idx == 0 else " ") + tokens[idx],
                    },
                )
                produced[rid] = idx + 1
                metrics.TOKENS.labels(direction="out").inc()

            metrics.BATCH_SIZE.observe(len(active))
            metrics.GPU_UTILIZATION.set(len(active) / max(1, capacity))
            metrics.WORKER_INFLIGHT.set(len(active))

            # ---- RETIRE: finished requests leave the batch ---------------------
            for rid in producing:
                if produced[rid] >= len(iterators[rid]):
                    await _retire(rid)

            await asyncio.sleep(step_ms / 1000 * random.uniform(0.9, 1.1))

    except WorkerCrash as crash:
        # The node "died": anything unfinished goes back on the queue
        # (at-least-once), unless it has already burned its retry budget.
        print(f"[worker] simulated crash on {crash}")
        for rid in list(active):
            item, handle = active[rid], handles[rid]
            if handle.attempt + 1 >= MAX_ATTEMPTS:
                await _finish_failed(
                    bus,
                    handle,
                    item,
                    code="worker_crash",
                    message="Inference node failed repeatedly; request abandoned.",
                    tokens_out=seq.get(rid, -1) + 1,
                    latency_ms=(time.monotonic() - admitted_at[rid]) * 1000,
                )
            else:
                crashed.append((handle, item))
            _discard(rid, handle, item)
    except Exception as exc:  # unexpected engine fault: fail active jobs, keep the worker alive
        print(f"[worker] batch fault: {exc!r}")
        for rid in list(active):
            item, handle = active[rid], handles[rid]
            await _finish_failed(
                bus, handle, item, code="internal_error", message="Unexpected worker error."
            )
            _discard(rid, handle, item)
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
                    "scheduler": settings.scheduler,
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
    print(
        f"[worker {worker_id}] ready "
        f"(batch={settings.batch_size}, wait={settings.batch_wait_ms}ms, "
        f"scheduler={settings.scheduler})"
    )

    try:
        while not stop.is_set():
            # Admission policy is read every batch cycle so a running worker can
            # be flipped (benchmarks/tests) without a restart.
            continuous = settings.scheduler != "static"
            batch = await bus.claim(
                JOBS, worker_id, settings.batch_size, settings.batch_wait_ms
            )
            if not batch:
                continue
            try:
                requeue = await run_batch(
                    bus,
                    batch,
                    step_ms,
                    worker_id=worker_id,
                    capacity=settings.batch_size,
                    continuous=continuous,
                    stop=stop,
                    inflight=inflight,
                )
                for handle, item in requeue:
                    await bus.requeue(handle, item)
            except Exception:
                # run_batch handles its own faults; this net only fires if it
                # faulted while unwinding. Fail everything still tracked.
                seen = {item["request_id"] for _, item in inflight}
                victims = list(inflight) + [
                    x for x in batch if x[1]["request_id"] not in seen
                ]
                for handle, item in victims:
                    await _finish_failed(
                        bus, handle, item, code="internal_error", message="Unexpected worker error."
                    )
                inflight.clear()
    except asyncio.CancelledError:
        raise
    finally:
        stop.set()
        hb_task.cancel()
        jan_task.cancel()
        # Whatever a cancelled run never finished goes back on the queue.
        for handle, item in list(inflight):
            await bus.requeue(handle, item)
        inflight.clear()
        await bus.heartbeat(worker_id, {"inflight": 0, "dead": True})
        print(f"[worker {worker_id}] stopped")


def install_stop(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):  # pragma: no cover
            pass
