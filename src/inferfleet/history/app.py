from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from starlette.concurrency import run_in_threadpool

from .. import metrics
from ..billing import settle
from ..bus import Bus
from ..db import session_scope
from ..streams import RESULTS

MAX_SETTLE_ATTEMPTS = 5


def _settle_one(item: dict) -> None:
    """Idempotent by generation id: a replayed result can never double-bill."""
    generation = {
        "id": item["id"],
        "user_id": item["user_id"],
        "key_id": item.get("key_id"),
        "prompt": item.get("prompt", ""),
        "completion": item.get("completion", ""),
        "model": item.get("model", "unknown"),
        "tokens_in": int(item.get("tokens_in", 0)),
        "tokens_out": int(item.get("tokens_out", 0)),
        "latency_ms": float(item.get("latency_ms", 0.0)),
        "cached": bool(item.get("cached", False)),
        "status": item.get("status", "ok"),
        "created_at": datetime.now(timezone.utc),
    }
    with session_scope() as session:
        settle(
            session,
            item["user_id"],
            reserved=int(item.get("reserved", 0)),
            actual=generation["tokens_in"] + generation["tokens_out"],
            generation=generation,
        )


async def history_loop(bus: Bus, stop: asyncio.Event) -> None:
    consumer = "history-0"
    print("[history] settlement worker ready")
    while not stop.is_set():
        batch = await bus.claim(RESULTS, consumer, count=32, block_ms=250)
        if not batch:
            continue
        for handle, item in batch:
            try:
                await run_in_threadpool(_settle_one, item)
                await bus.ack(handle)
                metrics.SETTLEMENTS.labels(status=item.get("status", "ok")).inc()
            except Exception as exc:  # durable retry with bounded attempts
                if handle.attempt + 1 >= MAX_SETTLE_ATTEMPTS:
                    print(f"[history] dropped result {item.get('id')}: {exc}")
                    await bus.ack(handle)
                    metrics.SETTLEMENTS.labels(status="dropped").inc()
                else:
                    await bus.requeue(handle, item)
    print("[history] stopped")
