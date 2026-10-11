from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from .. import metrics
from ..auth import (
    Principal,
    RateLimited,
    authenticate,
    check_rate_limit,
    est_tokens,
    rate_limit_http,
)
from ..billing import QuotaExceeded, reserve, result_event, usage_snapshot
from ..bus import Bus, Subscription, make_bus
from ..cache import ResponseCache
from ..circuit import CircuitBreaker
from ..config import TIERS, settings
from ..db import (
    ApiKey,
    init_db,
    new_id,
    seed_demo_data,
    session_scope,
)
from ..schemas import (
    ApiKeyOut,
    Choice,
    CreateKeyRequest,
    GenerateRequest,
    GenerateResponse,
    HistoryItem,
    HistoryPage,
    MeResponse,
    Usage,
)
from ..streams import JOBS, RESULTS, generation_channel

JOB_TIMEOUT_WARNING = 5.0


@dataclass
class GatewayState:
    bus: Bus
    cache: ResponseCache
    breaker: CircuitBreaker
    started_at: float = field(default_factory=time.time)
    recent: deque = field(default_factory=lambda: deque(maxlen=50))
    latencies: deque = field(default_factory=lambda: deque(maxlen=500))

    async def close(self) -> None:
        await self.bus.close()


# ---------------------------------------------------------------------------
# synchronous DB helpers (always executed inside the threadpool)
# ---------------------------------------------------------------------------

def _auth_sync(raw_key: str) -> Principal:
    with session_scope() as session:
        return authenticate(session, raw_key)


def _reserve_sync(principal: Principal, tokens: int) -> None:
    with session_scope() as session:
        reserve(session, principal.user_id, principal.monthly_quota_tokens, tokens)


def _history_sync(user_id: str, limit: int, before: Optional[str]) -> tuple[list[HistoryItem], Optional[str]]:
    from datetime import datetime

    from sqlalchemy import desc

    from ..db import Generation

    with session_scope() as session:
        q = session.query(Generation).filter(Generation.user_id == user_id)
        if before:
            q = q.filter(Generation.created_at < datetime.fromisoformat(before))
        rows = q.order_by(desc(Generation.created_at), desc(Generation.id)).limit(limit + 1).all()
    items = [
        HistoryItem(
            id=r.id,
            model=r.model,
            prompt=r.prompt,
            completion=r.completion,
            tokens_in=r.tokens_in,
            tokens_out=r.tokens_out,
            latency_ms=r.latency_ms,
            cached=r.cached,
            status=r.status,
            created_at=r.created_at.isoformat(),
        )
        for r in rows[:limit]
    ]
    cursor = rows[limit - 1].created_at.isoformat() if len(rows) > limit and items else None
    return items, cursor


def _usage_sync(principal: Principal) -> dict:
    with session_scope() as session:
        return usage_snapshot(session, principal.user_id, principal.monthly_quota_tokens)


def _list_keys_sync(user_id: str) -> list[ApiKeyOut]:
    with session_scope() as session:
        rows = session.query(ApiKey).filter(ApiKey.user_id == user_id).order_by(ApiKey.created_at).all()
    return [
        ApiKeyOut(
            id=r.id,
            label=r.label,
            tier=r.tier,
            key_prefix="ifk_live_••••" + r.id[-4:],
            revoked=r.revoked,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]


def _create_key_sync(user_id: str, tier: str, label: str) -> str:
    import uuid as _uuid

    raw = f"ifk_live_{_uuid.uuid4().hex}"
    from ..db import hash_key

    with session_scope() as session:
        session.add(
            ApiKey(
                id=new_id("key"),
                key_hash=hash_key(raw),
                user_id=user_id,
                label=label,
                tier=tier,
            )
        )
    return raw


def _revoke_key_sync(user_id: str, key_id: str) -> bool:
    with session_scope() as session:
        row = session.get(ApiKey, key_id)
        if row is None or row.user_id != user_id:
            return False
        row.revoked = True
        return True


# ---------------------------------------------------------------------------
# dependencies
# ---------------------------------------------------------------------------

def get_state(request: Request) -> GatewayState:
    return request.app.state.gw


async def require_principal(request: Request) -> Principal:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail={"code": "missing_api_key", "message": "Pass `Authorization: Bearer <key>`."},
        )
    raw = header[7:].strip()
    state = get_state(request)
    principal = await run_in_threadpool(_auth_sync, raw)
    try:
        await check_rate_limit(state.bus, principal)
    except RateLimited as exc:
        raise rate_limit_http(exc.retry_after)
    return principal


# ---------------------------------------------------------------------------
# core generation flow
# ---------------------------------------------------------------------------

@dataclass
class Admission:
    model: str
    request_id: str
    t0: float
    tokens_in: int
    reserved: int = 0
    cached: Optional[dict] = None
    sub: Optional[Subscription] = None
    job: Optional[dict] = None


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class GenerationError(Exception):
    def __init__(self, exc: HTTPException):
        self.exc = exc
        super().__init__(str(exc.detail))


async def _admit(state: GatewayState, principal: Principal, body: GenerateRequest) -> Admission:
    t0 = time.perf_counter()
    model = body.model or settings.model_name
    request_id = new_id("gen")
    tokens_in = est_tokens(body.prompt)
    bus = state.bus

    # 1. admission control: never queue work we cannot drain
    depth = await bus.depth(JOBS)
    metrics.QUEUE_DEPTH.set(depth)
    if depth >= settings.queue_max:
        metrics.QUEUE_SHED.inc()
        metrics.REQUESTS.labels(endpoint="generate", status="503", tier=principal.tier).inc()
        raise GenerationError(
            HTTPException(
                status_code=503,
                detail={"code": "overloaded", "message": "Queue is full, please retry."},
                headers={"Retry-After": "5"},
            )
        )

    # 2. cache lookup (cheap, local — always worth trying)
    cached = await state.cache.get(model, body.prompt, body.temperature, body.max_tokens)

    # 3. circuit breaker gate + capacity check (only for work that needs GPUs)
    if cached is None:
        if not state.breaker.allow():
            metrics.REQUESTS.labels(endpoint="generate", status="503", tier=principal.tier).inc()
            raise GenerationError(
                HTTPException(
                    status_code=503,
                    detail={
                        "code": "degraded",
                        "message": "Inference fleet is temporarily unavailable. Cached responses only.",
                    },
                    headers={"Retry-After": "15"},
                )
            )
        healthy_workers = await bus.workers()
        if not healthy_workers:
            metrics.REQUESTS.labels(endpoint="generate", status="503", tier=principal.tier).inc()
            raise GenerationError(
                HTTPException(
                    status_code=503,
                    detail={"code": "no_capacity", "message": "No healthy inference workers."},
                    headers={"Retry-After": "10"},
                )
            )

    if cached is not None:
        metrics.CACHE_HITS.labels(kind=cached["_cache_kind"]).inc()
        return Admission(
            model=model,
            request_id=request_id,
            t0=t0,
            tokens_in=cached.get("tokens_in", tokens_in),
            cached=cached,
        )

    # 4. strict billing reservation (row-locked transaction)
    reservation = tokens_in + body.max_tokens
    try:
        await run_in_threadpool(_reserve_sync, principal, reservation)
    except QuotaExceeded as exc:
        metrics.QUOTA_REJECTED.inc()
        metrics.REQUESTS.labels(endpoint="generate", status="402", tier=principal.tier).inc()
        raise GenerationError(
            HTTPException(
                status_code=402,
                detail={
                    "code": "quota_exceeded",
                    "message": "Monthly token budget exhausted for your tier.",
                    "used": exc.used,
                    "quota": exc.quota,
                },
            )
        )

    # 5. subscribe BEFORE enqueue so no event can be missed
    sub = await bus.subscribe(generation_channel(request_id))
    job = {
        "request_id": request_id,
        "user_id": principal.user_id,
        "key_id": principal.key_id,
        "prompt": body.prompt,
        "model": model,
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
        "tokens_in": tokens_in,
        "reserved": reservation,
        "enqueued_at": time.time(),
        "attempt": 0,
    }
    await bus.enqueue(JOBS, job)
    return Admission(
        model=model,
        request_id=request_id,
        t0=t0,
        tokens_in=tokens_in,
        reserved=reservation,
        sub=sub,
        job=job,
    )


async def _event_stream(adm: Admission, timeout_s: float) -> AsyncIterator[dict]:
    """Deduplicated event stream from the worker.

    Yields token/restart/done/error events, raises asyncio.TimeoutError on
    deadline, and always closes the subscription.
    """
    assert adm.sub is not None
    last_seq = -1
    deadline = adm.t0 + timeout_s
    iterator = adm.sub.__aiter__()
    try:
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise asyncio.TimeoutError
            try:
                event = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                raise asyncio.TimeoutError from None
            kind = event.get("type")
            if kind == "token":
                seq = int(event.get("seq", -1))
                if seq <= last_seq:
                    continue  # replayed by a retried worker; drop the duplicate
                last_seq = seq
            elif kind == "restart":
                last_seq = int(event.get("seq", -1))
            yield event
            if kind in ("done", "error"):
                return
    finally:
        await adm.sub.aclose()


async def _collect(adm: Admission, timeout_s: float) -> tuple[dict, float, str]:
    """Drain the stream to completion. Returns (terminal_event, ttfb_s, text)."""
    ttfb = 0.0
    parts: list[str] = []
    terminal: Optional[dict] = None
    stream = _event_stream(adm, timeout_s)
    try:
        async for event in stream:
            kind = event.get("type")
            if kind == "token":
                if ttfb == 0.0:
                    ttfb = time.perf_counter() - adm.t0
                parts.append(event.get("text", ""))
            elif kind in ("done", "error"):
                terminal = event
    finally:
        await stream.aclose()
    if terminal is None:
        raise asyncio.TimeoutError
    return terminal, ttfb or (time.perf_counter() - adm.t0), "".join(parts)


async def _finish_timeout(
    state: GatewayState, principal: Principal, adm: Admission, prompt: str
) -> HTTPException:
    # Release the reservation through the idempotent settlement path.
    await state.bus.enqueue(
        RESULTS,
        result_event(
            request_id=adm.request_id,
            user_id=principal.user_id,
            key_id=principal.key_id,
            prompt=prompt,
            completion="",
            model=adm.model,
            tokens_in=adm.tokens_in,
            tokens_out=0,
            latency_ms=(time.perf_counter() - adm.t0) * 1000,
            cached=False,
            status="timeout",
            reserved=adm.reserved,
        ),
    )
    state.breaker.record_failure()
    metrics.REQUESTS.labels(endpoint="generate", status="504", tier=principal.tier).inc()
    return HTTPException(
        status_code=504,
        detail={"code": "generation_timeout", "message": "Inference did not finish in time."},
        headers={"Retry-After": "5"},
    )


# ---------------------------------------------------------------------------
# application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if settings.auto_seed:
        seed_demo_data()
    bus = make_bus()
    await bus.start()
    state = GatewayState(
        bus=bus,
        cache=ResponseCache(bus, ttl_s=settings.cache_ttl_s, threshold=settings.semantic_threshold),
        breaker=CircuitBreaker(),
    )
    app.state.gw = state

    # Dev/test mode runs every microservice in-process (see docker-compose for
    # the multi-process topology). Same code, different deployment shape.
    stop = asyncio.Event()
    background: list[asyncio.Task] = []
    if settings.inline_workers:
        from ..history.app import history_loop
        from ..worker.main import worker_loop

        background = [
            asyncio.create_task(worker_loop(bus, stop), name="worker"),
            asyncio.create_task(history_loop(bus, stop), name="history"),
        ]
        # wait until the worker has registered so the first request is not shed
        for _ in range(100):
            if await bus.workers():
                break
            await asyncio.sleep(0.02)
    try:
        yield
    finally:
        stop.set()
        if background:
            await asyncio.wait(background, timeout=5)
            for task in background:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*background, return_exceptions=True)
        await state.close()


app = FastAPI(
    title="InferFleet",
    version="1.0.0",
    description="Distributed LLM inference serving platform: gateway, queue, batching, caching, billing.",
    lifespan=lifespan,
)


@app.get("/", response_class=JSONResponse)
async def root():
    return {
        "service": "inferfleet-gateway",
        "version": "1.0.0",
        "model": settings.model_name,
        "docs": "/docs",
        "dashboard": "/dashboard",
    }


@app.get("/healthz")
async def healthz(request: Request):
    state = get_state(request)
    workers = await state.bus.workers()
    depth = await state.bus.depth(JOBS)
    ok = bool(workers) or settings.is_memory
    payload = {
        "status": "ok" if ok else "degraded",
        "workers": len(workers),
        "queue_depth": depth,
        "circuit": state.breaker.state.name,
    }
    return JSONResponse(payload, status_code=200 if ok else 503)


@app.get("/metrics")
async def prometheus_metrics():
    body, content_type = metrics.render()
    return PlainTextResponse(body.decode(), media_type=content_type)


@app.post("/v1/generate", response_model=None)
async def generate(
    body: GenerateRequest,
    principal: Principal = Depends(require_principal),
    state: GatewayState = Depends(get_state),
):
    try:
        adm = await _admit(state, principal, body)
    except GenerationError as err:
        raise err.exc

    # ---- cache hit: respond immediately, settle usage asynchronously ----
    if adm.cached is not None:
        completion = adm.cached["completion"]
        usage = Usage(
            prompt_tokens=adm.cached.get("tokens_in", adm.tokens_in),
            completion_tokens=adm.cached.get("tokens_out", len(completion.split())),
            total_tokens=0,
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        latency_ms = (time.perf_counter() - adm.t0) * 1000
        await state.bus.enqueue(
            RESULTS,
            result_event(
                request_id=adm.request_id,
                user_id=principal.user_id,
                key_id=principal.key_id,
                prompt=body.prompt,
                completion=completion,
                model=adm.model,
                tokens_in=usage.prompt_tokens,
                tokens_out=usage.completion_tokens,
                latency_ms=latency_ms,
                cached=True,
                status="ok",
                reserved=0,
            ),
        )
        state.recent.append({"id": adm.request_id, "status": "cache_hit", "ms": round(latency_ms, 1)})
        metrics.TTFB.observe(latency_ms / 1000)
        metrics.REQUESTS.labels(endpoint="generate", status="200", tier=principal.tier).inc()
        if body.stream:
            return await _stream_cached(adm, body, completion, usage, latency_ms)
        return JSONResponse(
            GenerateResponse(
                id=adm.request_id,
                model=adm.model,
                choices=[Choice(text=completion)],
                usage=usage,
                cached=True,
                cache_kind=adm.cached["_cache_kind"],
                latency_ms=round(latency_ms, 2),
            ).model_dump()
        )

    # ---- live inference ----
    if body.stream:
        return StreamingResponse(
            _stream_live(state, principal, adm, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        event, ttfb, streamed_text = await _collect(adm, settings.job_timeout_s)
    except asyncio.TimeoutError:
        raise await _finish_timeout(state, principal, adm, body.prompt)

    latency_ms = (time.perf_counter() - adm.t0) * 1000
    if event.get("type") == "error":
        state.breaker.record_failure()
        metrics.REQUESTS.labels(endpoint="generate", status="502", tier=principal.tier).inc()
        state.recent.append({"id": adm.request_id, "status": "error", "ms": round(latency_ms, 1)})
        raise HTTPException(
            status_code=502,
            detail={"code": event.get("code", "inference_error"), "message": event.get("message", "generation failed")},
        )

    completion = event.get("completion", "") or streamed_text
    usage = Usage(prompt_tokens=adm.tokens_in, completion_tokens=event.get("tokens_out", 0), total_tokens=0)
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
    state.breaker.record_success()
    state.recent.append({"id": adm.request_id, "status": "ok", "ms": round(latency_ms, 1)})
    state.latencies.append(latency_ms / 1000)
    metrics.TTFB.observe(ttfb)
    metrics.REQUEST_DURATION.observe(latency_ms / 1000)
    metrics.REQUESTS.labels(endpoint="generate", status="200", tier=principal.tier).inc()

    await state.cache.put(
        request_id=adm.request_id,
        model=adm.model,
        prompt=body.prompt,
        completion=completion,
        tokens_in=adm.tokens_in,
        tokens_out=usage.completion_tokens,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
    )
    return JSONResponse(
        GenerateResponse(
            id=adm.request_id,
            model=adm.model,
            choices=[Choice(text=completion)],
            usage=usage,
            cached=False,
            queue_time_ms=round(ttfb * 1000, 1),
            latency_ms=round(latency_ms, 2),
        ).model_dump()
    )


async def _stream_live(
    state: GatewayState, principal: Principal, adm: Admission, body: GenerateRequest
) -> AsyncIterator[str]:
    yield _sse("meta", {"id": adm.request_id, "model": adm.model})
    stream = _event_stream(adm, settings.job_timeout_s)
    ttfb = 0.0
    parts: list[str] = []
    terminal: Optional[dict] = None
    try:
        async for event in stream:
            kind = event.get("type")
            if kind == "token":
                if ttfb == 0.0:
                    ttfb = time.perf_counter() - adm.t0
                parts.append(event.get("text", ""))
                yield _sse("token", {"seq": event.get("seq"), "text": event.get("text", "")})
            elif kind in ("done", "error"):
                terminal = event
                break
    except asyncio.TimeoutError:
        await stream.aclose()
        exc = await _finish_timeout(state, principal, adm, body.prompt)
        yield _sse("error", exc.detail)
        return
    finally:
        await stream.aclose()

    latency_ms = (time.perf_counter() - adm.t0) * 1000
    ttfb = ttfb or latency_ms / 1000
    if terminal is None or terminal.get("type") == "error":
        state.breaker.record_failure()
        metrics.REQUESTS.labels(endpoint="stream", status="502", tier=principal.tier).inc()
        detail = {"code": "inference_error", "message": "generation failed"}
        if terminal:
            detail = {"code": terminal.get("code"), "message": terminal.get("message")}
        yield _sse("error", detail)
        return

    completion = terminal.get("completion", "") or "".join(parts)
    usage = {"prompt_tokens": adm.tokens_in, "completion_tokens": terminal.get("tokens_out", 0)}
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    state.breaker.record_success()
    metrics.TTFB.observe(ttfb)
    metrics.REQUEST_DURATION.observe(latency_ms / 1000)
    metrics.REQUESTS.labels(endpoint="stream", status="200", tier=principal.tier).inc()
    state.recent.append({"id": adm.request_id, "status": "stream_ok", "ms": round(latency_ms, 1)})
    state.latencies.append(latency_ms / 1000)
    await state.cache.put(
        request_id=adm.request_id,
        model=adm.model,
        prompt=body.prompt,
        completion=completion,
        tokens_in=adm.tokens_in,
        tokens_out=usage["completion_tokens"],
        temperature=body.temperature,
        max_tokens=body.max_tokens,
    )
    yield _sse("done", {"id": adm.request_id, "completion": completion, "usage": usage})


async def _stream_cached(
    adm: Admission, body: GenerateRequest, completion: str, usage: Usage, latency_ms: float
) -> StreamingResponse:
    async def gen() -> AsyncIterator[str]:
        yield _sse("meta", {"id": adm.request_id, "model": adm.model, "cached": True})
        words = completion.split(" ")
        for i, word in enumerate(words):
            yield _sse("token", {"seq": i, "text": ("" if i == 0 else " ") + word})
            if i % 16 == 15:
                await asyncio.sleep(0)
        yield _sse(
            "done",
            {
                "id": adm.request_id,
                "completion": completion,
                "usage": usage.model_dump(),
                "cached": True,
            },
        )

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/v1/history", response_model=HistoryPage)
async def history(
    limit: int = Query(default=20, ge=1, le=100),
    before: Optional[str] = None,
    principal: Principal = Depends(require_principal),
):
    items, cursor = await run_in_threadpool(_history_sync, principal.user_id, limit, before)
    return HistoryPage(items=items, next_cursor=cursor)


@app.get("/v1/me", response_model=MeResponse)
async def me(principal: Principal = Depends(require_principal)):
    snap = await run_in_threadpool(_usage_sync, principal)
    return MeResponse(
        user_id=principal.user_id,
        key_id=principal.key_id,
        tier=principal.tier,
        rate_limit_rpm=principal.rate_limit_rpm,
        monthly_quota_tokens=principal.monthly_quota_tokens,
        tokens_used=snap["tokens_used"],
        tokens_reserved=snap["tokens_reserved"],
        remaining_tokens=snap["remaining"],
    )


@app.get("/v1/keys", response_model=list[ApiKeyOut])
async def list_keys(principal: Principal = Depends(require_principal)):
    return await run_in_threadpool(_list_keys_sync, principal.user_id)


@app.post("/v1/keys", response_model=dict)
async def create_key(
    body: CreateKeyRequest,
    principal: Principal = Depends(require_principal),
):
    order = {t: i for i, t in enumerate(TIERS)}
    if order[body.tier] > order[principal.tier]:
        raise HTTPException(
            status_code=403,
            detail={"code": "tier_upgrade_required", "message": "Cannot mint keys above your own tier."},
        )
    raw = await run_in_threadpool(_create_key_sync, principal.user_id, body.tier, body.label)
    return {"key": raw, "tier": body.tier, "warning": "Store this key now — it is not retrievable again."}


@app.delete("/v1/keys/{key_id}", response_model=dict)
async def revoke_key(
    key_id: str,
    principal: Principal = Depends(require_principal),
):
    removed = await run_in_threadpool(_revoke_key_sync, principal.user_id, key_id)
    if not removed:
        raise HTTPException(status_code=404, detail={"code": "not_found"})
    return {"revoked": True, "id": key_id}


@app.get("/v1/admin/stats")
async def admin_stats(state: GatewayState = Depends(get_state)):
    depth = await state.bus.depth(JOBS)
    workers = await state.bus.workers()
    lat = sorted(state.latencies)
    def pct(p: float) -> float:
        if not lat:
            return 0.0
        return round(lat[min(len(lat) - 1, int(p * len(lat)))], 4)
    return {
        "uptime_s": round(time.time() - state.started_at, 1),
        "queue_depth": depth,
        "queue_max": settings.queue_max,
        "workers": workers,
        "circuit": state.breaker.state.name,
        "semantic_cache_entries": state.cache.size(),
        "p50_latency_s": pct(0.50),
        "p95_latency_s": pct(0.95),
        "p99_latency_s": pct(0.99),
        "recent": list(state.recent)[-10:],
        "mode": settings.mode,
        "scheduler": settings.scheduler,
    }


DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>InferFleet</title>
<meta http-equiv="refresh" content="3">
<style>
 body{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#0b1020;color:#e6edf3;margin:0;padding:2rem}
 h1{font-size:1.2rem;color:#7aa2f7} h1 span{color:#9ece6a}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;margin:1.5rem 0}
 .card{background:#111832;border:1px solid #1f2a4a;border-radius:10px;padding:1rem}
 .k{color:#8b93b8;font-size:.75rem;text-transform:uppercase;letter-spacing:.08em}
 .v{font-size:1.6rem;margin-top:.3rem}
 .ok{color:#9ece6a}.warn{color:#e0af68}.bad{color:#f7768e}
 table{width:100%;border-collapse:collapse;font-size:.85rem}
 td,th{padding:.35rem .5rem;border-bottom:1px solid #1f2a4a;text-align:left}
 code{color:#7aa2f7}</style></head><body>
<h1>InferFleet <span>/ ops dashboard</span></h1>
<div class="grid" id="cards"></div>
<div class="card"><div class="k">workers</div><div id="workers"></div></div>
<div class="card" style="margin-top:1rem"><div class="k">recent requests</div>
<table><thead><tr><th>id</th><th>status</th><th>ms</th></tr></thead><tbody id="recent"></tbody></table></div>
<script>
async function tick(){
 const r = await fetch('/v1/admin/stats'); if(!r.ok){return}
 const s = await r.json();
 const c = s.circuit;
 const cls = c==='CLOSED'?'ok':(c==='HALF_OPEN'?'warn':'bad');
 document.getElementById('cards').innerHTML = `
  <div class="card"><div class="k">queue depth</div><div class="v">${s.queue_depth} / ${s.queue_max}</div></div>
  <div class="card"><div class="k">circuit breaker</div><div class="v ${cls}">${c}</div></div>
  <div class="card"><div class="k">p50 / p99 latency</div><div class="v">${s.p50_latency_s}s / ${s.p99_latency_s}s</div></div>
  <div class="card"><div class="k">semantic cache</div><div class="v">${s.semantic_cache_entries}</div></div>
  <div class="card"><div class="k">uptime</div><div class="v">${s.uptime_s}s</div></div>
  <div class="card"><div class="k">mode</div><div class="v">${s.mode}</div></div>`;
 document.getElementById('workers').innerHTML = (s.workers||[]).map(w =>
   `<div>${w.model||'?'} — <b>${w.inflight??0}</b> inflight</div>`).join('') || '<div class="bad">none</div>';
 document.getElementById('recent').innerHTML = (s.recent||[]).reverse().map(x =>
   `<tr><td>${(x.id||'').slice(0,18)}</td><td>${x.status}</td><td>${x.ms}</td></tr>`).join('');
}
tick(); setInterval(tick, 3000);
</script></body></html>"""


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)
