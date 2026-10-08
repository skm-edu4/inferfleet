from __future__ import annotations

import asyncio
import json

from inferfleet.config import settings


def parse_sse(raw: str) -> list[tuple[str, dict]]:
    events = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        ev_type, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev_type = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if ev_type and data is not None:
            events.append((ev_type, data))
    return events


# ---------------------------------------------------------------------------
# core generation
# ---------------------------------------------------------------------------

async def test_generate_end_to_end(pro):
    r = await pro.post(
        "/v1/generate",
        json={"prompt": "How do I upgrade my plan?", "max_tokens": 32, "temperature": 0.9},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "text.completion"
    assert body["choices"][0]["text"].strip()
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )
    assert body["cached"] is False
    assert body["latency_ms"] > 0


async def test_generate_respects_ttfb_budget(pro):
    r = await pro.post(
        "/v1/generate",
        json={"prompt": "Explain idempotency keys in one paragraph.", "max_tokens": 48, "temperature": 0.9},
    )
    assert r.status_code == 200
    # queue_time_ms measures time-to-first-token inside the platform
    assert r.json()["queue_time_ms"] < settings.ttfb_budget_ms * 4


async def test_streaming_sse(pro):
    async with pro.stream(
        "POST",
        "/v1/generate",
        json={"prompt": "Stream a short answer about caching.", "stream": True, "max_tokens": 40, "temperature": 0.9},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = ""
        async for chunk in resp.aiter_text():
            raw += chunk
    events = parse_sse(raw)
    kinds = [k for k, _ in events]
    assert kinds[0] == "meta"
    assert kinds[-1] == "done"
    assert kinds.count("token") >= 5
    done = events[-1][1]
    assert done["completion"].strip()
    # streamed tokens must concatenate to the final completion
    streamed = "".join(d["text"] for k, d in events if k == "token")
    assert streamed.strip() == done["completion"].strip()


async def test_streaming_from_cache(pro):
    payload = {"prompt": "cache streamed reply", "max_tokens": 24, "temperature": 0.2}
    assert (await pro.post("/v1/generate", json=payload)).status_code == 200
    async with pro.stream("POST", "/v1/generate", json={**payload, "stream": True}) as resp:
        raw = ""
        async for chunk in resp.aiter_text():
            raw += chunk
    events = parse_sse(raw)
    assert events[-1][0] == "done"
    assert events[-1][1]["cached"] is True


# ---------------------------------------------------------------------------
# history (eventual consistency) + usage (strict consistency)
# ---------------------------------------------------------------------------

async def test_history_written_eventually(pro):
    prompt = "Remember me: unique-history-prompt-42"
    r = await pro.post("/v1/generate", json={"prompt": prompt, "max_tokens": 24, "temperature": 0.9})
    assert r.status_code == 200

    item = None
    for _ in range(30):  # settlement worker is asynchronous by design
        page = (await pro.get("/v1/history", params={"limit": 5})).json()
        item = next((i for i in page["items"] if i["prompt"] == prompt), None)
        if item:
            break
        await asyncio.sleep(0.1)
    assert item is not None, "history should be written asynchronously"
    assert item["status"] == "ok"
    assert item["tokens_out"] > 0


async def test_usage_ledger_strictly_updated(pro):
    before = (await pro.get("/v1/me")).json()
    r = await pro.post(
        "/v1/generate",
        json={"prompt": "billing probe prompt", "max_tokens": 32, "temperature": 0.9},
    )
    assert r.status_code == 200
    usage = r.json()["usage"]

    after = before
    for _ in range(30):
        await asyncio.sleep(0.1)
        after = (await pro.get("/v1/me")).json()
        if after["tokens_used"] >= before["tokens_used"] + usage["total_tokens"]:
            break
    assert after["tokens_used"] >= before["tokens_used"] + usage["total_tokens"]
    assert after["tokens_reserved"] == before["tokens_reserved"]


async def test_history_pagination(pro):
    for i in range(3):
        await pro.post(
            "/v1/generate",
            json={"prompt": f"page test {i}", "max_tokens": 8, "temperature": 0.9},
        )
    page = (await pro.get("/v1/history", params={"limit": 1})).json()
    assert len(page["items"]) <= 1
    if page["next_cursor"]:
        older = (await pro.get("/v1/history", params={"limit": 1, "before": page["next_cursor"]})).json()
        assert len(older["items"]) <= 1


# ---------------------------------------------------------------------------
# failure handling & graceful degradation
# ---------------------------------------------------------------------------

async def test_engine_failure_returns_502(pro):
    r = await pro.post("/v1/generate", json={"prompt": "__FAIL__ boom", "max_tokens": 16})
    assert r.status_code == 502
    assert r.json()["detail"]["code"] == "engine_error"


async def test_crashed_worker_eventually_fails_cleanly(pro):
    r = await pro.post("/v1/generate", json={"prompt": "__CRASH__ node died", "max_tokens": 32})
    assert r.status_code == 502
    assert r.json()["detail"]["code"] == "worker_crash"


async def test_circuit_breaker_degrades(pro):
    # warm the cache while the fleet is healthy
    cached_prompt = {"prompt": "degrade-safe cached answer", "max_tokens": 16, "temperature": 0.2}
    first = await pro.post("/v1/generate", json=cached_prompt)
    assert first.status_code == 200
    assert first.json()["cached"] is False

    # trip the breaker with repeated engine failures
    for i in range(6):
        r = await pro.post("/v1/generate", json={"prompt": f"__FAIL__ degrade {i}", "max_tokens": 8})
        if i < 5:
            assert r.status_code == 502
        else:
            assert r.status_code == 503
            assert r.json()["detail"]["code"] == "degraded"
            assert r.headers.get("retry-after")

    # a cached answer is still served while degraded (graceful degradation)
    second = await pro.post("/v1/generate", json=cached_prompt)
    assert second.status_code == 200
    assert second.json()["cached"] is True

    # but fresh GPU work fails fast instead of queueing behind a broken fleet
    third = await pro.post("/v1/generate", json={"prompt": "fresh prompt while degraded", "max_tokens": 8})
    assert third.status_code == 503
    assert third.json()["detail"]["code"] == "degraded"


async def test_queue_shedding(pro, monkeypatch):
    object.__setattr__(settings, "queue_max", 0)
    try:
        r = await pro.post("/v1/generate", json={"prompt": "shed me", "max_tokens": 8})
        assert r.status_code == 503
        assert r.json()["detail"]["code"] == "overloaded"
    finally:
        object.__setattr__(settings, "queue_max", 1000)


async def test_timeout_refunds_reservation(pro, monkeypatch):
    baseline = (await pro.get("/v1/me")).json()
    object.__setattr__(settings, "job_timeout_s", 0.2)
    try:
        r = await pro.post(
            "/v1/generate",
            json={"prompt": "timeout probe with a fairly long answer", "max_tokens": 512, "temperature": 0.9},
        )
        assert r.status_code == 504
    finally:
        object.__setattr__(settings, "job_timeout_s", 6.0)

    # reservation must be released through the async settlement path
    released = False
    for _ in range(40):
        await asyncio.sleep(0.15)
        now = (await pro.get("/v1/me")).json()
        if now["tokens_reserved"] <= baseline["tokens_reserved"]:
            released = True
            break
    assert released, "timed-out reservation was never refunded"


async def test_quota_exceeded(pro, monkeypatch):
    snap = (await pro.get("/v1/me")).json()
    monkeypatch.setitem(settings.monthly_quotas, snap["tier"], snap["tokens_used"] + 40)
    r = await pro.post("/v1/generate", json={"prompt": "no budget left", "max_tokens": 256})
    assert r.status_code == 402
    assert r.json()["detail"]["code"] == "quota_exceeded"


# ---------------------------------------------------------------------------
# ops endpoints
# ---------------------------------------------------------------------------

async def test_health_and_metrics(pro):
    h = await pro.get("/healthz")
    assert h.status_code == 200
    assert h.json()["workers"] >= 1

    await pro.post("/v1/generate", json={"prompt": "metrics probe", "max_tokens": 8, "temperature": 0.9})
    m = await pro.get("/metrics")
    assert m.status_code == 200
    assert "inferfleet_requests_total" in m.text
    assert "inferfleet_batch_size" in m.text


async def test_admin_stats(pro):
    await pro.post("/v1/generate", json={"prompt": "stats probe", "max_tokens": 8, "temperature": 0.9})
    r = await pro.get("/v1/admin/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["circuit"] in {"CLOSED", "HALF_OPEN", "OPEN"}
    assert body["workers"]
    assert body["mode"] == "memory"


async def test_dashboard_renders(client):
    r = await client.get("/dashboard")
    assert r.status_code == 200
    assert "InferFleet" in r.text
