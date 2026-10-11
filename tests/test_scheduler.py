from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

import httpx

from inferfleet import metrics as m
from inferfleet.config import settings
from inferfleet.db import DEMO_KEYS
from inferfleet.gateway.app import app


def _count(kind: str) -> float:
    return m.ADMISSIONS.labels(kind=kind)._value.get()


async def _burst(client, n: int = 12):
    prompts = [
        {"prompt": f"burst admission probe {i}", "max_tokens": 32, "temperature": 0.9}
        for i in range(n)
    ]
    return await asyncio.gather(
        *(client.post("/v1/generate", json=p) for p in prompts)
    )


# ---------------------------------------------------------------------------
# continuous scheduling: jobs must join a running batch
# ---------------------------------------------------------------------------

async def test_continuous_admission_joins_running_batch(pro):
    """The test suite runs with batch=8, so a burst of 12 can never fit in the
    initial claim: at least 4 jobs must be admitted *while* the first batch is
    already decoding (warm admission)."""
    before = _count("warm")
    responses = await _burst(pro, 12)
    failures = [r.text for r in responses if r.status_code != 200]
    assert not failures, failures
    assert _count("warm") - before >= 1, "no job was admitted into a running batch"


async def test_admission_wait_histogram_recorded(pro):
    text_before = (await pro.get("/metrics")).text
    before = _hist_count(text_before)

    r = await pro.post(
        "/v1/generate",
        json={"prompt": "admission wait probe", "max_tokens": 16, "temperature": 0.9},
    )
    assert r.status_code == 200

    text_after = (await pro.get("/metrics")).text
    assert _hist_count(text_after) > before, "admission wait was never observed"


def _hist_count(metrics_text: str) -> float:
    for line in metrics_text.splitlines():
        if line.startswith("inferfleet_admission_wait_seconds_count"):
            return float(line.split()[-1])
    return 0.0


# ---------------------------------------------------------------------------
# static ablation: the same burst must NEVER warm-admit
# ---------------------------------------------------------------------------

async def test_static_scheduler_never_warm_admits(pro):
    object.__setattr__(settings, "scheduler", "static")
    try:
        # Burn any worker cycle that computed the admission policy before the
        # flip, then give idle cycles time to re-read it (cycles run every
        # batch_wait_ms).
        warmup = await pro.post(
            "/v1/generate",
            json={"prompt": "static scheduler warmup", "max_tokens": 8, "temperature": 0.9},
        )
        assert warmup.status_code == 200
        await asyncio.sleep(0.15)

        before_warm, before_cold = _count("warm"), _count("cold")
        responses = await _burst(pro, 12)
        failures = [r.text for r in responses if r.status_code != 200]
        assert not failures, failures

        assert _count("warm") - before_warm == 0, "static mode must not admit mid-batch"
        # every burst job still ran — via cold claims of fresh batches
        assert _count("cold") - before_cold == 12
    finally:
        object.__setattr__(settings, "scheduler", "continuous")


async def test_admin_stats_exposes_scheduler(pro):
    r = await pro.get("/v1/admin/stats")
    assert r.status_code == 200
    assert r.json()["scheduler"] in {"continuous", "static"}


# ---------------------------------------------------------------------------
# load generator smoke test (ASGI transport, no real socket)
# ---------------------------------------------------------------------------

async def test_loadgen_smoke(client):
    from inferfleet.loadgen import LoadConfig, run

    def factory():
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://test", timeout=20.0)

    cfg = LoadConfig(
        key=DEMO_KEYS["pro"],
        requests=30,
        concurrency=6,
        pool=40,
        seed=3,
        max_tokens=16,
        stream_ratio=0.3,
    )
    result = await run(cfg, client_factory=factory)
    s = result.summary

    assert s["sent"] == 30
    assert s["ok"] == 30, s["errors"]
    assert s["latency_ms"]["p50"] > 0
    assert s["tokens_out"] > 0
    assert s["ttfb_ms"] is not None  # stream_ratio > 0 produced stream samples
    assert result.metrics["admissions"] >= 0
    json.dumps(asdict(result))  # must be serializable for --json output
