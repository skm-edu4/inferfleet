from __future__ import annotations

import pytest

from inferfleet.auth import Principal, RateLimited, check_rate_limit
from inferfleet.bus import MemoryBus
from inferfleet.cache import cosine, embed
from inferfleet.circuit import CircuitBreaker, State

# ---------------------------------------------------------------------------
# authentication
# ---------------------------------------------------------------------------

async def test_missing_key_is_rejected(client):
    r = await client.post("/v1/generate", json={"prompt": "hello"})
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "missing_api_key"


async def test_bad_key_is_rejected(client):
    client.headers["Authorization"] = "Bearer ifk_live_nope"
    r = await client.post("/v1/generate", json={"prompt": "hello"})
    assert r.status_code == 401


async def test_valid_key_accepted(pro):
    r = await pro.get("/v1/me")
    assert r.status_code == 200
    body = r.json()
    assert body["tier"] == "pro"
    assert body["rate_limit_rpm"] > 0


async def test_key_management_flow(pro):
    created = await pro.post("/v1/keys", json={"label": "ci", "tier": "pro"})
    assert created.status_code == 200
    raw = created.json()["key"]
    assert raw.startswith("ifk_live_")

    keys = await pro.get("/v1/keys")
    assert keys.status_code == 200
    assert len(keys.json()) >= 2

    # the freshly minted key works immediately
    fresh = await pro.post(
        "/v1/generate",
        json={"prompt": "ping", "max_tokens": 16},
        headers={"Authorization": f"Bearer {raw}"},
    )
    assert fresh.status_code == 200

    # ...and can be revoked
    key_id = next(k["id"] for k in (await pro.get("/v1/keys")).json() if k["label"] == "ci")
    revoked = await pro.delete(f"/v1/keys/{key_id}")
    assert revoked.status_code == 200
    after = await pro.post(
        "/v1/generate",
        json={"prompt": "ping", "max_tokens": 16},
        headers={"Authorization": f"Bearer {raw}"},
    )
    assert after.status_code == 401


async def test_rate_limit_enforced_per_tier(client):
    from inferfleet.db import DEMO_KEYS

    # mint a dedicated free-tier key so we do not poison other tests' bucket
    created = await client.post(
        "/v1/keys",
        json={"label": "rl", "tier": "free"},
        headers={"Authorization": f"Bearer {DEMO_KEYS['enterprise']}"},
    )
    assert created.status_code == 200
    raw = created.json()["key"]

    codes = []
    retry_after = None
    for _ in range(25):
        r = await client.post(
            "/v1/generate",
            json={"prompt": "hi", "max_tokens": 8},
            headers={"Authorization": f"Bearer {raw}"},
        )
        codes.append(r.status_code)
        if r.status_code == 429:
            retry_after = r.headers.get("retry-after")
            break
    assert 429 in codes, "free tier must hit its rate limit"
    assert retry_after is not None and int(retry_after) > 0


async def test_rate_limiter_unit():
    bus = MemoryBus()
    principal = Principal(
        key_id="k", user_id="u", tier="free", label="t", rate_limit_rpm=3, monthly_quota_tokens=1000
    )
    for _ in range(3):
        await check_rate_limit(bus, principal)
    with pytest.raises(RateLimited):
        await check_rate_limit(bus, principal)


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------

async def test_exact_cache_hit(pro):
    payload = {"prompt": "What is our refund policy exactly?", "max_tokens": 24, "temperature": 0.2}
    first = await pro.post("/v1/generate", json=payload)
    assert first.status_code == 200
    assert first.json()["cached"] is False

    second = await pro.post("/v1/generate", json=payload)
    assert second.status_code == 200
    body = second.json()
    assert body["cached"] is True
    assert body["cache_kind"] == "exact"
    assert body["choices"][0]["text"] == first.json()["choices"][0]["text"]


async def test_semantic_cache_hit(pro):
    base = {"prompt": "how do I reset my account password", "max_tokens": 24, "temperature": 0.2}
    first = await pro.post("/v1/generate", json=base)
    assert first.status_code == 200
    assert first.json()["cached"] is False

    near = dict(base, prompt="how do I reset my account password please")
    second = await pro.post("/v1/generate", json=near)
    assert second.status_code == 200
    assert second.json()["cached"] is True
    assert second.json()["cache_kind"] == "semantic"


def test_embedding_similarity_sanity():
    a = embed("reset my account password")
    b = embed("please reset my account password")
    assert cosine(a, b) > 0.8
    c = embed("summarize the quarterly earnings report")
    assert cosine(a, c) < 0.5


def test_circuit_breaker_transitions():
    cb = CircuitBreaker(failure_threshold=3, window_s=10, cooldown_s=0.01)
    assert cb.allow()
    cb.record_failure()
    cb.record_failure()
    assert cb.state == State.CLOSED
    cb.record_failure()
    assert cb.state == State.OPEN
    assert not cb.allow()
    import time

    time.sleep(0.02)
    assert cb.allow()  # half-open probe
    cb.record_success()
    assert cb.state == State.CLOSED
