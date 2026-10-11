# InferFleet — System Design Document
### Designing a Distributed AI Model Serving Platform

| | |
|---|---|
| **Author** | MKS, Lead Backend Engineer |
| **Format** | STAR-framed system design (45-minute whiteboard ready) |
| **Companion code** | This repository — every component described here is implemented and tested (34 tests) |
| **Stack** | Python · FastAPI · Redis Streams · PostgreSQL/SQLite · SQLAlchemy · Prometheus |

---

## 1. SITUATION — The Context & Requirements

### 1.1 Business context

We are a fast-growing AI startup. The team has just finished training a state-of-the-art
LLM, and the business goal is to expose it to the public through an API — the same
product shape as OpenAI's completions endpoint. The model itself is a solved problem;
**the hard problem is serving it**: GPUs are expensive, inference is slow, and the
product promise (instant, always-available responses) fights against both.

### 1.2 Functional requirements

| # | Requirement |
|---|---|
| F1 | Users send text prompts and receive AI-generated text (streamed and non-streamed). |
| F2 | Users can view their past generation history. |
| F3 | Users can manage API keys and belong to a billing tier: **Free, Pro, Enterprise**. |

### 1.3 Non-functional requirements (the scale)

| Requirement | Target | Design pressure it creates |
|---|---|---|
| **Traffic** | 10M DAU, peak **50,000 RPS** | Horizontal scaling everywhere; no shared bottleneck on the hot path |
| **Latency** | **TTFB < 300 ms** | Streaming is mandatory; queueing must be bounded; caching is not optional |
| **Availability** | **99.99 %** (~52 min downtime/year) | Redundancy, graceful degradation, circuit breakers, retries |
| **Consistency** | Eventual for history; **strict for billing** | Two different write paths: async settlement vs. synchronous row-locked reservation |

### 1.4 The three numbers that drive the entire design

```
50,000 RPS  ×  ~1.5 s average GPU time per request
            =  ~75,000 GPU-seconds of work arriving per second
```

No GPU fleet on earth absorbs that naively. Everything below — the queue, the dynamic
batcher, the cache, the router, the degrader — exists to **shrink that number** before
it reaches the GPUs, and to **smooth the remainder** so the fleet runs at high
utilization instead of thrashing.

---

## 2. TASK — The Scope & Constraints

### 2.1 Core objective

> Design the backend infrastructure that **routes user requests to GPU clusters**,
> **handles queuing during high traffic**, and **stores user data efficiently** —
> under hard cost and reliability constraints.

### 2.2 Constraints

| Constraint | Design consequence |
|---|---|
| **GPU compute is expensive and finite** | Never let a request reach a GPU if a cheaper answer exists (cache), and pack many requests per GPU pass (dynamic batching) |
| **Microservices architecture** | Gateway / Scheduler-worker / Settlement are separate processes with independent scaling; contract between them is a durable queue, not a function call |
| **Graceful degradation if the model is down** | Circuit breaker + cache fallback + polite `503` with `Retry-After`; never leave a client hanging against a dead fleet |

### 2.3 Explicitly out of scope (stated to the interviewer)

- Training / fine-tuning pipelines.
- The model runtime itself (vLLM/TensorRT-LLM would plug into our worker interface).
- Web console UI (only the ops dashboard and public API are in scope).
- Multi-region active-active (listed under Future Improvements).

---

## 3. ACTION — The Design & Execution

### 3.1 High-level architecture

```
                            ┌──────────────────────────────┐
   clients                  │   L7 Load Balancer (NGINX)   │
  (SDK / curl) ────────────►│   TLS termination, retries   │
                            └──────────────┬───────────────┘
                                           │ 50k RPS spread across pods
                            ┌──────────────▼───────────────┐
                            │   API GATEWAY  (FastAPI)     │  ◄── stateless, N replicas
                            │  ─ auth (API keys)           │
                            │  ─ rate limiting (tiered)    │
                            │  ─ billing reservation       │      ┌────────────────┐
                            │  ─ admission control         │─────►│  PostgreSQL    │
                            │  ─ response cache lookup     │      │  keys + ledger │
                            │  ─ circuit breaker           │      │  (strict)      │
                            └───────┬──────────────▲───────┘      └────────────────┘
                     enqueue job    │              │ SSE tokens (fan-out)
                     subscribe first│              │
                            ┌───────▼──────────────┴───────┐
                            │  REDIS STREAMS  (q:jobs)     │  ◄── durable queue, backpressure
                            │  + Pub/Sub  gen:{request_id} │
                            └───────┬──────────────▲───────┘
                        XREADGROUP  │              │ token events + heartbeats
                  (pull = built-in  │              │
                   load balancing)  │              │
                     ┌──────────────▼───┐  ┌───────┴────────┐   ┌───────────────────┐
                     │  GPU WORKER #1   │  │  GPU WORKER #N │   │  SETTLEMENT SVC   │
                     │  dynamic batcher │  │  (scale to 3+) │   │  (q:results)      │
                     │  + mock engine*  │  │                │   │  async history +  │
                     └──────┬───────────┘  └───────┬────────┘   │  billing settle   │
                            │                      │           └─────────┬─────────┘
                            └──────────┬───────────┘                     │
                                       ▼                                 ▼
                              Prometheus /metrics  ◄──────────────  SQLite / PostgreSQL
                              ops dashboard                       history (eventual)
```

\* The mock engine implements the exact interface a real runtime (vLLM) would: it
receives a batch of prompts and advances them token-by-token. Swapping in a real model
touches **one file** (`worker/engine.py`).

**Request lifecycle (happy path):**

```
POST /v1/generate
   │  1. authenticate API key                (~1 ms, DB read, hashed lookup)
   │  2. rate limit (fixed window, shared)   (~0.3 ms, Redis INCR)
   │  3. admission control (queue depth)     (~0.2 ms, XLEN)
   │  4. cache lookup (exact → semantic)     (~0.5 ms, local + Redis GET)
   │  5. billing reservation (row lock)      (~1.5 ms, BEGIN … FOR UPDATE)
   │  6. subscribe to token channel          (~0.2 ms)   ← subscribe BEFORE enqueue
   │  7. enqueue job                         (~0.5 ms)
   │                                         ──────────── TTFB budget so far: ~5 ms
   │  8. worker claims batch, one GPU pass   (batch wait ≤ 20 ms + first token)
   │  9. tokens fan out over Pub/Sub → SSE   (first token at ~30–80 ms)
   │ 10. done → history written asynchronously (settlement service)
   ▼
200 JSON  or  text/event-stream
```

### 3.2 API design

REST for control-plane, streaming over SSE for the data plane (see §3.5 for why).

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/generate` | Generate. Body: `prompt, max_tokens, temperature, stream, model`. Returns JSON, or SSE when `stream=true` |
| `GET` | `/v1/history?limit&before` | Cursor-paginated generation history (eventual consistency) |
| `GET` | `/v1/me` | Tier, rate limit, quota, usage, remaining budget |
| `GET` | `/v1/keys` | List keys (masked — raw key shown **once** at creation) |
| `POST` | `/v1/keys` | Mint a key (cannot exceed your own tier) |
| `DELETE` | `/v1/keys/{id}` | Revoke a key (immediate, checked per request) |
| `GET` | `/v1/admin/stats` | Queue depth, circuit state, worker fleet, p50/p99 |
| `GET` | `/healthz` · `/metrics` · `/dashboard` | Liveness · Prometheus · ops dashboard |

Example:

```bash
curl -sN localhost:8000/v1/generate \
  -H "Authorization: Bearer $INFERFLEET_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"How do I reset my password?","stream":true,"temperature":0.2}'
```

```text
event: meta
data: {"id": "gen_9f2c…", "model": "inferfleet-8x7b-instruct"}

event: token
data: {"seq": 0, "text": "Thanks"}

event: token
data: {"seq": 1, "text": " for"}

event: done
data: {"id": "gen_9f2c…", "completion": "Thanks for …", "usage": {"prompt_tokens": 6, "completion_tokens": 64, "total_tokens": 70}}
```

**SSE design notes (defensible details):**
- Every token event carries a monotonic `seq`. A retried worker emits `restart` first,
  so the gateway can reset its dedupe cursor — **redelivery never corrupts the stream**.
- The gateway **subscribes before it enqueues** — no event can be lost in the gap.
- `done` always carries the authoritative completion + usage, so a client that missed
  tokens still converges to the correct final state.

### 3.3 Database design

Two databases, two consistency regimes, chosen per access pattern:

| Data | Store | Why |
|---|---|---|
| Users, API keys, **billing ledger** | **PostgreSQL** (SQLite in dev) | Strong transactions; ledger needs `SELECT … FOR UPDATE` row locking |
| Generation history | Same PG store, append-only table, written **asynchronously** | Eventual consistency is explicitly allowed; keeps the hot path free of history writes |
| Exact response cache | **Redis KV** with TTL | Shared across gateway replicas; sub-millisecond |
| Semantic cache index | In-process vector index (hashing-trick embeddings, cosine) | Warm per replica; avoids a network hop on the hottest lookup (trade-off in §4.1) |
| Rate-limit counters | **Redis** `INCR` + `EXPIRE` | Shared across replicas |

**Schema (implemented in `src/inferfleet/db.py`):**

```sql
users            (id PK, email UNIQUE, created_at)

api_keys         (id PK, key_hash UNIQUE -- sha256, never the raw key --,
                  user_id FK, label, tier, revoked, created_at)

generations      (id PK,                  -- == request id: idempotency key
                  user_id idx, key_id,
                  prompt, completion, model,
                  tokens_in, tokens_out, latency_ms,
                  cached bool, status, created_at idx)

usage_ledger     (user_id PK, period PK 'YYYY-MM',   -- composite PK = one row per billing period
                  tokens_used, tokens_reserved, requests, updated_at)
```

**The ledger is the strict-consistency showpiece.** Two-phase usage accounting:

```
gateway (synchronous, strict)                 settlement service (asynchronous, strict)
──────────────────────────────                ─────────────────────────────────────────
BEGIN;
  SELECT … FOR UPDATE  (row lock)
  if used + reserved + est > quota → 402      BEGIN;
  tokens_reserved += est;                       if generation.id already exists → COMMIT (idempotent)
COMMIT;                                         SELECT … FOR UPDATE
                                                tokens_reserved -= est
                                                tokens_used      += actual
                                                INSERT generations (…)
                                              COMMIT;
```

Why reservations instead of charging at the end? Because with 50k RPS and slow GPUs,
the window between *admission* and *completion* is seconds long — without a reservation,
a user with 0 remaining tokens could fire 100 concurrent requests and drain the system
before the first settlement lands. **Reservation makes quota enforcement linearizable
at admission time; settlement later converts reservations into actuals.**

Duplicate settlement (queue redelivery, gateway timeout racing a slow worker) is
defused by the idempotency guard on `generations.id` — a replayed result can never
double-bill. This is tested (`test_timeout_refunds_reservation`,
`test_usage_ledger_strictly_updated`).

### 3.4 Deep dive: the "AI bottleneck"

> *Inference takes seconds; users expect instant. How?*

Four mechanisms, in the order they fire:

**1. Streaming (SSE) decouples TTFB from total latency.**
The product promise is "first token < 300 ms", not "whole answer < 300 ms". Tokens
fan out over Redis Pub/Sub on channel `gen:{request_id}` — one publish, many gateway
subscribers — straight into the client's SSE connection. Sync clients still get the
full JSON when `done` arrives.

**2. Dynamic batching packs the GPU — with iteration-level (continuous) admission.**
One forward pass over a batch of 16 prompts costs almost the same as one pass over a
single prompt. The worker therefore doesn't process jobs one at a time:

```
claim(batch ≤ 16)                      # first job opens the ≤ 20 ms fill window
loop, per decode step:                 # iteration-level scheduling (vLLM-style)
    ADMIT     ── non-blocking claim for any free slots (batch < 16?)
                (skipped once shutdown begins: drain, don't admit)
    DECODE    ── one token for every live request   # one "GPU step", ~18 ms
    RETIRE    ── requests that hit their end-of-sample publish done, ack,
                and free their slot for a waiting job
```

Consequences you can point at in code:
- At low traffic, batch = 1 → latency stays minimal (the window is bounded by 20 ms).
- At peak, batch = 16 → **~16× tokens per GPU second** for the same wall-clock step.
- The window *starts at the first arrival*, so an idle fleet never adds delay.
- Because completion times are ragged (requests stop at different steps), slots
  free **mid-batch** — and `IF_SCHEDULER=continuous` refills them on the next
  step, while the `static` fallback runs each batch to completion first
  (kept as an A/B baseline). Measured on an identical workload: **+34 % throughput,
  −35 % p99, −41 % admission wait** ([BENCHMARKS.md](BENCHMARKS.md)).
- Metrics: `inferfleet_batch_size` histogram (per-step occupancy),
  `inferfleet_gpu_utilization`, `inferfleet_batch_admissions_total{kind=cold|warm}`,
  `inferfleet_admission_wait_seconds` (queue → GPU).

**3. Queuing smooths spikes.**
Jobs land on a Redis Stream (`q:jobs`). Workers **pull** with `XREADGROUP` — this is
load balancing *and* backpressure in one primitive: no worker can be overwhelmed
because it only asks for work when it is ready. A 2× traffic spike becomes queue
depth, not 2× GPU nodes.

**4. The cache removes work entirely.**
Two tiers (see §3.6): exact match via shared Redis KV, near-duplicate match via
cosine similarity over embeddings. Only for `temperature ≤ 0.5` — we never serve a
cached sample when the user asked for stochastic decoding (that would be a correctness
bug, not an optimization).

**What we deliberately do *not* do:** speculative execution and prefix/KV-cache reuse
across requests — real wins, but they belong to the model runtime layer (vLLM), below
our scheduling layer. Listed in Future Improvements.

### 3.5 Scaling & performance

**Caching (GPU cost saver).**
- Exact key: `sha256(model | temperature | max_tokens | normalize(prompt))` in Redis, TTL 15 min.
- Semantic: hashing-trick embeddings (unigrams + bigrams, L2-normalized), cosine ≥ 0.90,
  plus a containment rule ("what is X" inside "what is X please"), LRU-capped at 5k entries.
- Hit path skips queue, GPU **and** the reservation (usage is still recorded, asynchronously).
- Measured in tests: exact and semantic hits both return `cached: true` in < 5 ms.

**Message queues (traffic smoother).**
- `q:jobs` — Redis Stream, consumer group `workers`. Streams give us: durable backlog,
  per-consumer PEL (pending entries list), `XAUTOCLAIM` for dead-consumer recovery,
  and `XLEN` for admission control — all in one data structure.
- `q:results` — settlement stream between gateway/worker and the history service.
- Stream lengths capped (`MAXLEN 100k`) so an outage cannot grow Redis without bound.

**Database scaling (history → terabytes).**
1. **Phase 0 (this repo):** single PG, partitioning-ready schema (`period` on the
   ledger, `created_at` index on generations).
2. **Phase 1:** monthly range-partition `generations`; add a read replica for history
   reads (they're eventual anyway).
3. **Phase 2:** shard by `user_id` hash — history queries are already scoped to a
   single user, so a shard key on `user_id` keeps every read single-shard.
4. **Phase 3:** cold generations → object storage (S3) with a query layer (Trino /
   ClickHouse); the ledger stays in PG forever — it's tiny (≤ 1 row/user/month) and
   it's the part that must be strictly consistent.

**Statelessness = horizontal scale.** Gateways hold zero durable state (cache index
is a warm optimization, rate limits live in Redis). `docker compose up --scale worker=N`.

**Measured performance (reproducible — see [BENCHMARKS.md](BENCHMARKS.md)).**
A load generator (`inferfleet.loadgen`, closed- and open-loop, seeded workloads)
drives the all-in-one deployment and scrapes `/metrics` before/after each run:
- Continuous vs static batching, identical 400-request workload (both runs generated
  exactly 17,205 tokens): **14.3 → 19.2 req/s (+34 %)**, p99 **3,164 → 2,058 ms**,
  batch occupancy **11.7 → 15.6/16**, **25 % fewer GPU steps** for the same work.
- Load sweep: GPU capacity ≈ 18–19 req/s (92 % of the theoretical 20.8) and it
  **plateaus rather than collapses** across a 12× overload range — the 120 rps run
  still completes 16.3 req/s while shedding/timeouts happen at the edge.
- Hot cache (82.8 % hit rate) at 40 rps offered: p50 **2.2 ms** (vs 6.05 s uncached
  at the same offered load) and **−64 % GPU token work**.

### 3.6 Failure handling

| Failure | Detection | Response | Client experience |
|---|---|---|---|
| **GPU node crashes mid-generation** | Job idles in the stream's PEL | `XAUTOCLAIM` reclaims after 60 s (in-process requeue in dev mode); job retried ≤ 3 attempts; worker emits `restart` + `seq` reset so duplicates are deduped | Brief pause, then a correct answer — or a clean `502 worker_crash` after 3 attempts |
| **Engine returns an error** | Error event on the request channel | `502` with machine-readable code; failure recorded in circuit breaker | Fast, explicit failure |
| **Fleet-wide outage** | Circuit breaker: 5 failures / 30 s | **OPEN** → fail fast (no queueing behind a dead backend); after 15 s cooldown → **HALF_OPEN** probe; success → **CLOSED** | `503 degraded` + `Retry-After: 15` in ~1 ms; **cached responses still served** (tested) |
| **No healthy workers** | Heartbeat registry empty (TTL 8 s) | Reject at admission: `503 no_capacity` | Never hangs on an empty queue |
| **Overload (queue full)** | `XLEN ≥ queue_max` | Admission control sheds: `503 overloaded` + `Retry-After: 5` | Controlled degradation, no cascade |
| **Request exceeds deadline** | Gateway timer (30 s budget) | `504` + reservation released through the idempotent settlement path | Explicit timeout; **quota refunded** (tested) |
| **PostgreSQL down** | Pool errors | Reservations fail closed → `503`; we never grant quota we can't account for. History/settlement retries with bounded attempts (outbox-style) | Temporarily degraded, never financially inconsistent |
| **Redis down** | Bus errors | Memory mode runs fully in-process (dev); in prod, queue errors → breaker path → degrade. Failures are **fail-closed for billing, fail-open for nothing** | `503` rather than wrong answers |
| **Slow consumer (client disconnects)** | SSE generator cancelled | Subscription closed, job continues, result still settles billing | No leaked subscriptions |

The invariant worth stating out loud: **every code path that reserves quota has a
paired path that releases it** — success (convert to actuals), failure (release),
timeout (release), crash (release via retry-then-fail). Idempotency on
`generations.id` makes double-release impossible.

### 3.7 Microservice boundaries (and why)

| Service | Responsibility | Scales on | State it owns |
|---|---|---|---|
| **gateway** | Auth, rate limits, admission, cache, breaker, SSE fan-out | Request rate (CPU + sockets) | none (stateless) |
| **worker** | Batch scheduling, GPU execution, token publishing | GPU availability & memory | none (pulls from queue) |
| **history** | Async settlement: ledger conversion + history append | Write throughput | none (owns the write *logic*, not the data) |
| **data stores** | PG (strict), Redis (fast/shared) | — | all durable state |

Contract between services = **durable streams**, never synchronous calls (except
control-plane reads). That is what lets us restart, scale, and redeploy each service
independently — and what makes the failure table above implementable in ~50 lines each.

---

## 4. RESULT — The Evaluation & Trade-offs

### 4.1 Trade-offs made (say these unprompted)

| Decision | What we gained | What we sacrificed | Why it's right here |
|---|---|---|---|
| **SSE instead of WebSockets** | One-way server→client fits generation perfectly; plain HTTP (works through proxies/CDNs/load balancers); trivially re-connectable | No client→server channel mid-stream (we don't need one) | Cheapest primitive that satisfies TTFB; OpenAI/Anthropic use the same choice |
| **Pull-based queue (consumer groups) instead of a push dispatcher** | Backpressure and load balancing for free; no separate scheduler service to operate | Slight per-poll overhead; a busy-loop risk if misconfigured (bounded here by blocking reads) | At 50k RPS a push dispatcher is another stateful service to fail; the broker already has the logic |
| **PostgreSQL for billing + append-only history** | Serializable ledger, row locks, composite-PK idempotency | History joins/analytic queries are weaker than a warehouse | Only billing *needs* strict consistency — so only billing pays for it |
| **Eventual history via async settlement** | Hot path never blocks on history writes; history outage can't take down generation | Users may not see their last request for ~100 ms | Explicitly permitted by the requirements |
| **Semantic cache index in-process, not a vector DB** | Zero extra hop on the hottest path; no infra dependency | Index is per-replica (a hit on replica A isn't visible to B); no ANN at 10M-scale | Local warm cache + shared exact cache captures most hits; a Qdrant/Redis-JSON tier is a config-level addition (Future Work) |
| **Semantic caching gated at `temperature ≤ 0.5`** | Never wrong answers | Cache hit-rate is lower for creative workloads | Correctness beats hit-rate |
| **Fixed-window rate limiting (`INCR`+`EXPIRE`)** | One round-trip, atomic across replicas | Up to 2× burst across window boundaries | Token buckets per tier need Lua/Redis-side state machines; we documented the burst caveat instead of hiding it |
| **Mock engine behind a runtime interface** | Testable, deterministic, zero model downloads | Doesn't prove real GPU scheduling | The *serving platform* is the artifact under review; vLLM plugs into `engine.plan()` |
| **Fail-closed on billing unavailability** | Financial correctness never compromised | PG outage degrades availability below 99.99 % for *new* requests | Charging blindly at 50k RPS is a business-ending bug; degradation is recoverable, un-billed usage is not |

### 4.2 Bottlenecks — where the weakest links are

1. **GPU inference latency is the irreducible floor.** Everything else is scheduling
   around it. No architecture makes a 1.5 s generation instant; it makes the *first
   token* instant and hides the rest behind streaming.
2. **Gateway as the SSE fan-out point** — long-lived connections pin sockets to one
   replica; a deploy restarts them (clients auto-reconnect with `Last-Event-ID` in
   Future Work). Mitigated by many stateless replicas behind the LB.
3. **Single Redis for queue + pub/sub + counters.** Throughput is high, but it is a
   shared fate domain. Mitigation path: Redis Cluster / shard the job streams by
   hash slot, move pub/sub to a dedicated instance.
4. **PG write path for the ledger** — one row per user per period, hot rows under
   concurrency (`FOR UPDATE`). Fine at 10k RPS of *admissions*, but it's the first
   thing to shard if we grow 10×; the composite period key was chosen exactly so
   hot rows spread across months.
5. **Cache invalidation on model updates** — a new model version must flush keys;
   we version the key with `model` in the hash, which prevents *wrong* answers but
   not *stale* ones across versions.

### 4.3 Monitoring & metrics (Prometheus / Datadog)

| Metric | Type | Alert condition |
|---|---|---|
| `inferfleet_ttfb_seconds` (p99) | histogram | p99 > 300 ms for 5 min |
| `inferfleet_queue_depth` | gauge | > 80 % of `queue_max` for 2 min (pre-shed warning) |
| `inferfleet_gpu_utilization` | gauge | < 30 % during peak (over-provisioned / cache too good) or > 90 % sustained (add GPUs) |
| `inferfleet_circuit_state` | gauge | != CLOSED (page: fleet degraded) |
| `inferfleet_requests_total{status}` | counter | 5xx rate > 0.1 % over 5 min; 503 (shed) > 1 % |
| `inferfleet_batch_size` (p50) | histogram | p50 = 1 during peak → batching is broken |
| `inferfleet_admission_wait_seconds` (p99) | histogram | p99 climbing while occupancy < 12 → admission policy regression |
| `inferfleet_batch_admissions_total{kind="warm"}` | counter | warm share → 0 under load → workers are draining batches instead of refilling (continuous batching off?) |
| `inferfleet_cache_hits_total{kind}` | counter | semantic hit-rate collapse → embedding/index regression |
| `inferfleet_quota_rejected_total` | counter | spike → tier/pricing issue, not a system issue |
| SLO: availability = `2xx+4xx / total` | derived | burn-rate alerting on 99.99 % budget |

Every request carries a `gen_*` id visible in gateway logs, worker logs, the history
row, and the settlement event — **one trace id end to end**, joinable across services
in Datadog/LangSmith-style tooling.

### 4.4 Future improvements (6 months, unlimited budget)

1. **Real runtime integration** — vLLM/TensorRT-LLM behind `engine.plan()` (our
   scheduler already does iteration-level admission; paged KV-cache and cross-request
   prefix caching live in the runtime layer below us).
2. **Multi-region active-active** — regional gateway pools, geo-DNS, region-local
   caches; ledger stays single-region (strict consistency is easier to keep linearizable
   in one write region with async replication).
3. **Global semantic cache** — Redis-JSON/Qdrant tier with ANN search + cross-replica
   invalidation, keyed by model version.
4. **Priority queues** — Enterprise jobs on a separate stream with weighted fair
   scheduling; Free tier gets strict FIFO with deeper sheds.
5. **Speculative decoding & KV-cache reuse** across requests with shared prefixes.
6. **Autoscaling policy** — predictive: scale workers on queue-depth slope, not just
   level (spikes at 50k RPS arrive faster than reactive autoscalers react).
7. **Canary model rollout** — stream-level routing % between model versions with
   automated rollback on TTFB/5xx regression.
8. **OpenTelemetry traces** — spans per gateway stage (auth/limit/cache/queue) so the
   300 ms budget is attributable per-hop, not just measured end-to-end.

### 4.5 What the tests prove (verification, not vibes)

| Claim in this document | Test |
|---|---|
| Streaming works and tokens reconstruct the answer | `test_streaming_sse` |
| Exact + semantic caching skip the GPU | `test_exact_cache_hit`, `test_semantic_cache_hit` |
| History is eventual, ledger is strict | `test_history_written_eventually`, `test_usage_ledger_strictly_updated` |
| Rate limits are per-tier and shared | `test_rate_limit_enforced_per_tier`, `test_rate_limiter_unit` |
| Quota cannot be overspent | `test_quota_exceeded` |
| Circuit breaker degrades gracefully, cache still serves | `test_circuit_breaker_degrades` |
| Timeouts refund reservations | `test_timeout_refunds_reservation` |
| Queue shedding protects the fleet | `test_queue_shedding` |
| Worker crash = clean failure, not a hung client | `test_crashed_worker_eventually_fails_cleanly` |
| Jobs join a *running* GPU batch (continuous admission) | `test_continuous_admission_joins_running_batch` |
| Static scheduler never admits mid-batch (A/B switch is real) | `test_static_scheduler_never_warm_admits` |
| Queue→GPU wait is measured per admission | `test_admission_wait_histogram_recorded` |
| Load generator runs a full closed-loop workload in-process | `test_loadgen_smoke` |
| Broker semantics (FIFO, redelivery attempts, pre-buffered fan-out) | `tests/test_bus.py` (8 tests) |

**39 tests, all passing:** `make test`. Benchmark harness: `make bench`.

---

## Appendix A — 45-minute talk track

| Min | Slide |
|---|---|
| 0–3 | Requirements: the three numbers (50k RPS, 300 ms TTFB, 99.99 %) and the GPU math |
| 3–7 | Context diagram: LB → gateway → queue → workers → settlement; the two data stores |
| 7–14 | Request walkthrough: 10 hops, which ones are synchronous, where the 5 ms goes |
| 14–22 | Deep dive #1 — the AI bottleneck: streaming + dynamic batching + queue + cache |
| 22–28 | Deep dive #2 — billing: reservation vs settlement, idempotency, fail-closed |
| 28–34 | Failure table: walk 4 failures, show the circuit breaker state machine |
| 34–40 | Scaling: cache tiers, stream consumer groups, DB sharding plan |
| 40–45 | Trade-offs, bottlenecks, metrics, future work — then invite the attack questions |

## Appendix B — Anticipated hard questions

**"Why not WebSockets?"** — Generation is one-directional; SSE is plain HTTP, survives
proxies/CDNs, and reconnects trivially. WebSockets earn their keep when the client
talks back mid-stream (interactive agents) — that's a Future Work feature, not a
current requirement.

**"What happens when Redis dies?"** — Job enqueue fails → admission rejects with 503
(fail closed, no lost quota); the breaker opens; once Redis returns, the recovery path
is identical to any fleet restart. Billing is never touched by Redis state, so no
financial inconsistency is possible.

**"Isn't a single row per user a hot row in the ledger?"** — One `FOR UPDATE` per
*admission*, held for ~1 ms. At 50k RPS spread over millions of users, per-row
contention is negligible; we batch nothing on purpose because admission must be
linearizable. If one whale user saturates their row, we can split their reservations
into a per-request side table — noted as a scaling step.

**"How do you prevent a poisoned prompt from always hitting the cache with a wrong
answer?"** — Cache keys include model + temperature + max_tokens + normalized prompt;
semantic hits are capped at cosine ≥ 0.90 *and* only at `temperature ≤ 0.5`; entries
expire at 15 min. The QA risk of a *wrong but similar* answer is the real cost of a
semantic cache — we scoped it to low-temperature, factual workloads where it's safe.

**"Why is the mock engine acceptable on a CV?"** — Because the artifact is the
*serving platform*: admission control, batching, streaming, failure handling, billing
consistency. The engine is behind a 1-function interface; a real vLLM server is a
drop-in. Honest scoping beats a half-integrated 7B model that OOMs the reviewer's
laptop.

**"Scale to 500k RPS?"** — Gateways and workers are stateless/pull-based → pure
horizontal scale; shard `q:jobs` streams by hash slot; move pub/sub to its own Redis;
partition PG history monthly then shard by `user_id`; the ledger stays in PG (it
grows with *users*, not *requests*). The design has no single in-process bottleneck
left after that point.
