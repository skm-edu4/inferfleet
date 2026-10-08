# InferFleet

**A distributed LLM inference serving platform** — API gateway, dynamic GPU batching,
two-tier response caching, tiered rate limiting, strict billing consistency, circuit-breaker
degradation, and async history settlement.

> Interviews want to see that you can *design* a system. This project shows you can
> **build** it: every pattern in [`DESIGN.md`](DESIGN.md) is implemented here and
> covered by **34 passing tests**.

```
client ──► gateway(auth · rate-limit · quota · cache · breaker) ──► Redis Stream ──► GPU workers(batched)
                 │                                                     │                    │
                 └── SSE tokens ◄──── Pub/Sub ◄─────────────────────────┴────────────────────┘
                 │
                 └──► settlement service ──► history (eventual) + billing ledger (strict)
```

---

## Quickstart (zero external dependencies)

```bash
make install          # creates .venv, installs deps
make test             # 34 tests
make run              # all-in-one: gateway + worker + settlement on :8000
```

Then, in a second terminal:

```bash
make demo             # scripted tour: health → generate → cache hit → stream → history
open http://localhost:8000/dashboard   # live ops dashboard
open http://localhost:8000/docs        # OpenAPI
```

Demo API keys (auto-seeded, deterministic):

| Tier | Key | Rate limit | Monthly quota |
|---|---|---|---|
| free | `ifk_live_free000000000000000000000000001` | 20 rpm | 100k tokens |
| pro | `ifk_live_pro00000000000000000000000000001` | 120 rpm | 5M tokens |
| enterprise | `ifk_live_ent00000000000000000000000000001` | 1200 rpm | 200M tokens |

```bash
export KEY=ifk_live_pro00000000000000000000000000001

# sync generation
curl -s localhost:8000/v1/generate -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"How do I reset my password?","temperature":0.2}'

# streaming (SSE, first token in milliseconds)
curl -sN localhost:8000/v1/generate -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain dynamic batching.","stream":true}'

# usage & quota
curl -s localhost:8000/v1/me -H "Authorization: Bearer $KEY"
```

## Production-shaped mode (4 microservices + Redis + Postgres)

```bash
make up               # docker compose: gateway, 3× worker, settlement, redis, postgres
make logs
make down
```

Same code, different deployment shape — `IF_MODE=memory` runs every service in one
process (dev/test), `IF_MODE=redis` splits them across containers. That split is
itself a talking point: **the bus is an interface with two implementations**
(`src/inferfleet/bus.py`).

---

## What each layer demonstrates

| Layer | Implemented here | Interview topic |
|---|---|---|
| **Gateway** | `gateway/app.py` | API design, auth (hashed keys), tiered rate limits, admission control, SSE, TTFB measurement |
| **Billing** | `billing.py`, `db.py` | Row-locked reservations, idempotent async settlement, fail-closed on DB outage |
| **Queue** | `bus.py` (Redis Streams + in-memory) | Pull-based load balancing, consumer groups, PEL/`XAUTOCLAIM` crash recovery, `XLEN` backpressure |
| **Workers** | `worker/` | Dynamic batching (≤20 ms window, 16-way), lockstep token emission, restart/seq dedupe, heartbeats |
| **Cache** | `cache.py` | Exact (shared Redis KV) + semantic (cosine ≥ 0.90), temperature-gated, LRU-capped |
| **Reliability** | `circuit.py` | Closed → open → half-open breaker, fail fast, cache-served degradation |
| **Observability** | `metrics.py` | Prometheus: TTFB histogram, queue depth, batch size, GPU utilization, breaker state |
| **Ops** | `/dashboard`, `/v1/admin/stats` | Queue depth, p50/p99, fleet health at a glance |

## Project structure

```
inferfleet/
├── DESIGN.md                  # STAR system design document (interview artifact)
├── README.md
├── docker-compose.yml         # gateway + 3 workers + settlement + redis + postgres
├── Makefile
├── scripts/
│   ├── demo.sh                # scripted live demo
│   └── seed.py                # idempotent tenant seeding
├── src/inferfleet/
│   ├── gateway/app.py         # API gateway: auth, quota, cache, breaker, SSE
│   ├── worker/                # GPU worker: dynamic batching + mock engine
│   ├── history/app.py         # async settlement (history + billing)
│   ├── bus.py                 # data plane: MemoryBus | RedisBus
│   ├── billing.py             # reserve / settle / refund (idempotent)
│   ├── cache.py               # exact + semantic response cache
│   ├── circuit.py             # circuit breaker state machine
│   ├── auth.py                # key auth + rate limiting
│   ├── db.py                  # SQLAlchemy models + demo seed
│   └── metrics.py             # Prometheus instrumentation
└── tests/                     # 34 tests: bus unit + auth/cache + full e2e
```

## Test suite

```bash
make test
# 34 passed in ~18s
```

Coverage highlights: SSE token reconstruction, exact & semantic cache hits, rate-limit
enforcement, quota 402, circuit-breaker degradation (cache still served), queue shedding,
timeout reservation refunds, worker-crash recovery, history idempotency, broker FIFO /
redelivery / fan-out semantics.

## Design rationale

The full STAR-framed design document — requirements math, architecture, schemas,
failure matrix, trade-off table, metrics, and anticipated interview questions — is in
**[DESIGN.md](DESIGN.md)**.

## CV-ready bullet points

> - Designed and built **InferFleet**, a distributed LLM serving platform handling a
>   simulated 50K RPS / 300 ms TTFB SLO: stateless API gateway, Redis Stream job queue
>   with pull-based load balancing, and GPU workers using **dynamic batching**
>   (≤20 ms windows, 16-way) for ~16× GPU throughput.
> - Implemented **two-tier response caching** (shared exact + local semantic cosine
>   search) and a **circuit breaker** with graceful degradation — cached answers keep
>   serving while the fleet is down; admissions shed with `Retry-After` instead of cascading.
> - Enforced **strict billing consistency** with row-locked quota reservations and an
>   **idempotent async settlement pipeline** (exactly-once accounting under at-least-once
>   delivery); history stays eventually consistent off the hot path.
> - Instrumented end-to-end with **Prometheus** (TTFB, queue depth, batch size, GPU
>   utilization, breaker state) + live ops dashboard; **34 integration/unit tests**;
>   runs as one process for dev and **4 microservices** (gateway/workers/settlement + Redis/Postgres) via Docker Compose.

## License

MIT
