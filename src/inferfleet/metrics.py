from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry(auto_describe=True)

# ---- gateway ----
REQUESTS = Counter(
    "inferfleet_requests_total",
    "Requests handled by the gateway.",
    ["endpoint", "status", "tier"],
    registry=REGISTRY,
)
TTFB = Histogram(
    "inferfleet_ttfb_seconds",
    "Time to first token seen by the client.",
    buckets=(0.02, 0.05, 0.1, 0.15, 0.3, 0.5, 1.0, 2.0, 5.0),
    registry=REGISTRY,
)
REQUEST_DURATION = Histogram(
    "inferfleet_request_duration_seconds",
    "End-to-end request duration.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0, 30.0),
    registry=REGISTRY,
)
CACHE_HITS = Counter(
    "inferfleet_cache_hits_total",
    "Responses served from cache.",
    ["kind"],
    registry=REGISTRY,
)
QUEUE_DEPTH = Gauge(
    "inferfleet_queue_depth",
    "Jobs waiting in the inference queue.",
    registry=REGISTRY,
)
QUEUE_SHED = Counter(
    "inferfleet_queue_shed_total",
    "Requests rejected by admission control (load shedding).",
    registry=REGISTRY,
)
CIRCUIT_STATE = Gauge(
    "inferfleet_circuit_state",
    "Circuit breaker state: 0 closed, 1 half-open, 2 open.",
    registry=REGISTRY,
)
QUOTA_REJECTED = Counter(
    "inferfleet_quota_rejected_total",
    "Requests rejected because the monthly token budget was exhausted.",
    registry=REGISTRY,
)

# ---- worker / GPU ----
BATCH_SIZE = Histogram(
    "inferfleet_batch_size",
    "Dynamic batch size at which a GPU step ran.",
    buckets=(1, 2, 4, 8, 16, 32, 64),
    registry=REGISTRY,
)
GPU_UTILIZATION = Gauge(
    "inferfleet_gpu_utilization",
    "Fraction of the GPU batch capacity currently occupied.",
    registry=REGISTRY,
)
WORKER_INFLIGHT = Gauge(
    "inferfleet_worker_inflight",
    "Requests currently being generated.",
    registry=REGISTRY,
)
TOKENS = Counter(
    "inferfleet_tokens_total",
    "Tokens produced or consumed.",
    ["direction"],
    registry=REGISTRY,
)
JOBS = Counter(
    "inferfleet_jobs_total",
    "Inference jobs processed by workers.",
    ["status"],
    registry=REGISTRY,
)

# ---- history / billing ----
SETTLEMENTS = Counter(
    "inferfleet_settlements_total",
    "Billing settlements written by the history service.",
    ["status"],
    registry=REGISTRY,
)


def render() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
