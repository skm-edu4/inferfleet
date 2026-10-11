"""Async load generator for InferFleet.

Closed-loop mode (--requests/--concurrency): N workers pull from a shared
counter — models patient clients, hides queueing (good for throughput).

Open-loop mode (--rps/--duration): requests are fired on a fixed schedule —
models real traffic, exposes queueing (good for latency curves). Scheduling
lag is recorded separately so coordinated omission stays visible.

Both modes sample prompts from a seeded pool (uniform or Zipf-skewed) so
cache-hit ratios are reproducible, and scrape /metrics before/after to report
admission wait, batch occupancy, token throughput and cache hits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# config & result types
# ---------------------------------------------------------------------------


@dataclass
class LoadConfig:
    base_url: str = "http://127.0.0.1:8000"
    key: str = ""
    # closed loop
    requests: int = 200
    concurrency: int = 16
    # open loop
    rps: float = 0.0
    duration_s: float = 10.0
    # workload shape
    pool: int = 500
    zipf: float = 0.0  # 0 = uniform; 1.2 = heavy hot-prompt skew (cache-friendly)
    seed: int = 7
    stream_ratio: float = 0.0
    max_tokens: int = 64
    temperature: float = 0.9
    timeout_s: float = 30.0
    json_out: str = ""


@dataclass
class RequestResult:
    status: int
    latency_ms: float
    ttfb_ms: Optional[float] = None
    cached: bool = False
    tokens_out: int = 0
    lag_ms: float = 0.0
    error: str = ""


@dataclass
class LoadResult:
    config: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(p * len(ordered)))], 2)


def stats(values: list[float]) -> dict:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": round(max(values), 2),
        "mean": round(sum(values) / len(values), 2),
    }


_PROMPT_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+)$')


def parse_metrics(text: str) -> dict[tuple[str, str], float]:
    """Minimal Prometheus text parser: (name, raw_labels) -> value."""
    out: dict[tuple[str, str], float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _PROMPT_RE.match(line)
        if not m:
            continue
        out[(m.group(1), m.group(2) or "")] = float(m.group(3))
    return out


def _sum_labels(samples: dict[tuple[str, str], float], name: str, label_key: str) -> float:
    """Sum every series of `name` that carries `label_key` (e.g. kind=)."""
    return sum(v for (n, labels), v in samples.items() if n == name and label_key in labels)


async def scrape_metrics(client: httpx.AsyncClient) -> dict[tuple[str, str], float]:
    resp = await client.get("/metrics")
    resp.raise_for_status()
    return parse_metrics(resp.text)


def metrics_delta(before: dict, after: dict, elapsed_s: float) -> dict:
    def d(name: str, labels: str = "") -> float:
        return after.get((name, labels), 0.0) - before.get((name, labels), 0.0)

    wait_sum = d("inferfleet_admission_wait_seconds_sum")
    wait_n = d("inferfleet_admission_wait_seconds_count")
    batch_sum = d("inferfleet_batch_size_sum")
    batch_n = d("inferfleet_batch_size_count")
    ttfb_sum = d("inferfleet_ttfb_seconds_sum")
    ttfb_n = d("inferfleet_ttfb_seconds_count")
    tokens = d("inferfleet_tokens_total", '{direction="out"}')
    cache_hits = _sum_labels(after, "inferfleet_cache_hits_total", "kind=") - _sum_labels(
        before, "inferfleet_cache_hits_total", "kind="
    )
    shed = _sum_labels(after, "inferfleet_queue_shed_total", "") - _sum_labels(
        before, "inferfleet_queue_shed_total", ""
    )
    return {
        "admission_wait_avg_ms": round(wait_sum / wait_n * 1000, 2) if wait_n else None,
        "admissions": int(wait_n),
        "batch_occupancy_avg": round(batch_sum / batch_n, 2) if batch_n else None,
        "batch_steps": int(batch_n),
        "ttfb_avg_ms": round(ttfb_sum / ttfb_n * 1000, 2) if ttfb_n else None,
        "tokens_out": int(tokens),
        "tokens_per_s": round(tokens / elapsed_s, 1) if elapsed_s else 0.0,
        "cache_hits": int(cache_hits),
        "queue_shed": int(shed),
    }


class PromptPool:
    def __init__(self, cfg: LoadConfig):
        self._rng = random.Random(cfg.seed)
        self._size = max(1, cfg.pool)
        if cfg.zipf and cfg.zipf > 0:
            weights = [1.0 / (rank**cfg.zipf) for rank in range(1, self._size + 1)]
            total = sum(weights)
            self._cum = []
            acc = 0.0
            for w in weights:
                acc += w / total
                self._cum.append(acc)
        else:
            self._cum = None

    def next_prompt(self) -> str:
        if self._cum is None:
            idx = self._rng.randrange(self._size)
        else:
            idx = self._pick()
        return f"Explain request topic-{idx} to a customer in plain language."

    def _pick(self) -> int:
        # binary search over precomputed cumulative weights
        u = self._rng.random()
        lo, hi = 0, len(self._cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cum[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        return lo


# ---------------------------------------------------------------------------
# request execution
# ---------------------------------------------------------------------------


async def _one_sync(
    client: httpx.AsyncClient, cfg: LoadConfig, prompt: str, lag_ms: float = 0.0
) -> RequestResult:
    payload = {
        "prompt": prompt,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
    }
    t0 = time.perf_counter()
    try:
        r = await client.post("/v1/generate", json=payload, timeout=cfg.timeout_s)
    except Exception as exc:
        return RequestResult(status=0, latency_ms=0.0, lag_ms=lag_ms, error=type(exc).__name__)
    latency = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        return RequestResult(status=r.status_code, latency_ms=latency, lag_ms=lag_ms)
    body = r.json()
    return RequestResult(
        status=200,
        latency_ms=latency,
        cached=bool(body.get("cached")),
        tokens_out=int(body.get("usage", {}).get("completion_tokens", 0)),
        lag_ms=lag_ms,
    )


async def _one_stream(
    client: httpx.AsyncClient, cfg: LoadConfig, prompt: str, lag_ms: float = 0.0
) -> RequestResult:
    payload = {
        "prompt": prompt,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttfb_ms: Optional[float] = None
    cached = False
    saw_error = False
    try:
        async with client.stream(
            "POST", "/v1/generate", json=payload, timeout=cfg.timeout_s
        ) as r:
            if r.status_code != 200:
                await r.aread()
                return RequestResult(
                    status=r.status_code,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    lag_ms=lag_ms,
                )
            expecting_done = False
            async for line in r.aiter_lines():
                if ttfb_ms is None and line.startswith("event: token"):
                    ttfb_ms = (time.perf_counter() - t0) * 1000
                elif line.startswith("event: done"):
                    expecting_done = True
                elif line.startswith("event: error"):
                    saw_error = True
                elif expecting_done and line.startswith("data:"):
                    cached = '"cached": true' in line
    except Exception as exc:
        return RequestResult(status=0, latency_ms=0.0, lag_ms=lag_ms, error=type(exc).__name__)
    return RequestResult(
        status=200 if not saw_error else 502,
        latency_ms=(time.perf_counter() - t0) * 1000,
        ttfb_ms=ttfb_ms,
        cached=cached,
        lag_ms=lag_ms,
    )


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------


async def run(
    cfg: LoadConfig, client_factory=None
) -> LoadResult:
    """Execute the load. `client_factory()` may return an injected client
    (tests use an ASGI transport); default talks to cfg.base_url over HTTP."""
    pool = PromptPool(cfg)

    def _factory() -> httpx.AsyncClient:
        if client_factory is not None:
            return client_factory()
        return httpx.AsyncClient(base_url=cfg.base_url, timeout=cfg.timeout_s)

    async with _factory() as client:
        headers = {"Authorization": f"Bearer {cfg.key}"} if cfg.key else {}
        client.headers.update(headers)

        before = await scrape_metrics(client)
        started = time.perf_counter()
        results: list[RequestResult] = []

        open_loop = cfg.rps > 0
        if open_loop:
            n = max(1, int(cfg.duration_s * cfg.rps))
            interval = 1.0 / cfg.rps
            t_start = time.perf_counter()

            async def _fire(i: int) -> RequestResult:
                target = t_start + i * interval
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                lag_ms = max(0.0, (time.perf_counter() - target) * 1000)
                prompt = pool.next_prompt()
                use_stream = cfg.stream_ratio and (
                    random.Random(cfg.seed + i).random() < cfg.stream_ratio
                )
                fn = _one_stream if use_stream else _one_sync
                return await fn(client, cfg, prompt, lag_ms)

            results = await asyncio.gather(*(_fire(i) for i in range(n)))
        else:
            counter = iter(range(cfg.requests))

            async def _worker() -> None:
                for i in counter:
                    prompt = pool.next_prompt()
                    use_stream = cfg.stream_ratio and (
                        random.Random(cfg.seed + i).random() < cfg.stream_ratio
                    )
                    fn = _one_stream if use_stream else _one_sync
                    results.append(await fn(client, cfg, prompt))

            await asyncio.gather(
                *(_worker() for _ in range(max(1, cfg.concurrency)))
            )

        elapsed = time.perf_counter() - started
        after = await scrape_metrics(client)

    return LoadResult(
        config=asdict(cfg),
        summary=_summarize(results, elapsed, cfg),
        metrics=metrics_delta(before, after, elapsed),
    )


def _summarize(results: list[RequestResult], elapsed: float, cfg: LoadConfig) -> dict:
    ok = [r for r in results if r.status == 200]
    errors: dict[str, int] = {}
    for r in results:
        if r.status != 200:
            errors[str(r.status) or r.error] = errors.get(str(r.status) or r.error, 0) + 1
    latencies = [r.latency_ms for r in ok]
    ttfbs = [r.ttfb_ms for r in ok if r.ttfb_ms is not None]
    lags = [r.lag_ms for r in results if r.lag_ms > 0]
    offered = cfg.rps if cfg.rps > 0 else len(results) / elapsed if elapsed else 0.0
    return {
        "sent": len(results),
        "ok": len(ok),
        "errors": errors,
        "elapsed_s": round(elapsed, 2),
        "offered_rps": round(offered, 1),
        "achieved_rps": round(len(ok) / elapsed, 1) if elapsed else 0.0,
        "latency_ms": stats(latencies),
        "ttfb_ms": stats(ttfbs) if ttfbs else None,
        "scheduling_lag_ms": stats(lags) if lags else None,
        "cached_pct": round(100 * sum(1 for r in ok if r.cached) / len(ok), 1) if ok else 0.0,
        "tokens_out": sum(r.tokens_out for r in ok),
    }


def _print_report(result: LoadResult) -> None:
    s, m, c = result.summary, result.metrics, result.config
    mode = f"open-loop {c['rps']} rps × {c['duration_s']}s" if c["rps"] > 0 else \
        f"closed-loop {c['concurrency']} workers × {c['requests']} req"
    print(f"\n== InferFleet load report ({mode}) ==")
    print(f"  sent/ok/errors : {s['sent']} / {s['ok']} / {s['errors'] or '{}'}")
    print(f"  achieved rps   : {s['achieved_rps']}  (offered {s['offered_rps']})")
    lat = s["latency_ms"]
    print(f"  latency ms     : p50={lat['p50']} p90={lat['p90']} p99={lat['p99']} max={lat['max']}")
    if s.get("ttfb_ms"):
        t = s["ttfb_ms"]
        print(f"  ttfb ms        : p50={t['p50']} p90={t['p90']} p99={t['p99']}")
    if s.get("scheduling_lag_ms"):
        g = s["scheduling_lag_ms"]
        print(f"  sched lag ms   : p50={g['p50']} p99={g['p99']}")
    print(f"  cache hits     : {s['cached_pct']}%  ({m.get('cache_hits')} in-window)")
    print(
        f"  admission wait : avg={m.get('admission_wait_avg_ms')}ms "
        f"(n={m.get('admissions')})"
    )
    print(f"  batch occupancy: avg={m.get('batch_occupancy_avg')} over {m.get('batch_steps')} steps")
    print(f"  token throughput: {m.get('tokens_per_s')} tok/s  (shed={m.get('queue_shed')})")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="InferFleet load generator")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--key", default="")
    p.add_argument("--requests", type=int, default=200, help="closed-loop: total requests")
    p.add_argument("--concurrency", type=int, default=16, help="closed-loop: workers")
    p.add_argument("--rps", type=float, default=0.0, help="open-loop: offered requests/s")
    p.add_argument("--duration", type=float, default=10.0, help="open-loop: seconds")
    p.add_argument("--pool", type=int, default=500, help="unique prompt pool size")
    p.add_argument("--zipf", type=float, default=0.0, help="zipf skew (0=uniform, 1.2=hot set)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--stream-ratio", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="<=0.5 enables the response cache; >0.5 always bypasses it",
    )
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--json", dest="json_out", default="")
    args = p.parse_args(argv)

    cfg = LoadConfig(
        base_url=args.url,
        key=args.key,
        requests=args.requests,
        concurrency=args.concurrency,
        rps=args.rps,
        duration_s=args.duration,
        pool=args.pool,
        zipf=args.zipf,
        seed=args.seed,
        stream_ratio=args.stream_ratio,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout,
        json_out=args.json_out,
    )
    result = asyncio.run(run(cfg))
    _print_report(result)
    if cfg.json_out:
        with open(cfg.json_out, "w") as fh:
            json.dump(asdict(result), fh, indent=2)
        print(f"  wrote {cfg.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
