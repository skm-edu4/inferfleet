from __future__ import annotations

import hashlib
import math
import re
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from .bus import Bus

_TOKEN_RE = re.compile(r"[a-z0-9]+")
DIM = 512
MAX_SEMANTIC_ENTRIES = 5_000


def normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def embed(text: str) -> dict[int, float]:
    """Deterministic hashing-trick embedding (unigrams + bigrams), L2-normalized.

    Stands in for a learned embedding model so the project runs with zero
    model downloads; the interface matches what a real embedder would return.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return {}
    vec: dict[int, float] = {}
    for tok in tokens:
        idx = zlib.crc32(tok.encode()) % DIM
        vec[idx] = vec.get(idx, 0.0) + 1.0
    for a, b in zip(tokens, tokens[1:]):
        idx = DIM + zlib.crc32(f"{a}_{b}".encode()) % DIM
        vec[idx] = vec.get(idx, 0.0) + 1.5
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if len(b) < len(a):
        a, b = b, a
    return sum(v * b[k] for k, v in a.items() if k in b)


@dataclass
class CacheEntry:
    request_id: str
    model: str
    prompt: str
    completion: str
    tokens_in: int
    tokens_out: int
    vec: dict[int, float]
    expires_at: float

    @property
    def expired(self) -> bool:
        return self.expires_at <= time.monotonic()


def cache_key(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    raw = f"{model}|{temperature:.2f}|{max_tokens}|{normalize(prompt)}"
    return "cache:gen:" + hashlib.sha256(raw.encode()).hexdigest()


class ResponseCache:
    """Two-tier cache.

    Tier 1 - exact match, shared across gateway replicas through the bus KV
             (Redis in production, in-process dict in dev).
    Tier 2 - semantic near-duplicate match via cosine similarity over a local
             vector index (warm per replica; see DESIGN.md for the trade-off).
    """

    def __init__(self, bus: Bus, ttl_s: int = 900, threshold: float = 0.90):
        self._bus = bus
        self._ttl = ttl_s
        self._threshold = threshold
        self._index: "OrderedDict[str, CacheEntry]" = OrderedDict()

    async def get(
        self, model: str, prompt: str, temperature: float, max_tokens: int
    ) -> Optional[dict]:
        if temperature > 0.5:
            return None  # never serve cached samples for creative decoding
        key = cache_key(model, prompt, temperature, max_tokens)
        raw = await self._bus.kv_get(key)
        if raw:
            try:
                import json

                entry = json.loads(raw)
                entry["_cache_kind"] = "exact"
                return entry
            except ValueError:  # pragma: no cover
                pass
        return self._semantic_lookup(prompt, model)

    def _semantic_lookup(self, prompt: str, model: str) -> Optional[dict]:
        norm_prompt = normalize(prompt)
        query = embed(norm_prompt)
        best: Optional[CacheEntry] = None
        best_score = 0.0
        stale: list[str] = []
        for key, entry in list(self._index.items()):
            if entry.expired or entry.model != model:
                stale.append(key)
                continue
            score = cosine(query, entry.vec)
            # containment rule: "what is X" inside "what is X please" ~ same intent
            if norm_prompt in normalize(entry.prompt) or normalize(entry.prompt) in norm_prompt:
                score = max(score, 0.999)
            if score >= self._threshold and score > best_score:
                best, best_score = entry, score
        for key in stale:
            self._index.pop(key, None)
        if best is None:
            return None
        self._index.move_to_end(best.request_id)
        return {
            "request_id": best.request_id,
            "model": best.model,
            "prompt": best.prompt,
            "completion": best.completion,
            "tokens_in": best.tokens_in,
            "tokens_out": best.tokens_out,
            "_cache_kind": "semantic",
        }

    async def put(
        self,
        *,
        request_id: str,
        model: str,
        prompt: str,
        completion: str,
        tokens_in: int,
        tokens_out: int,
        temperature: float,
        max_tokens: int,
    ) -> None:
        if temperature > 0.5:
            return
        import json

        key = cache_key(model, prompt, temperature, max_tokens)
        payload = json.dumps(
            {
                "request_id": request_id,
                "model": model,
                "prompt": prompt,
                "completion": completion,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            }
        )
        await self._bus.kv_set(key, payload, ttl_s=self._ttl)
        entry = CacheEntry(
            request_id=request_id,
            model=model,
            prompt=prompt,
            completion=completion,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            vec=embed(normalize(prompt)),
            expires_at=time.monotonic() + self._ttl,
        )
        self._index[request_id] = entry
        self._index.move_to_end(request_id)
        while len(self._index) > MAX_SEMANTIC_ENTRIES:
            self._index.popitem(last=False)

    def size(self) -> int:
        return len(self._index)
