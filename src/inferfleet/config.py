from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


TIERS = ("free", "pro", "enterprise")

# Per-key request ceilings (fixed 60s window). Enterprise pays for headroom.
DEFAULT_RATE_LIMITS: dict[str, int] = {
    "free": 20,
    "pro": 120,
    "enterprise": 1200,
}

# Monthly token budgets enforced with strict (serializable) billing transactions.
DEFAULT_MONTHLY_QUOTAS: dict[str, int] = {
    "free": 100_000,
    "pro": 5_000_000,
    "enterprise": 200_000_000,
}


@dataclass(frozen=True)
class Settings:
    mode: str = "memory"
    service: str = "gateway"
    database_url: str = "sqlite:///./data/inferfleet.db"
    redis_url: str = "redis://localhost:6379/0"

    queue_max: int = 1000
    batch_size: int = 16
    batch_wait_ms: int = 20
    job_timeout_s: float = 30.0
    ttfb_budget_ms: int = 300

    cache_ttl_s: int = 900
    semantic_threshold: float = 0.90

    failure_rate: float = 0.0
    auto_seed: bool = True
    inline_workers: bool = True

    model_name: str = "inferfleet-8x7b-instruct"
    rate_limits: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_RATE_LIMITS)
    )
    monthly_quotas: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_MONTHLY_QUOTAS)
    )

    @property
    def is_memory(self) -> bool:
        return self.mode != "redis"

    @property
    def uses_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


def load_settings() -> Settings:
    mode = os.getenv("IF_MODE", "memory").strip().lower()
    rate_limits = {
        tier: max(1, _int(f"IF_RATE_{tier.upper()}", limit))
        for tier, limit in DEFAULT_RATE_LIMITS.items()
    }
    quotas = {
        tier: max(1, _int(f"IF_QUOTA_{tier.upper()}", limit))
        for tier, limit in DEFAULT_MONTHLY_QUOTAS.items()
    }
    inline_flag = os.getenv("IF_INLINE", "auto").strip().lower()
    if inline_flag == "auto":
        inline = mode != "redis"
    elif inline_flag in {"1", "true", "yes", "on"}:
        inline = True
    else:
        inline = False
    return Settings(
        mode=mode,
        service=os.getenv("IF_SERVICE", "gateway"),
        database_url=os.getenv("DATABASE_URL", "sqlite:///./data/inferfleet.db"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        queue_max=_int("IF_QUEUE_MAX", 1000),
        batch_size=max(1, _int("IF_BATCH_SIZE", 16)),
        batch_wait_ms=max(1, _int("IF_BATCH_WAIT_MS", 20)),
        job_timeout_s=max(0.5, _float("IF_JOB_TIMEOUT_S", 30.0)),
        ttfb_budget_ms=max(1, _int("IF_TTFB_BUDGET_MS", 300)),
        cache_ttl_s=_int("IF_CACHE_TTL_S", 900),
        semantic_threshold=min(1.0, max(0.5, _float("IF_SEMANTIC_THRESHOLD", 0.90))),
        failure_rate=min(1.0, max(0.0, _float("IF_FAILURE_RATE", 0.0))),
        auto_seed=_bool("IF_AUTO_SEED", "true"),
        inline_workers=inline,
        rate_limits=rate_limits,
        monthly_quotas=quotas,
    )


settings = load_settings()
