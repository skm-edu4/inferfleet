from __future__ import annotations

import time
from dataclasses import dataclass

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from .bus import Bus
from .config import settings
from .db import ApiKey, User, hash_key

WINDOW_S = 60


@dataclass(frozen=True)
class Principal:
    key_id: str
    user_id: str
    tier: str
    label: str
    rate_limit_rpm: int
    monthly_quota_tokens: int


class RateLimited(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"rate limited, retry in {retry_after}s")


def authenticate(session: Session, presented: str) -> Principal:
    row = (
        session.query(ApiKey)
        .filter(ApiKey.key_hash == hash_key(presented))
        .one_or_none()
    )
    if row is None or row.revoked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "invalid_api_key",
                "message": "No API key matched. Pass `Authorization: Bearer <key>`.",
            },
        )
    user = session.get(User, row.user_id)
    if user is None:  # pragma: no cover - referential integrity
        raise HTTPException(status_code=401, detail={"code": "invalid_api_key"})
    return Principal(
        key_id=row.id,
        user_id=row.user_id,
        tier=row.tier,
        label=row.label,
        rate_limit_rpm=settings.rate_limits.get(row.tier, settings.rate_limits["free"]),
        monthly_quota_tokens=settings.monthly_quotas.get(
            row.tier, settings.monthly_quotas["free"]
        ),
    )


async def check_rate_limit(bus: Bus, principal: Principal) -> None:
    """Fixed 60s window counter shared across gateway replicas (Redis INCR)."""
    bucket = int(time.time()) // WINDOW_S
    key = f"rl:{principal.key_id}:{bucket}"
    count = await bus.incr_window(key, ttl_s=WINDOW_S * 2)
    if count > principal.rate_limit_rpm:
        retry_after = WINDOW_S - (int(time.time()) % WINDOW_S)
        raise RateLimited(retry_after)


def rate_limit_http(retry_after: int) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail={
            "code": "rate_limited",
            "message": "Too many requests for your billing tier.",
            "retry_after_s": retry_after,
        },
        headers={"Retry-After": str(retry_after)},
    )


def est_tokens(text: str) -> int:
    """Cheap planner estimate (~4 chars/token). The real tokenizer runs on the worker."""
    return max(1, len(text) // 4 + 4)


def require_ok(response, context: str) -> None:
    if response.status_code >= 400:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail={"code": "upstream_unavailable", "message": context})
