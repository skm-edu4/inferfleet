from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from .db import Generation, UsageLedger, current_period

MAX_ATTEMPTS = 3
LEDGER_RETRIES = 5


class QuotaExceeded(Exception):
    def __init__(self, used: int, quota: int):
        self.used = used
        self.quota = quota
        super().__init__(f"quota exceeded: {used}/{quota}")


def _ledger(session: Session, user_id: str) -> UsageLedger:
    """Row-locked get-or-create for the current period's ledger row.

    Concurrent *first* writes of a period race on the INSERT: SQLite has no
    FOR UPDATE (the loser sees UNIQUE violation or SQLITE_BUSY_SNAPSHOT from a
    stale WAL read), Postgres blocks on the unique index until the winner
    commits. Rolling back to a fresh snapshot and retrying serializes both —
    without it, strict-consistency billing silently loses updates under load.
    """
    period = current_period()
    last_exc: Exception | None = None
    for _ in range(LEDGER_RETRIES):
        row = session.execute(
            select(UsageLedger)
            .where(UsageLedger.user_id == user_id, UsageLedger.period == period)
            .with_for_update()
        ).scalar_one_or_none()
        if row is not None:
            return row
        try:
            row = UsageLedger(
                user_id=user_id, period=period, tokens_used=0, tokens_reserved=0, requests=0
            )
            session.add(row)
            session.flush()
            return row
        except (IntegrityError, OperationalError) as exc:
            last_exc = exc
            session.rollback()  # winner committed (or snapshot went stale): re-read
    raise RuntimeError(f"could not acquire usage ledger row for {user_id}") from last_exc


def reserve(session: Session, user_id: str, quota: int, tokens: int) -> None:
    """Strict-consistency admission: row-locked check-and-reserve.

    Postgres takes a FOR UPDATE lock (serializable across gateway replicas);
    SQLite serializes writers at the database level and the retry loop in
    `_ledger` resolves concurrent first-writes.
    """
    row = _ledger(session, user_id)
    if row.tokens_used + row.tokens_reserved + tokens > quota:
        raise QuotaExceeded(row.tokens_used + row.tokens_reserved, quota)
    row.tokens_reserved += tokens
    row.requests += 1


def settle(
    session: Session,
    user_id: str,
    reserved: int,
    actual: int,
    generation: Optional[dict] = None,
) -> None:
    """Convert a reservation into real usage + append history.

    Idempotent: if this request id was already recorded, the ledger is left
    untouched, so redelivered queue messages can never double-bill a customer.
    """
    if generation is not None and session.get(Generation, generation["id"]) is not None:
        return
    row = _ledger(session, user_id)
    row.tokens_reserved = max(0, row.tokens_reserved - reserved)
    row.tokens_used = max(0, row.tokens_used + actual)
    if generation is not None:
        session.add(Generation(**generation))


def refund(session: Session, user_id: str, reserved: int) -> None:
    row = _ledger(session, user_id)
    row.tokens_reserved = max(0, row.tokens_reserved - reserved)


def usage_snapshot(session: Session, user_id: str, quota: int) -> dict:
    row = _ledger(session, user_id)
    used = row.tokens_used + row.tokens_reserved
    return {
        "period": row.period,
        "tokens_used": row.tokens_used,
        "tokens_reserved": row.tokens_reserved,
        "requests": row.requests,
        "quota": quota,
        "remaining": max(0, quota - used),
    }


# ---------------------------------------------------------------------------
# Result events flowing worker -> history/billing settlement
# ---------------------------------------------------------------------------

def result_event(
    *,
    request_id: str,
    user_id: str,
    key_id: str,
    prompt: str,
    completion: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: float,
    cached: bool,
    status: str,
    reserved: int,
) -> dict:
    return {
        "id": request_id,
        "user_id": user_id,
        "key_id": key_id,
        "prompt": prompt,
        "completion": completion,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": round(latency_ms, 2),
        "cached": cached,
        "status": status,
        "reserved": reserved,
        "attempt": 0,
    }
