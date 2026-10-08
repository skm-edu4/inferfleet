from __future__ import annotations

import contextlib
import hashlib
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Iterator, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from .config import TIERS, settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("users.id"), index=True
    )
    label: Mapped[str] = mapped_column(String(64), default="default")
    tier: Mapped[str] = mapped_column(String(20), default="free")
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    @property
    def rate_limit_rpm(self) -> int:
        from .config import settings

        return settings.rate_limits.get(self.tier, settings.rate_limits["free"])

    @property
    def monthly_quota_tokens(self) -> int:
        from .config import settings

        return settings.monthly_quotas.get(self.tier, settings.monthly_quotas["free"])


class Generation(Base):
    """Append-only history log. Written asynchronously (eventual consistency)."""

    __tablename__ = "generations"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(40), index=True)
    key_id: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    prompt: Mapped[str] = mapped_column(Text)
    completion: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(64))
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(20), default="ok")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


class UsageLedger(Base):
    """Strict-consistency billing ledger: one row per (user, billing period)."""

    __tablename__ = "usage_ledger"

    user_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("users.id"), primary_key=True
    )
    period: Mapped[str] = mapped_column(String(7), primary_key=True)  # YYYY-MM
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    tokens_reserved: Mapped[int] = mapped_column(Integer, default=0)
    requests: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


_engine = None
_SessionLocal: Optional[sessionmaker] = None


def _build_engine(url: str):
    if url.startswith("sqlite"):
        os.makedirs(os.path.dirname(url.split("///")[-1]) or ".", exist_ok=True)
    engine = create_engine(
        url,
        pool_pre_ping=True,
        future=True,
        connect_args=(
            {"check_same_thread": False, "timeout": 30}
            if url.startswith("sqlite")
            else {}
        ),
    )
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()
    return engine


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = _build_engine(settings.database_url)
        _SessionLocal = sessionmaker(
            bind=_engine, autoflush=False, expire_on_commit=False, future=True
        )
    return _engine


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)


@contextlib.contextmanager
def session_scope() -> Iterator[Session]:
    get_engine()
    assert _SessionLocal is not None
    session: Session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def current_period(now: Optional[datetime] = None) -> str:
    now = now or utcnow()
    return now.strftime("%Y-%m")


# ---------------------------------------------------------------------------
# Seed data: deterministic demo keys so curl examples in the README just work.
# ---------------------------------------------------------------------------

DEMO_KEYS = {
    "free": "ifk_live_free000000000000000000000000001",
    "pro": "ifk_live_pro00000000000000000000000000001",
    "enterprise": "ifk_live_ent000000000000000000000000001",
}


def seed_demo_data() -> dict[str, str]:
    """Idempotently create one user + API key per tier. Returns raw keys."""
    with session_scope() as session:
        for tier in TIERS:
            email = f"{tier}@inferfleet.dev"
            user = session.query(User).filter_by(email=email).one_or_none()
            if user is None:
                user = User(id=new_id("usr"), email=email)
                session.add(user)
                session.flush()
            raw = DEMO_KEYS[tier]
            exists = session.query(ApiKey).filter_by(key_hash=hash_key(raw)).one_or_none()
            if exists is None:
                session.add(
                    ApiKey(
                        id=new_id("key"),
                        key_hash=hash_key(raw),
                        user_id=user.id,
                        label=f"{tier}-demo",
                        tier=tier,
                    )
                )
    return dict(DEMO_KEYS)


def create_key(email: str, tier: str, label: str, raw_key: Optional[str] = None):
    with session_scope() as session:
        user = session.query(User).filter_by(email=email).one_or_none()
        if user is None:
            user = User(id=new_id("usr"), email=email)
            session.add(user)
            session.flush()
        raw = raw_key or f"ifk_live_{uuid.uuid4().hex}"
        key = ApiKey(
            id=new_id("key"),
            key_hash=hash_key(raw),
            user_id=user.id,
            label=label,
            tier=tier if tier in TIERS else "free",
        )
        session.add(key)
        session.flush()
        return key.id, raw


def uptime_seconds(start: float) -> float:
    return round(time.time() - start, 3)
