from __future__ import annotations

import os
import tempfile

# Environment must be configured before inferfleet.config is imported.
_TMP = tempfile.mkdtemp(prefix="inferfleet-test-")
os.environ.setdefault("IF_MODE", "memory")
os.environ.setdefault("IF_INLINE", "true")
os.environ.setdefault("IF_AUTO_SEED", "true")
os.environ.setdefault("IF_JOB_TIMEOUT_S", "6")
os.environ.setdefault("IF_BATCH_WAIT_MS", "5")
os.environ.setdefault("IF_BATCH_SIZE", "8")
os.environ.setdefault("IF_QUEUE_MAX", "1000")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP}/test.db")

import httpx  # noqa: E402
import pytest  # noqa: E402

from inferfleet.db import DEMO_KEYS  # noqa: E402
from inferfleet.gateway.app import app  # noqa: E402

PRO_KEY = DEMO_KEYS["pro"]
FREE_KEY = DEMO_KEYS["free"]
ENT_KEY = DEMO_KEYS["enterprise"]


def auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
async def client():
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=20.0
        ) as c:
            yield c


@pytest.fixture
async def pro(client):
    client.headers.update(auth(PRO_KEY))
    return client
