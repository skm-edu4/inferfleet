from __future__ import annotations

JOBS = "q:jobs"
RESULTS = "q:results"


def generation_channel(request_id: str) -> str:
    return f"gen:{request_id}"
