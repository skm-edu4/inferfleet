from __future__ import annotations

"""InferFleet all-in-one entrypoint.

Runs gateway + GPU worker + history settlement in one process (memory bus).
This is the zero-dependency dev/demo mode; production runs the same code as
separate microservices via docker-compose (see README).
"""


def main() -> None:
    import os

    import uvicorn

    os.environ.setdefault("IF_INLINE", "true")
    from .gateway.app import app

    uvicorn.run(
        app,
        host=os.getenv("IF_HOST", "127.0.0.1"),
        port=int(os.getenv("IF_PORT", "8000")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
