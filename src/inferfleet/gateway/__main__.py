from __future__ import annotations


def main() -> None:
    import uvicorn

    from .app import app

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
