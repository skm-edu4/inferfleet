from __future__ import annotations

import asyncio
import signal

from ..bus import make_bus
from .app import history_loop


async def main() -> None:
    bus = make_bus()
    await bus.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):  # pragma: no cover
            pass
    try:
        await history_loop(bus, stop)
    finally:
        await bus.close()


if __name__ == "__main__":
    asyncio.run(main())
