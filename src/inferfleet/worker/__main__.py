from __future__ import annotations

import asyncio

from ..bus import make_bus
from .main import install_stop, worker_loop


async def main() -> None:
    bus = make_bus()
    await bus.start()
    stop = asyncio.Event()
    install_stop(stop)
    try:
        await worker_loop(bus, stop)
    finally:
        await bus.close()


if __name__ == "__main__":
    asyncio.run(main())
