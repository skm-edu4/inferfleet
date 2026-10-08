#!/usr/bin/env python
"""Seed (or re-seed) the demo tenants and print their API keys."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inferfleet.db import DEMO_KEYS, init_db, seed_demo_data  # noqa: E402


def main() -> None:
    init_db()
    keys = seed_demo_data()
    print("Demo API keys (idempotent — safe to re-run):\n")
    for tier, key in keys.items():
        print(f"  {tier:<11} {key}")
    print("\nExport one to try the API:\n")
    print(f"  export INFERFLEET_KEY={keys['pro']}")
    print('  curl -s localhost:8000/v1/me -H "Authorization: Bearer $INFERFLEET_KEY"')


if __name__ == "__main__":
    main()
