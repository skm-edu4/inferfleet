from __future__ import annotations

import time
from enum import IntEnum

from .metrics import CIRCUIT_STATE


class State(IntEnum):
    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


class CircuitBreaker:
    """Protects clients from a sick GPU fleet.

    OPEN  -> fail fast (no queueing behind a broken backend), degrade gracefully.
    HALF_OPEN -> after cooldown, let a probe through to test recovery.
    CLOSED -> normal operation.
    """

    def __init__(self, failure_threshold: int = 5, window_s: float = 30.0, cooldown_s: float = 15.0):
        self.failure_threshold = failure_threshold
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self.state = State.CLOSED
        self._failures: list[float] = []
        self._opened_at = 0.0
        self._probe_in_flight = False
        CIRCUIT_STATE.set(int(self.state))

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        self._failures = [t for t in self._failures if t >= cutoff]

    def allow(self) -> bool:
        now = time.monotonic()
        if self.state == State.CLOSED:
            return True
        if self.state == State.OPEN:
            if now - self._opened_at >= self.cooldown_s:
                self._transition(State.HALF_OPEN)
                self._probe_in_flight = True
                return True
            return False
        # HALF_OPEN: single probe at a time
        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        return True

    def record_success(self) -> None:
        self._failures.clear()
        self._probe_in_flight = False
        self._transition(State.CLOSED)

    def record_failure(self) -> None:
        now = time.monotonic()
        self._probe_in_flight = False
        self._prune(now)
        self._failures.append(now)
        if self.state == State.HALF_OPEN or len(self._failures) >= self.failure_threshold:
            self._opened_at = now
            self._transition(State.OPEN)

    def _transition(self, new_state: State) -> None:
        if new_state != self.state:
            self.state = new_state
            CIRCUIT_STATE.set(int(new_state))

    @property
    def degraded(self) -> bool:
        return self.state == State.OPEN
