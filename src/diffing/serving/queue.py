"""Admission control for the one GPU behind the server.

The backend holds a lock, which is enough for correctness and nothing else: a plain lock is
not fair (a caller can be skipped repeatedly), it accepts an unbounded crowd, and it makes a
queue of forty look exactly like a queue of one from the outside. That matters here because
the callers are not a browser tab — they are a sharded battery, a dashboard, and whatever an
agent is doing, all pointed at the same GPU on purpose.

So requests take a ticket and are served in order. Three consequences worth the code:

* FIFO, so a long battery cannot starve the person clicking in the dashboard, and a client's
  wait is bounded by the work ahead of it rather than by luck;
* bounded, so a server that is already forty requests deep says so immediately instead of
  accepting a forty-first that will wait half an hour — a refusal a client can act on beats a
  timeout it cannot;
* observable: `snapshot()` is what lets /status say how deep the queue is and what it has
  served, which is the difference between "the GPU is slow" and "there are nine ahead of you".
"""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


class QueueFull(RuntimeError):
    """More requests are waiting than the server accepts; `.waiting` is how many."""

    def __init__(self, waiting: int, capacity: int):
        super().__init__(
            f"{waiting} requests already waiting for the GPU (capacity {capacity}); "
            f"retry shortly"
        )
        self.waiting = waiting
        self.capacity = capacity


@dataclass
class GpuQueue:
    """One GPU, one queue, served first-come-first-served.

    Args:
        capacity: How many requests may WAIT. The one being served is not counted, so
            capacity=0 is a valid setting meaning "serve one, refuse everything else".
    """

    capacity: int = 32
    _cv: threading.Condition = field(default_factory=threading.Condition, repr=False)
    _next_ticket: int = 0
    _now_serving: int = 0
    _waiting: int = 0
    _running: bool = False
    _served: int = 0
    _rejected: int = 0
    _wait_seconds: float = 0.0

    def __post_init__(self) -> None:
        assert self.capacity >= 0, f"capacity must be non-negative, got {self.capacity}"

    @contextmanager
    def slot(self):
        """Hold the GPU for the duration of the block, after waiting for the tickets ahead.

        Raises:
            QueueFull: if the queue is already at capacity — before waiting, never after.
        """
        with self._cv:
            # tickets handed out minus tickets finished = the request being served plus the
            # ones queued behind it. `capacity` counts WAITERS, so an idle queue admits its
            # first request at any capacity, including 0.
            ahead = self._next_ticket - self._now_serving
            if ahead > self.capacity:
                self._rejected += 1
                raise QueueFull(ahead - 1, self.capacity)
            ticket = self._next_ticket
            self._next_ticket += 1
            self._waiting += 1
            started = time.monotonic()
            while self._now_serving != ticket:
                self._cv.wait()
            self._waiting -= 1
            self._running = True
            self._wait_seconds += time.monotonic() - started
        try:
            yield
        finally:
            with self._cv:
                self._now_serving += 1
                self._running = False
                self._served += 1
                self._cv.notify_all()

    def snapshot(self) -> dict:
        """Queue state for /status. `waiting` excludes the request being served."""
        with self._cv:
            return {
                "waiting": self._waiting,
                "capacity": self.capacity,
                "running": self._running,
                "served": self._served,
                "rejected": self._rejected,
                "mean_wait_s": (round(self._wait_seconds / self._served, 3)
                                if self._served else 0.0),
            }
