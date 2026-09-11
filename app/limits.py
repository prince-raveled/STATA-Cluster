"""Admission control: a bounded run queue and a per-client rate limit.

SPEC §20 rules out Celery and Redis, so the queue is in-process — a semaphore plus a
counter. That is enough for the shape of load this server sees, and it fixes the real
problem measured in `tests/reference/load_test.py`: with no bound, six simultaneous
runs each took 63 s against ~10 s alone, because every run contended for the same
cores. Bounding concurrency makes latency predictable instead of degrading everyone.

Nothing here is a security control. There are no accounts (§21), so a determined
caller can change address; the point is to keep one careless script from occupying the
whole machine.
"""
from __future__ import annotations

import contextlib
import os
import threading
import time
from collections import deque

#: Concurrent multiverse runs. Beyond the core count they only contend.
MAX_CONCURRENT_RUNS = int(os.environ.get(
    "MICROVERSE_MAX_CONCURRENT", max(2, min(4, (os.cpu_count() or 2)))
))
#: Runs waiting for a slot. Past this the server says so instead of queueing forever.
MAX_QUEUED_RUNS = int(os.environ.get("MICROVERSE_MAX_QUEUED", "24"))
#: Per-client budget, applied to the endpoints that start work or accept uploads.
RATE_LIMIT_REQUESTS = int(os.environ.get("MICROVERSE_RATE_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("MICROVERSE_RATE_WINDOW", "300"))


class RunQueue:
    """Bounds how many runs execute at once, and how many may wait."""

    def __init__(self, max_concurrent: int = MAX_CONCURRENT_RUNS,
                 max_queued: int = MAX_QUEUED_RUNS):
        self.max_concurrent = max(1, max_concurrent)
        self.max_queued = max(0, max_queued)
        self._semaphore = threading.BoundedSemaphore(self.max_concurrent)
        self._lock = threading.Lock()
        self._running = 0
        self._waiting = 0

    @property
    def running(self) -> int:
        with self._lock:
            return self._running

    @property
    def waiting(self) -> int:
        with self._lock:
            return self._waiting

    def has_room(self) -> bool:
        with self._lock:
            return self._running + self._waiting < self.max_concurrent + self.max_queued

    def acquire(self, timeout: float = None) -> bool:
        """Block until a slot frees. Returns False if it never does."""
        with self._lock:
            self._waiting += 1
        try:
            acquired = self._semaphore.acquire(timeout=timeout)
        finally:
            with self._lock:
                self._waiting -= 1
        if acquired:
            with self._lock:
                self._running += 1
        return acquired

    def release(self) -> None:
        with self._lock:
            self._running = max(0, self._running - 1)
        # Released more than acquired can only happen if a caller double-releases;
        # the count is already corrected above, so there is nothing further to do.
        with contextlib.suppress(ValueError):
            self._semaphore.release()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self._running,
                "waiting": self._waiting,
                "max_concurrent": self.max_concurrent,
                "max_queued": self.max_queued,
            }


class RateLimiter:
    """A fixed-window counter per client key."""

    def __init__(self, limit: int = RATE_LIMIT_REQUESTS,
                 window: int = RATE_LIMIT_WINDOW_SECONDS):
        self.limit = max(1, limit)
        self.window = max(1, window)
        self._hits: dict = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple:
        """Return (allowed, seconds until the oldest hit expires)."""
        now = time.monotonic()
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and now - bucket[0] > self.window:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False, int(self.window - (now - bucket[0])) + 1
            bucket.append(now)

            # Opportunistic cleanup so idle clients do not accumulate.
            if len(self._hits) > 4096:
                for client, hits in list(self._hits.items()):
                    if not hits or now - hits[-1] > self.window:
                        self._hits.pop(client, None)
        return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def client_key(request) -> str:
    """Identify the caller. Honours one proxy hop, which is what a PaaS gives us."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


run_queue = RunQueue()
rate_limiter = RateLimiter()


def enforce_rate_limit(request) -> None:
    """Raise a `TooManyRequests` when a caller exceeds its budget.

    Applied to uploading and to starting a run — the two things that cost real work.
    Reading results is not limited: a shared results link should keep working.
    """
    from .core.validation import DatasetError

    allowed, retry_after = rate_limiter.check(client_key(request))
    if allowed:
        return
    error = DatasetError(
        "Too many requests from this address.",
        f"MicroVerse allows {RATE_LIMIT_REQUESTS} uploads or runs every "
        f"{RATE_LIMIT_WINDOW_SECONDS // 60} minutes. Try again in about "
        f"{max(1, retry_after // 60)} minute(s). Results you have already started are "
        f"unaffected — their links still work.",
    )
    error.status_code = 429
    error.retry_after = retry_after
    raise error


def queue_state() -> dict:
    """Queue depth, for /healthz and the job page."""
    return run_queue.snapshot()
