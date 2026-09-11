"""Admission control — the bounded run queue and the per-client rate limit.

Both exist because of a measurement, not a guess: `tests/reference/load_test.py` showed
six unbounded simultaneous runs taking 63 s each against ~10 s alone.
"""
from __future__ import annotations

import threading
import time

import pytest

from app.limits import RateLimiter, RunQueue, client_key


class _Request:
    """Just enough of a Starlette request for `client_key`."""

    def __init__(self, host="1.2.3.4", forwarded=None):
        self.headers = {"x-forwarded-for": forwarded} if forwarded else {}
        self.client = type("C", (), {"host": host})()


# --- the run queue --------------------------------------------------------
def test_queue_admits_up_to_the_limit_and_no_further():
    queue = RunQueue(max_concurrent=2, max_queued=0)
    assert queue.acquire(timeout=1)
    assert queue.acquire(timeout=1)
    assert queue.snapshot()["running"] == 2
    # The third has nowhere to go and must not block forever.
    assert not queue.acquire(timeout=0.2)
    queue.release()
    assert queue.acquire(timeout=1)
    queue.release()
    queue.release()
    assert queue.snapshot()["running"] == 0


def test_queue_serialises_work_rather_than_dropping_it():
    """A waiting run must still run — the queue delays, it does not discard."""
    queue = RunQueue(max_concurrent=2, max_queued=10)
    order, lock = [], threading.Lock()
    start = threading.Barrier(4)

    def worker(index):
        start.wait()
        assert queue.acquire(timeout=10), index
        with lock:
            order.append(index)
        time.sleep(0.05)
        queue.release()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert sorted(order) == [0, 1, 2, 3], "every queued run must eventually execute"
    assert queue.snapshot()["running"] == 0


def test_queue_never_exceeds_its_concurrency_bound():
    queue = RunQueue(max_concurrent=3, max_queued=20)
    peak, current, lock = 0, 0, threading.Lock()

    def worker():
        nonlocal peak, current
        assert queue.acquire(timeout=10)
        with lock:
            current += 1
            peak = max(peak, current)
        time.sleep(0.02)
        with lock:
            current -= 1
        queue.release()

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert peak <= 3, f"ran {peak} concurrently with a bound of 3"


def test_has_room_reflects_the_backlog():
    queue = RunQueue(max_concurrent=1, max_queued=1)
    assert queue.has_room()
    assert queue.acquire(timeout=1)
    assert queue.has_room()          # one slot taken, one queue place free
    queue.release()


# --- the rate limiter -----------------------------------------------------
def test_rate_limiter_allows_up_to_the_budget():
    limiter = RateLimiter(limit=3, window=60)
    for _ in range(3):
        allowed, _ = limiter.check("client")
        assert allowed
    allowed, retry_after = limiter.check("client")
    assert not allowed
    assert 0 < retry_after <= 61


def test_rate_limiter_is_per_client():
    limiter = RateLimiter(limit=1, window=60)
    assert limiter.check("a")[0]
    assert not limiter.check("a")[0]
    assert limiter.check("b")[0], "one client must not consume another's budget"


def test_rate_limiter_window_expires():
    limiter = RateLimiter(limit=1, window=1)
    assert limiter.check("client")[0]
    assert not limiter.check("client")[0]
    time.sleep(1.1)
    assert limiter.check("client")[0], "the window must roll forward"


def test_rate_limiter_is_thread_safe():
    limiter = RateLimiter(limit=50, window=60)
    granted, lock = 0, threading.Lock()

    def worker():
        nonlocal granted
        for _ in range(20):
            if limiter.check("shared")[0]:
                with lock:
                    granted += 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert granted == 50, f"budget of 50 granted {granted} times under contention"


# --- client identity ------------------------------------------------------
def test_client_key_prefers_the_forwarded_address():
    assert client_key(_Request(host="10.0.0.1")) == "10.0.0.1"
    assert client_key(_Request(host="10.0.0.1", forwarded="203.0.113.9, 10.0.0.1")) \
        == "203.0.113.9"


# --- through the app ------------------------------------------------------
def test_rate_limited_upload_answers_429_with_retry_after():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from app import limits
    from app.main import app

    original = limits.rate_limiter
    limits.rate_limiter = RateLimiter(limit=2, window=300)
    try:
        with TestClient(app) as client:
            files = {
                "abundance": ("a.tsv", b"nope", "text/plain"),
                "metadata": ("m.tsv", b"sample_id\tgroup\nS1\ta\n", "text/plain"),
            }
            # The first two are refused on their content (422), not their rate.
            for _ in range(2):
                assert client.post("/upload", files=files).status_code == 422
            response = client.post("/upload", files=files)
            assert response.status_code == 429
            assert response.headers.get("Retry-After")
            assert "Too many requests" in response.text
    finally:
        limits.rate_limiter = original


def test_healthz_reports_queue_depth():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        payload = client.get("/healthz").json()
    assert set(payload["queue"]) == {"running", "waiting", "max_concurrent", "max_queued"}
    assert payload["queue"]["max_concurrent"] >= 1
