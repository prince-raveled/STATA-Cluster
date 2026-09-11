"""Concurrency and scale check for the running server.

SPEC §20 puts jobs in SQLite behind FastAPI BackgroundTasks with no Celery and no
Redis, and §6 O6 budgets Quick mode at under 90 s. Neither had ever been exercised with
more than one caller, or with a table near the §8 ceiling of 1,500 taxa.

    .venv/Scripts/python -m uvicorn app.main:app --port 8077     # in another shell
    .venv/Scripts/python tests/reference/load_test.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

BASE = os.environ.get("MICROVERSE_URL", "http://127.0.0.1:8077")
FAILURES: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def report(name: str, detail: str) -> None:
    print(f"  [ -- ] {name} — {detail}")


def _get(path: str, timeout: float = 120.0):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as response:
        return response.status, response.read(), response.url


def _post_form(path: str, fields, timeout: float = 600.0):
    body = urllib.parse.urlencode(fields).encode()
    request = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.url


def _multipart(files: dict, fields: dict) -> tuple:
    boundary = "----microverseload"
    buffer = io.BytesIO()

    def write(text):
        buffer.write(text.encode() if isinstance(text, str) else text)

    for name, value in fields.items():
        write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
    for name, (filename, payload) in files.items():
        write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
              f"filename=\"{filename}\"\r\nContent-Type: text/tab-separated-values\r\n\r\n")
        write(payload)
        write("\r\n")
    write(f"--{boundary}--\r\n")
    return buffer.getvalue(), f"multipart/form-data; boundary={boundary}"


def run_job(demo: str, mode: str = "quick", timeout: float = 300.0) -> dict:
    """Start a demo run and wait for it, returning the job record."""
    started = time.perf_counter()
    _, _, url = _get(f"/demo/{demo}")
    token = url.rstrip("/").rsplit("/", 1)[-1]
    _post_form(f"/run/{token}", [("mode", mode)])

    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        _, payload, _ = _get(f"/api/jobs/{token}")
        job = json.loads(payload)
        if job["status"] in ("done", "error"):
            job["wall_seconds"] = time.perf_counter() - started
            return job
        time.sleep(0.4)
    return {"status": "timeout", "token": token, "wall_seconds": timeout}


# ---------------------------------------------------------------------------
def test_concurrent_runs(n: int = 6) -> None:
    """SQLite has one writer. Several runs at once is the case that would expose it."""
    print(f"\nconcurrency — {n} runs started simultaneously")
    results: list = []
    barrier = threading.Barrier(n)

    def worker(index: int) -> None:
        demo = ("ibd_genus", "gut_species", "t2d_covariates")[index % 3]
        barrier.wait()                       # release them together
        try:
            results.append(run_job(demo))
        except Exception as exc:             # noqa: BLE001 - reported, not raised
            results.append({"status": f"exception: {type(exc).__name__}: {exc}"})

    started = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started

    done = [r for r in results if r.get("status") == "done"]
    check("every concurrent run finished", len(done) == n,
          f"{len(done)}/{n} done — statuses: {sorted({r['status'] for r in results})}")
    if done:
        slowest = max(r["wall_seconds"] for r in done)
        report("wall clock", f"{elapsed:.1f}s total, slowest run {slowest:.1f}s")
        check("no run exceeded the 90 s Quick-mode budget", slowest < 90.0,
              f"slowest = {slowest:.1f}s")
        check("results are distinct jobs", len({r["token"] for r in done}) == len(done))


def test_repeated_polling(n: int = 120) -> None:
    """The job page polls once a second; a stale or missing token must not 500."""
    print(f"\npolling — {n} status requests in a tight loop")
    _, _, url = _get("/demo/ibd_genus")
    token = url.rstrip("/").rsplit("/", 1)[-1]
    codes = set()
    started = time.perf_counter()
    for _ in range(n):
        status, _, _ = _get(f"/job/{token}/progress")
        codes.add(status)
    elapsed = time.perf_counter() - started
    check("every poll returned 200", codes == {200}, f"codes seen: {sorted(codes)}")
    report("throughput", f"{n / elapsed:.0f} polls/second")


def test_large_table(n_taxa: int = 1500, n_samples: int = 60) -> None:
    """The §8 ceiling, end to end: 1,500 taxa is the largest table MicroVerse accepts."""
    print(f"\nscale — a table at the §8 ceiling ({n_taxa} taxa x {n_samples} samples)")
    rng = np.random.default_rng(4)
    counts = rng.negative_binomial(4, 0.03, size=(n_taxa, n_samples))
    counts = counts * (rng.random(counts.shape) > 0.55)
    counts[0] += 400                                     # keep every sample non-empty

    header = "taxon\t" + "\t".join(f"S{j:03d}" for j in range(n_samples))
    lines = [header]
    for i in range(n_taxa):
        lines.append(f"T{i:05d}\t" + "\t".join(str(int(v)) for v in counts[i]))
    abundance = ("\n".join(lines) + "\n").encode()
    metadata = ("sample_id\tgroup\n" + "".join(
        f"S{j:03d}\t{'control' if j < n_samples // 2 else 'disease'}\n"
        for j in range(n_samples)
    )).encode()
    report("upload size", f"{len(abundance) / 1e6:.1f} MB")

    body, content_type = _multipart(
        {"abundance": ("big.tsv", abundance), "metadata": ("meta.tsv", metadata)},
        {"group_column": "group"},
    )
    request = urllib.request.Request(BASE + "/upload", data=body,
                                     headers={"Content-Type": content_type})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        token = response.url.rstrip("/").rsplit("/", 1)[-1]
    report("parse + validate", f"{time.perf_counter() - started:.1f}s")

    started = time.perf_counter()
    _post_form(f"/run/{token}", [("mode", "quick")], timeout=900)
    deadline = time.perf_counter() + 600
    job = {}
    while time.perf_counter() < deadline:
        _, payload, _ = _get(f"/api/jobs/{token}", timeout=60)
        job = json.loads(payload)
        if job["status"] in ("done", "error"):
            break
        time.sleep(1.0)
    elapsed = time.perf_counter() - started

    check("the run completed", job.get("status") == "done",
          job.get("error") or job.get("status", "unknown"))
    if job.get("status") == "done":
        report("engine runtime", f"{job['runtime_seconds']:.1f}s over "
                                 f"{job['n_specs']:,} specifications")
        check("inside the 90 s Quick-mode budget (O6)", elapsed < 90.0,
              f"{elapsed:.1f}s wall clock including polling")

        started = time.perf_counter()
        status, page, _ = _get(f"/results/{token}", timeout=300)
        check("the results page renders", status == 200 and len(page) > 10_000,
              f"{len(page) / 1e6:.1f} MB in {time.perf_counter() - started:.1f}s")

        started = time.perf_counter()
        status, blob, _ = _get(f"/download/{token}/bundle", timeout=600)
        check("the bundle downloads", status == 200,
              f"{len(blob) / 1e6:.1f} MB in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    try:
        status, payload, _ = _get("/healthz", timeout=10)
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"No server at {BASE} ({exc}). Start uvicorn first.")
        sys.exit(2)
    print(f"server at {BASE}: {json.loads(payload)['version']}")

    test_concurrent_runs()
    test_repeated_polling()
    test_large_table()

    print("\n" + ("FAILED: " + ", ".join(FAILURES) if FAILURES else "All checks passed."))
    sys.exit(1 if FAILURES else 0)
