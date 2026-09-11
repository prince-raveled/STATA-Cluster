"""Compute benchmark — SPEC §23 validation 5.

"Runtime vs samples x taxa. Publish the curve — a reviewer will ask."

Measures each phase separately, because they scale differently and the headline number
hides that: the multiverse itself is close to linear in taxa, while the exports are
linear in (specifications x taxa) and dominate at the ceiling.

    .venv/Scripts/python tests/reference/benchmark.py            # full sweep
    .venv/Scripts/python tests/reference/benchmark.py --quick    # a short sweep
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.attribution import attribute  # noqa: E402
from app.core.parsers.base import AbundanceTable  # noqa: E402
from app.core.report import build_zip, long_results_csv_gz  # noqa: E402
from app.core.robustness import compute_robustness  # noqa: E402
from app.core.runner import run_multiverse  # noqa: E402
from app.core.validation import validate_dataset  # noqa: E402

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "docs", "benchmark.json")


def make_dataset(n_taxa: int, n_samples: int, seed: int = 2):
    rng = np.random.default_rng(seed)
    counts = rng.negative_binomial(4, 0.03, size=(n_taxa, n_samples))
    counts = counts * (rng.random(counts.shape) > 0.55)
    counts[0] += 400                                        # no empty sample
    taxa = [f"T{i:05d}" for i in range(n_taxa)]
    samples = [f"S{j:04d}" for j in range(n_samples)]
    table = AbundanceTable(
        counts=pd.DataFrame(counts, index=taxa, columns=samples), value_type="counts"
    )
    metadata = pd.DataFrame(
        {"group": ["control"] * (n_samples // 2) + ["disease"] * (n_samples - n_samples // 2)},
        index=samples,
    )
    metadata.index.name = "sample_id"
    return validate_dataset(table, metadata, "group")


def measure(n_taxa: int, n_samples: int, mode: str = "quick") -> dict:
    dataset = make_dataset(n_taxa, n_samples)
    timings: dict = {}

    def phase(name, fn):
        started = time.perf_counter()
        value = fn()
        timings[name] = round(time.perf_counter() - started, 3)
        return value

    run = phase("multiverse", lambda: run_multiverse(dataset, mode=mode))
    summary = phase("robustness", lambda: compute_robustness(run))
    attribution = phase("attribution", lambda: attribute(run))
    long_gz = phase("export_long_csv", lambda: long_results_csv_gz(run))
    bundle = phase("export_bundle", lambda: build_zip(run, summary, attribution,
                                                      long_csv_gz=long_gz))

    return {
        "n_taxa": n_taxa,
        "n_samples": n_samples,
        "mode": mode,
        "n_specifications": run.grid_report.n_valid,
        "n_result_rows": int(len(run.long)),
        "bundle_mb": round(len(bundle) / 1e6, 2),
        "timings": timings,
        "total_seconds": round(sum(timings.values()), 2),
    }


def describe_machine() -> dict:
    """Enough about this machine that someone can judge whether the timings transfer.

    A core count alone does not: twelve cores of a laptop under thermal limits and
    twelve cores of a server are different numbers. Everything here is read from the
    running system — nothing is filled in by hand, so a field that cannot be determined
    says "unknown" rather than carrying a guess.
    """
    import platform

    machine = {
        "python": sys.version.split()[0],
        "cpus": os.cpu_count(),
        "cpu_model": "unknown",
        "memory_gb": "unknown",
        "platform": f"{platform.system()} {platform.release()}",
        "architecture": platform.machine() or "unknown",
    }

    processor = (platform.processor() or "").strip()
    if processor and processor.lower() not in ("", "unknown"):
        machine["cpu_model"] = processor

    if platform.system() == "Windows":
        try:
            import winreg

            key = r"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
                machine["cpu_model"] = winreg.QueryValueEx(
                    handle, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        machine["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass

    try:                                    # psutil is not a dependency; do without it
        import psutil

        machine["memory_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except ImportError:
        try:
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                machine["memory_gb"] = round(status.ullTotalPhys / 1024 ** 3, 1)
        except Exception:                                    # noqa: BLE001
            pass

    for package in ("numpy", "scipy", "pandas"):
        try:
            module = __import__(package)
            machine[f"{package}_version"] = getattr(module, "__version__", "unknown")
        except ImportError:
            continue
    return machine


def main() -> None:
    quick = "--quick" in sys.argv
    grid = (
        [(120, 40), (380, 120), (1500, 60)]
        if quick else
        [(60, 20), (120, 40), (250, 60), (380, 120), (600, 100),
         (1000, 80), (1500, 60), (1500, 200)]
    )

    print(f"{'taxa':>6} {'samples':>8} {'specs':>7} {'rows':>10} "
          f"{'multiv':>8} {'robust':>7} {'attrib':>7} {'export':>7} {'TOTAL':>8} {'bundle':>8}")
    print("-" * 92)

    rows = []
    for n_taxa, n_samples in grid:
        row = measure(n_taxa, n_samples)
        rows.append(row)
        t = row["timings"]
        export = t["export_long_csv"] + t["export_bundle"]
        print(f"{row['n_taxa']:>6} {row['n_samples']:>8} {row['n_specifications']:>7,} "
              f"{row['n_result_rows']:>10,} {t['multiverse']:>7.2f}s {t['robustness']:>6.2f}s "
              f"{t['attribution']:>6.2f}s {export:>6.2f}s {row['total_seconds']:>7.2f}s "
              f"{row['bundle_mb']:>7.1f}M")

    # The claim SPEC §6 O6 makes, checked at the §8 ceiling.
    ceiling = [r for r in rows if r["n_taxa"] == 1500]
    print()
    for row in ceiling:
        verdict = "within" if row["total_seconds"] < 90 else "OVER"
        print(f"O6 (Quick mode < 90 s) at the §8 ceiling of 1,500 taxa x "
              f"{row['n_samples']} samples: {row['total_seconds']:.1f}s — {verdict} budget")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump({"machine": describe_machine(), "runs": rows}, handle, indent=2)
    print(f"\nwritten to {os.path.relpath(OUTPUT)}")


if __name__ == "__main__":
    main()
