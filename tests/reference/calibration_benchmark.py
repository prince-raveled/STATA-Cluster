"""How long calibration takes on the IBD demo — v3 plan §27.4.

Budget: a calibrated Quick run on the IBD demo (120 samples × 380 genera, 1,560
specifications, B = 2,000) in at most 60 s on 2 vCPU, on top of the Quick run itself.

The measurement is written into docs/benchmark.json under "calibration", beside (never
over) the v2 sweep that tests/reference/benchmark.py records, with this machine's
description: a timing means nothing without the machine it was taken on.

    python tests/reference/calibration_benchmark.py
    python tests/reference/calibration_benchmark.py --threads 2    # the budget's 2 vCPU
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUTPUT = os.path.join(ROOT, "docs", "benchmark.json")
BUDGET_SECONDS = 60.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=0,
                        help="limit BLAS threads (0 = the library default)")
    args = parser.parse_args()
    if args.threads:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = str(args.threads)

    sys.path.insert(0, ROOT)
    from app.core import inference
    from app.core.robustness import compute_robustness
    from app.core.runner import run_multiverse
    from app.services import load_demo

    dataset = load_demo("ibd_genus")
    started = time.perf_counter()
    run = run_multiverse(dataset, mode="quick")
    summary = compute_robustness(run)
    quick = time.perf_counter() - started
    weights = summary.spec_summary.sort_values("spec_id")["weight"].to_numpy()
    clock = time.perf_counter()
    result = inference.calibrate(run, dataset, weights, n_permutations=args.permutations)
    calibration = time.perf_counter() - clock

    entry = {
        "script": "tests/reference/calibration_benchmark.py",
        "dataset": "ibd_genus demo, Quick mode",
        "n_samples": int(dataset.n_samples), "n_taxa": int(dataset.n_taxa),
        "n_specs": int(run.n_specs), "n_permutations": int(args.permutations),
        "seconds_quick_run": round(quick, 2),
        "seconds_calibration": round(calibration, 2),
        "phases": result.timings,
        "budget_seconds": BUDGET_SECONDS,
        "within_budget": bool(calibration <= BUDGET_SECONDS),
        "threads": args.threads or "library default",
        "machine": {"platform": platform.platform(), "python": platform.python_version(),
                    "cpus": os.cpu_count()},
        "n_selected": int(result.n_selected), "n_certified": int(result.n_certified),
    }
    record = {}
    if os.path.exists(OUTPUT):
        with open(OUTPUT, encoding="utf-8") as handle:
            record = json.load(handle)
    record["calibration"] = entry
    with open(OUTPUT, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(f"Quick run {quick:.1f}s; calibration with B = {args.permutations}: "
          f"{calibration:.1f}s ({'within' if entry['within_budget'] else 'OVER'} the "
          f"{BUDGET_SECONDS:.0f}s budget) on {os.cpu_count()} CPUs, threads "
          f"{entry['threads']}")
    print(f"phases: {result.timings}")
    print(f"wrote {os.path.relpath(OUTPUT, ROOT)} (calibration)")
    return 0 if entry["within_budget"] else 1


if __name__ == "__main__":
    sys.exit(main())
