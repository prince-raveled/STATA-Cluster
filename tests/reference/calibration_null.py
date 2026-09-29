"""Are the calibrated family p-values uniform under the null? — v3 plan §27.6.

Acceptance criterion: "Under simulated nulls, family p-values are uniform (KS p > 0.01
across 200 replicates)". This is an implementation check on simulated data, not V9 (the
pre-registered validation on real data, plan §33), which is run separately.

Design, fixed before running:

  * 200 replicate tables from the test suite's simulator (`tests/conftest.make_counts`)
    with no differential taxa: both groups draw from one composition, and library sizes
    do not depend on the group, so the samples are exchangeable by construction.
    40 taxa, 20 samples per group, seed = replicate number.
  * Each replicate: a Quick-mode multiverse (rule set v3), then `inference.calibrate`
    with B = 999 permutations, BH at q = 0.05, decision-tree weights.
  * Primary: one family p-value per replicate — the most prevalent taxon's, so the 200
    values are independent — tested against Uniform(0, 1) with Kolmogorov-Smirnov.
    Passes if the KS p-value exceeds 0.01.
  * Reported beside it: the share of replicates with any discovery (under the complete
    null, BH's FDR equals the familywise rate, so it should be near or below q), and the
    share with any CERTIFIED ROBUST taxon.

    python tests/reference/calibration_null.py
    python tests/reference/calibration_null.py --replicates 20     # quick look, not the record
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from conftest import make_counts  # noqa: E402

from app.core import inference  # noqa: E402
from app.core.parsers.base import AbundanceTable  # noqa: E402
from app.core.robustness import compute_robustness  # noqa: E402
from app.core.runner import run_multiverse  # noqa: E402
from app.core.validation import validate_dataset  # noqa: E402

OUTPUT = os.path.join(ROOT, "docs", "calibration_null.json")
SMOKE_OUTPUT = os.path.join(ROOT, "data", "calibration_null_smoke.json")
REGISTERED = {"replicates": 200, "permutations": 999, "taxa": 40, "per_group": 20}


def _shown(path) -> str:
    """A path for the console: relative to the repository where it can be, as given
    otherwise (Windows has no relative path between two drives)."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return str(path)


def null_dataset(seed: int, n_taxa: int, per_group: int):
    counts, groups, _, _ = make_counts(n_taxa=n_taxa, n_per_group=per_group,
                                       n_differential=0, seed=seed)
    taxa = [f"T{i:03d}" for i in range(n_taxa)]
    samples = [f"S{j:03d}" for j in range(counts.shape[1])]
    table = AbundanceTable(counts=pd.DataFrame(counts, index=taxa, columns=samples),
                           source_format="simulated", value_type="counts")
    metadata = pd.DataFrame({"group": np.where(groups == 1, "b", "a")}, index=samples)
    metadata.index.name = "sample_id"
    return validate_dataset(table, metadata, "group"), counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=REGISTERED["replicates"])
    parser.add_argument("--permutations", type=int, default=REGISTERED["permutations"])
    args = parser.parse_args()
    registered = (args.replicates == REGISTERED["replicates"]
                  and args.permutations == REGISTERED["permutations"])

    print("Calibrated family p-values under a simulated complete null")
    print(f"{args.replicates} replicates, B = {args.permutations}, "
          f"{REGISTERED['taxa']} taxa, {REGISTERED['per_group']} per group\n")
    started = time.perf_counter()
    primary, any_selected, any_certified, all_p = [], [], [], []
    for replicate in range(args.replicates):
        dataset, counts = null_dataset(replicate + 1, REGISTERED["taxa"],
                                       REGISTERED["per_group"])
        run = run_multiverse(dataset, mode="quick")
        summary = compute_robustness(run)
        weights = summary.spec_summary.sort_values("spec_id")["weight"].to_numpy()
        result = inference.calibrate(run, dataset, weights,
                                     n_permutations=args.permutations,
                                     seed=10_000 + replicate)
        taxa = result.taxa.set_index("taxon_id")
        prevalence = (counts > 0).mean(axis=1)
        target = int(np.argmax(prevalence))          # input-rank taxon ids are row order
        primary.append(float(taxa.loc[target, "p_family"]))
        any_selected.append(bool(taxa["selected"].any()))
        any_certified.append(bool(taxa["certified_robust"].any()))
        all_p.extend(taxa["p_family"].tolist())
        if (replicate + 1) % 20 == 0:
            print(f"  {replicate + 1:>4} replicates, {time.perf_counter() - started:.0f}s")

    ks = stats.kstest(primary, "uniform")
    payload = {
        "check": "calibrated family p-values are uniform under a simulated null",
        "plan": "docs/MICROVERSE_V3_PLAN.md §27.6",
        "script": "tests/reference/calibration_null.py",
        "registered_design": registered,
        "design": {**REGISTERED, "replicates": args.replicates,
                   "permutations": args.permutations, "q": inference.DEFAULT_Q,
                   "discovery": "bh", "weights": "decision_tree", "mode": "quick",
                   "primary": "the most prevalent taxon's family p-value, one per replicate"},
        "ks_statistic": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "passed": bool(ks.pvalue > 0.01),
        "share_below": {str(a): float(np.mean(np.array(primary) <= a))
                        for a in (0.01, 0.05, 0.1)},
        "share_replicates_with_any_discovery": float(np.mean(any_selected)),
        "share_replicates_with_any_certified": float(np.mean(any_certified)),
        "all_taxa_share_below_0_05": float(np.mean(np.array(all_p) <= 0.05)),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    print()
    print(f"  KS against Uniform(0,1): D = {ks.statistic:.3f}, p = {ks.pvalue:.3f} "
          f"-> {'PASS' if payload['passed'] else 'FAIL'} (criterion p > 0.01)")
    print(f"  share of primary p-values <= 0.05: {payload['share_below']['0.05']:.3f}")
    print(f"  replicates with any discovery (BH, q = 0.05): "
          f"{payload['share_replicates_with_any_discovery']:.3f}")
    print(f"  replicates with any CERTIFIED ROBUST taxon: "
          f"{payload['share_replicates_with_any_certified']:.3f}")
    output = OUTPUT if registered else SMOKE_OUTPUT
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"\nwrote {_shown(output)}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
