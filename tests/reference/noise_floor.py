"""What does the multiverse report on a table with no group difference at all?

The results page tells a reader that a taxon in UNSTABLE is not weak evidence of an
effect, and quotes a number for how many taxa a pure-noise table still puts there. That
number has to come from somewhere, so it comes from here.

The point it supports: "significant in at least one of 1,596 analyses" is an almost
unmissable bar. The analyses are correlated re-runs of the same samples, so a taxon only
has to cross the line once in a large, correlated family. UNSTABLE is what that looks
like, and it is the tier a null table fills.

This is not a defect in the tiers — it is the reason the tiers exist, and precisely why
ROBUST and CONDITIONAL are the only two with a measured replication rate behind them
(SPEC §24.6). It is recorded so the interface can say so with a figure attached.

Run:  .venv/Scripts/python tests/reference/noise_floor.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from app.core.parsers.base import AbundanceTable  # noqa: E402
from app.core.robustness import compute_robustness, n_detected, n_stable  # noqa: E402
from app.core.runner import run_multiverse  # noqa: E402
from app.core.validation import validate_dataset  # noqa: E402

RECORD = Path(__file__).resolve().parents[2] / "docs" / "noise_floor.json"

#: The configuration quoted in the interface. Changing it means changing the copy.
HEADLINE_SEED = 11
HEADLINE_TAXA = 80
HEADLINE_PER_GROUP = 25


def null_dataset(n_per_group: int, n_taxa: int, seed: int):
    """Both groups drawn from one distribution — no difference exists to be found."""
    rng = np.random.default_rng(seed)
    n = n_per_group * 2
    props = rng.dirichlet(np.full(n_taxa, 0.4))
    depths = rng.integers(4_000, 12_000, n)
    counts = np.vstack([rng.multinomial(int(d), props) for d in depths]).T.astype(float)
    frame = pd.DataFrame(
        counts,
        index=[f"taxon_{i:03d}" for i in range(n_taxa)],
        columns=[f"s{j:03d}" for j in range(n)])
    metadata = pd.DataFrame(
        {"group": ["control"] * n_per_group + ["case"] * n_per_group},
        index=frame.columns)
    table = AbundanceTable(counts=frame, lineages={}, source_format="simulated",
                           value_type="counts")
    return validate_dataset(table, metadata, "group")


def measure(n_per_group: int, n_taxa: int, seed: int) -> dict:
    dataset = null_dataset(n_per_group, n_taxa, seed)
    run = run_multiverse(dataset, mode="quick")
    summary = compute_robustness(run)
    counts = summary.tier_counts
    tested = int(len(summary.table))
    return {
        "seed": seed, "n_per_group": n_per_group, "n_taxa": n_taxa,
        "n_taxa_tiered": tested,
        "n_specifications": int(summary.n_specs_total),
        "tiers": {k: int(v) for k, v in counts.items()},
        "n_detected": int(n_detected(summary)),
        "n_stable": int(n_stable(summary)),
        "unstable_share": round(counts["UNSTABLE"] / max(tested, 1), 4),
    }


def main() -> int:
    print("Null tables through the multiverse — no group difference by construction\n"
          + "=" * 72)
    headline = measure(HEADLINE_PER_GROUP, HEADLINE_TAXA, HEADLINE_SEED)
    print("\nHeadline configuration (the one the interface quotes):")
    print(f"  {headline['n_per_group']} per group, {headline['n_taxa']} taxa, "
          f"{headline['n_specifications']:,} analyses")
    print(f"  UNSTABLE       {headline['tiers']['UNSTABLE']:>4d} of "
          f"{headline['n_taxa_tiered']} taxa")
    print(f"  NOT DETECTED   {headline['tiers']['NOT DETECTED']:>4d}")
    print(f"  ROBUST         {headline['tiers']['ROBUST']:>4d}")
    print(f"  CONDITIONAL    {headline['tiers']['CONDITIONAL']:>4d}")

    print("\nAcross further seeds:")
    replicates = [measure(HEADLINE_PER_GROUP, HEADLINE_TAXA, s) for s in (12, 13, 14)]
    for row in replicates:
        print(f"  seed {row['seed']}: UNSTABLE {row['tiers']['UNSTABLE']:>3d}/"
              f"{row['n_taxa_tiered']}, ROBUST {row['tiers']['ROBUST']}, "
              f"CONDITIONAL {row['tiers']['CONDITIONAL']}")

    every = [headline, *replicates]
    stable_total = sum(r["n_stable"] for r in every)
    print(f"\nROBUST or CONDITIONAL across all {len(every)} null runs: {stable_total}")
    print("The two tiers carrying a replication claim stay empty on pure noise, while\n"
          "UNSTABLE fills up. That is the separation the interface relies on.")

    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(
        {"headline": headline, "replicates": replicates,
         "n_stable_across_all_runs": stable_total,
         "interface_claim": (
             f"on a table simulated with no difference between the groups at all, "
             f"{headline['tiers']['UNSTABLE']} of {headline['n_taxa_tiered']} taxa "
             f"still landed in UNSTABLE")},
        indent=2), encoding="utf-8")
    print(f"\nRecorded to {RECORD.relative_to(RECORD.parents[1])}")
    return 0 if stable_total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
