"""Is the §17 fork attribution stable enough to support a claim? — sensitivity analysis.

SPEC §24.3 reported "rarefaction leads in 7 of 17 cohorts, DA method in 1" as a finding.
That sentence rests entirely on a variance decomposition that had never been checked for
stability. This checks it, three ways, across every cohort:

  1. **Does the fork model explain anything?** The shares are percentages of *explained*
     variance. If the within-taxon R² is 3%, the leading fork leads a decomposition of
     noise, and naming it is not a finding.
  2. **Do independent estimators agree?** Three run on every analysis — §17's MixedLM,
     taxon fixed effects, and §17's group-means fallback. If they rank the forks
     differently, no single ranking is the answer.
  3. **Is the leader stable under resampling?** A cluster bootstrap over taxa gives the
     probability that each fork comes first. A leader that wins 55% of draws is not a
     leader.

The output decides how §24.3 and the README are allowed to describe fork attribution.
This script exists because the honest answer was not known when that claim was written.

    .venv/Scripts/python tests/reference/attribution_sensitivity.py
    .venv/Scripts/python tests/reference/attribution_sensitivity.py --cohorts cdi_schubert
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from app.core.attribution import (  # noqa: E402
    EFFECT_FORKS,
    MIN_ESTIMATOR_AGREEMENT,
    MIN_EXPLAINED_VARIANCE,
    SIGNIFICANCE_FORKS,
    attribute,
    within_attribution,
)
from app.core.runner import run_multiverse  # noqa: E402
from tests.reference.microbiomehd import COHORTS, load_dataset  # noqa: E402

OUTPUT = os.path.join(ROOT, "docs", "attribution_sensitivity.json")

FAILURES: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def report(name: str, detail: str) -> None:
    print(f"  [ -- ] {name} — {detail}")


def leader_stability(run, forks, value_col: str, draws: int = 300,
                     seed: int = 17) -> dict:
    """How often does each fork come first, resampling taxa with replacement?"""
    specs = run.specs_frame.copy()
    specs["rarefaction"] = [s.rarefaction_label for s in run.specs]
    frame = run.long.merge(specs, on="spec_id", how="left")
    frame["prev_filter"] = frame["prev_filter"].map(lambda v: f"{float(v):.0%}")
    frame["fdr_threshold"] = frame["fdr_threshold"].map(lambda v: f"{float(v):g}")
    frame["significant"] = frame["significant"].astype(float)
    active = [f for f in forks if frame[f].nunique() > 1]
    if not active:
        return {}

    taxa = pd.unique(frame["taxon"])
    if len(taxa) < 5:
        return {}
    subset = taxa if len(taxa) <= 60 else np.random.default_rng(seed).choice(
        taxa, size=60, replace=False)
    # Not dict(groupby): pandas exposes `.keys` as the grouping key, so dict() calls it.
    blocks = {  # noqa: C416
        t: part for t, part in frame[frame["taxon"].isin(subset)].groupby("taxon")}

    rng = np.random.default_rng(seed)
    winners: Counter = Counter()
    for _ in range(draws):
        picked = rng.choice(list(blocks), size=len(blocks), replace=True)
        parts = []
        for i, t in enumerate(picked):
            block = blocks[t].copy()
            block["taxon"] = f"{t}__{i}"
            parts.append(block)
        shares, _, _ = within_attribution(pd.concat(parts, ignore_index=True),
                                          active, value_col)
        if shares:
            winners[max(shares, key=shares.get)] += 1
    total = sum(winners.values())
    return {fork: count / total for fork, count in winners.most_common()} if total else {}


def study_one(cohort: str) -> dict:
    dataset, description = load_dataset(cohort)
    started = time.perf_counter()
    run = run_multiverse(dataset, mode="quick")
    attribution = attribute(run, bootstrap=False)

    record = {"cohort": cohort, "description": description,
              "n_samples": int(dataset.n_samples), "n_taxa": int(dataset.n_taxa),
              "runtime_seconds": round(time.perf_counter() - started, 1), "targets": {}}

    for target, forks, value_col in (("effect", EFFECT_FORKS, "effect_h"),
                                     ("significance", SIGNIFICANCE_FORKS, "significant")):
        result = attribution[target]
        stability = leader_stability(run, forks, value_col)
        leader, share = (result.ranked[0] if result.ranked else (None, float("nan")))
        record["targets"][target] = {
            "leader": leader,
            "leader_share": float(share),
            "explained_variance": float(result.explained_variance),
            "estimator_agreement": float(result.estimator_agreement()),
            "converged": bool(result.converged),
            "evidence": result.evidence,
            "leader_win_probability": (
                stability.get(leader, float("nan")) if leader else float("nan")),
            "leader_distribution": stability,
            "mixedlm_leader": (max(result.mixedlm_shares, key=result.mixedlm_shares.get)
                               if result.mixedlm_shares else None),
            "within_leader": (max(result.within_shares, key=result.within_shares.get)
                              if result.within_shares else None),
            "fallback_leader": (max(result.fallback_shares, key=result.fallback_shares.get)
                                if result.fallback_shares else None),
        }
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", default="", help="comma-separated; default all")
    args = parser.parse_args()
    selected = ([c.strip() for c in args.cohorts.split(",") if c.strip()]
                if args.cohorts else list(COHORTS))

    print("Is the §17 fork attribution stable enough to support a claim?")
    print(f"{len(selected)} MicrobiomeHD cohorts, Quick mode.")
    print()
    print(f"  {'cohort':<20}{'target':<14}{'leader':<14}{'R2':>7}{'agree':>7}"
          f"{'wins':>7}  estimators")
    print("  " + "-" * 86)

    records = []
    for cohort in selected:
        try:
            record = study_one(cohort)
        except Exception as exc:                        # noqa: BLE001 — report, continue
            print(f"  {cohort:<20}SKIPPED — {type(exc).__name__}: {exc}")
            continue
        records.append(record)
        for target, block in record["targets"].items():
            leaders = {block["mixedlm_leader"], block["within_leader"],
                       block["fallback_leader"]} - {None}
            agree_mark = "same" if len(leaders) == 1 else f"{len(leaders)} differ"
            print(f"  {cohort:<20}{target:<14}{str(block['leader']):<14}"
                  f"{block['explained_variance']:>7.3f}"
                  f"{block['estimator_agreement']:>7.2f}"
                  f"{block['leader_win_probability']:>7.2f}  {agree_mark}")

    if not records:
        print("\nNo cohort produced a result.")
        return 1

    print()
    summary = {}
    for target in ("effect", "significance"):
        blocks = [r["targets"][target] for r in records if target in r["targets"]]
        explained = np.array([b["explained_variance"] for b in blocks], dtype=float)
        agreement = np.array([b["estimator_agreement"] for b in blocks], dtype=float)
        wins = np.array([b["leader_win_probability"] for b in blocks], dtype=float)
        all_agree = [
            len({b["mixedlm_leader"], b["within_leader"], b["fallback_leader"]}
                - {None}) == 1
            for b in blocks
        ]
        usable = [
            b for b in blocks
            if np.isfinite(b["explained_variance"])
            and b["explained_variance"] >= MIN_EXPLAINED_VARIANCE
            and np.isfinite(b["estimator_agreement"])
            and b["estimator_agreement"] >= MIN_ESTIMATOR_AGREEMENT
        ]
        print(f"{target}:")
        report("within-taxon R² across cohorts",
               f"median {np.nanmedian(explained):.3f}, "
               f"range {np.nanmin(explained):.3f}–{np.nanmax(explained):.3f}")
        report("estimator rank agreement",
               f"median {np.nanmedian(agreement):+.2f}, "
               f"range {np.nanmin(agreement):+.2f}–{np.nanmax(agreement):+.2f}")
        report("three estimators pick the same leading fork",
               f"{sum(all_agree)}/{len(blocks)} cohorts")
        report("leader wins the bootstrap",
               f"median {np.nanmedian(wins):.2f} of draws")
        report("cohorts meeting both thresholds for a reportable ranking",
               f"{len(usable)}/{len(blocks)}")
        summary[target] = {
            "median_explained_variance": float(np.nanmedian(explained)),
            "median_agreement": float(np.nanmedian(agreement)),
            "same_leader_cohorts": int(sum(all_agree)),
            "n_cohorts": int(len(blocks)),
            "median_leader_win_probability": float(np.nanmedian(wins)),
            "n_reportable": len(usable),
            "leader_counts": dict(Counter(
                b["leader"] for b in blocks if b["leader"])),
        }
        print()

    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump({"cohorts": records, "summary": summary,
                   "thresholds": {"min_explained_variance": MIN_EXPLAINED_VARIANCE,
                                  "min_estimator_agreement": MIN_ESTIMATOR_AGREEMENT}},
                  handle, indent=2, default=float)

    print("Verdict — what §24.3 and the README are allowed to say")
    significance = summary["significance"]
    effect = summary["effect"]
    check("the fork model explains enough variance for a ranking to mean something "
          "(significance)",
          significance["median_explained_variance"] >= MIN_EXPLAINED_VARIANCE,
          f"median R² = {significance['median_explained_variance']:.3f}")
    check("independent estimators agree on the leading fork in most cohorts "
          "(significance)",
          significance["same_leader_cohorts"] > significance["n_cohorts"] / 2,
          f"{significance['same_leader_cohorts']}/{significance['n_cohorts']} cohorts")
    check("the leading fork is stable under resampling (significance)",
          significance["median_leader_win_probability"] >= 0.80,
          f"median win probability {significance['median_leader_win_probability']:.2f}")
    check("the same holds for the effect-size decomposition",
          effect["median_explained_variance"] >= MIN_EXPLAINED_VARIANCE
          and effect["same_leader_cohorts"] > effect["n_cohorts"] / 2,
          f"R² {effect['median_explained_variance']:.3f}, "
          f"{effect['same_leader_cohorts']}/{effect['n_cohorts']} agree")

    print()
    print(f"wrote {os.path.relpath(OUTPUT, ROOT)}")
    print()
    if FAILURES:
        print("FAILED: " + ", ".join(FAILURES))
        print()
        print("This is the intended outcome of a sensitivity analysis that finds an")
        print("unstable estimate. It is not a bug: it means fork attribution must be")
        print("reported as exploratory, and any claim of the form 'fork X leads in N of")
        print("M cohorts' must be withdrawn or heavily qualified.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
