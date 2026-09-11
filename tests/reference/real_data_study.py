"""SPEC §23 validation 2 — the multiverse on real published cohorts.

"Turn on Forks 1-5 and quantify the *additional* instability. Their limitations section
predicted this; nobody has measured it. This is a genuine finding, not just a tool
paper."

Tierney et al. 2022 vibrated over covariates only, with linear models, on an HPC
cluster, and reported (their Table 1 / §3):

    taxa flipping association sign substantially        1 in 3
    mean fraction of models nominally significant       38%
    mean fraction FDR significant                       16%

This runs MicroVerse's Quick-mode grid — rarefaction depth and seed, prevalence filter,
transformation, taxonomic rank, four DA methods and three FDR settings — over 17
published case-control cohorts from MicrobiomeHD (Duvallet et al. 2017), and computes
the same three quantities plus the fork attribution.

Read the comparison carefully: the denominators differ. Tierney's percentages are over
581 *literature-reported* associations; ours are over every taxon the grid tested, and
also over the subset significant somewhere, which is the closer analogue of "a result
somebody would have written down". Both are reported.

    .venv/Scripts/python tests/reference/real_data_study.py
    .venv/Scripts/python tests/reference/real_data_study.py --cohorts cdi_schubert,crc_zhao
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.attribution import attribute  # noqa: E402
from app.core.robustness import compute_robustness  # noqa: E402
from app.core.runner import run_multiverse  # noqa: E402
from app.core.validation import DatasetError  # noqa: E402
from tests.reference.microbiomehd import COHORTS, load_dataset  # noqa: E402

OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docs", "real_data_study.json",
)

#: Tierney et al. 2022, PLOS Biol 20(3):e3001556 — the numbers we are extending.
TIERNEY = {
    "sign_flip_fraction": 1 / 3,
    "mean_fraction_nominal": 0.38,
    "mean_fraction_fdr": 0.16,
}


def study_one(cohort: str, keep_otu_level: bool = False) -> dict:
    dataset, description = load_dataset(cohort, keep_otu_level=keep_otu_level)
    started = time.perf_counter()
    run = run_multiverse(dataset, mode="quick")
    summary = compute_robustness(run)
    attribution = attribute(run)
    elapsed = time.perf_counter() - started

    table = summary.table
    tested = table[table["n_specs_tested"] >= 10]          # §15: below this, untiered
    detected = tested[tested["frac_significant"] > 0]      # significant somewhere

    def leader(target):
        shares = attribution[target].shares
        if not shares:
            return None, 0.0
        fork = max(shares, key=shares.get)
        return fork, float(shares[fork])

    significance_fork, significance_share = leader("significance")
    effect_fork, effect_share = leader("effect")
    draw = attribution.get("rarefaction_draw") or {}

    return {
        "cohort": cohort,
        "description": description,
        "n_samples": int(dataset.n_samples),
        "n_group_a": int(dataset.group_sizes[0]),
        "n_group_b": int(dataset.group_sizes[1]),
        "n_taxa": int(dataset.n_taxa),
        "n_specifications": int(run.grid_report.n_valid),
        "n_enumerated": int(run.grid_report.n_enumerated),
        "runtime_seconds": round(elapsed, 2),
        "n_taxa_tested": int(len(tested)),
        "n_taxa_detected": int(len(detected)),
        "n_rank_levels": int(run.specs_frame["rank"].nunique()),
        # Numerators and denominators, so the aggregate can be pooled (§24.3).
        "n_flip": int((tested["sign_consistency"] < 0.95).sum()),
        "n_flip_detected": int((detected["sign_consistency"] < 0.95).sum()),
        "n_robust_detected": int((detected["robustness_tier"] == "ROBUST").sum()),
        "sum_frac_nominal": float(tested["frac_nominal"].sum()),
        "sum_frac_fdr": float(tested["frac_significant"].sum()),
        "sum_frac_nominal_detected": float(detected["frac_nominal"].sum()),
        "sum_frac_fdr_detected": float(detected["frac_significant"].sum()),
        # --- the three Tierney quantities -------------------------------
        # "Flipping substantially": the direction is not settled by the data.
        "sign_flip_fraction": float((tested["sign_consistency"] < 0.95).mean()),
        "sign_flip_fraction_detected": (
            float((detected["sign_consistency"] < 0.95).mean()) if len(detected) else 0.0
        ),
        "mean_fraction_nominal": float(tested["frac_nominal"].mean()),
        "mean_fraction_fdr": float(tested["frac_significant"].mean()),
        "mean_fraction_nominal_detected": (
            float(detected["frac_nominal"].mean()) if len(detected) else 0.0
        ),
        "mean_fraction_fdr_detected": (
            float(detected["frac_significant"].mean()) if len(detected) else 0.0
        ),
        # --- robustness ---------------------------------------------------
        "tiers": {k: int(v) for k, v in summary.tier_counts.items()},
        "fraction_robust_of_detected": (
            float((detected["robustness_tier"] == "ROBUST").mean()) if len(detected) else 0.0
        ),
        # --- attribution ---------------------------------------------------
        "significance_leading_fork": significance_fork,
        "significance_leading_share": round(significance_share, 1),
        "effect_leading_fork": effect_fork,
        "effect_leading_share": round(effect_share, 1),
        "rarefaction_seed_share": round(float(draw.get("seed_share", float("nan"))), 1),
    }


def main() -> int:
    selected = list(COHORTS)
    keep_otu_level = False
    for argument in sys.argv[1:]:
        if argument.startswith("--cohorts="):
            selected = argument.split("=", 1)[1].split(",")
        elif argument == "--cohorts":
            index = sys.argv.index(argument)
            selected = sys.argv[index + 1].split(",")
        elif argument == "--otu":
            # Trim to the §8 ceiling instead of collapsing to genus, so fork 4 has
            # two levels. See §24.3: on the default path most cohorts have one.
            keep_otu_level = True

    print(f"MicroVerse Quick mode over {len(selected)} MicrobiomeHD cohorts"
          f"{' at OTU level (fork 4 has two ranks)' if keep_otu_level else ''}")
    print(f"{'cohort':<20}{'n':>5}{'taxa':>6}{'specs':>7}{'detected':>9}"
          f"{'flip':>7}{'nom':>7}{'fdr':>7}{'robust':>8}{'time':>7}")
    print("-" * 84)

    results, failures = [], []
    for cohort in selected:
        try:
            row = study_one(cohort, keep_otu_level=keep_otu_level)
        except DatasetError as exc:
            failures.append((cohort, f"refused by §8: {exc.message}"))
            print(f"{cohort:<20}  refused: {exc.message[:52]}")
            continue
        except Exception as exc:                            # noqa: BLE001 - reported
            failures.append((cohort, f"{type(exc).__name__}: {exc}"))
            print(f"{cohort:<20}  FAILED: {type(exc).__name__}: {str(exc)[:44]}")
            continue

        results.append(row)
        print(f"{row['cohort']:<20}{row['n_samples']:>5}{row['n_taxa']:>6}"
              f"{row['n_specifications']:>7,}{row['n_taxa_detected']:>9}"
              f"{row['sign_flip_fraction']:>7.0%}{row['mean_fraction_nominal']:>7.0%}"
              f"{row['mean_fraction_fdr']:>7.0%}"
              f"{row['fraction_robust_of_detected']:>8.0%}"
              f"{row['runtime_seconds']:>6.1f}s")

    if not results:
        print("\nNo cohort completed.")
        return 1

    def mean(key):
        return float(np.mean([r[key] for r in results]))

    def pooled(numerator: str, denominator: str, rows=None) -> float:
        rows = results if rows is None else rows
        bottom = sum(r[denominator] for r in rows)
        return float(sum(r[numerator] for r in rows) / bottom) if bottom else float("nan")

    def bootstrap(numerator: str, denominator: str, draws: int = 2000,
                  seed: int = 7) -> tuple:
        """Resample whole cohorts, not taxa: taxa inside a cohort are not independent.

        Seventeen cohorts is a small cluster count, so this is a confidence interval on
        the pooled fraction across *these* cohorts, not a population estimate.
        """
        rng = np.random.default_rng(seed)
        n = len(results)
        if n < 3:
            return float("nan"), float("nan")
        values = []
        for _ in range(draws):
            sample = [results[i] for i in rng.integers(0, n, size=n)]
            value = pooled(numerator, denominator, sample)
            if np.isfinite(value):
                values.append(value)
        if not values:
            return float("nan"), float("nan")
        return (float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)))

    print("\n" + "=" * 84)
    print("AGGREGATE — MicroVerse (forks 1-5 + FDR) vs Tierney et al. 2022 (covariates only)")
    print("=" * 84)
    rows = [
        ("taxa flipping sign (sign consistency < 0.95)", "sign_flip_fraction",
         TIERNEY["sign_flip_fraction"], "1 in 3"),
        ("mean fraction of specifications nominally significant", "mean_fraction_nominal",
         TIERNEY["mean_fraction_nominal"], "38%"),
        ("mean fraction FDR significant", "mean_fraction_fdr",
         TIERNEY["mean_fraction_fdr"], "16%"),
    ]
    #: Pooled numerator/denominator for each quantity, so a cohort with 300 taxa is
    #: not given the same weight as one with 40 (§24.3 caveat).
    POOLED = {
        "sign_flip_fraction": ("n_flip", "n_taxa_tested"),
        "mean_fraction_nominal": ("sum_frac_nominal", "n_taxa_tested"),
        "mean_fraction_fdr": ("sum_frac_fdr", "n_taxa_tested"),
    }
    print(f"{'quantity':<52}{'pooled':>9}{'95% CI':>16}{'mean':>8}{'Tierney':>10}")
    for label, key, _value, printed in rows:
        top, bottom = POOLED[key]
        low, high = bootstrap(top, bottom)
        print(f"{label:<52}{pooled(top, bottom):>8.0%}"
              f"{f'{low:.0%} - {high:.0%}':>16}{mean(key):>8.0%}{printed:>10}")
    print("  pooled = one taxon one vote across all cohorts; mean = one cohort one "
          "vote (the two differ when cohorts differ in size)")
    print("  CI = 2.5-97.5 percentile of 2,000 bootstrap resamples of the cohorts")

    print()
    print(f"{'the same, over taxa significant in >=1 specification':<52}"
          f"{'pooled':>9}{'95% CI':>16}{'mean':>8}")
    for label, numerator, denominator, mean_key in (
        ("  taxa flipping sign", "n_flip_detected", "n_taxa_detected",
         "sign_flip_fraction_detected"),
        ("  mean fraction nominally significant", "sum_frac_nominal_detected",
         "n_taxa_detected", "mean_fraction_nominal_detected"),
        ("  mean fraction FDR significant", "sum_frac_fdr_detected",
         "n_taxa_detected", "mean_fraction_fdr_detected"),
        ("  fraction reaching ROBUST", "n_robust_detected", "n_taxa_detected",
         "fraction_robust_of_detected"),
    ):
        low, high = bootstrap(numerator, denominator)
        print(f"{label:<52}{pooled(numerator, denominator):>8.0%}"
              f"{f'{low:.0%} - {high:.0%}':>16}{mean(mean_key):>8.0%}")

    forks = {}
    for row in results:
        forks[row["significance_leading_fork"]] = forks.get(row["significance_leading_fork"], 0) + 1
    print("\nleading fork for significance, across cohorts:")
    for fork, count in sorted(forks.items(), key=lambda kv: -kv[1]):
        print(f"  {str(fork):<22} {count}/{len(results)} cohorts")
    seed_shares = [r["rarefaction_seed_share"] for r in results
                   if np.isfinite(r["rarefaction_seed_share"])]
    if seed_shares:
        print(f"\nshare of rarefaction variance that is the random draw, not the depth: "
              f"{np.mean(seed_shares):.0f}% (median {np.median(seed_shares):.0f}%)")

    total_specs = sum(r["n_specifications"] for r in results)
    total_fits = sum(r["n_specifications"] * r["n_taxa"] for r in results)
    print(f"\n{len(results)} cohorts, {total_specs:,} specifications, "
          f"{total_fits:,} taxon-level results, "
          f"{sum(r['runtime_seconds'] for r in results):.0f}s total on one laptop.")

    if failures:
        print(f"\n{len(failures)} cohort(s) not analysed:")
        for cohort, reason in failures:
            print(f"  {cohort}: {reason[:88]}")

    # The two rank modes answer different questions and are both worth keeping, so a
    # --otu run must not overwrite the genus record.
    destination = OUTPUT if not keep_otu_level else OUTPUT.replace(".json", "_otu.json")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump({
            "source": "MicrobiomeHD, Duvallet et al. 2017, Zenodo 1146764 (CC-BY-NC-4.0)",
            "comparison": "Tierney et al. 2022, PLOS Biol 20(3):e3001556",
            "tierney_reference": TIERNEY,
            "rank_mode": "otu" if keep_otu_level else "genus",
            "cohorts": results,
            "failures": failures,
        }, handle, indent=2)
    print()
    print(f"written to {os.path.relpath(destination)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
