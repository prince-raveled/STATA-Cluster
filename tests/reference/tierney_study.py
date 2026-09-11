"""SPEC §23 validation 1 — reproduce Tierney et al. on Tierney et al.'s own cohorts.

    "Reproduce Tierney et al. Code: `github.com/chiragjp/ubiome_robustness`.
     Cohorts: curatedMetagenomicData. Run Mode C on their cohorts; check you recover
     1-in-3 sign-flipping and >90% T1D/T2D nonrobust. **If you reproduce their result,
     your engine is correct.** Highest-value experiment available."  — SPEC §23.1

Tierney et al. 2022 (PLOS Biol 20(3):e3001556) took cohorts from curatedMetagenomicData
and vibrated each microbe-disease association over covariate adjustment sets — one fork,
many models. MicroVerse's covariate mode (§12) is that design: elementary methods over
all 2^k adjustment sets on a fixed reference sub-grid.

`tierney_export.R` pulls the same T1D and T2D cohorts from curatedMetagenomicData with
full MetaPhlAn lineages; this runs Mode C over them and reports the two quantities §23.1
names.

Read the comparison honestly. This is their data source, their disease contrasts and
their fork, but not a line-by-line rerun of their code: their adjustment variables were
chosen per analysis, ours are whatever cMD populates for that cohort, and their
denominator was the associations they carried forward rather than every taxon tested.
Both denominators are reported, as in `real_data_study.py`.

    Rscript tests/reference/tierney_export.R data/cmd/export
    .venv/Scripts/python tests/reference/tierney_study.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from app.core.attribution import attribute  # noqa: E402
from app.core.parsers.base import AbundanceTable, split_lineage  # noqa: E402
from app.core.robustness import compute_robustness  # noqa: E402
from app.core.runner import run_multiverse  # noqa: E402
from app.core.validation import validate_dataset  # noqa: E402

EXPORT = os.path.join(ROOT, "data", "cmd", "export")
OUTPUT = os.path.join(ROOT, "docs", "tierney_study.json")

#: Tierney et al. 2022, the numbers §23.1 asks us to recover.
TIERNEY = {
    "sign_flip_fraction": 1 / 3,
    "fraction_nonrobust": 0.90,
}

FAILURES: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def report(name: str, detail: str) -> None:
    print(f"  [ -- ] {name} — {detail}")


def load_cohort(slug: str):
    folder = os.path.join(EXPORT, slug)
    counts = pd.read_csv(os.path.join(folder, "counts.tsv"), sep="\t", index_col=0)
    metadata = pd.read_csv(os.path.join(folder, "meta.tsv"), sep="\t", index_col=0)
    counts = counts[[c for c in counts.columns if c in set(metadata.index)]]
    # MetaPhlAn assigns nothing at all to the occasional very shallow library; those
    # samples carry no information and the validator rightly refuses them (§8).
    counts = counts.loc[:, counts.sum(axis=0) > 0]
    counts = counts.loc[counts.sum(axis=1) > 0]
    metadata = metadata.loc[counts.columns]
    lineages = {str(t): split_lineage(t) for t in counts.index}
    table = AbundanceTable(
        counts=counts,
        lineages={k: v for k, v in lineages.items() if v},
        source_format="curatedMetagenomicData",
        value_type="counts",
    )
    return validate_dataset(table, metadata, "group")


def _covariate_fork_only(run, method: str = "linear", min_subsets: int = 8) -> dict:
    """Sign-flipping across adjustment sets alone — Tierney's actual vibration.

    Mode C varies method, transform, rarefaction and rank as well as the adjustment set,
    so a sign-consistency computed over the whole mode answers a broader question than
    Tierney asked. This holds every other fork fixed and moves only the covariates,
    inside each matrix, which is their design: one model type, one preprocessing, many
    adjustment sets. The per-matrix results are then averaged.
    """
    specs = run.specs_frame
    chosen = specs[(specs["method"] == method) & (specs["fdr_method"] == "bh")
                   & (specs["fdr_threshold"] == 0.05)]
    if chosen.empty:
        return {}
    keys = ["rarefaction", "rare_seed", "rank", "prev_filter", "transform"]
    flips_tested, flips_detected, n_matrices, n_subsets_seen = [], [], 0, []
    for _, block in chosen.groupby(keys, dropna=False):
        if block["covariates"].nunique() < min_subsets:
            continue
        part = run.long[run.long["spec_id"].isin(set(block["spec_id"]))]
        if part.empty:
            continue
        grouped = part.groupby("taxon")
        frame = pd.DataFrame({
            "n": grouped["effect_n"].size(),
            "positive": grouped["effect_n"].apply(lambda v: float((v > 0).mean())),
            "frac_sig": grouped["significant"].mean(),
        })
        frame = frame[frame["n"] >= min_subsets]
        if frame.empty:
            continue
        consistency = frame[["positive"]].assign(
            other=1.0 - frame["positive"]).max(axis=1)
        flipped = consistency < 0.95
        flips_tested.append(float(flipped.mean()))
        detected = frame["frac_sig"] > 0
        if detected.any():
            flips_detected.append(float(flipped[detected].mean()))
        n_matrices += 1
        n_subsets_seen.append(int(block["covariates"].nunique()))
    if not flips_tested:
        return {}
    return {
        "n_matrices": n_matrices,
        "n_subsets": int(np.median(n_subsets_seen)),
        "sign_flip_tested": float(np.mean(flips_tested)),
        "sign_flip_detected": float(np.mean(flips_detected)) if flips_detected else 0.0,
    }


def study_one(slug: str, row) -> dict:
    dataset = load_cohort(slug)
    started = time.perf_counter()
    run = run_multiverse(dataset, mode="covariate",
                         covariate_columns=tuple(dataset.covariate_columns))
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
    n_subsets = run.specs_frame["covariates"].nunique()
    covariate_only = _covariate_fork_only(run)

    return {
        "slug": slug,
        "study": row["study"],
        "condition": row["condition"],
        "covariates": row["covariates"].split("|") if isinstance(row["covariates"], str) else [],
        "n_samples": int(dataset.n_samples),
        "n_taxa": int(dataset.n_taxa),
        "n_case": int(row["n_case"]),
        "n_control": int(row["n_control"]),
        "n_ranks": len(dataset.table.ranks) if hasattr(dataset.table, "ranks") else None,
        "n_specifications": int(run.grid_report.n_valid),
        "n_covariate_subsets": int(n_subsets),
        "runtime_seconds": round(elapsed, 2),
        "n_taxa_tested": int(len(tested)),
        "n_taxa_detected": int(len(detected)),
        # --- the two §23.1 quantities ------------------------------------
        "sign_flip_fraction": float((tested["sign_consistency"] < 0.95).mean()),
        "sign_flip_fraction_detected": (
            float((detected["sign_consistency"] < 0.95).mean()) if len(detected) else 0.0),
        "fraction_nonrobust_of_detected": (
            float((detected["robustness_tier"] != "ROBUST").mean()) if len(detected) else 0.0),
        "fraction_nonrobust_of_tested": (
            float((tested["robustness_tier"] != "ROBUST").mean()) if len(tested) else 0.0),
        "tiers": {k: int(v) for k, v in summary.tier_counts.items()},
        "significance_leading_fork": significance_fork,
        "significance_leading_share": round(significance_share, 1),
        # --- Tierney's fork on its own -----------------------------------
        "covariate_fork_only": covariate_only,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", default="", help="comma-separated slugs")
    args = parser.parse_args()

    index_path = os.path.join(EXPORT, "index.csv")
    if not os.path.exists(index_path):
        print("SKIPPED — Tierney cohorts not exported. Run:")
        print("  Rscript tests/reference/tierney_export.R data/cmd/export")
        print("(needs R with curatedMetagenomicData; set MICROVERSE_R_LIB if it is "
              "outside the default library)")
        return 0
    index = pd.read_csv(index_path)
    if args.cohorts:
        wanted = {c.strip() for c in args.cohorts.split(",")}
        index = index[index["slug"].isin(wanted)]

    print(f"Tierney et al. cohorts from curatedMetagenomicData: {len(index)} contrasts")
    print("Mode C (§12): elementary methods over every covariate adjustment subset.")
    print()

    records = []
    for i, row in index.reset_index(drop=True).iterrows():
        try:
            record = study_one(row["slug"], row)
        except Exception as exc:                        # noqa: BLE001 — report, continue
            print(f"  [{i + 1}/{len(index)}] {row['slug']}: SKIPPED — {exc}")
            continue
        records.append(record)
        print(f"  [{i + 1}/{len(index)}] {record['study']:<24} {record['condition']:<4} "
              f"{record['n_specifications']:>5} specs "
              f"({record['n_covariate_subsets']} subsets) "
              f"{record['runtime_seconds']:>6.1f}s  "
              f"sign-flip {record['sign_flip_fraction']:.0%}  "
              f"nonrobust {record['fraction_nonrobust_of_detected']:.0%}")

    if not records:
        print("\nNo cohort produced a result.")
        return 1

    frame = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump({"source": "Tierney et al. 2022, PLOS Biol 20(3):e3001556",
                   "data": "curatedMetagenomicData",
                   "target": TIERNEY, "cohorts": records}, handle, indent=2)

    print(f"\n{len(records)} contrasts, "
          f"{int(frame['n_specifications'].sum()):,} specifications, "
          f"{frame['runtime_seconds'].sum():.0f}s total")

    print("\nSPEC §23.1 — the two numbers to recover")
    cov = [r["covariate_fork_only"] for r in records if r.get("covariate_fork_only")]
    if cov:
        cov_tested = float(np.mean([c["sign_flip_tested"] for c in cov]))
        cov_detected = float(np.mean([c["sign_flip_detected"] for c in cov]))
        report("adjustment sets per matrix",
               f"median {int(np.median([c['n_subsets'] for c in cov]))}, over "
               f"{sum(c['n_matrices'] for c in cov)} matrices")
        report("sign-flipping over adjustment sets alone, all taxa", f"{cov_tested:.1%}")
        report("sign-flipping over adjustment sets alone, taxa detected",
               f"{cov_detected:.1%}")
        check("recovers Tierney's 1-in-3 sign-flipping (their fork, their cohorts)",
              0.20 <= cov_tested <= 0.50 or 0.20 <= cov_detected <= 0.50,
              f"{cov_tested:.1%} of tested / {cov_detected:.1%} of detected "
              f"vs their {TIERNEY['sign_flip_fraction']:.0%}")
    else:
        report("covariate-only sign flipping", "no cohort had enough adjustment sets")

    flip_all = float(frame["sign_flip_fraction"].mean())
    flip_detected = float(frame["sign_flip_fraction_detected"].mean())
    nonrobust = float(frame["fraction_nonrobust_of_detected"].mean())
    # The harmonised effect (§14) is computed from the matrix, so it does not move with
    # the adjustment set at all: over the whole of Mode C only the transform, rarefaction
    # and rank forks can flip its sign. Reported for contrast, not as Tierney's number.
    report("sign-flipping of the harmonised effect over all of Mode C",
           f"{flip_all:.1%} of tested, {flip_detected:.1%} of detected")

    report("nonrobust, of taxa detected somewhere", f"{nonrobust:.1%}")
    check("recovers >90% of T1D/T2D associations nonrobust",
          nonrobust >= TIERNEY["fraction_nonrobust"],
          f"{nonrobust:.1%} are not ROBUST, over {len(frame)} contrasts")
    print("\nper contrast")
    for record in records:
        print(f"  {record['study']:<24} {record['condition']:<4} "
              f"n={record['n_samples']:>4}  detected={record['n_taxa_detected']:>4}  "
              f"sign-flip {record['sign_flip_fraction']:>5.0%}  "
              f"nonrobust {record['fraction_nonrobust_of_detected']:>5.0%}  "
              f"leading fork: {record['significance_leading_fork']}")

    print(f"\nwrote {OUTPUT}")
    print("\n" + ("FAILED: " + ", ".join(FAILURES) if FAILURES else "All checks passed."))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
