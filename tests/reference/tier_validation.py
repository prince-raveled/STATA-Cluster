"""Do the robustness tiers predict out-of-sample replication? — SPEC §16.2 validation.

The tiers of §16.2 — ROBUST / CONDITIONAL / FRAGILE / UNSTABLE / INSUFFICIENT — are
thresholds on `frac_significant` and `sign_consistency`. `tests/test_worked_example.py`
proves they are *implemented* as written. Nothing proved they *mean* anything: that a
taxon called ROBUST is likelier to hold up in new data than one called FRAGILE. That is
the claim the interface makes every time it prints a tier, and until this experiment it
was unevidenced. It is the difference between internal consistency and external validity.

Design — discovery/validation:

  1. Take a published cohort and split the samples 50/50, stratified on the group, so
     both halves keep the case/control ratio. Repeat with several seeds per cohort.
  2. Half A is discovery: run the Quick-mode multiverse (the product default) and assign
     every taxon a tier from half A alone.
  3. Half B is validation, disjoint by construction. Replication is defined there
     *tier-blind*, two independent ways, because the definition is a researcher's choice
     and the conclusion should not hinge on ours:
       - `majority`  — nominally significant in most half-B specifications, same sign.
       - `reference` — significant in one pre-specified default analysis on half B
                       (no rarefaction, 10% prevalence, raw counts, Wilcoxon, BH 0.05),
                       same sign. What someone would actually run next.
  4. A null benchmark permutes the half-B group labels, leaving discovery untouched, so
     the replication rate attributable to chance is measured rather than assumed.
  5. Uncertainty is a cluster bootstrap over *cohorts*: taxa within a cohort share
     samples, and repeated splits of one cohort are not independent observations.
  6. Discrimination is an effect size — AUC of the tier ordering against replication,
     and a risk ratio — not only a p-value.

Cohorts are filtered to at least `--min-per-group` samples in the smaller group, because
a 30-sample half produces no FDR-significant result under any specification, so no tier
above FRAGILE is ever assigned and there is nothing to discriminate. That is a real
limit on where tiers are meaningful, and it is reported rather than worked around.

    .venv/Scripts/python tests/reference/tier_validation.py
    .venv/Scripts/python tests/reference/tier_validation.py --max-cohorts 4 --repeats 1
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from app.core.parsers.base import AbundanceTable  # noqa: E402
from app.core.robustness import MIN_SPECS_FOR_TIER, compute_robustness  # noqa: E402
from app.core.runner import run_multiverse  # noqa: E402
from app.core.validation import validate_dataset  # noqa: E402

EXPORT = os.path.join(ROOT, "data", "pelto", "export")
OUTPUT = os.path.join(ROOT, "docs", "tier_validation.json")
ROWS_OUTPUT = os.path.join(ROOT, "docs", "tier_validation_rows.csv.gz")

#: Tiers that assert something about the data, strongest first. INSUFFICIENT (too few
#: specifications) and NOT DETECTED (never significant) assert nothing, so they are
#: reported but kept out of the ordering test.
CLAIM_TIERS = ("ROBUST", "CONDITIONAL", "FRAGILE", "UNSTABLE")

#: The "what a researcher would run next" analysis behind the reference definition.
REFERENCE_SPEC = {"rarefaction": "none", "prev_filter": 0.10, "transform": "raw",
                  "method": "wilcoxon", "fdr_method": "bh", "fdr_threshold": 0.05}

FAILURES: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def report(name: str, detail: str) -> None:
    print(f"  [ -- ] {name} — {detail}")


# --- data ------------------------------------------------------------------
def read_cohort(slug: str):
    """Counts (taxa x samples) and metadata for one whole cohort, unsplit."""
    folder = os.path.join(EXPORT, slug)
    counts = pd.read_csv(os.path.join(folder, "counts.tsv"), sep="\t", index_col=0)
    metadata = pd.read_csv(os.path.join(folder, "meta.tsv"), sep="\t", index_col=0)
    metadata["group"] = metadata["group"].map({"control": "a_control", "case": "b_case"})
    metadata = metadata.dropna(subset=["group"])
    counts = counts[[c for c in counts.columns if c in set(metadata.index)]]
    counts = counts.loc[counts.sum(axis=1) > 0]
    return counts, metadata.loc[counts.columns, ["group"]].copy()


def stratified_halves(metadata: pd.DataFrame, seed: int):
    """Two disjoint sample lists, each keeping the cohort's case/control ratio."""
    rng = np.random.default_rng(seed)
    first, second = [], []
    for _, block in metadata.groupby("group"):
        samples = block.index.to_numpy().copy()
        rng.shuffle(samples)
        cut = len(samples) // 2
        first.extend(samples[:cut])
        second.extend(samples[cut:])
    return sorted(first), sorted(second)


def dataset_from(counts: pd.DataFrame, metadata: pd.DataFrame, samples,
                 permute_groups: int = 0):
    """A validated Dataset over one subset of the samples."""
    subset = counts[list(samples)]
    subset = subset.loc[subset.sum(axis=1) > 0]
    meta = metadata.loc[list(samples), ["group"]].copy()
    if permute_groups:
        rng = np.random.default_rng(permute_groups)
        meta["group"] = rng.permutation(meta["group"].to_numpy())
    table = AbundanceTable(counts=subset, lineages={}, source_format="pelto",
                           value_type="counts")
    return validate_dataset(table, meta, "group")


# --- the experiment --------------------------------------------------------
def _reference_spec_ids(specs_frame: pd.DataFrame) -> set:
    """Specification ids matching REFERENCE_SPEC, ignoring forks it does not pin."""
    mask = pd.Series(True, index=specs_frame.index)
    for column, value in REFERENCE_SPEC.items():
        if column not in specs_frame.columns:
            continue
        values = specs_frame[column]
        if column in ("prev_filter", "fdr_threshold"):
            mask &= np.isclose(values.astype(float), float(value))
        else:
            mask &= values.astype(str) == str(value)
    return set(specs_frame.loc[mask, "spec_id"])


def evaluate_validation_half(run) -> pd.DataFrame:
    """Per-taxon replication evidence from the held-out half. Never sees a tier."""
    long = run.long
    grouped = long.groupby("taxon", sort=True)
    frame = pd.DataFrame({
        "n_specs": grouped.size(),
        "frac_nominal": grouped["p_raw"].apply(lambda s: float((s < 0.05).mean())),
        "frac_significant": grouped["significant"].mean(),
        "median_effect": grouped["effect_h"].median(),
    })

    reference_ids = _reference_spec_ids(run.specs_frame)
    if reference_ids:
        part = long[long["spec_id"].isin(reference_ids)]
        frame = frame.join(
            part.groupby("taxon").agg(ref_significant=("significant", "max")), how="left")
    else:
        frame["ref_significant"] = np.nan

    frame["name"] = [run.taxa_names[i] for i in frame.index]
    return frame.set_index("name")


def one_split(counts, metadata, row, seed: int, mode: str = "quick",
              permute: int = 0) -> pd.DataFrame:
    """Tier on half A, score on half B. One row per taxon present in both."""
    first, second = stratified_halves(metadata, seed)

    discovery = run_multiverse(dataset_from(counts, metadata, first), mode=mode)
    table = compute_robustness(discovery).table.set_index("taxon")

    validation = run_multiverse(
        dataset_from(counts, metadata, second, permute_groups=permute), mode=mode)
    held_out = evaluate_validation_half(validation)

    shared = sorted(set(table.index) & set(held_out.index))
    if not shared:
        return pd.DataFrame()
    table, held_out = table.loc[shared], held_out.loc[shared]

    discovery_effect = table["median_effect"].to_numpy()
    same_sign = np.sign(discovery_effect) == np.sign(held_out["median_effect"].to_numpy())
    same_sign &= discovery_effect != 0.0   # no direction to replicate

    return pd.DataFrame({
        "cohort": row["study"],
        "type": row["type"],
        "seed": seed,
        "n_discovery": int(len(first)),
        "n_validation": int(len(second)),
        "taxon": shared,
        "tier": table["robustness_tier"].to_numpy(),
        "n_specs_tested": table["n_specs_tested"].to_numpy(),
        "frac_significant": table["frac_significant"].to_numpy(),
        "sign_consistency": table["sign_consistency"].to_numpy(),
        "discovery_effect": discovery_effect,
        "validation_effect": held_out["median_effect"].to_numpy(),
        "validation_frac_nominal": held_out["frac_nominal"].to_numpy(),
        "same_sign": same_sign,
        "replicated_majority": (held_out["frac_nominal"].to_numpy() > 0.5) & same_sign,
        "replicated_reference": (
            held_out["ref_significant"].fillna(0).to_numpy().astype(bool) & same_sign),
        "permuted": bool(permute),
    })


# --- analysis --------------------------------------------------------------
def cluster_bootstrap(rows: pd.DataFrame, column: str, tier: str,
                      draws: int = 2000, seed: int = 11) -> tuple:
    """Resample cohorts, not taxa: taxa within a cohort share samples."""
    subset = rows[rows["tier"] == tier]
    cohorts = subset["cohort"].unique() if len(subset) else []
    if len(cohorts) < 3:
        return float("nan"), float("nan")
    by_cohort = {c: subset[subset["cohort"] == c][column].to_numpy() for c in cohorts}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        pooled = np.concatenate(
            [by_cohort[c] for c in rng.choice(cohorts, size=len(cohorts), replace=True)])
        if pooled.size:
            values.append(pooled.mean())
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def tier_table(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    records = []
    for tier in (*CLAIM_TIERS, "INSUFFICIENT", "NOT DETECTED"):
        subset = rows[rows["tier"] == tier]
        if subset.empty:
            continue
        low, high = cluster_bootstrap(rows, column, tier)
        records.append({
            "tier": tier,
            "n_taxa": int(len(subset)),
            "n_cohorts": int(subset["cohort"].nunique()),
            "replication_rate": float(subset[column].mean()),
            "ci_low": low,
            "ci_high": high,
        })
    return pd.DataFrame(records)


def leave_one_cohort_out(rows: pd.DataFrame, column: str) -> list:
    """Drop each cohort in turn. A conclusion that rests on one cohort is not a conclusion.

    Tier prevalence is extremely uneven — a single cohort with a large true effect can
    supply most of the ROBUST calls in the whole experiment — so the rate is recomputed
    without each cohort that contributes any.
    """
    records = []
    contributors = sorted(rows.loc[rows["tier"] == "ROBUST", "cohort"].unique())
    for cohort in contributors:
        kept = rows[rows["cohort"] != cohort]
        block = {"dropped": cohort}
        for tier in CLAIM_TIERS:
            subset = kept[kept["tier"] == tier]
            block[tier] = float(subset[column].mean()) if len(subset) else float("nan")
            block[f"n_{tier}"] = int(len(subset))
        block["auc"] = discrimination(kept, column).get("auc", float("nan"))
        records.append(block)
    return records


def discrimination(rows: pd.DataFrame, column: str) -> dict:
    """AUC of the tier ordering against replication, plus a trend test and risk ratio."""
    subset = rows[rows["tier"].isin(CLAIM_TIERS)].copy()
    if subset.empty:
        return {}
    order = {tier: len(CLAIM_TIERS) - 1 - i for i, tier in enumerate(CLAIM_TIERS)}
    scores = subset["tier"].map(order).to_numpy(dtype=float)
    outcome = subset[column].to_numpy().astype(bool)
    if outcome.all() or not outcome.any():
        return {"n": int(len(subset)), "auc": float("nan"),
                "note": "every taxon has the same outcome; AUC undefined"}

    positive, negative = scores[outcome], scores[~outcome]
    result = stats.mannwhitneyu(positive, negative, alternative="two-sided")
    trend = stats.spearmanr(scores, outcome.astype(float))
    robust = subset[subset["tier"] == "ROBUST"][column].to_numpy().astype(bool)
    rest = subset[subset["tier"] != "ROBUST"][column].to_numpy().astype(bool)
    risk_ratio = (float(robust.mean() / rest.mean())
                  if robust.size and rest.size and rest.mean() > 0 else float("nan"))
    return {
        "n": int(len(subset)),
        "n_replicated": int(outcome.sum()),
        "auc": float(result.statistic / (positive.size * negative.size)),
        "auc_p": float(result.pvalue),
        "trend_rho": float(trend.statistic),
        "trend_p": float(trend.pvalue),
        "robust_rate": float(robust.mean()) if robust.size else float("nan"),
        "other_rate": float(rest.mean()) if rest.size else float("nan"),
        "risk_ratio": risk_ratio,
    }


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cohorts", type=int, default=0, help="0 = every eligible")
    parser.add_argument("--repeats", type=int, default=3,
                        help="random 50/50 splits per cohort")
    parser.add_argument("--min-per-group", type=int, default=40,
                        help="smallest group a cohort must have to be split at all")
    parser.add_argument("--mode", default="quick", choices=("quick", "full"))
    parser.add_argument("--no-null", action="store_true")
    parser.add_argument("--reanalyse", action="store_true",
                        help="recompute the statistics from the saved per-taxon rows "
                             "instead of rerunning every multiverse")
    args = parser.parse_args()

    if args.reanalyse:
        if not os.path.exists(ROWS_OUTPUT):
            print(f"SKIPPED — {os.path.relpath(ROWS_OUTPUT, ROOT)} does not exist; "
                  "run without --reanalyse first.")
            return 0
        saved = pd.read_csv(ROWS_OUTPUT)
        rows = saved[~saved["permuted"]].reset_index(drop=True)
        nulls = saved[saved["permuted"]].reset_index(drop=True)
        print("Reanalysing the saved per-taxon rows; no multiverse was rerun.")
        return analyse(rows, nulls, {
            "design": "stratified 50/50 discovery/validation splits of published cohorts",
            "mode": args.mode, "reanalysed": True,
            "n_splits": int(rows.groupby(["cohort", "seed"]).ngroups),
            "n_cohorts": int(rows["cohort"].nunique()),
            "n_observations": int(len(rows)),
            "reference_spec": REFERENCE_SPEC,
            "min_specs_for_tier": MIN_SPECS_FOR_TIER,
            "definitions": {},
        })

    index_path = os.path.join(EXPORT, "index.csv")
    if not os.path.exists(index_path):
        print("SKIPPED — Pelto cohorts not exported. Run:")
        print("  python tests/reference/pelto_fetch.py")
        print("  Rscript tests/reference/pelto_export.R "
              "data/pelto/data_171023.rds data/pelto/export")
        return 0

    index = pd.read_csv(index_path)
    whole = index[index["kind"] == "whole"].copy()
    whole["min_group"] = whole[["n_case", "n_control"]].min(axis=1)
    eligible = whole[whole["min_group"] >= args.min_per_group].sort_values(
        "n_samples", ascending=False)
    if args.max_cohorts:
        eligible = eligible.head(args.max_cohorts)

    print("Do the SPEC 16.2 robustness tiers predict out-of-sample replication?")
    print(f"{len(eligible)} cohorts with at least {args.min_per_group} samples in the "
          f"smaller group, {args.repeats} stratified 50/50 splits each, "
          f"{'Quick' if args.mode == 'quick' else 'Full'} mode.")
    print("Tiers come from the discovery half; the held-out half is scored tier-blind.")
    print()

    started = time.perf_counter()
    frames, null_frames = [], []
    for i, (_, row) in enumerate(eligible.iterrows()):
        try:
            counts, metadata = read_cohort(row["slug"])
        except Exception as exc:                        # noqa: BLE001 — report, continue
            print(f"  [{i + 1}/{len(eligible)}] {row['study']}: unreadable — {exc}")
            continue
        for repeat in range(args.repeats):
            seed = 101 + repeat
            try:
                frame = one_split(counts, metadata, row, seed, mode=args.mode)
            except Exception as exc:                    # noqa: BLE001
                print(f"  [{i + 1}/{len(eligible)}] {row['study']} seed {seed}: "
                      f"SKIPPED — {type(exc).__name__}: {exc}")
                continue
            if frame.empty:
                continue
            frames.append(frame)
            if not args.no_null:
                with contextlib.suppress(Exception):
                    null_frames.append(one_split(counts, metadata, row, seed,
                                                 mode=args.mode,
                                                 permute=7000 + seed + i))
            tiers = frame["tier"].value_counts()
            print(f"  [{i + 1}/{len(eligible)}] {row['study']:<24} seed {seed}  "
                  f"n={frame['n_discovery'].iloc[0]:>3}/{frame['n_validation'].iloc[0]:<3} "
                  f"{len(frame):>4} taxa  ROBUST {int(tiers.get('ROBUST', 0)):>3}  "
                  f"CONDITIONAL {int(tiers.get('CONDITIONAL', 0)):>3}  "
                  f"replicated {frame['replicated_majority'].mean():.0%}")

    if not frames:
        print()
        print("No split produced a result.")
        return 1

    rows = pd.concat(frames, ignore_index=True)
    nulls = pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame()
    elapsed = time.perf_counter() - started

    os.makedirs(os.path.dirname(ROWS_OUTPUT), exist_ok=True)
    (pd.concat([rows, nulls], ignore_index=True) if len(nulls) else rows).to_csv(
        ROWS_OUTPUT, index=False, compression="gzip")

    print()
    print(f"{len(rows):,} taxon observations, {rows['cohort'].nunique()} cohorts, "
          f"{len(frames)} splits, {elapsed:.0f}s")
    print()

    payload = {
        "design": "stratified 50/50 discovery/validation splits of published cohorts",
        "source": "Pelto et al., arXiv:2404.02691; Zenodo 10.5281/zenodo.15047338",
        "mode": args.mode,
        "min_per_group": args.min_per_group,
        "repeats": args.repeats,
        "n_splits": len(frames),
        "n_cohorts": int(rows["cohort"].nunique()),
        "n_observations": int(len(rows)),
        "reference_spec": REFERENCE_SPEC,
        "min_specs_for_tier": MIN_SPECS_FOR_TIER,
        "definitions": {},
    }
    return analyse(rows, nulls, payload)





def analyse(rows: pd.DataFrame, nulls: pd.DataFrame, payload: dict) -> int:
    """Statistics and verdict, separated from execution so they can be rerun."""

    for column, label in (("replicated_majority", "majority of held-out specifications"),
                          ("replicated_reference", "one pre-specified held-out analysis")):
        print(f"Replication defined as: {label}")
        table = tier_table(rows, column)
        print(f"  {'tier':<14}{'taxa':>7}{'cohorts':>9}{'replicated':>12}{'95% CI':>18}")
        for _, r in table.iterrows():
            ci = (f"{r['ci_low']:.0%} - {r['ci_high']:.0%}"
                  if np.isfinite(r["ci_low"]) else "n/a")
            print(f"  {r['tier']:<14}{int(r['n_taxa']):>7}{int(r['n_cohorts']):>9}"
                  f"{r['replication_rate']:>11.0%}{ci:>18}")

        block = discrimination(rows, column)
        if block and np.isfinite(block.get("auc", np.nan)):
            report("discrimination (AUC of the tier ordering)",
                   f"{block['auc']:.3f}  (0.5 = the tiers carry no information)")
            report("trend across ordered tiers",
                   f"Spearman rho = {block['trend_rho']:+.3f}, p = {block['trend_p']:.2g}")
            report("ROBUST vs every other claim tier",
                   f"{block['robust_rate']:.0%} vs {block['other_rate']:.0%}, "
                   f"risk ratio {block['risk_ratio']:.2f}")
        null_rate = float(nulls[column].mean()) if len(nulls) else None
        if null_rate is not None:
            null_table = tier_table(nulls, column)
            null_robust = null_table.loc[null_table["tier"] == "ROBUST",
                                         "replication_rate"]
            extra = (f", {float(null_robust.iloc[0]):.0%} among ROBUST"
                     if len(null_robust) else "")
            report("label-permuted null", f"{null_rate:.1%} overall{extra}")

        sensitivity = leave_one_cohort_out(rows, column)
        if sensitivity:
            report("leave-one-cohort-out (cohorts supplying any ROBUST call)", "")
            for entry in sensitivity:
                rates = "  ".join(
                    f"{t[:4]} {entry[t]:.0%}" for t in CLAIM_TIERS
                    if np.isfinite(entry.get(t, np.nan)))
                print(f"        without {entry['dropped']:<22} "
                      f"n(ROBUST)={entry['n_ROBUST']:>3}  {rates}  "
                      f"AUC {entry['auc']:.3f}")

        payload["definitions"][column] = {
            "label": label,
            "by_tier": table.to_dict(orient="records"),
            "discrimination": block,
            "null_rate": null_rate,
            "leave_one_cohort_out": sensitivity,
        }
        print()

    primary = payload["definitions"]["replicated_majority"]
    by_tier = {r["tier"]: r for r in primary["by_tier"]}
    ordered = [by_tier[t]["replication_rate"] for t in CLAIM_TIERS if t in by_tier]
    auc = primary["discrimination"].get("auc", float("nan"))

    print("Verdict")
    check("the tier ordering carries information about replication",
          np.isfinite(auc) and auc > 0.55, f"AUC = {auc:.3f}")
    monotone = all(a >= b - 1e-9 for a, b in zip(ordered, ordered[1:], strict=False))
    check("replication decreases down the tiers",
          len(ordered) >= 3 and monotone, " > ".join(f"{r:.0%}" for r in ordered))
    robust_rate = by_tier.get("ROBUST", {}).get("replication_rate", float("nan"))
    null_mean = float(nulls["replicated_majority"].mean()) if len(nulls) else float("nan")
    check("ROBUST clears the label-permuted null",
          bool(len(nulls)) and np.isfinite(robust_rate) and robust_rate > null_mean * 1.5,
          f"{robust_rate:.0%} vs {null_mean:.0%} null" if len(nulls)
          else "null benchmark not run")

    payload["verdict"] = {"failures": list(FAILURES)}
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print()
    print(f"wrote {os.path.relpath(OUTPUT, ROOT)} and "
          f"{os.path.relpath(ROWS_OUTPUT, ROOT)}")
    print()
    print("FAILED: " + ", ".join(FAILURES) if FAILURES else "All checks passed.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
