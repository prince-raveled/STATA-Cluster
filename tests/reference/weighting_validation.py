"""V8 — does neutral weighting change the labels, and do they still predict replication?

Pre-registered in docs/MICROVERSE_V3_PLAN.md §33:

    Question   Does neutral weighting change labels, and does it keep within-study
               predictive value?
    Data       25 Pelto cohorts, 75 splits (the v2 design)
    Metrics    held-out replication AUC per scheme; share of taxa changing label;
               Kendall tau between schemes
    Success    decision_tree AUC non-inferior to uniform, margin 0.03
    If it fails
               report it. The default stays decision_tree, which was chosen on
               principle, not on AUC: choosing the scheme with the best AUC would
               itself be a fork.

Design. The v2 tier validation (tests/reference/tier_validation.py) is reused exactly:
the same cohorts, the same stratified 50/50 splits and seeds, the same eligibility floor
and the same two tier-blind definitions of replication on the held-out half. What
changes is only how the discovery half is labelled:

  * discovery, rule set v3, labelled under `uniform`, `flat_tree` and `decision_tree`
    (`app/core/weights.py`) — the comparison the success criterion is about;
  * discovery, rule set v2, one vote per specification — the baseline that
    tier_validation.py measured, run again here so every label is scored on the same
    held-out results.

The held-out half stays on rule set v2. Its `reference` definition is one
pre-specified analysis, unrarefied raw counts with Wilcoxon, which is exactly what rule
R8 prunes; running it under v3 would change the outcome being predicted along with the
labels predicting it.

Operational choices the plan leaves open, fixed here before any data were seen (and
logged in docs/V3_DEVIATIONS.md):

  * Primary comparison: decision_tree against uniform, both under rule set v3, so the
    difference is the weighting alone. The v2 baseline is reported beside them.
  * Non-inferiority: the lower limit of the 95% paired cluster-bootstrap interval for
    AUC(decision_tree) - AUC(uniform), resampling cohorts, must exceed -0.03.
  * Replication is scored with each scheme's own discovery direction (its weighted
    median effect), because that is the direction the label is attached to.
  * Kendall tau-b between schemes on the tier order NOT DETECTED < UNSTABLE < FRAGILE <
    CONDITIONAL < ROBUST, over observations tiered under both (INSUFFICIENT dropped);
    and on frac_significant.

    python tests/reference/weighting_validation.py
    python tests/reference/weighting_validation.py --max-cohorts 4 --repeats 1   # smoke

Only the full pre-registered run writes docs/weighting_validation.json (and the copy in
app/core/records/ that the site reads, because docs/ is not deployed). A smaller run
writes under data/, which is git-ignored, so a smoke test can never overwrite the record.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import tier_validation as tv  # noqa: E402

from app.core.evidence import V8_PREREGISTRATION  # noqa: E402
from app.core.robustness import compute_robustness  # noqa: E402
from app.core.runner import run_multiverse  # noqa: E402

OUTPUT = os.path.join(ROOT, "docs", "weighting_validation.json")
ROWS_OUTPUT = os.path.join(ROOT, "docs", "weighting_validation_rows.csv.gz")
APP_COPY = os.path.join(ROOT, "app", "core", "records", "weighting_validation.json")
SMOKE_OUTPUT = os.path.join(ROOT, "data", "weighting_validation_smoke.json")
SMOKE_ROWS = os.path.join(ROOT, "data", "weighting_validation_smoke_rows.csv.gz")
TIER_RECORD = os.path.join(ROOT, "docs", "tier_validation.json")

#: (label, rule set, weighting scheme). The label is the key in every output.
LABELLINGS = (
    ("v3_uniform", "v3", "uniform"),
    ("v3_flat_tree", "v3", "flat_tree"),
    ("v3_decision_tree", "v3", "decision_tree"),
    ("v2_uniform", "v2", "uniform"),
)
PRIMARY = ("v3_decision_tree", "v3_uniform")
#: The pre-registered margin, read from the one place the site also states it.
MARGIN = V8_PREREGISTRATION["margin"]
BOOTSTRAP_DRAWS = 2000

#: The pre-registered design: what makes a run the record rather than a smoke test.
REGISTERED = {"max_cohorts": 0, "repeats": 3, "min_per_group": 40, "mode": "quick"}

TIER_SCORE = {"NOT DETECTED": 0, "UNSTABLE": 1, "FRAGILE": 2, "CONDITIONAL": 3,
              "ROBUST": 4}
OUTCOMES = ("replicated_majority", "replicated_reference")

FAILURES: list = []


def _shown(path) -> str:
    """A path for the console: relative to the repository where it can be, as given
    otherwise (Windows has no relative path between two drives)."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return str(path)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def report(name: str, detail: str) -> None:
    print(f"  [ -- ] {name} — {detail}")


# --- one split -------------------------------------------------------------------
def discovery_tables(dataset) -> dict:
    """label -> per-taxon table (indexed by taxon name) for every labelling."""
    runs = {"v3": run_multiverse(dataset, mode="quick", ruleset="v3"),
            "v2": run_multiverse(dataset, mode="quick", ruleset="v2")}
    tables = {}
    for label, ruleset, scheme in LABELLINGS:
        run = runs[ruleset]
        summary = (compute_robustness(run) if ruleset == "v2"
                   else compute_robustness(run, scheme=scheme))
        tables[label] = summary.table.set_index("taxon")
    return tables


def score(tables: dict, held_out: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """One row per taxon present in every labelling and in the held-out half."""
    shared = set(held_out.index)
    for table in tables.values():
        shared &= set(table.index)
    shared = sorted(shared)
    if not shared:
        return pd.DataFrame()
    held = held_out.loc[shared]
    frame = pd.DataFrame({**meta, "taxon": shared,
                          "validation_effect": held["median_effect"].to_numpy(),
                          "validation_frac_nominal": held["frac_nominal"].to_numpy()})
    for label, table in tables.items():
        table = table.loc[shared]
        effect = table["median_effect"].to_numpy(dtype=float)
        same_sign = (np.sign(effect) == np.sign(held["median_effect"].to_numpy())) & (
            effect != 0.0)
        frame[f"tier_{label}"] = table["robustness_tier"].to_numpy()
        frame[f"frac_significant_{label}"] = table["frac_significant"].to_numpy()
        frame[f"sign_consistency_{label}"] = table["sign_consistency"].to_numpy()
        frame[f"discovery_effect_{label}"] = effect
        frame[f"replicated_majority_{label}"] = (
            (held["frac_nominal"].to_numpy() > 0.5) & same_sign)
        frame[f"replicated_reference_{label}"] = (
            held["ref_significant"].fillna(0).to_numpy().astype(bool) & same_sign)
    return frame


def one_split(counts, metadata, row, seed: int, null_seed: int = 0):
    """Label the discovery half every way, score each on the held-out half.

    Returns (rows, null rows). The discovery half is run once and scored twice: against
    the real held-out labels and, when `null_seed` is set, against permuted ones.
    """
    first, second = tv.stratified_halves(metadata, seed)
    tables = discovery_tables(tv.dataset_from(counts, metadata, first))
    meta = {"cohort": row["study"], "type": row["type"], "seed": seed,
            "n_discovery": int(len(first)), "n_validation": int(len(second))}

    held = tv.evaluate_validation_half(run_multiverse(
        tv.dataset_from(counts, metadata, second), mode="quick", ruleset="v2"))
    rows = score(tables, held, {**meta, "permuted": False})

    nulls = pd.DataFrame()
    if null_seed:
        permuted = tv.evaluate_validation_half(run_multiverse(
            tv.dataset_from(counts, metadata, second, permute_groups=null_seed),
            mode="quick", ruleset="v2"))
        nulls = score(tables, permuted, {**meta, "permuted": True})
    return rows, nulls


# --- analysis ----------------------------------------------------------------------
def view(rows: pd.DataFrame, label: str) -> pd.DataFrame:
    """One labelling in the column layout tier_validation.py's statistics expect."""
    out = rows[["cohort", "seed", "taxon"]].copy()
    out["tier"] = rows[f"tier_{label}"].to_numpy()
    for outcome in OUTCOMES:
        out[outcome] = rows[f"{outcome}_{label}"].to_numpy().astype(bool)
    return out


def auc(frame: pd.DataFrame, outcome: str) -> float:
    """AUC of the claim-tier ordering against replication (tier_validation's definition)."""
    subset = frame[frame["tier"].isin(tv.CLAIM_TIERS)]
    order = {tier: len(tv.CLAIM_TIERS) - 1 - i for i, tier in enumerate(tv.CLAIM_TIERS)}
    scores = subset["tier"].map(order).to_numpy(dtype=float)
    positive = subset[outcome].to_numpy().astype(bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def paired_auc_difference(rows: pd.DataFrame, first: str, second: str, outcome: str,
                          draws: int = BOOTSTRAP_DRAWS, seed: int = 17) -> dict:
    """AUC(first) - AUC(second) with a paired cluster bootstrap over cohorts.

    Paired: each draw resamples cohorts once and recomputes both AUCs on the same rows,
    so what is shared between the labellings cancels in the difference.
    """
    a, b = view(rows, first), view(rows, second)
    observed = auc(a, outcome) - auc(b, outcome)
    cohorts = rows["cohort"].unique()
    positions = {c: np.flatnonzero(rows["cohort"].to_numpy() == c) for c in cohorts}
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        picked = np.concatenate([positions[c] for c in rng.choice(cohorts, len(cohorts))])
        value = auc(a.iloc[picked], outcome) - auc(b.iloc[picked], outcome)
        if np.isfinite(value):
            differences.append(value)
    if len(differences) < draws // 2:
        return {"difference": observed, "ci": [float("nan"), float("nan")],
                "draws": len(differences)}
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"difference": observed, "ci": [float(low), float(high)],
            "draws": len(differences)}


def label_changes(rows: pd.DataFrame, reference: str) -> dict:
    """Share of observations whose tier differs from `reference`'s, and the transitions."""
    out = {}
    for label, _, _ in LABELLINGS:
        if label == reference:
            continue
        before = rows[f"tier_{reference}"]
        after = rows[f"tier_{label}"]
        changed = before != after
        transitions = (pd.crosstab(before, after).stack().rename("n").reset_index())
        transitions = transitions[(transitions["n"] > 0)
                                  & (transitions.iloc[:, 0] != transitions.iloc[:, 1])]
        out[label] = {
            "share_changed": float(changed.mean()),
            "n_changed": int(changed.sum()),
            "n": int(len(rows)),
            "transitions": [
                {"from": str(r.iloc[0]), "to": str(r.iloc[1]), "n": int(r["n"])}
                for _, r in transitions.sort_values("n", ascending=False).iterrows()],
        }
    return out


def kendall(rows: pd.DataFrame) -> dict:
    """Kendall tau-b between every pair of labellings, on tier order and frac_significant."""
    out = {}
    labels = [label for label, _, _ in LABELLINGS]
    for i, first in enumerate(labels):
        for second in labels[i + 1:]:
            a = rows[f"tier_{first}"].map(TIER_SCORE)
            b = rows[f"tier_{second}"].map(TIER_SCORE)
            both = a.notna() & b.notna()
            tier_tau = stats.kendalltau(a[both], b[both], variant="b")
            fa = rows[f"frac_significant_{first}"].to_numpy(dtype=float)
            fb = rows[f"frac_significant_{second}"].to_numpy(dtype=float)
            finite = np.isfinite(fa) & np.isfinite(fb)
            frac_tau = stats.kendalltau(fa[finite], fb[finite], variant="b")
            out[f"{first}~{second}"] = {
                "tier_tau_b": float(tier_tau.statistic), "tier_n": int(both.sum()),
                "frac_significant_tau_b": float(frac_tau.statistic),
                "frac_significant_n": int(finite.sum()),
            }
    return out


def commit() -> dict:
    """The code the numbers came from (plan §33: every record names its commit)."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    return {"sha": git("rev-parse", "HEAD") or None,
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def analyse(rows: pd.DataFrame, nulls: pd.DataFrame, payload: dict) -> dict:
    """Every statistic in the record, from the saved per-taxon rows."""
    payload["labellings"] = {}
    for label, ruleset, scheme in LABELLINGS:
        frame = view(rows, label)
        block = {"ruleset": ruleset, "scheme": scheme, "definitions": {}}
        for outcome in OUTCOMES:
            table = tv.tier_table(frame, outcome)
            null_rate = (float(view(nulls, label)[outcome].mean()) if len(nulls)
                         else None)
            block["definitions"][outcome] = {
                "by_tier": table.to_dict(orient="records"),
                "discrimination": tv.discrimination(frame, outcome),
                "null_rate": null_rate,
            }
        tiers = frame["tier"].value_counts()
        block["tier_counts"] = {t: int(tiers.get(t, 0)) for t in
                                (*tv.CLAIM_TIERS, "INSUFFICIENT", "NOT DETECTED")}
        payload["labellings"][label] = block

    payload["non_inferiority"] = {}
    for outcome in OUTCOMES:
        result = paired_auc_difference(rows, *PRIMARY, outcome)
        result["margin"] = -MARGIN
        low = result["ci"][0]
        result["non_inferior"] = bool(np.isfinite(low) and low > -MARGIN)
        payload["non_inferiority"][outcome] = result
    payload["against_v2_baseline"] = {
        outcome: paired_auc_difference(rows, "v3_decision_tree", "v2_uniform", outcome)
        for outcome in OUTCOMES}
    payload["label_changes"] = {"from_v3_uniform": label_changes(rows, "v3_uniform"),
                                "from_v2_uniform": label_changes(rows, "v2_uniform")}
    payload["kendall_tau"] = kendall(rows)
    return payload


def print_summary(payload: dict) -> None:
    for outcome in OUTCOMES:
        print(f"Replication defined as: {outcome.replace('replicated_', '')}")
        print(f"  {'labelling':<20}{'AUC':>8}   by tier (ROBUST / CONDITIONAL / FRAGILE / "
              "UNSTABLE)")
        for label, block in payload["labellings"].items():
            definition = block["definitions"][outcome]
            rates = {r["tier"]: r["replication_rate"] for r in definition["by_tier"]}
            text = " / ".join(f"{rates[t]:.0%}" if t in rates else "—"
                              for t in tv.CLAIM_TIERS)
            value = definition["discrimination"].get("auc", float("nan"))
            print(f"  {label:<20}{value:>8.3f}   {text}")
        result = payload["non_inferiority"][outcome]
        report("decision_tree - uniform (v3 rules)",
               f"{result['difference']:+.3f}, 95% CI {result['ci'][0]:+.3f} to "
               f"{result['ci'][1]:+.3f}, margin -{MARGIN}")
        print()
    for label, change in payload["label_changes"]["from_v3_uniform"].items():
        report(f"label changed, v3_uniform -> {label}",
               f"{change['share_changed']:.1%} ({change['n_changed']:,} of {change['n']:,})")
    for pair, tau in payload["kendall_tau"].items():
        report(f"Kendall tau-b {pair}",
               f"tiers {tau['tier_tau_b']:.3f}, frac_significant "
               f"{tau['frac_significant_tau_b']:.3f}")


def verdict(payload: dict, registered: bool) -> None:
    print()
    print("Verdict (pre-registered: decision_tree AUC non-inferior to uniform, "
          f"margin {MARGIN})")
    primary = payload["non_inferiority"]["replicated_majority"]
    check("decision_tree AUC is non-inferior to uniform (majority definition)",
          primary["non_inferior"],
          f"lower 95% limit {primary['ci'][0]:+.3f} against -{MARGIN}")
    secondary = payload["non_inferiority"]["replicated_reference"]
    report("the same on the reference definition (secondary)",
           f"lower 95% limit {secondary['ci'][0]:+.3f}: "
           f"{'non-inferior' if secondary['non_inferior'] else 'not shown non-inferior'}")

    # The v2 baseline is the tier_validation.py experiment rerun; on the registered
    # design it must reproduce that record, or the two are not measuring the same thing.
    if registered and os.path.exists(TIER_RECORD):
        with open(TIER_RECORD, encoding="utf-8") as handle:
            record = json.load(handle)
        expected = record["definitions"]["replicated_majority"]["discrimination"]["auc"]
        got = payload["labellings"]["v2_uniform"]["definitions"]["replicated_majority"][
            "discrimination"].get("auc", float("nan"))
        check("the v2 baseline reproduces docs/tier_validation.json",
              np.isfinite(got) and abs(got - expected) < 1e-9,
              f"AUC {got:.4f} against {expected:.4f}")
    else:
        report("v2 baseline against docs/tier_validation.json",
               "not compared: this is not the registered design")
    payload["verdict"] = {"failures": list(FAILURES)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cohorts", type=int, default=0, help="0 = every eligible")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-per-group", type=int, default=40)
    parser.add_argument("--no-null", action="store_true")
    parser.add_argument("--osf", default="",
                        help="OSF registration URL for the record (a human input)")
    parser.add_argument("--export", default=tv.EXPORT,
                        help="directory written by pelto_export.R")
    parser.add_argument("--reanalyse", action="store_true",
                        help="recompute every statistic from the saved per-taxon rows")
    args = parser.parse_args()

    registered = (args.max_cohorts == REGISTERED["max_cohorts"]
                  and args.repeats == REGISTERED["repeats"]
                  and args.min_per_group == REGISTERED["min_per_group"]
                  and args.export == tv.EXPORT)
    output, rows_output = (OUTPUT, ROWS_OUTPUT) if registered else (SMOKE_OUTPUT,
                                                                     SMOKE_ROWS)

    if args.reanalyse:
        if not os.path.exists(rows_output):
            print(f"SKIPPED — {_shown(rows_output)} does not exist; "
                  "run without --reanalyse first.")
            return 0
        saved = pd.read_csv(rows_output)
        rows = saved[~saved["permuted"]].reset_index(drop=True)
        nulls = saved[saved["permuted"]].reset_index(drop=True)
        payload = {"reanalysed": True}
    else:
        index_path = os.path.join(args.export, "index.csv")
        if not os.path.exists(index_path):
            print("SKIPPED — Pelto cohorts not exported. Run:")
            print("  python tests/reference/pelto_fetch.py")
            print("  Rscript tests/reference/pelto_export.R "
                  "data/pelto/data_171023.rds data/pelto/export")
            return 0
        tv.EXPORT = args.export
        index = pd.read_csv(index_path)
        whole = index[index["kind"] == "whole"].copy()
        whole["min_group"] = whole[["n_case", "n_control"]].min(axis=1)
        eligible = whole[whole["min_group"] >= args.min_per_group].sort_values(
            "n_samples", ascending=False)
        if args.max_cohorts:
            eligible = eligible.head(args.max_cohorts)

        print("V8: does neutral weighting change the labels, and keep their predictive "
              "value?")
        print(f"{len(eligible)} cohorts, {args.repeats} stratified 50/50 splits each; "
              "discovery labelled four ways, held-out half scored once under v2 rules.")
        print()
        started = time.perf_counter()
        frames, null_frames = [], []
        for i, (_, row) in enumerate(eligible.iterrows()):
            try:
                counts, metadata = tv.read_cohort(row["slug"])
            except Exception as exc:                    # noqa: BLE001 — report, continue
                print(f"  [{i + 1}/{len(eligible)}] {row['study']}: unreadable — {exc}")
                continue
            for repeat in range(args.repeats):
                seed = 101 + repeat          # tier_validation.py's seeds, so splits match
                try:
                    frame, null = one_split(
                        counts, metadata, row, seed,
                        null_seed=0 if args.no_null else 7000 + seed + i)
                except Exception as exc:                # noqa: BLE001
                    print(f"  [{i + 1}/{len(eligible)}] {row['study']} seed {seed}: "
                          f"SKIPPED — {type(exc).__name__}: {exc}")
                    continue
                if frame.empty:
                    continue
                frames.append(frame)
                if len(null):
                    null_frames.append(null)
                changed = (frame["tier_v3_decision_tree"] != frame["tier_v3_uniform"]).mean()
                print(f"  [{i + 1}/{len(eligible)}] {row['study']:<24} seed {seed}  "
                      f"{len(frame):>4} taxa  label changed by weighting {changed:.0%}")
        if not frames:
            print("\nNo split produced a result.")
            return 1
        rows = pd.concat(frames, ignore_index=True)
        nulls = pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame()
        os.makedirs(os.path.dirname(rows_output), exist_ok=True)
        (pd.concat([rows, nulls], ignore_index=True) if len(nulls) else rows).to_csv(
            rows_output, index=False, compression="gzip")
        payload = {"elapsed_seconds": round(time.perf_counter() - started, 1)}
        print(f"\n{len(rows):,} taxon observations, {rows['cohort'].nunique()} cohorts, "
              f"{len(frames)} splits\n")

    payload.update({
        "experiment": "V8",
        "script": "tests/reference/weighting_validation.py",
        "preregistration": {
            "plan": "docs/MICROVERSE_V3_PLAN.md §33, V8",
            "osf": args.osf or None,
            "success": f"decision_tree AUC non-inferior to uniform, margin {MARGIN}",
            "operationalised": "lower limit of the 95% paired cluster-bootstrap interval "
                               "(cohorts resampled) for AUC(v3 decision_tree) - AUC(v3 "
                               f"uniform) above -{MARGIN}, majority definition",
        },
        "code_commit": commit(),
        "registered_design": registered,
        "design": REGISTERED | {"max_cohorts": args.max_cohorts, "repeats": args.repeats,
                                "min_per_group": args.min_per_group},
        "source": ("Pelto et al., arXiv:2404.02691; Zenodo 10.5281/zenodo.15047338"
                   if args.export == tv.EXPORT else f"the export at {args.export}"),
        "held_out_ruleset": "v2",
        "reference_spec": tv.REFERENCE_SPEC,
        "n_splits": int(rows.groupby(["cohort", "seed"]).ngroups),
        "n_cohorts": int(rows["cohort"].nunique()),
        "n_observations": int(len(rows)),
    })
    payload = analyse(rows, nulls, payload)
    print_summary(payload)
    verdict(payload, registered)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, default=float)
        handle.write("\n")
    written = [output, rows_output]
    if registered:
        os.makedirs(os.path.dirname(APP_COPY), exist_ok=True)
        shutil.copyfile(output, APP_COPY)
        written.append(APP_COPY)
    print()
    print("wrote " + ", ".join(_shown(p) for p in written))
    print("FAILED: " + ", ".join(FAILURES) if FAILURES else "All checks passed.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
