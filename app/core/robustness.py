"""Robustness metrics and tiers — SPEC §15, §16.

The denominators are the part that silently corrupts everything if you get them wrong:
`frac_significant` and `sign_consistency` divide by `n_specs_tested`, never by the run's
total specification count.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import ordinal

TIERS = {
    "ROBUST": {"emoji": "🟢", "class": "robust",
               "blurb": "significant in at least 80% of specifications, "
                        "with a consistent direction"},
    "CONDITIONAL": {"emoji": "🟡", "class": "conditional",
                    "blurb": "significant in 30-80% of specifications; the answer "
                             "depends on the analytical choice"},
    "FRAGILE": {"emoji": "🟠", "class": "fragile",
                "blurb": "significant in fewer than 30% of specifications"},
    "UNSTABLE": {"emoji": "🔴", "class": "unstable",
                 "blurb": "the direction of effect is not determined by the data"},
    "INSUFFICIENT": {"emoji": "⚪", "class": "insufficient",
                     "blurb": "testable in fewer than 10 specifications"},
    "NOT DETECTED": {"emoji": "⚫", "class": "undetected",
                     "blurb": "never reached significance under any defensible specification"},
}

TIER_ORDER = ["ROBUST", "CONDITIONAL", "FRAGILE", "UNSTABLE", "INSUFFICIENT", "NOT DETECTED"]

MIN_SPECS_FOR_TIER = 10  # §15


def assign_tier(n_specs_tested: int, frac_significant: float, sign_consistency: float) -> str:
    """SPEC §16.2, evaluated in order, first match wins.

    Two gap-fillers are needed because the four published criteria do not partition the
    space; both are logged in SPEC §24:
      * frac_significant == 0 with a consistent sign matches no tier -> NOT DETECTED;
      * sign_consistency in [0.80, 0.95) with frac_significant >= 0.30 matches no tier
        -> CONDITIONAL (significant often, direction not fully pinned down).
    """
    if n_specs_tested < MIN_SPECS_FOR_TIER:
        return "INSUFFICIENT"
    if not (np.isfinite(frac_significant) and np.isfinite(sign_consistency)):
        # NaN compares false against every threshold, so an unguarded cascade falls
        # through to its last branch and hands undefined evidence a named tier. Since
        # each tier now carries a measured replication rate (90% for ROBUST, 51% for
        # CONDITIONAL — tests/reference/tier_validation.py), that would attach a claim
        # to a taxon nothing is known about. INSUFFICIENT is the honest answer.
        return "INSUFFICIENT"
    if frac_significant >= 0.80 and sign_consistency >= 0.95:
        return "ROBUST"
    if 0.30 <= frac_significant < 0.80 and sign_consistency >= 0.95:
        return "CONDITIONAL"
    if 0.0 < frac_significant < 0.30 and sign_consistency >= 0.80:
        return "FRAGILE"
    if sign_consistency < 0.80:
        return "UNSTABLE"
    if frac_significant <= 0.0:
        return "NOT DETECTED"
    return "CONDITIONAL"


@dataclass
class RobustnessSummary:
    table: pd.DataFrame
    spec_summary: pd.DataFrame
    tier_counts: dict
    n_specs_total: int
    n_specs_skipped: int = 0
    declared_percentile: float = float("nan")
    declared_rank: int = -1
    declared_n_significant: int = -1


def _sign_consistency(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    positive = int((values > 0).sum())
    negative = int((values < 0).sum())
    return max(positive, negative) / values.size


def compute_robustness(run) -> RobustnessSummary:
    """Per-taxon metrics (§16.2) plus the per-specification summary the curve needs."""
    long = run.long

    # A specification whose matrix was unusable (rarefaction can drop a group below the
    # minimum) produces no rows. Counting it as a denominator would understate every
    # taxon's frac_tested, so eligibility is measured over the specifications that
    # actually ran, and the shortfall is reported.
    ran = set(long["spec_id"].unique().tolist())
    n_specs_total = len(ran)
    n_specs_skipped = len(run.specs) - n_specs_total

    spec_rank = np.array([s.rank for s in run.specs])
    ran_mask = np.array([i in ran for i in range(len(run.specs))])
    eligible_by_rank = {
        rank: int(((spec_rank == rank) & ran_mask).sum()) for rank in set(spec_rank)
    }

    grouped = long.groupby("taxon", sort=True)
    n_tested = grouped.size()
    n_significant = grouped["significant"].sum()
    n_nominal = grouped["p_raw"].apply(lambda s: int((s < 0.05).sum()))
    median_effect = grouped["effect_h"].median()
    q1 = grouped["effect_h"].quantile(0.25)
    q3 = grouped["effect_h"].quantile(0.75)
    min_p = grouped["p_adj"].min()
    sign_consistency = grouped["effect_h"].apply(lambda s: _sign_consistency(s.to_numpy()))

    taxa_idx = n_tested.index.to_numpy()
    names = [run.taxa_names[i] for i in taxa_idx]
    display = [run.taxa_display[i] for i in taxa_idx]
    ranks = [run.taxa_rank[i] for i in taxa_idx]
    eligible = np.array([eligible_by_rank.get(r, n_specs_total) for r in ranks], dtype=float)

    frame = pd.DataFrame(
        {
            "taxon_id": taxa_idx,
            "taxon": names,
            "label": display,
            "rank": ranks,
            "n_specs_tested": n_tested.to_numpy(),
            "n_specs_eligible": eligible.astype(int),
            "frac_tested": n_tested.to_numpy() / np.maximum(eligible, 1),
            "frac_significant": n_significant.to_numpy() / np.maximum(n_tested.to_numpy(), 1),
            "frac_nominal": n_nominal.to_numpy() / np.maximum(n_tested.to_numpy(), 1),
            "sign_consistency": sign_consistency.to_numpy(),
            "median_effect": median_effect.to_numpy(),
            "iqr_low": q1.to_numpy(),
            "iqr_high": q3.to_numpy(),
            "min_p_adjusted": min_p.to_numpy(),
        }
    )
    frame["robustness_tier"] = [
        assign_tier(int(n), float(f), float(s))
        for n, f, s in zip(frame["n_specs_tested"], frame["frac_significant"],
                           frame["sign_consistency"], strict=True)
    ]
    frame["direction"] = np.where(
        frame["median_effect"] > 0, f"higher in {run.group_labels[1]}",
        np.where(frame["median_effect"] < 0, f"higher in {run.group_labels[0]}", "no difference"),
    )
    frame = frame.sort_values(
        by=["robustness_tier", "frac_significant", "n_specs_tested"],
        key=lambda col: col.map(TIER_ORDER.index) if col.name == "robustness_tier" else col,
        ascending=[True, False, False],
    ).reset_index(drop=True)

    tier_counts = {tier: int((frame["robustness_tier"] == tier).sum()) for tier in TIER_ORDER}

    # Per-specification summary: how many taxa each specification calls significant.
    spec_grouped = long.groupby("spec_id", sort=True)
    spec_summary = pd.DataFrame(
        {
            "spec_id": spec_grouped.size().index.to_numpy(),
            "n_taxa_tested": spec_grouped.size().to_numpy(),
            "n_significant": spec_grouped["significant"].sum().to_numpy(),
        }
    )
    # Keep a row for every specification, so exports and counts line up with the grid.
    spec_summary = run.specs_frame.merge(spec_summary, on="spec_id", how="left")
    spec_summary[["n_taxa_tested", "n_significant"]] = (
        spec_summary[["n_taxa_tested", "n_significant"]].fillna(0).astype(int)
    )
    spec_summary["ran"] = spec_summary["spec_id"].isin(ran)

    summary = RobustnessSummary(
        table=frame,
        spec_summary=spec_summary,
        tier_counts=tier_counts,
        n_specs_total=n_specs_total,
        n_specs_skipped=n_specs_skipped,
    )

    if run.declared_spec_id >= 0:
        row = spec_summary.loc[spec_summary["spec_id"] == run.declared_spec_id]
        if len(row):
            declared_n = int(row["n_significant"].iloc[0])
            values = spec_summary.loc[spec_summary["ran"], "n_significant"].to_numpy()
            rank = int((values < declared_n).sum() + 0.5 * (values == declared_n).sum())
            summary.declared_n_significant = declared_n
            summary.declared_rank = rank
            summary.declared_percentile = 100.0 * rank / max(1, len(values))

    return summary


def locate_declared(run, taxon_id: int) -> dict:
    """SPEC §16.4 — where the user's own pipeline sits for one taxon."""
    if run.declared_spec_id < 0:
        return {}
    rows = run.long[run.long["taxon"] == taxon_id]
    if rows.empty:
        return {}
    declared = rows[rows["spec_id"] == run.declared_spec_id]
    if declared.empty:
        return {
            "tested": False,
            "note": "Your declared pipeline filters this taxon out, so it has no result there.",
        }
    p_declared = float(declared["p_raw"].iloc[0])
    scores = -np.log10(np.clip(rows["p_raw"].to_numpy(dtype=float), 1e-300, 1.0))
    score = -np.log10(max(p_declared, 1e-300))
    n = scores.size
    rank = int((scores < score).sum() + 0.5 * (scores == score).sum())
    return {
        "tested": True,
        "p_raw": p_declared,
        "p_adjusted": float(declared["p_adj"].iloc[0]),
        "significant": bool(declared["significant"].iloc[0]),
        "effect_h": float(declared["effect_h"].iloc[0]),
        "rank": rank,
        "n_specs_tested": int(n),
        "percentile": 100.0 * rank / max(1, n),
    }


def verdict_sentence(run, summary: RobustnessSummary) -> str:
    """SPEC §16.1 — one line, at the top of the page."""
    counts = summary.tier_counts
    detected = sum(counts[t] for t in ("ROBUST", "CONDITIONAL", "FRAGILE", "UNSTABLE"))
    unit = "taxa" if detected != 1 else "taxon"
    parts = [
        f"Of the {detected} {unit} significant in at least one specification, "
        f"{counts['ROBUST']} are ROBUST, {counts['CONDITIONAL']} CONDITIONAL, "
        f"{counts['FRAGILE']} FRAGILE and {counts['UNSTABLE']} UNSTABLE"
    ]
    if counts["INSUFFICIENT"]:
        parts.append(f"{counts['INSUFFICIENT']} could not be tiered (tested in <10 specifications)")
    sentence = "; ".join(parts) + f", across {summary.n_specs_total:,} valid specifications."

    if run.declared_spec_id >= 0 and np.isfinite(summary.declared_percentile):
        sentence += (
            f" Your reported pipeline sits at the {ordinal(summary.declared_percentile)} "
            f"percentile of significance — it calls {summary.declared_n_significant} taxa "
            f"significant, more than {summary.declared_percentile:.0f}% of defensible "
            f"alternatives."
        )
    return sentence
