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
    #: The weighting the tiers were assigned under (weights.SCHEMES). Class-level
    #: defaults, so a summary stored before v3 unpickles as what it was: uniform.
    scheme: str = "uniform"
    ruleset: str = "v2"
    #: {scheme: weights.GridComposition} and {scheme: taxa whose tier differs from the
    #: primary scheme's}, for the grid-composition panel (plan §26.2). None under v2.
    composition: object = None
    tier_changes: object = None

    @property
    def weighted(self) -> bool:
        return self.ruleset != "v2"


def _sign_consistency(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    positive = int((values > 0).sum())
    negative = int((values < 0).sum())
    return max(positive, negative) / values.size


def compute_robustness(run, scheme: str = None, custom: dict = None) -> RobustnessSummary:
    """Per-taxon metrics (§16.2) plus the per-specification summary the curve needs.

    A v2 run (ruleset "v2", which includes every run stored before v3) is summarised
    exactly as v2 did it, one vote per specification. A v3 run is summarised under
    every weighting scheme; its tiers come from `scheme` (default `decision_tree`,
    plan §26.2) and the table says which taxa another scheme would label differently.
    """
    long = run.long
    ruleset = getattr(run, "ruleset", "v2")
    legacy = ruleset == "v2"
    if legacy and scheme not in (None, "uniform"):
        raise ValueError("A v2 run is summarised with one vote per specification.")
    from .weights import DEFAULT_SCHEME
    scheme = "uniform" if legacy else (scheme or DEFAULT_SCHEME)

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
    # Keep a row for every taxon, so exports and counts line up with the input --
    # the same reason spec_summary keeps a row for every specification below.
    #
    # A taxon filtered out of *every* specification produces no rows in `long`, so the
    # groupby above never sees it and it used to vanish: no row, no tier, and no
    # mention anywhere. The counts then did not add up to the table the person
    # uploaded, and nothing said why. SPEC §16.2 is explicit that "a taxon with
    # n_specs_tested < 10 is reported as INSUFFICIENT, not tiered", and nought is
    # less than ten; assign_tier already answers INSUFFICIENT for it. The taxon only
    # had to reach assign_tier to be reported.
    absent = [i for i in range(len(run.taxa_names)) if i not in set(taxa_idx.tolist())]
    if absent:
        frame = pd.concat([frame, pd.DataFrame({
            "taxon_id": absent,
            "taxon": [run.taxa_names[i] for i in absent],
            "label": [run.taxa_display[i] for i in absent],
            "rank": [run.taxa_rank[i] for i in absent],
            "n_specs_tested": 0,
            "n_specs_eligible": [eligible_by_rank.get(run.taxa_rank[i], n_specs_total)
                                 for i in absent],
            "frac_tested": 0.0,
            "frac_significant": 0.0,
            "frac_nominal": 0.0,
            "sign_consistency": 0.0,
            "median_effect": np.nan,
            "iqr_low": np.nan,
            "iqr_high": np.nan,
            "min_p_adjusted": np.nan,
        })], ignore_index=True)

    extras = None
    if not legacy:
        frame, extras = _apply_weighting(run, frame, scheme, custom)

    frame["robustness_tier"] = [
        assign_tier(int(n), float(f), float(s))
        for n, f, s in zip(frame["n_specs_tested"], frame["frac_significant"],
                           frame["sign_consistency"], strict=True)
    ]
    if extras is not None:
        _stability_columns(frame, extras["schemes"])
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
    if extras is not None:
        spec_summary["weight"] = extras["weights"][scheme]
        for name, weights in extras["weights"].items():
            spec_summary[f"weight_{name}"] = weights

    summary = RobustnessSummary(
        table=frame,
        spec_summary=spec_summary,
        tier_counts=tier_counts,
        n_specs_total=n_specs_total,
        n_specs_skipped=n_specs_skipped,
        scheme=scheme,
        ruleset=ruleset,
    )
    if extras is not None:
        summary.composition = extras["composition"]
        summary.tier_changes = {
            other: int((frame[f"tier_{other}"] != frame["robustness_tier"]).sum())
            for other in extras["schemes"] if other != scheme}

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


def _apply_weighting(run, frame: pd.DataFrame, scheme: str, custom: dict):
    """Replace the v2 counts with the primary scheme's weighted metrics, and add the
    per-scheme columns the stability flag and the CSV need (plan §26.2)."""
    from .weighted import pseudocount_flip, weighting_table

    columns, weights, comps = weighting_table(run, primary=scheme, custom=custom)
    columns = columns.reindex(frame["taxon_id"].to_numpy())
    frame = frame.copy()
    # The v2 count stays, named for what it is, beside the weighted value that the tier
    # now uses.
    frame["frac_significant_unweighted"] = frame["frac_significant"]
    frame["frac_significant"] = columns[f"frac_significant_{scheme}"].to_numpy()
    frame["frac_nominal"] = columns[f"frac_nominal_{scheme}"].to_numpy()
    frame["sign_consistency"] = columns[f"sign_consistency_{scheme}"].to_numpy()
    if scheme != "uniform":
        frame["median_effect"] = columns["median_effect_w"].to_numpy()
        frame["iqr_low"] = columns["iqr_low_w"].to_numpy()
        frame["iqr_high"] = columns["iqr_high_w"].to_numpy()
    frame["n_eff_specs"] = columns[f"n_eff_{scheme}"].to_numpy()
    untested = frame["n_specs_tested"].to_numpy() == 0
    for column in ("frac_significant", "frac_nominal", "sign_consistency"):
        # v2 wrote 0.0 for a taxon nothing tested; keep that, so its tier is unchanged.
        frame.loc[untested, column] = 0.0

    for other in weights:
        frac = columns[f"frac_significant_{other}"].to_numpy()
        sign = columns[f"sign_consistency_{other}"].to_numpy()
        frame[f"tier_{other}"] = [
            assign_tier(int(n), 0.0 if u else float(f), 0.0 if u else float(c))
            for n, f, c, u in zip(frame["n_specs_tested"], frac, sign, untested,
                                  strict=True)]

    low = columns.get("median_effect_pc_x0.1")
    high = columns.get("median_effect_pc_x10")
    if low is not None and high is not None:
        frame["median_effect_pc_low"] = low.to_numpy()
        frame["median_effect_pc_high"] = high.to_numpy()
        frame["pseudocount_sign_flip"] = pseudocount_flip(
            frame["median_effect"].to_numpy(dtype=float), low.to_numpy(), high.to_numpy())
    return frame, {"schemes": list(weights), "weights": weights, "composition": comps}


def _stability_columns(frame: pd.DataFrame, schemes: list) -> None:
    """weight_stable: the same tier under every built-in scheme (plan §26.2)."""
    from .weights import SCHEMES

    built_in = [s for s in SCHEMES if s in schemes]
    tiers = frame[[f"tier_{s}" for s in built_in]]
    frame["weight_stable"] = tiers.nunique(axis=1).eq(1).to_numpy()


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


def n_detected(summary: RobustnessSummary) -> int:
    """Taxa significant in at least one specification — the four assignable tiers."""
    counts = summary.tier_counts
    return sum(counts[t] for t in ("ROBUST", "CONDITIONAL", "FRAGILE", "UNSTABLE"))


def n_stable(summary: RobustnessSummary) -> int:
    """Taxa that held up: ROBUST or CONDITIONAL.

    The distinction that matters to a reader. "Significant in at least one of 1,596
    analyses" is a very low bar — a table simulated with no group difference at all
    still put 57 of 79 taxa in UNSTABLE — so a page can be full of tiers and contain
    no finding. This counts the tiers that survive their own definition.
    """
    counts = summary.tier_counts
    return int(counts["ROBUST"]) + int(counts["CONDITIONAL"])


def verdict_sentence(run, summary: RobustnessSummary) -> str:
    """SPEC §16.1 — one line, at the top of the page."""
    counts = summary.tier_counts
    detected = n_detected(summary)

    # A cohort where nothing reached significance is a real result, and one worth
    # reporting. Rendering it through the tier template gives "Of the 0 taxa
    # significant in at least one specification, 0 are ROBUST, 0 CONDITIONAL..." —
    # arithmetically true and indistinguishable from a malfunction.
    if detected == 0:
        tested = int(counts.get("INSUFFICIENT", 0)) + int(counts.get("NOT DETECTED", 0))
        return (
            f"No taxon reached significance in any of the "
            f"{summary.n_specs_total:,} valid specifications. That is a finding, not a "
            f"failure: across every defensible way of analysing this table, "
            f"{tested:,} taxa were examined and none produced a difference between the "
            f"groups that survived multiple-testing correction. The most common reason "
            f"is that the study is too small or too sparse to settle the question — the "
            f"readiness checks on the configure page say which applies here."
        )

    unit = "taxa" if detected != 1 else "taxon"
    parts = [
        f"Of the {detected} {unit} significant in at least one specification, "
        f"{counts['ROBUST']} are ROBUST, {counts['CONDITIONAL']} CONDITIONAL, "
        f"{counts['FRAGILE']} FRAGILE and {counts['UNSTABLE']} UNSTABLE"
    ]
    if counts["INSUFFICIENT"]:
        # "A further": these are not among the taxa counted above, and reading the clause
        # as a share of them made the sentence's numbers appear not to add up.
        n = int(counts["INSUFFICIENT"])
        parts.append(f"a further {n} {'taxon' if n == 1 else 'taxa'} could not be tiered "
                     "(tested in fewer than 10 specifications)")
    skipped = int(getattr(summary, "n_specs_skipped", 0) or 0)
    across = (f"across the {summary.n_specs_total:,} of {summary.n_specs_total + skipped:,} "
              "valid specifications that produced a result" if skipped
              else f"across {summary.n_specs_total:,} valid specifications")
    sentence = "; ".join(parts) + f", {across}."

    if run.declared_spec_id >= 0 and np.isfinite(summary.declared_percentile):
        sentence += (
            f" Your reported pipeline sits at the {ordinal(summary.declared_percentile)} "
            f"percentile of significance — it calls {summary.declared_n_significant} taxa "
            f"significant, more than {summary.declared_percentile:.0f}% of defensible "
            f"alternatives."
        )
    return sentence
