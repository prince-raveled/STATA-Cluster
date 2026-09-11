"""Why did this taxon's answer change? — per-taxon, per-choice instability.

The specification curve shows *that* a taxon's conclusion moves. It does not say *which
choice moved it*: a reader has to eyeball the fork panel underneath and guess. This
module computes the answer directly, for one taxon at a time.

DESIGN NOTE — why this is descriptive rather than a model.

`attribution.py` fits a variance decomposition across all taxa at once, and a sensitivity
analysis (`tests/reference/attribution_sensitivity.py`) found it explains a median 1% of
the within-taxon variance with estimators that disagree about the ranking. So it is
graded exploratory and must not be the basis of a per-taxon claim.

What is computed here needs no model and has nothing to converge: it is the observed
swing in the significance rate when one choice is varied and everything else is held
fixed. If a taxon is significant in 95% of unrarefied specifications and 12% of those
rarefied to 1,000 reads, that is not an estimate — it is what happened, and it is the
honest answer to "why did my result change".

Two versions of the swing are reported, because they answer different questions:

  * **conditional** — hold every other choice fixed, vary this one, record how much the
    significance rate moves, then average over all those held-fixed cells. This is the
    number that means "changing this choice, and nothing else, flips the answer this
    often". It is the primary figure.
  * **marginal** — the spread of the significance rate across this choice's levels,
    averaging over everything else. Easier to read off the curve, but it absorbs the
    other choices' influence, so it can overstate.

Neither is a probability, and neither licenses a causal claim about the biology. They
describe the analysis, not the organism.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import FORK_LABELS

#: The choices a taxon-level answer can hinge on. `rarefaction` is the label form
#: (depth plus seed) so "rarefied to 1,000, draw 2" reads as its own level.
INSTABILITY_FORKS = ["rarefaction", "rank", "prev_filter", "transform", "method",
                     "fdr_method", "fdr_threshold"]

#: Below this many specifications at a level, its rate is too noisy to show.
MIN_SPECS_PER_LEVEL = 3

#: A swing this large or more is worth telling the researcher about unprompted.
NOTABLE_SWING = 0.25


@dataclass
class ForkInfluence:
    """How much one analytical choice moves one taxon's conclusion."""

    fork: str
    label: str
    n_levels: int
    conditional_swing: float
    marginal_swing: float
    direction_swing: float
    effect_swing: float
    levels: list = field(default_factory=list)

    @property
    def most_significant(self) -> dict:
        return max(self.levels, key=lambda level: level["frac_significant"], default={})

    @property
    def least_significant(self) -> dict:
        return min(self.levels, key=lambda level: level["frac_significant"], default={})

    def sentence(self) -> str:
        """One plain line a researcher can read without the table."""
        if not self.levels or self.conditional_swing < 0.01:
            return f"{self.label} makes no difference to this taxon."
        high, low = self.most_significant, self.least_significant
        return (
            f"{self.label}: significant in {high['frac_significant']:.0%} of analyses "
            f"using {high['label']}, but {low['frac_significant']:.0%} using "
            f"{low['label']}."
        )


@dataclass
class TaxonInstability:
    """Everything known about why one taxon's answer does or does not move."""

    taxon_id: int
    name: str
    n_tested: int
    n_eligible: int
    frac_significant: float
    direction_agreement: float
    effect_median: float
    effect_iqr: tuple
    influences: list = field(default_factory=list)
    method_agreement: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def ranked(self) -> list:
        return sorted(self.influences, key=lambda i: i.conditional_swing, reverse=True)

    @property
    def leading(self):
        ranked = self.ranked
        return ranked[0] if ranked and ranked[0].conditional_swing >= 0.01 else None

    def headline(self) -> str:
        """The one sentence this whole page exists to produce."""
        if self.n_tested < 10:
            return ("Too few analyses tested this taxon to say anything about "
                    "stability.")
        if self.frac_significant >= 0.95:
            return "Significant under essentially every analysis tried."
        if self.frac_significant <= 0.05:
            return "Not significant under essentially any analysis tried."
        leader = self.leading
        if leader is None or leader.conditional_swing < 0.10:
            return ("The result varies, but no single analytical choice explains it — "
                    "the variation is spread across several choices at once.")
        label = leader.label
        # FORK_LABELS holds things like "DA method"; lowercasing wholesale gives
        # "da method", which reads as a typo.
        spoken = label if label[:2].isupper() else label[0].lower() + label[1:]
        return (
            f"The result depends mainly on {spoken}: changing only that choice moves "
            f"the significance rate by {leader.conditional_swing:.0%} on average."
        )


def _level_label(fork: str, value) -> str:
    """A human label for one level of one choice."""
    if fork == "prev_filter":
        return f"{float(value):.0%} prevalence filter"
    if fork == "fdr_threshold":
        return f"threshold {float(value):g}"
    if fork == "fdr_method":
        return {"bh": "Benjamini-Hochberg", "by": "Benjamini-Yekutieli"}.get(
            str(value), str(value))
    if fork == "transform":
        return {"raw": "raw counts", "tss": "proportions (TSS)",
                "clr": "log-ratio (CLR)", "tmm": "TMM"}.get(str(value), str(value))
    if fork == "rarefaction":
        return "no rarefaction" if str(value) == "none" else str(value)
    if fork == "method":
        from .models import METHOD_LABELS
        return METHOD_LABELS.get(str(value), str(value))
    if fork == "rank":
        return f"{value} level"
    return str(value)


def _conditional_swing(frame: pd.DataFrame, fork: str, others: list) -> float:
    """Average change in the significance rate from varying one choice alone.

    Group by every *other* choice, so within a group the only thing that differs is
    `fork`. The range of the significance rate inside a group is what changing this one
    choice did, with everything else held fixed. Averaged over groups.
    """
    if not others or frame[fork].nunique() < 2:
        return 0.0
    # One grouped mean, then a max-minus-min per cell. Looping over the cells in Python
    # costs about a second per taxon on a full grid, which is too slow to serve on
    # demand; this is the same arithmetic in one pass.
    rates = frame.groupby([*others, fork], observed=True, dropna=False)[
        "significant"].mean()
    if rates.empty:
        return 0.0
    by_cell = rates.groupby(level=list(range(len(others))), observed=True)
    swing = by_cell.max() - by_cell.min()
    # Cells where this choice had only one level available contribute nothing to
    # "what happens if I change it", so they are dropped rather than counted as zero.
    usable = by_cell.count() > 1
    swing = swing[usable]
    return float(swing.mean()) if len(swing) else 0.0


def analyse_taxon(run, taxon_id: int) -> TaxonInstability:
    """Per-choice instability for one taxon. Pure description, no model."""
    rows = run.long[run.long["taxon"] == taxon_id]
    name = run.taxa_display[taxon_id]
    if rows.empty:
        return TaxonInstability(
            taxon_id=taxon_id, name=name, n_tested=0, n_eligible=0,
            frac_significant=float("nan"), direction_agreement=float("nan"),
            effect_median=float("nan"), effect_iqr=(float("nan"), float("nan")),
            notes=["This taxon was filtered out of every specification."])

    specs = run.specs_frame.copy()
    specs["rarefaction"] = [s.rarefaction_label for s in run.specs]
    frame = rows.merge(specs, on="spec_id", how="left")
    frame["significant"] = frame["significant"].astype(float)

    active = [f for f in INSTABILITY_FORKS
              if f in frame.columns and frame[f].nunique() > 1]

    influences = []
    for fork in active:
        others = [f for f in active if f != fork]
        levels = []
        for value, block in frame.groupby(fork, observed=True, dropna=False):
            if len(block) < MIN_SPECS_PER_LEVEL:
                continue
            effects = block["effect_h"].to_numpy(dtype=float)
            levels.append({
                "value": str(value),
                "label": _level_label(fork, value),
                "n": int(len(block)),
                "frac_significant": float(block["significant"].mean()),
                "frac_positive": float((effects > 0).mean()),
                "median_effect": float(np.median(effects)) if effects.size else float("nan"),
            })
        if len(levels) < 2:
            continue
        rates = [level["frac_significant"] for level in levels]
        positives = [level["frac_positive"] for level in levels]
        medians = [level["median_effect"] for level in levels
                   if np.isfinite(level["median_effect"])]
        influences.append(ForkInfluence(
            fork=fork,
            label=FORK_LABELS.get(fork, fork),
            n_levels=len(levels),
            conditional_swing=_conditional_swing(frame, fork, others),
            marginal_swing=float(max(rates) - min(rates)),
            direction_swing=float(max(positives) - min(positives)),
            effect_swing=float(max(medians) - min(medians)) if len(medians) > 1 else 0.0,
            levels=sorted(levels, key=lambda level: level["frac_significant"],
                          reverse=True),
        ))

    # Do the statistical tests agree with each other? Reported separately because
    # "methods disagree" is a different worry from "preprocessing moved it".
    method_agreement = {}
    if "method" in frame.columns and frame["method"].nunique() > 1:
        from .models import METHOD_LABELS
        for value, block in frame.groupby("method", observed=True):
            method_agreement[str(value)] = {
                "label": METHOD_LABELS.get(str(value), str(value)),
                "n": int(len(block)),
                "frac_significant": float(block["significant"].mean()),
                "median_effect": float(block["effect_h"].median()),
            }

    effects = frame["effect_h"].to_numpy(dtype=float)
    positive = float((effects > 0).mean()) if effects.size else float("nan")

    eligible = int(run.specs_frame["spec_id"].nunique())
    notes = []
    if len(frame) < eligible:
        notes.append(
            f"Tested in {len(frame):,} of {eligible:,} analyses — a prevalence filter "
            f"removed it from the rest, so those are not counted against it.")
    if method_agreement:
        rates = [m["frac_significant"] for m in method_agreement.values()]
        if max(rates) - min(rates) >= NOTABLE_SWING:
            notes.append(
                "The statistical tests disagree with each other about this taxon; see "
                "the method breakdown below before trusting any single one.")

    return TaxonInstability(
        taxon_id=taxon_id,
        name=name,
        n_tested=int(len(frame)),
        n_eligible=eligible,
        frac_significant=float(frame["significant"].mean()),
        direction_agreement=max(positive, 1.0 - positive),
        effect_median=float(np.median(effects)) if effects.size else float("nan"),
        effect_iqr=(float(np.percentile(effects, 25)), float(np.percentile(effects, 75)))
        if effects.size else (float("nan"), float("nan")),
        influences=influences,
        method_agreement=method_agreement,
        notes=notes,
    )


# ---------------------------------------------------------------------------
def stability_fingerprint(row) -> dict:
    """Four separate stability dimensions for one taxon, deliberately not combined.

    A single tier compresses detection and direction into one word and drops magnitude
    entirely. These are kept apart so a researcher can see, for instance, that a taxon's
    direction is settled while its magnitude is not — which one label cannot express.

    Each is a share between 0 and 1 describing the *analyses*, not a probability that
    the biology is real.
    """
    spread = float(row["iqr_high"]) - float(row["iqr_low"])
    magnitude = abs(float(row["median_effect"]))
    # Effect stability: how tight the middle half of the estimates is, relative to the
    # size of the effect itself. A wide spread around a large effect is less alarming
    # than the same spread around a small one.
    if not np.isfinite(spread) or not np.isfinite(magnitude):
        effect_stability = float("nan")
    elif magnitude < 1e-9:
        effect_stability = 0.0
    else:
        effect_stability = float(max(0.0, 1.0 - spread / (2.0 * magnitude)))

    return {
        "detection": float(row["frac_significant"]),
        "direction": float(row["sign_consistency"]),
        "effect": effect_stability,
        "coverage": float(row["frac_tested"]),
    }


FINGERPRINT_LABELS = {
    "detection": ("Detection", "How often the analyses called it significant."),
    "direction": ("Direction", "How often they agreed which group had more of it."),
    "effect": ("Magnitude", "How tightly the effect-size estimates cluster, relative to "
                            "the size of the effect."),
    "coverage": ("Coverage", "How many of the analyses could test it at all, rather "
                             "than filtering it out."),
}
