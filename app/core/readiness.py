"""Is this dataset suitable for the question being asked? — scientific readiness.

`validation.py` answers a different question: *can the software process this file*. It
enforces the §8 floors (10 samples, 5 per group, 10 taxa) and refuses anything below
them. Those are technical gates, and passing them says nothing about whether the
analysis will mean anything.

This module answers the second question, and never blocks. A 12-sample study clears
every §8 floor and will still produce an almost empty result — the tier validation
(§24.6) found that below roughly 40 samples per group the multiverse assigns no ROBUST
or CONDITIONAL calls at all, because nothing reaches significance in 30% of
specifications. A researcher deserves to know that before waiting for the run, not
after reading a page of NOT DETECTED.

Every check returns a level and a reason:

  ok       Nothing to say.
  note     Worth knowing; will not distort the analysis.
  caution  Will visibly limit what the analysis can conclude.
  serious  The analysis will run and its output will be hard to interpret.

Nothing here refuses an analysis. Refusal belongs in §8, where the reasons are
technical and absolute; this is advice, and a researcher can have good reasons to
proceed anyway.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

OK = "ok"
NOTE = "note"
CAUTION = "caution"
SERIOUS = "serious"

LEVEL_ORDER = {OK: 0, NOTE: 1, CAUTION: 2, SERIOUS: 3}
LEVEL_LABEL = {OK: "Fine", NOTE: "Worth knowing", CAUTION: "Limits the analysis",
               SERIOUS: "Hard to interpret"}

#: Below this many samples in the smaller group, the tier validation observed no ROBUST
#: or CONDITIONAL calls at all. Not a rule, an observation — see SPEC §24.6.
TIER_INFORMATIVE_PER_GROUP = 40

#: A group ratio beyond this is imbalanced enough to cost power noticeably.
IMBALANCE_RATIO = 3.0

#: Library sizes spanning more than this factor make the rarefaction fork consequential.
DEPTH_SPAN_WARNING = 10.0

#: Above this share of zeros, most methods are working mostly with absence/presence.
SPARSITY_CAUTION = 0.80
SPARSITY_SERIOUS = 0.92


@dataclass
class Check:
    """One readiness finding."""

    key: str
    title: str
    level: str
    detail: str
    advice: str = ""

    @property
    def label(self) -> str:
        return LEVEL_LABEL[self.level]


@dataclass
class ReadinessReport:
    checks: list = field(default_factory=list)

    @property
    def worst(self) -> str:
        return max((c.level for c in self.checks), key=lambda level: LEVEL_ORDER[level],
                   default=OK)

    @property
    def flagged(self) -> list:
        return [c for c in self.checks if c.level != OK]

    @property
    def counts(self) -> dict:
        return {level: sum(1 for c in self.checks if c.level == level)
                for level in (SERIOUS, CAUTION, NOTE, OK)}

    def headline(self) -> str:
        flagged = self.flagged
        if not flagged:
            return "Nothing about this dataset limits what the analysis can conclude."
        serious = [c for c in flagged if c.level == SERIOUS]
        caution = [c for c in flagged if c.level == CAUTION]
        if serious:
            return (f"{len(serious)} issue{'s' if len(serious) > 1 else ''} will make "
                    f"the results hard to interpret. The analysis will still run.")
        if caution:
            return (f"{len(caution)} thing{'s' if len(caution) > 1 else ''} will limit "
                    f"what this analysis can conclude.")
        return "A few things worth knowing before you read the results."


def _group_check(dataset) -> list:
    checks = []
    n_a, n_b = dataset.group_sizes
    smaller = min(n_a, n_b)
    a, b = dataset.group_labels

    if smaller < TIER_INFORMATIVE_PER_GROUP:
        checks.append(Check(
            key="sample_size",
            title="Sample size for stability testing",
            level=SERIOUS if smaller < 20 else CAUTION,
            detail=(f"The smaller group has {smaller} samples. In the held-out "
                    f"validation of the robustness tiers, studies below about "
                    f"{TIER_INFORMATIVE_PER_GROUP} per group produced no ROBUST or "
                    f"CONDITIONAL findings at all — not because the tiers failed, but "
                    f"because nothing reached significance often enough to earn one."),
            advice=("Expect mostly NOT DETECTED and FRAGILE. That is a truthful "
                    "report that the data cannot settle the question, not a bug."),
        ))
    else:
        checks.append(Check(
            key="sample_size", title="Sample size for stability testing", level=OK,
            detail=f"{n_a} and {n_b} samples — enough for tiers to be informative."))

    ratio = max(n_a, n_b) / max(1, smaller)
    if ratio > IMBALANCE_RATIO:
        checks.append(Check(
            key="balance", title="Group balance",
            level=CAUTION if ratio > 5 else NOTE,
            detail=(f"{n_a} {a} against {n_b} {b} — a {ratio:.1f}:1 imbalance. Power is "
                    f"set by the smaller group, so this behaves closer to a "
                    f"{smaller}-per-group study than a {(n_a + n_b) // 2}-per-group one."),
            advice="Tests differ in how well they cope; watch for method disagreement.",
        ))
    else:
        checks.append(Check(key="balance", title="Group balance", level=OK,
                            detail=f"{n_a} against {n_b} — reasonably balanced."))
    return checks


def _depth_check(dataset) -> list:
    libraries = np.asarray(dataset.table.library_sizes, dtype=float)
    libraries = libraries[np.isfinite(libraries) & (libraries > 0)]
    if libraries.size == 0:
        return [Check(key="depth", title="Sequencing depth", level=NOTE,
                      detail="No usable library sizes — the values may not be counts.")]

    low, high = float(libraries.min()), float(libraries.max())
    span = high / max(low, 1.0)
    if span > DEPTH_SPAN_WARNING:
        return [Check(
            key="depth", title="Sequencing depth varies widely",
            level=CAUTION if span > 50 else NOTE,
            detail=(f"Library sizes run from {low:,.0f} to {high:,.0f} reads — a "
                    f"{span:.0f}-fold span. Rarefying to a common depth discards most "
                    f"of the deepest samples; not rarefying leaves depth confounded "
                    f"with abundance."),
            advice=("This is exactly the disagreement the rarefaction fork exists to "
                    "expose. Look at how much of your result it drives."),
        )]
    return [Check(key="depth", title="Sequencing depth", level=OK,
                  detail=f"{low:,.0f} to {high:,.0f} reads — a {span:.1f}-fold span.")]


def _sparsity_check(dataset) -> list:
    counts = np.asarray(dataset.table.counts.to_numpy(), dtype=float)
    if counts.size == 0:
        return []
    zeros = float((counts == 0).mean())
    level = OK
    if zeros >= SPARSITY_SERIOUS:
        level = SERIOUS
    elif zeros >= SPARSITY_CAUTION:
        level = CAUTION
    detail = f"{zeros:.0%} of the table is zero."
    if level == OK:
        return [Check(key="sparsity", title="Sparsity", level=OK,
                      detail=detail + " Typical for this kind of data.")]
    return [Check(
        key="sparsity", title="Sparsity", level=level,
        detail=(detail + " At this density most taxa are absent from most samples, so "
                "several methods are effectively testing presence and absence rather "
                "than abundance."),
        advice=("Raising the prevalence filter will remove the emptiest taxa. Watch "
                "whether your findings depend on where you set it."),
    )]


def _prevalence_check(dataset) -> list:
    counts = np.asarray(dataset.table.counts.to_numpy(), dtype=float)
    if counts.size == 0:
        return []
    prevalence = (counts > 0).mean(axis=1)
    rare = float((prevalence < 0.10).mean())
    if rare > 0.5:
        return [Check(
            key="prevalence", title="Most taxa are very rare", level=CAUTION,
            detail=(f"{rare:.0%} of taxa appear in fewer than 10% of samples. They "
                    f"carry little information and inflate the multiple-testing "
                    f"burden, which makes every other taxon harder to detect."),
            advice="The prevalence-filter fork will matter a lot here.",
        )]
    return [Check(key="prevalence", title="Taxon prevalence", level=OK,
                  detail=f"{rare:.0%} of taxa appear in under 10% of samples.")]


def _rank_check(dataset) -> list:
    if dataset.table.can_collapse_to_genus:
        return [Check(key="rank", title="Taxonomy", level=OK,
                      detail="Lineages present, so the analysis can be repeated at "
                             "genus level as well as the level supplied.")]
    return [Check(
        key="rank", title="No taxonomy supplied", level=NOTE,
        detail=("Without lineages the taxonomic-rank choice has only one level, so "
                "that fork cannot be varied and the grid is correspondingly smaller."),
        advice=("Supplying a taxonomy file adds a genuine analytical dimension — and "
                "on published data, rank changed the conclusions more than the choice "
                "of statistical test did."),
    )]


def _covariate_check(dataset) -> list:
    available = list(dataset.covariate_columns)
    if not available:
        return [Check(
            key="covariates", title="No covariates available", level=NOTE,
            detail=("No metadata column is usable as a covariate, so confounding "
                    "cannot be examined and covariate mode is unavailable."),
            advice=("Age, sex, BMI, batch or sequencing depth, if you have them. "
                    "Tierney et al. found covariate choice alone flips about one "
                    "association in three."),
        )]

    checks = [Check(key="covariates", title="Covariates available", level=OK,
                    detail=f"{len(available)} usable: {', '.join(available[:6])}"
                           + (" …" if len(available) > 6 else ""))]

    # A covariate that is nearly collinear with the grouping cannot be adjusted for
    # without removing the very contrast being tested.
    metadata = dataset.metadata
    groups = np.asarray(dataset.groups)
    for column in available:
        series = metadata[column]
        if series.dtype.kind in "ifu":
            values = series.to_numpy(dtype=float)
            usable = np.isfinite(values)
            if usable.sum() < 10 or len(set(groups[usable])) < 2:
                continue
            first = values[usable & (groups == 0)]
            second = values[usable & (groups == 1)]
            if first.size < 3 or second.size < 3:
                continue
            pooled = np.std(values[usable])
            if pooled <= 0:
                continue
            separation = abs(first.mean() - second.mean()) / pooled
            if separation > 1.5:
                checks.append(Check(
                    key=f"covariate_{column}",
                    title=f"'{column}' tracks the grouping closely",
                    level=CAUTION,
                    detail=(f"Its mean differs between the groups by "
                            f"{separation:.1f} standard deviations. Adjusting for it "
                            f"removes much of the contrast being tested."),
                    advice=("Covariate mode will show what happens both ways. Treat a "
                            "result that only survives without this adjustment "
                            "carefully — and one that only survives with it, equally "
                            "so."),
                ))
    return checks


def _metadata_check(dataset) -> list:
    metadata = dataset.metadata
    if metadata is None or metadata.empty:
        return []
    missing = metadata.isna().mean()
    worst = missing[missing > 0.2]
    if len(worst):
        listed = ", ".join(f"{name} ({share:.0%})" for name, share in
                           worst.sort_values(ascending=False).head(4).items())
        return [Check(
            key="metadata", title="Incomplete metadata", level=NOTE,
            detail=f"Columns with substantial missing values: {listed}.",
            advice=("Samples missing a covariate are dropped from the models that use "
                    "it, so adjustment sets end up fitted on different samples."),
        )]
    return []


def assess(dataset) -> ReadinessReport:
    """Scientific readiness for the dataset the validator has already accepted."""
    checks = []
    for builder in (_group_check, _depth_check, _sparsity_check, _prevalence_check,
                    _rank_check, _covariate_check, _metadata_check):
        try:
            checks.extend(builder(dataset))
        except Exception as exc:                        # noqa: BLE001
            # A readiness check failing must never stop an analysis the validator
            # already accepted. Say so and carry on.
            checks.append(Check(
                key=builder.__name__.strip("_"),
                title="A readiness check could not run",
                level=NOTE,
                detail=f"{type(exc).__name__}: {exc}",
            ))
    # Worst first. LEVEL_ORDER is the single source of severity, used here and by the
    # `worst` property — a second ordering dict is how the two drift apart.
    checks.sort(key=lambda c: LEVEL_ORDER[c.level], reverse=True)
    return ReadinessReport(checks=checks)
