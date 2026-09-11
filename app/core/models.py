"""Core data structures — SPEC §13.

`Specification` is frozen and hashable so it can key the preprocessed-matrix cache.
`TaxonResult` is the per-taxon, per-specification record; the engine stores these
columnar (see `ResultBlock`) because a Quick-mode run produces ~1.2M of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Rarefaction = Literal["none", "min", "1000", "5000", "10000"]
Rank = Literal["genus", "input"]
Transform = Literal["tss", "clr", "raw", "tmm"]
Method = Literal["wilcoxon", "ttest", "logistic", "linear", "ancombc", "aldex2", "pydeseq2"]
FDRMethod = Literal["bh", "by"]
Mode = Literal["quick", "full", "covariate"]

#: Human labels used everywhere in the UI and exports.
FORK_LABELS = {
    "rarefaction": "Rarefaction depth",
    "rank": "Taxonomic rank",
    "prev_filter": "Prevalence filter",
    "transform": "Transformation",
    "method": "DA method",
    "fdr_method": "FDR method",
    "fdr_threshold": "FDR threshold",
    "covariates": "Covariate adjustment",
}

METHOD_LABELS = {
    "wilcoxon": "Wilcoxon rank-sum",
    "ttest": "Welch's t-test",
    "logistic": "Logistic regression (presence/absence)",
    "linear": "Linear regression (LinDA-style)",
    "ancombc": "ANCOM-BC",
    "aldex2": "ALDEx2",
    "pydeseq2": "PyDESeq2",
}

#: Compact labels for the specification-curve dot matrix, where a full method name
#: would take more horizontal room than the panel has.
METHOD_SHORT = {
    "wilcoxon": "Wilcoxon",
    "ttest": "Welch t",
    "logistic": "Logistic P/A",
    "linear": "Linear",
    "ancombc": "ANCOM-BC",
    "aldex2": "ALDEx2",
    "pydeseq2": "PyDESeq2",
}

#: The method's own effect measure — stored but never plotted on the spec curve (§14).
EFFECT_NATIVE_TYPE = {
    "wilcoxon": "rank_biserial",
    "ttest": "mean_difference",
    "logistic": "logOR",
    "linear": "coefficient",
    "ancombc": "ln_fold_change",
    "aldex2": "clr_diff_log2",  # ALDEx2 works in log2; the name says so, as ln_fold_change does
    "pydeseq2": "log2fc",
}

def ordinal(value) -> str:
    """1 -> 1st, 2 -> 2nd, 51 -> 51st, 11 -> 11th. Percentiles are read aloud."""
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return "-"
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


ELEMENTARY_METHODS: tuple[str, ...] = ("wilcoxon", "ttest", "logistic", "linear")
SOPHISTICATED_METHODS: tuple[str, ...] = ("ancombc", "aldex2", "pydeseq2")


@dataclass(frozen=True)
class Specification:
    """One point in the multiverse. Hashable, so it can key a cache."""

    rarefaction: Rarefaction
    rare_seed: int | None  # None when rarefaction == "none"
    rank: Rank
    prev_filter: float  # 0.0, 0.05, 0.10, 0.20
    transform: Transform
    method: Method
    fdr_method: FDRMethod
    fdr_threshold: float  # 0.05, 0.10
    covariates: tuple[str, ...] = ()  # empty in Modes A/B

    @property
    def matrix_key(self) -> tuple:
        """Identifies the cached preprocessed matrix (forks 1-4 only)."""
        return (self.rarefaction, self.rare_seed, self.rank, self.prev_filter, self.transform)

    @property
    def fit_key(self) -> tuple:
        """Identifies one model fit: a matrix, a method, and a covariate set.

        FDR is post-hoc (§10 fork 6), so the three FDR settings share one fit.
        """
        return self.matrix_key + (self.method, self.covariates)

    @property
    def rarefaction_label(self) -> str:
        if self.rarefaction == "none":
            return "none"
        depth = "min depth" if self.rarefaction == "min" else self.rarefaction
        return f"{depth} (seed {self.rare_seed})"

    @property
    def fdr_label(self) -> str:
        return f"{self.fdr_method.upper()}@{self.fdr_threshold:g}"

    def fork_levels(self) -> dict[str, str]:
        """Fork -> active level, for the specification-curve dot matrix (§16.3)."""
        return {
            "rarefaction": self.rarefaction_label,
            "rank": self.rank,
            "prev_filter": f"{self.prev_filter:.0%}",
            "transform": self.transform.upper(),
            "method": METHOD_SHORT[self.method],
            "fdr_method": self.fdr_method.upper(),
            "fdr_threshold": f"{self.fdr_threshold:g}",
        }

    def as_row(self) -> dict:
        return {
            "rarefaction": self.rarefaction,
            "rare_seed": self.rare_seed,
            "rank": self.rank,
            "prev_filter": self.prev_filter,
            "transform": self.transform,
            "method": self.method,
            "fdr_method": self.fdr_method,
            "fdr_threshold": self.fdr_threshold,
            "covariates": "|".join(self.covariates),
        }

    def describe(self) -> str:
        bits = [
            f"rarefaction={self.rarefaction_label}",
            f"rank={self.rank}",
            f"prevalence>={self.prev_filter:.0%}",
            f"transform={self.transform}",
            f"method={METHOD_LABELS[self.method]}",
            f"fdr={self.fdr_label}",
        ]
        if self.covariates:
            bits.append("covariates=" + "+".join(self.covariates))
        return ", ".join(bits)


@dataclass
class TaxonResult:
    """One taxon under one specification."""

    taxon: str
    spec_id: int
    tested: bool  # survived filtering?
    p_raw: float
    p_adjusted: float
    significant: bool
    effect_native: float  # the method's own statistic
    effect_native_type: str  # "log2fc" | "logOR" | "clr_diff" | ...
    effect_harmonized: float  # ALWAYS log2 fold change — see §14


@dataclass
class FitResult:
    """Columnar output of one model fit: arrays aligned to `taxa`.

    Taxa absent from this fit were removed by the prevalence filter and are simply
    not present in `taxa` — §15 depends on that distinction being kept.
    """

    taxa: np.ndarray  # int32 indices into the run's global taxon list
    p_raw: np.ndarray  # float64
    effect_native: np.ndarray  # float64
    effect_harmonized: np.ndarray  # float64
    effect_native_type: str
    n_taxa_input: int = 0


@dataclass
class GridReport:
    """Counts the user is entitled to see (§11: 'report both counts')."""

    mode: str
    n_enumerated: int
    n_valid: int
    n_pruned: int
    n_matrices: int
    n_fits: int
    pruned_reasons: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def prune_ratio(self) -> float:
        return 0.0 if not self.n_enumerated else self.n_pruned / self.n_enumerated
