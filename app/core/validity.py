"""Validity matrix — SPEC §11, implemented literally.

Pruning matters twice: it cuts compute, and it stops incoherent specifications from
artificially widening the distribution of answers. Both counts (enumerated and valid)
are reported to the user.
"""
from __future__ import annotations

from dataclasses import dataclass

INCOMPATIBLE = {
    # method: set of transforms it must NOT receive
    "pydeseq2": {"clr", "tss", "tmm"},  # requires integer counts
    "aldex2": {"clr"},                  # applies CLR internally
    "ancombc": {"clr", "tss", "tmm"},   # estimates sampling fractions itself
    "wilcoxon": set(),                  # accepts anything
    "ttest": set(),
    "logistic": set(),                  # uses presence/absence; transform irrelevant
    "linear": set(),
}

INCOMPATIBLE_REASON = {
    "pydeseq2": "PyDESeq2 requires integer counts",
    "aldex2": "ALDEx2 applies CLR internally",
    "ancombc": "ANCOM-BC estimates sampling fractions itself",
}

# Additional rules, checked separately:
RULES = [
    # (predicate, reason) — spec is INVALID if predicate is True
    (lambda s: s.method == "ancombc" and s.rarefaction != "none",
     "ANCOM-BC estimates sampling fractions; rarefaction double-corrects"),

    (lambda s: s.transform == "tmm" and s.rarefaction != "none",
     "TMM + rarefaction is double library-size correction"),

    (lambda s: s.method == "logistic" and s.transform != "raw",
     "presence/absence is transform-invariant; keep one canonical "
     "combination"),

    (lambda s: s.method == "pydeseq2" and s.rarefaction != "none",
     "DESeq2 models library size internally"),
]


@dataclass(frozen=True)
class Capabilities:
    """Dataset-level facts that make otherwise-valid specifications impossible.

    Kept separate from §11 so the published validity matrix stays literal: these
    prunings depend on the upload, not on statistics.
    """

    integer_counts: bool = True
    can_rarefy: bool = True
    can_collapse_to_genus: bool = True

    def reason(self, spec) -> str:
        if not self.can_rarefy and spec.rarefaction != "none":
            return "dataset values are not integers, so subsampling is impossible"
        if not self.integer_counts and spec.method in {"pydeseq2", "ancombc", "aldex2"}:
            return f"{spec.method} needs integer counts; this table holds relative abundances"
        if not self.integer_counts and spec.transform == "tmm":
            return "TMM needs integer counts; this table holds relative abundances"
        if not self.can_collapse_to_genus and spec.rank == "genus":
            return "no taxonomy supplied, so the table cannot be collapsed to genus"
        return ""


def invalid_reason(spec, capabilities: Capabilities = None) -> str:
    """Empty string when the specification is valid; otherwise why it was pruned."""
    forbidden = INCOMPATIBLE.get(spec.method, set())
    if spec.transform in forbidden:
        why = INCOMPATIBLE_REASON.get(spec.method, "transform is incompatible with the method")
        return f"{why}; {spec.transform.upper()} input is not defensible"
    for predicate, reason in RULES:
        if predicate(spec):
            return reason
    if capabilities is not None:
        return capabilities.reason(spec)
    return ""


def is_valid(spec, capabilities: Capabilities = None) -> bool:
    return not invalid_reason(spec, capabilities)
