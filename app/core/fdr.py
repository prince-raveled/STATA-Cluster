"""Multiple-testing correction — SPEC §10, fork 6.

Applied post-hoc to stored p-values, never as a separate model fit, which is why the
three FDR settings cost essentially nothing. Correction is always within a single
specification: the taxa tested under that specification are the family.
"""
from __future__ import annotations

import numpy as np


def adjust(p_values: np.ndarray, method: str = "bh") -> np.ndarray:
    """Benjamini-Hochberg or Benjamini-Yekutieli adjusted p-values.

    Non-finite inputs are excluded from the family rather than carried through it. A
    test that could not be computed is not a test that was performed, so it must
    neither inflate `n` nor receive an adjusted value — it comes back as NaN and the
    caller reads that as untested.

    This is a containment rule, not a change to BH/BY. The arithmetic below runs on the
    finite subset exactly as it always has. It matters because `np.minimum.accumulate`
    propagates a NaN backwards through the whole sorted vector: before this guard a
    single non-finite p-value silently turned *every* taxon in that specification
    non-significant, with no error and no recorded failure.
    """
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return p.copy()

    finite = np.isfinite(p)
    if not finite.all():
        out = np.full(p.shape, np.nan, dtype=float)
        if finite.any():
            out[finite] = adjust(p[finite], method)
        elif method not in ("bh", "by"):
            raise ValueError(f"Unknown FDR method: {method}")
        return out

    n = p.size
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    ranks = np.arange(1, n + 1, dtype=float)

    factor = 1.0
    if method == "by":
        factor = np.sum(1.0 / ranks)  # the Benjamini-Yekutieli c(n) penalty
    elif method != "bh":
        raise ValueError(f"Unknown FDR method: {method}")

    scaled = ranked * n * factor / ranks
    # Enforce monotonicity from the largest p-value down.
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def significant(p_values: np.ndarray, method: str, threshold: float):
    """Return (adjusted p-values, significance mask).

    A NaN adjusted p-value compares False, which is the right answer: a test that could
    not be computed is not a discovery.
    """
    adjusted = adjust(p_values, method)
    with np.errstate(invalid="ignore"):
        return adjusted, adjusted <= threshold
