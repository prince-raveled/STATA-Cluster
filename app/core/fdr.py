"""Multiple-testing correction — SPEC §10, fork 6.

Applied post-hoc to stored p-values, never as a separate model fit, which is why the
three FDR settings cost essentially nothing. Correction is always within a single
specification: the taxa tested under that specification are the family.
"""
from __future__ import annotations

import numpy as np


def adjust(p_values: np.ndarray, method: str = "bh") -> np.ndarray:
    """Benjamini-Hochberg or Benjamini-Yekutieli adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p.copy()
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
    """Return (adjusted p-values, significance mask)."""
    adjusted = adjust(p_values, method)
    return adjusted, adjusted <= threshold
