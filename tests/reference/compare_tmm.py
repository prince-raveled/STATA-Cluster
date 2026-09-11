"""Validate the in-tree TMM against analytic ground truth, and against `conorm`.

SPEC §10 fork 3 lists TMM but names no library, so `app/core/preprocess.py::tmm_factors`
implements edgeR's `.calcFactorTMM` (Robinson & Oshlack 2010) directly.

Two kinds of evidence, in order of weight:

1. **Ground truth.** A table is built where the correct normalisation factor is known
   in closed form: 20% of genes are genuinely up in one group, which mechanically
   deflates every other gene's *proportion*, and the factor TMM must recover is exactly
   the ratio of true totals. Agreeing with another implementation is weaker than
   recovering a known answer, so this is the test that decides correctness.

2. **Agreement with `conorm`**, an independent Python port. Reported for context. The
   two diverge on sparse tables, and the reason is identified rather than papered over:
   edgeR trims by *rank*, conorm trims by *quantile value*. With microbiome counts the
   M-values are full of ties, and quantile trimming discards every tied observation at
   the boundary. We follow edgeR.

    .venv/Scripts/python tests/reference/compare_tmm.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _restore_numpy_interpolation_kwarg() -> None:
    """conorm 1.2.0 calls `np.nanquantile(..., interpolation=...)`.

    NumPy renamed that argument to `method` in 1.22 and removed the alias in 2.0. The
    shim restores the alias so the reference implementation runs unmodified — it only
    renames a keyword, it does not change what conorm computes.
    """
    original = np.nanquantile

    def nanquantile(*args, **kwargs):
        if "interpolation" in kwargs:
            kwargs["method"] = kwargs.pop("interpolation")
        return original(*args, **kwargs)

    np.nanquantile = nanquantile


_restore_numpy_interpolation_kwarg()

import conorm  # noqa: E402

from app.core.preprocess import tmm_factors  # noqa: E402

FAILURES: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def report(name: str, detail: str) -> None:
    """Recorded for context, not asserted."""
    print(f"  [ -- ] {name} — {detail}")


# ---------------------------------------------------------------------------
# 1. Ground truth
# ---------------------------------------------------------------------------
def known_factor_table(n_genes=600, n_per_group=8, up_fraction=0.20, up_fold=4.0,
                       depth=4_000_000, sparsity=0.0, seed=17):
    """Counts whose correct TMM factor is known in closed form.

    Absolute abundance A is shared; in group B a `up_fraction` slice is `up_fold` times
    higher. Sequencing only sees *proportions*, so every other gene looks depleted in B
    by exactly total(A) / total(B). That ratio is the factor TMM has to recover.
    """
    rng = np.random.default_rng(seed)
    absolute = rng.lognormal(5.0, 1.2, n_genes)

    up = np.zeros(n_genes, dtype=bool)
    up[rng.choice(n_genes, size=int(n_genes * up_fraction), replace=False)] = True
    absolute_b = absolute * np.where(up, up_fold, 1.0)

    # What a correct TMM must return for a B sample, relative to an A sample.
    true_factor_b = absolute.sum() / absolute_b.sum()

    def draw(mean_abundance, n):
        p = mean_abundance / mean_abundance.sum()
        return np.array([rng.multinomial(depth, p) for _ in range(n)]).T.astype(float)

    counts = np.hstack([draw(absolute, n_per_group), draw(absolute_b, n_per_group)])
    if sparsity > 0:
        counts[rng.random(counts.shape) < sparsity] = 0.0
    groups = np.array([0] * n_per_group + [1] * n_per_group)
    return counts, groups, true_factor_b, up


def ground_truth(label: str, **kwargs) -> None:
    print(f"\nground truth — {label}")
    counts, groups, true_ratio, _ = known_factor_table(**kwargs)

    ours = tmm_factors(counts)
    frame = pd.DataFrame(counts)
    theirs = np.asarray(conorm.tmm_norm_factors(frame)).ravel().astype(float)

    # Factors are centred to a geometric mean of 1, so compare the B/A ratio.
    def ratio(factors):
        return float(np.exp(np.mean(np.log(factors[groups == 1])))
                     / np.exp(np.mean(np.log(factors[groups == 0]))))

    our_ratio, their_ratio = ratio(ours), ratio(theirs)
    our_error = abs(np.log2(our_ratio / true_ratio))
    their_error = abs(np.log2(their_ratio / true_ratio))

    print(f"    true B/A factor = {true_ratio:.4f}")
    check("recovers the known factor", our_error < 0.05,
          f"ours = {our_ratio:.4f} (error {our_error:.4f} log2)")
    report("conorm on the same table",
           f"{their_ratio:.4f} (error {their_error:.4f} log2)")
    # Both are approximations of the same estimator; what matters is that neither is
    # systematically worse. 0.01 log2 is 0.7% on the factor.
    check("as accurate as conorm", our_error <= their_error + 0.01,
          f"{our_error:.4f} vs {their_error:.4f} log2 error")


# ---------------------------------------------------------------------------
# 2. Agreement with conorm
# ---------------------------------------------------------------------------
def agreement(label: str, counts: np.ndarray, tolerance: float | None) -> None:
    print(f"\nagreement with conorm — {label}  "
          f"({counts.shape[0]} taxa x {counts.shape[1]} samples, "
          f"{(counts == 0).mean():.0%} zeros)")
    ours = tmm_factors(counts)
    theirs = np.asarray(conorm.tmm_norm_factors(pd.DataFrame(counts))).ravel().astype(float)

    check("unit geometric mean", np.isclose(np.exp(np.mean(np.log(ours))), 1.0),
          f"gm = {np.exp(np.mean(np.log(ours))):.6f}")
    rel = float(np.max(np.abs(ours - theirs) / theirs))
    if tolerance is None:
        report("max relative difference",
               f"{rel:.2%} — expected: edgeR trims by rank, conorm by quantile value")
        # No composition bias was built in, so every true factor is 1. Distance from 1
        # is the honest measure of which implementation is noisier here.
        our_drift = float(np.max(np.abs(np.log2(ours))))
        their_drift = float(np.max(np.abs(np.log2(theirs))))
        report("worst drift from the true factor of 1.0",
               f"ours {our_drift:.4f} log2, conorm {their_drift:.4f} log2")
        check("does not drift further than conorm on null data",
              our_drift <= their_drift + 0.05,
              f"{our_drift:.4f} vs {their_drift:.4f} log2")
    else:
        check("factors match conorm", rel < tolerance, f"max relative = {rel:.2%}")


def composition_bias_property() -> None:
    """The property that makes TMM worth having, stated as a test.

    Uses the *known* unchanged genes rather than a proxy: their true abundance is
    identical in both groups, so any apparent change is composition bias and nothing
    else. Selecting them by observed abundance instead would silently include the
    up-regulated genes and measure the wrong thing.
    """
    print("\nproperty — TMM undoes composition bias that TSS cannot")
    counts, groups, _, up = known_factor_table(sparsity=0.0)
    factors = tmm_factors(counts)
    libraries = counts.sum(axis=0)
    ordinary = counts[~up]

    tss = ordinary / libraries
    tmm = ordinary / (factors * libraries)
    tss_ratio = float(tss[:, groups == 1].mean() / tss[:, groups == 0].mean())
    tmm_ratio = float(tmm[:, groups == 1].mean() / tmm[:, groups == 0].mean())

    check("TMM moves unchanged genes back towards a ratio of 1",
          abs(np.log(tmm_ratio)) < abs(np.log(tss_ratio)),
          f"{tss_ratio:.3f} under TSS vs {tmm_ratio:.3f} under TMM (1.0 is correct)")
    check("and lands close to 1", abs(np.log2(tmm_ratio)) < 0.06,
          f"residual bias = {np.log2(tmm_ratio):+.4f} log2")


if __name__ == "__main__":
    print(f"conorm {conorm.__version__} on Python {sys.version.split()[0]}")

    ground_truth("20% of genes 4x up, dense", up_fraction=0.20, up_fold=4.0)
    ground_truth("30% of genes 3x up, dense", up_fraction=0.30, up_fold=3.0)
    ground_truth("20% of genes 4x up, 40% zeros", up_fraction=0.20, up_fold=4.0, sparsity=0.40)

    rng = np.random.default_rng(1)
    agreement("negative-binomial counts",
              rng.negative_binomial(6, 0.02, size=(500, 14)).astype(float), tolerance=0.05)
    agreement("sparse microbiome-like counts",
              (rng.negative_binomial(2, 0.05, size=(300, 20))
               * (rng.random((300, 20)) > 0.45)).astype(float), tolerance=None)

    composition_bias_property()
    print("\n" + ("FAILED: " + ", ".join(FAILURES) if FAILURES else "All assertions passed."))
    sys.exit(1 if FAILURES else 0)
