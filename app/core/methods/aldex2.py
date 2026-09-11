"""ALDEx2 (SPEC §10, fork 5, method 6).

Dirichlet Monte-Carlo over the count table, CLR per instance, a test per instance,
then the expected p-value across instances. Fewer instances than the R default (128)
because Full mode runs this over ~42 matrices; the count is reported in the methods
paragraph.

SPEC named scikit-bio as the source, but scikit-bio has no wheel for this Python
(its biom-format dependency fails to build). The algorithm is reimplemented here
against Fernandes et al. 2014. Logged in SPEC §24.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..models import EFFECT_NATIVE_TYPE, FitResult

DEFAULT_INSTANCES = 64
DIRICHLET_PRIOR = 0.5  # Fernandes et al. use the Jeffreys prior
PAIRING_STREAM = 0xA1DE  # keeps the effect-size pairing off the Dirichlet stream


def dirichlet_instances(counts: np.ndarray, n_instances: int, seed: int = 1) -> np.ndarray:
    """Draw `n_instances` posterior compositions per sample: (instances, taxa, samples)."""
    rng = np.random.default_rng(seed)
    alpha = np.asarray(counts, dtype=float) + DIRICHLET_PRIOR
    n_taxa, n_samples = alpha.shape
    draws = rng.gamma(shape=alpha[None, :, :], size=(n_instances, n_taxa, n_samples))
    totals = draws.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    return draws / totals


def clr_stack(compositions: np.ndarray) -> np.ndarray:
    """CLR per instance, in log2 — the base ALDEx2 works in, so diff.btw is a log2
    fold change. The base is a constant scaling, so it leaves both tests' p-values
    untouched (Welch t and Mann-Whitney are scale-invariant); it changes only the
    units of the reported effect, and ALDEx2 users read those as log2."""
    logged = np.log2(np.clip(compositions, 1e-300, None))
    return logged - logged.mean(axis=1, keepdims=True)


def run_aldex2(matrix, covariate_frame=None, n_instances: int = DEFAULT_INSTANCES,
               seed: int = 1) -> FitResult:
    """Two-group ALDEx2. `covariate_frame` is accepted for a uniform method signature
    and ignored: this is the two-group form, and covariate mode (§12) runs only the
    four elementary methods."""
    del covariate_frame
    counts = matrix.counts
    groups = matrix.groups
    n_taxa = counts.shape[0]
    if n_taxa == 0:
        return FitResult(matrix.taxa_idx, np.array([]), np.array([]), np.array([]),
                         EFFECT_NATIVE_TYPE["aldex2"], 0)

    instances = clr_stack(dirichlet_instances(counts, n_instances, seed))
    a = instances[:, :, groups == 0]
    b = instances[:, :, groups == 1]

    # One vectorised test over (instance x taxon) rows.
    flat_a = a.reshape(-1, a.shape[2])
    flat_b = b.reshape(-1, b.shape[2])
    with np.errstate(invalid="ignore"):
        wilcox = stats.mannwhitneyu(flat_a, flat_b, axis=1, alternative="two-sided",
                                    method="asymptotic")
        welch = stats.ttest_ind(flat_b, flat_a, axis=1, equal_var=False)

    p_wilcox = np.asarray(wilcox.pvalue, dtype=float).reshape(n_instances, n_taxa)
    p_welch = np.asarray(welch.pvalue, dtype=float).reshape(n_instances, n_taxa)
    p_wilcox = np.where(np.isfinite(p_wilcox), p_wilcox, 1.0)
    p_welch = np.where(np.isfinite(p_welch), p_welch, 1.0)

    # ALDEx2 reports expected p-values (wi.ep / we.ep); take the more conservative.
    p_values = np.maximum(p_wilcox.mean(axis=0), p_welch.mean(axis=0))

    # Effect: ALDEx2's diff.btw. Pool each group's CLR values over samples *and*
    # Monte-Carlo instances, pair the two pools at random, and take the median of the
    # paired differences. This is deliberately not the difference of the two medians:
    # CLR distributions on sparse data are left-skewed, and on real tables the two
    # estimators disagree by whole CLR units. aldex.effect does exactly this
    # (`l2d$btw <- smpl2 - smpl1` over independently permuted rows, then rowMedians);
    # verified against ALDEx2 1.44.0 in tests/reference/compare_r.py.
    pooled_a = np.moveaxis(a, 1, 0).reshape(n_taxa, -1)
    pooled_b = np.moveaxis(b, 1, 0).reshape(n_taxa, -1)
    n_pairs = min(pooled_a.shape[1], pooled_b.shape[1])
    pair_rng = np.random.default_rng((seed, PAIRING_STREAM))
    paired = (pair_rng.permuted(pooled_b, axis=1)[:, :n_pairs]
              - pair_rng.permuted(pooled_a, axis=1)[:, :n_pairs])
    effect = np.median(paired, axis=1)

    return FitResult(
        taxa=matrix.taxa_idx,
        p_raw=np.clip(p_values, 0.0, 1.0),
        effect_native=np.where(np.isfinite(effect), effect, 0.0),
        effect_harmonized=np.zeros(n_taxa),
        effect_native_type=EFFECT_NATIVE_TYPE["aldex2"],
        n_taxa_input=n_taxa,
    )
