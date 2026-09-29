"""Calibrated multiverse inference — v3 plan §27 (N1).

The tiers of SPEC §16.2 are descriptive: "significant in 80% of the weight" is a
statement about analytical stability, with no error rate attached. This module attaches
one. It answers two questions per taxon j, with the family F_j = {H_js : s tested j}:

  (a) discovery — does j differ between the groups under at least one defensible
      pipeline? (reject the global null H_j = ∩_s H_js, FDR ≤ q across taxa);
  (b) certified robustness — among the pipelines, where is that difference established
      with error control, and how much of the weight do they carry? (CR_j)

How, in the order of plan §27.3:

  1. Every preprocessed matrix is label-independent. Rarefaction, collapsing, the
     prevalence filter, TMM, CLR's zero replacement and the harmonised effect all use
     every sample and never the group, so labels enter only at the test. That is what
     makes permutation affordable: each matrix is built once and tested B + 1 times.
  2. One (B + 1) × n label matrix, row 0 the observed labels, rows 1..B permutations of
     them (within strata when a batch column is given), seeded and stored with the run.
     A matrix that dropped samples (rarefaction) uses the same rows restricted to the
     samples it kept, so every specification sees the same permutation.
  3. The four elementary tests are computed for all rows at once, from group sums —
     rank sums for Wilcoxon, sums and sums of squares for Welch and the linear model,
     presence counts for the 2×2 logistic test. Row 0 reproduces the scalar methods in
     `methods/elementary.py`; `tests/test_inference.py` holds them to 1e-10. Logistic
     regression adjusted for log library size (rule R9) is the exception: a Wald test
     needs an iterative fit per permutation, so the calibrated engine uses the score
     test, whose null model does not involve the labels and is fitted once.
     Each statistic becomes |z| = Φ⁻¹(1 − p/2) through its own parametric p-value; the
     scaling affects power, not validity, because validity comes from permuting the
     maximum.
  4. Streaming: per permutation and taxon, only the running maximum of |z| over the
     specifications is kept — a B × m array, never B × S × m.
  5. The family p-value is Westfall & Young's single-step maxT,
     p_j = (1 + #{b : M_j^b ≥ M_j}) / (B + 1). Where fewer than 10 permutation maxima
     reach the observed one, a generalised Pareto distribution is fitted to the tail of
     the maxima and the p-value is read from it; both values are reported.
  6. Discovery across taxa: BH at q (default), or BY, or e-BH with the p-to-e
     calibrator e = κ p^(κ-1), κ = 1/2. Selected set S, R = |S| of m tested taxa.
  7. Within a selected taxon, p̃_js = P(M_j ≥ |z_js|) from the same maxima, and H_js is
     rejected where p̃_js ≤ q·R/m (selective inference on families).
  8. CR_j = Σ_{s rejected} w_s / Σ_{s tested} w_s under the run's weights (§26), with the
     weighted sign agreement among the rejected specifications.
  9. CERTIFIED ROBUST requires j ∈ S, CR_j ≥ 0.80 and sign agreement ≥ 0.95 — v2's ROBUST
     thresholds, so the two labels can be compared.

Assumptions, stated wherever the result is shown: samples are exchangeable under the
null (permutation validity), and the per-taxon maxT inherits subset pivotality, which
strong compositional shifts in other taxa can violate (V9 tests this). Covariate-adjusted
calibration is not implemented; covariate-mode grids are not calibrated.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import special, stats

DEFAULT_PERMUTATIONS = 2000
DEFAULT_Q = 0.05
DEFAULT_SEED = 20260929
DISCOVERY_METHODS = ("bh", "by", "ebh")
#: The p-to-e calibrator's κ for e-BH, fixed before any data were seen.
EBH_KAPPA = 0.5

#: p is floored here before |z| = Φ⁻¹(1 − p/2), as effects.signed_z does.
P_FLOOR = 1e-300
Z_CAP = float(-special.ndtri(P_FLOOR / 2.0))

#: Below this many permutation maxima at or above the observed value, the tail is
#: fitted (plan §27.3 step 5).
TAIL_MIN_EXCEEDANCES = 10
TAIL_START = 250
TAIL_STEP = 10
TAIL_GOF_ALPHA = 0.05
#: A tail-fitted p-value is an extrapolation; below this it is reported as this rather
#: than as a number the permutations never measured. No decision can depend on the
#: floor: the smallest threshold in use, q/m, is above 1e-6 for q ≥ 0.01 and m ≤ 1,500.
TAIL_P_FLOOR = 1e-12

CERTIFIED_CR = 0.80
CERTIFIED_SIGN = 0.95

#: The tests the permutation engine can compute for all rows at once.
CALIBRATED_METHODS = ("wilcoxon", "ttest", "linear", "logistic")


# ---------------------------------------------------------------------------------------
# 2. Permutations
# ---------------------------------------------------------------------------------------
def permutation_matrix(groups, n_permutations: int = DEFAULT_PERMUTATIONS,
                       seed: int = DEFAULT_SEED, strata=None) -> np.ndarray:
    """(B + 1) × n int8 labels: row 0 observed, rows 1..B permuted (within strata)."""
    groups = np.asarray(groups, dtype=np.int8)
    rng = np.random.default_rng(seed)
    out = np.empty((int(n_permutations) + 1, groups.size), dtype=np.int8)
    out[0] = groups
    if n_permutations <= 0:
        return out
    if strata is None:
        out[1:] = rng.permuted(np.broadcast_to(groups, (n_permutations, groups.size)),
                               axis=1)
        return out
    strata = np.asarray(strata)
    out[1:] = groups
    for level in pd.unique(strata):
        idx = np.flatnonzero(strata == level)
        block = np.broadcast_to(groups[idx], (n_permutations, idx.size))
        out[1:, idx] = rng.permuted(block, axis=1)
    return out


# ---------------------------------------------------------------------------------------
# 3. Vectorised statistics. Each test has a core that returns its statistic for every
#    row of `labels` (rows × taxa). Two views are built on the cores:
#      statistics() — (p, effect) for every row, matching methods/elementary.py; the
#                     tests hold row 0 and permuted rows to 1e-10;
#      abs_z_rows() — |z| for every row, what the maxT engine streams, computed directly
#                     for the normal-based tests and through the p-value for the t-based
#                     ones, where it skips cells that cannot raise the running maximum.
# ---------------------------------------------------------------------------------------
def _group_counts(labels: np.ndarray):
    n_b = labels.sum(axis=1, dtype=np.int64).astype(float)[:, None]
    n_a = labels.shape[1] - n_b
    return n_a, n_b


def _tie_term(values: np.ndarray) -> np.ndarray:
    """Σ (t³ − t) over each row's tie groups, as scipy's Mann-Whitney U uses it."""
    ordered = np.sort(values, axis=1)
    rows, n = ordered.shape
    if n == 0:
        return np.zeros(rows)
    starts = np.ones_like(ordered, dtype=bool)
    starts[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    run_id = np.cumsum(starts, axis=1) - 1 + (np.arange(rows) * n)[:, None]
    sizes = np.bincount(run_id.ravel(), minlength=rows * n).astype(float)
    per_run = sizes ** 3 - sizes
    owner = np.repeat(np.arange(rows), n)
    return np.bincount(owner, weights=per_run, minlength=rows)


def _wilcoxon_core(values, labels):
    """Mann-Whitney U, asymptotic with tie and continuity correction, two-sided, as
    `scipy.stats.mannwhitneyu(a, b, method="asymptotic")`: returns the corrected z of
    max(U1, U2) and the rank-biserial effect."""
    ranks = stats.rankdata(values, axis=1)
    tie = _tie_term(values)[None, :]
    n = values.shape[1]
    n_a, n_b = _group_counts(labels)
    u1 = (1 - labels).astype(float) @ ranks.T - n_a * (n_a + 1) / 2.0
    u = np.maximum(u1, n_a * n_b - u1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.sqrt(n_a * n_b / 12.0 * ((n + 1) - tie / (n * (n - 1))))
        z = (u - n_a * n_b / 2.0 - 0.5) / s
        effect = 1.0 - 2.0 * u1 / np.maximum(1.0, n_a * n_b)
    return z, effect


def _sums(values: np.ndarray, labels: np.ndarray):
    """Group means and sums of squared deviations, per row and taxon.

    Values are centred per taxon first, so the sums of squares do not lose precision
    to a large mean (raw counts).
    """
    centred = values - values.mean(axis=1, keepdims=True)
    lab = labels.astype(float)
    n_a, n_b = _group_counts(labels)
    s1_b = lab @ centred.T
    s2_b = lab @ (centred ** 2).T
    s1_a = centred.sum(axis=1)[None, :] - s1_b
    s2_a = (centred ** 2).sum(axis=1)[None, :] - s2_b
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_a, mean_b = s1_a / n_a, s1_b / n_b
        ss_a = np.maximum(s2_a - s1_a * mean_a, 0.0)
        ss_b = np.maximum(s2_b - s1_b * mean_b, 0.0)
    return n_a, n_b, mean_a, mean_b, ss_a, ss_b


def _welch_core(values, labels):
    """Welch's t and its df, `scipy.stats.ttest_ind(b, a, equal_var=False)`."""
    n_a, n_b, mean_a, mean_b, ss_a, ss_b = _sums(values, labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        va, vb = ss_a / (n_a - 1) / n_a, ss_b / (n_b - 1) / n_b
        t = (mean_b - mean_a) / np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (n_a - 1) + vb ** 2 / (n_b - 1))
    return t, df, mean_b - mean_a


def _linear_core(values, labels):
    """OLS on group with no covariates — the pooled t-test `run_linear` reduces to,
    including its floor on the standard error."""
    n_a, n_b, mean_a, mean_b, ss_a, ss_b = _sums(values, labels)
    df = values.shape[1] - 2
    coefficient = mean_b - mean_a
    if df <= 0:
        return np.full_like(coefficient, np.nan), df, np.zeros_like(coefficient)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(np.maximum((ss_a + ss_b) / df * (1.0 / n_a + 1.0 / n_b), 1e-300))
        t = coefficient / se
    return t, df, coefficient


def _logistic_core(presence, labels):
    """The 2×2 Wald test of `_logistic_2x2`: z and log odds ratio, every row, NaN z
    where the method reports p = 1."""
    pres = presence.astype(float)
    n_a, n_b = _group_counts(labels)
    present_b = labels.astype(float) @ pres.T
    present_a = pres.sum(axis=1)[None, :] - present_b
    absent_a, absent_b = n_a - present_a, n_b - present_b
    empty = (present_a == 0) | (absent_a == 0) | (present_b == 0) | (absent_b == 0)
    c = np.where(empty, 0.5, 0.0)
    pa, na_, pb, nb_ = present_a + c, absent_a + c, present_b + c, absent_b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        log_or = np.log((pb * na_) / (pa * nb_))
        se = np.sqrt(1.0 / pa + 1.0 / na_ + 1.0 / pb + 1.0 / nb_)
        z = np.where(se > 0, log_or / se, 0.0)
    constant = (present_a + present_b == 0) | (absent_a + absent_b == 0)
    z = np.where(constant | ~np.isfinite(z), np.nan, z)
    return z, np.where(constant | ~np.isfinite(log_or), 0.0, log_or)


def _null_logistic(y: np.ndarray, covariate: np.ndarray, iterations: int = 50):
    """Fitted probabilities of presence ~ 1 + covariate, per taxon (labels not used)."""
    design = np.column_stack([np.ones_like(covariate), covariate])
    beta = np.zeros((y.shape[0], 2))
    ridge = 1e-9 * np.eye(2)
    for _ in range(iterations):
        mu = special.expit(np.clip(beta @ design.T, -30.0, 30.0))
        w = np.clip(mu * (1.0 - mu), 1e-12, None)
        score = (y - mu) @ design
        info = np.einsum("tn,np,nq->tpq", w, design, design) + ridge
        step = np.linalg.solve(info, score[:, :, None])[:, :, 0]
        beta += step
        if np.nanmax(np.abs(step)) < 1e-10:
            break
    return special.expit(np.clip(beta @ design.T, -30.0, 30.0))


def _score_core(presence, labels, log_depth):
    """Score test for group in presence ~ 1 + group + log depth (rule R9), every row.

    The null model presence ~ 1 + log depth does not involve the labels, so it is
    fitted once per taxon; each row then needs only group sums. U = g'(y − μ̂) and its
    variance is the efficient information g'Wg − g'WX (X'WX)⁻¹ X'Wg, X = [1, depth].
    """
    y = presence.astype(float)
    depth = (log_depth - log_depth.mean()) / (log_depth.std() or 1.0)
    mu = _null_logistic(y, depth)
    w = mu * (1.0 - mu)
    lab = labels.astype(float)
    score = lab @ (y - mu).T
    a0 = lab @ w.T
    a1 = lab @ (w * depth[None, :]).T
    s00, s01, s11 = w.sum(axis=1), (w * depth).sum(axis=1), (w * depth ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        quad = (a0 ** 2 * s11 - 2.0 * a0 * a1 * s01 + a1 ** 2 * s00) / (s00 * s11 - s01 ** 2)
        info = a0 - quad
        z = score / np.sqrt(info)
    constant = (y.sum(axis=1) == 0) | (y.sum(axis=1) == y.shape[1])
    bad = constant[None, :] | ~np.isfinite(z) | ~(info > 1e-12)
    return np.where(bad, np.nan, z), np.where(bad, 0.0, score)


def _t_p(t, df):
    with np.errstate(invalid="ignore"):
        p = 2.0 * special.stdtr(df, -np.abs(t))
    return np.where(np.isfinite(p), np.clip(p, 0.0, 1.0), 1.0)


def _normal_p(z):
    with np.errstate(invalid="ignore"):
        p = 2.0 * special.ndtr(-np.abs(z))
    return np.where(np.isfinite(p), np.clip(p, 0.0, 1.0), 1.0)


def statistics(method: str, matrix, labels: np.ndarray, log_depth=None):
    """(p, effect) for every row of `labels` on one preprocessed matrix."""
    if method == "wilcoxon":
        z, effect = _wilcoxon_core(matrix.values, labels)
        with np.errstate(invalid="ignore"):
            p = np.clip(2.0 * special.ndtr(-z), 0.0, 1.0)
        return np.where(np.isfinite(p), p, 1.0), effect
    if method == "ttest":
        t, df, effect = _welch_core(matrix.values, labels)
        return _t_p(t, df), effect
    if method == "linear":
        t, df, effect = _linear_core(matrix.values, labels)
        return _t_p(t, df), effect
    if method == "logistic":
        presence = matrix.counts > 0
        if log_depth is not None:
            z, effect = _score_core(presence, labels, log_depth)
        else:
            z, effect = _logistic_core(presence, labels)
        return _normal_p(z), effect
    raise ValueError(f"{method} is not calibrated (plan §27.4: Quick-mode tests only)")


# The public names the tests and the scalar comparison use.
def wilcoxon(values, labels):
    return statistics("wilcoxon", SimpleMatrix(values), labels)


def welch(values, labels):
    return statistics("ttest", SimpleMatrix(values), labels)


def linear(values, labels):
    return statistics("linear", SimpleMatrix(values), labels)


def logistic_2x2(presence, labels):
    return statistics("logistic", SimpleMatrix(presence, presence), labels)


def logistic_score(presence, labels, log_depth):
    return statistics("logistic", SimpleMatrix(presence, presence), labels, log_depth)


@dataclass
class SimpleMatrix:
    """The two fields of a PreparedMatrix the statistics read."""

    values: np.ndarray
    counts: np.ndarray = None


def abs_z(p: np.ndarray) -> np.ndarray:
    """|z| = Φ⁻¹(1 − p/2), from the p-value, floored as effects.signed_z floors it."""
    return -special.ndtri(np.clip(p, P_FLOOR, 1.0) / 2.0)


def _normal_abs_z(z, one_sided_max: bool = False):
    """|z| of a normal-based test as abs_z(p) would give it, without the round trip.

    `one_sided_max`: the Wilcoxon z is that of max(U1, U2) minus the continuity
    correction, so it is ≥ 0 except within the correction, where p is clipped to 1 and
    |z| is 0 — not the absolute value.
    """
    magnitude = np.maximum(z, 0.0) if one_sided_max else np.abs(z)
    return np.where(np.isfinite(z), np.minimum(magnitude, Z_CAP), 0.0)


def _t_abs_z(t, df, bound=None):
    """|z| from a t statistic through its p-value.

    Student's t has heavier tails than the normal for every df > 0 (it is a scale
    mixture of normals with E[scale] ≤ 1, and 2Φ̄(x·s) is convex and decreasing in s),
    so p_t ≥ p_normal and |z| ≤ |t|. A permuted cell whose |t| is at most the running
    maximum therefore cannot raise it, and its |z| is left at 0 rather than computed:
    the maximum is exactly what full evaluation gives, at a fraction of the cost.
    Row 0, the observed labels, is always computed.
    """
    abs_t = np.abs(t)
    need = np.isfinite(abs_t)
    if bound is not None:
        need[1:] &= abs_t[1:] > bound
    out = np.zeros(abs_t.shape)
    df_need = df[need] if np.ndim(df) else df
    with np.errstate(invalid="ignore"):
        p = 2.0 * special.stdtr(df_need, -abs_t[need])
    out[need] = abs_z(np.where(np.isfinite(p), p, 1.0))
    return out


def abs_z_rows(method: str, matrix, labels: np.ndarray, log_depth=None, bound=None):
    """|z| for every row of `labels`, and the observed row's signed z.

    `bound` (rows − 1 × taxa) is the running maximum the permuted rows will be compared
    with; cells that provably cannot exceed it may be returned as 0 (t-based tests).
    """
    if method == "wilcoxon":
        z, effect = _wilcoxon_core(matrix.values, labels)
        zabs = _normal_abs_z(z, one_sided_max=True)
    elif method in ("ttest", "linear"):
        core = _welch_core if method == "ttest" else _linear_core
        t, df, effect = core(matrix.values, labels)
        zabs = _t_abs_z(t, df, bound)
    elif method == "logistic":
        presence = matrix.counts > 0
        z, effect = (_score_core(presence, labels, log_depth) if log_depth is not None
                     else _logistic_core(presence, labels))
        zabs = _normal_abs_z(z)
    else:
        raise ValueError(f"{method} is not calibrated (plan §27.4: Quick-mode tests only)")
    return zabs, np.sign(effect[0]) * zabs[0]


# ---------------------------------------------------------------------------------------
# 5. Tail of the permutation maxima
# ---------------------------------------------------------------------------------------
@dataclass
class TailFit:
    """A generalised Pareto fit to the largest permutation maxima of one taxon."""

    threshold: float
    n_exceed: int
    n_permutations: int
    shape: float
    scale: float
    gof_p: float

    def sf(self, x) -> np.ndarray:
        """P(M ≥ x) for x beyond the threshold."""
        excess = np.maximum(np.asarray(x, dtype=float) - self.threshold, 0.0)
        tail = stats.genpareto.sf(excess, self.shape, loc=0.0, scale=self.scale)
        return self.n_exceed / self.n_permutations * tail


def fit_tail(maxima: np.ndarray) -> TailFit | None:
    """Fit a GPD to the exceedances of the largest maxima, fewer each time until the
    fit is acceptable (Cramér–von Mises p above 0.05), from 250 down to 10.

    None when no fit is acceptable; the caller then keeps the raw permutation p-value.
    """
    ordered = np.sort(np.asarray(maxima, dtype=float))[::-1]
    size = ordered.size
    n_exceed = min(TAIL_START, size // 4)
    n_exceed -= n_exceed % TAIL_STEP
    while n_exceed >= TAIL_MIN_EXCEEDANCES:
        threshold = 0.5 * (ordered[n_exceed - 1] + ordered[n_exceed])
        excess = ordered[:n_exceed] - threshold
        if np.ptp(excess) > 0:
            try:
                shape, _, scale = stats.genpareto.fit(excess, floc=0.0)
                gof = stats.cramervonmises(
                    excess, stats.genpareto(shape, loc=0.0, scale=scale).cdf).pvalue
            except (ValueError, RuntimeError, FloatingPointError):
                gof = 0.0
            if gof > TAIL_GOF_ALPHA and scale > 0:
                return TailFit(float(threshold), int(n_exceed), int(size), float(shape),
                               float(scale), float(gof))
        n_exceed -= TAIL_STEP
    return None


def exceedance_p(maxima: np.ndarray, observed: np.ndarray, tail: TailFit | None = None,
                 fit: bool = True):
    """Permutation p-values P(M ≥ observed) with the tail estimate where sparse.

    Returns (raw, estimate, tail_used, tail). `raw` is (1 + count) / (B + 1); `estimate`
    is the tail-fitted value where fewer than 10 maxima reach the observed one and a fit
    is acceptable, and `raw` otherwise.
    """
    maxima = np.sort(np.asarray(maxima, dtype=float))
    observed = np.atleast_1d(np.asarray(observed, dtype=float))
    b = maxima.size
    counts = b - np.searchsorted(maxima, observed, side="left")
    raw = (1.0 + counts) / (b + 1.0)
    estimate = raw.copy()
    sparse = counts < TAIL_MIN_EXCEEDANCES
    used = np.zeros_like(sparse)
    if sparse.any() and fit:
        tail = tail if tail is not None else fit_tail(maxima)
        if tail is not None:
            beyond = sparse & (observed > tail.threshold)
            estimate[beyond] = np.maximum(tail.sf(observed[beyond]), TAIL_P_FLOOR)
            used = beyond
    return raw, estimate, used, tail


# ---------------------------------------------------------------------------------------
# 6. Discovery across taxa
# ---------------------------------------------------------------------------------------
def step_up(p: np.ndarray, q: float, method: str = "bh"):
    """Selected mask and adjusted values for BH, BY or e-BH (p-to-e, κ = 1/2)."""
    p = np.asarray(p, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    if method == "ebh":
        e = EBH_KAPPA * np.power(np.maximum(p, P_FLOOR), EBH_KAPPA - 1.0)
        order = np.argsort(-e)
        ranks = np.arange(1, m + 1)
        passes = e[order] >= m / (q * ranks)
        k = int(ranks[passes].max()) if passes.any() else 0
        selected = np.zeros(m, dtype=bool)
        selected[order[:k]] = True
        adjusted = np.empty(m)
        adjusted[order] = np.minimum.accumulate((m / (ranks * e[order]))[::-1])[::-1]
        return selected, np.minimum(adjusted, 1.0)
    factor = 1.0 if method == "bh" else float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(p)
    ranked = p[order] * m * factor / np.arange(1, m + 1)
    adjusted = np.empty(m)
    adjusted[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    return adjusted <= q, adjusted


# ---------------------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------------------
@dataclass
class CalibrationResult:
    """Everything the calibrated run reports. `taxa` is one row per tested taxon;
    `rejected` one row per (specification, taxon) pair certified within a selected
    family; `specs_excluded` counts specifications outside the calibrated family."""

    n_permutations: int
    seed: int
    q: float
    discovery: str
    scheme: str
    strata: str | None
    taxa: pd.DataFrame
    rejected: pd.DataFrame
    n_taxa_tested: int
    n_selected: int
    within_threshold: float
    specs_calibrated: int
    specs_excluded: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    #: SHA-256 of the (B + 1) × n label matrix, so a rerun can prove it used the same one.
    permutations_sha256: str = ""

    @property
    def n_certified(self) -> int:
        return int(self.taxa["certified_robust"].sum()) if len(self.taxa) else 0


def _fits(run):
    """fit key -> (matrix key, method, depth_adjusted, [spec ids]) for the calibrated
    family. Specifications differing only in their FDR setting share one fit: the
    calibrated procedure replaces per-specification FDR, so they are one hypothesis."""
    fits: dict = {}
    excluded: dict = {}
    for spec_id, spec in enumerate(run.specs):
        if spec.method not in CALIBRATED_METHODS:
            excluded[spec.method] = excluded.get(spec.method, 0) + 1
            continue
        if spec.covariates:
            excluded["covariate-adjusted"] = excluded.get("covariate-adjusted", 0) + 1
            continue
        key = spec.fit_key
        entry = fits.setdefault(key, {"matrix_key": spec.matrix_key, "method": spec.method,
                                      "depth": bool(getattr(spec, "depth_adjusted", False)),
                                      "spec_ids": []})
        entry["spec_ids"].append(spec_id)
    return fits, excluded


def calibrate(run, dataset, weights: np.ndarray, *, n_permutations: int = DEFAULT_PERMUTATIONS,
              seed: int = DEFAULT_SEED, q: float = DEFAULT_Q, discovery: str = "bh",
              strata=None, strata_name: str | None = None, scheme: str = "decision_tree",
              progress=None) -> CalibrationResult:
    """Run §27.3 on a finished multiverse run. `weights` are the run's specification
    weights (one per spec, the scheme the tiers use)."""
    from .preprocess import MatrixBuilder

    if discovery not in DISCOVERY_METHODS:
        raise ValueError(f"Unknown discovery procedure: {discovery}")
    started = time.perf_counter()
    timings: dict = {}
    builder = MatrixBuilder(dataset)
    labels = permutation_matrix(builder.groups, n_permutations, seed, strata)
    labels_sha256 = hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()
    libraries = builder.base_counts.sum(axis=0)
    log_depth = np.log(np.maximum(libraries, 1.0))

    fits, excluded = _fits(run)
    by_matrix: dict = {}
    for key, entry in fits.items():
        by_matrix.setdefault(entry["matrix_key"], []).append((key, entry))

    n_global = len(run.taxa_names)
    maxima = np.zeros((n_permutations, n_global))
    observed_max = np.full(n_global, -np.inf)
    observed: dict = {}           # fit key -> (global taxa, signed z)
    tested = np.zeros(n_global, dtype=bool)
    offsets = {"input": 0, "genus": len(builder.taxa_names("input"))}

    keys = sorted(by_matrix, key=lambda k: (str(k[0]), k[1] or 0, k[2], k[3], k[4]))
    for position, matrix_key in enumerate(keys):
        matrix = builder.build(*matrix_key)
        if not matrix.usable:
            continue
        rows = labels[:, matrix.sample_idx]
        global_idx = matrix.taxa_idx + offsets[matrix_key[2]]
        for fit_key, entry in by_matrix[matrix_key]:
            depth = log_depth[matrix.sample_idx] if entry["depth"] else None
            current = maxima[:, global_idx]           # a copy: array indexing copies
            z, signed = abs_z_rows(entry["method"], matrix, rows, depth, bound=current)
            maxima[:, global_idx] = np.maximum(current, z[1:])
            observed_max[global_idx] = np.maximum(observed_max[global_idx], z[0])
            observed[fit_key] = (global_idx, signed)
            tested[global_idx] = True
        if progress and position % max(1, len(keys) // 20) == 0:
            progress(position / max(1, len(keys)),
                     f"Permuting: matrix {position + 1} of {len(keys)}")
    timings["permutations"] = round(time.perf_counter() - started, 2)

    # 5. family p-values, with the tail where the permutations run out.
    clock = time.perf_counter()
    taxa = np.flatnonzero(tested)
    raw = np.ones(taxa.size)
    estimate = np.ones(taxa.size)
    used = np.zeros(taxa.size, dtype=bool)
    tails: dict = {}
    for i, j in enumerate(taxa):
        r, e, u, tail = exceedance_p(maxima[:, j], observed_max[j])
        raw[i], estimate[i], used[i] = r[0], e[0], u[0]
        if tail is not None:
            tails[j] = tail
    timings["tails"] = round(time.perf_counter() - clock, 2)

    # 6. discovery across taxa.
    selected, adjusted = step_up(estimate, q, discovery)
    m, r_sel = int(taxa.size), int(selected.sum())
    threshold = q * r_sel / m if m else 0.0

    # 7-8. within the selected families.
    clock = time.perf_counter()
    spec_weights = np.asarray(weights, dtype=float)
    rejected_rows = []
    cr = np.zeros(taxa.size)
    sign = np.full(taxa.size, np.nan)
    n_rejected = np.zeros(taxa.size, dtype=int)
    position_of = {int(j): i for i, j in enumerate(taxa)}
    per_taxon: dict = {int(j): [] for j in taxa[selected]}
    tested_weight = np.zeros(n_global)
    for fit_key, (global_idx, z_signed) in observed.items():
        ids = fits[fit_key]["spec_ids"]
        w = spec_weights[ids]
        tested_weight[global_idx] += w.sum()
        for local, j in enumerate(global_idx):
            if int(j) in per_taxon:
                per_taxon[int(j)].append((ids, w, z_signed[local]))
    for j, entries in per_taxon.items():
        i = position_of[j]
        z_abs = np.array([abs(z) for _, _, z in entries])
        _, p_tilde, _, _ = exceedance_p(maxima[:, j], z_abs, tail=tails.get(j))
        weight_rejected, positive, negative = 0.0, 0.0, 0.0
        for (ids, w, z), pt in zip(entries, p_tilde, strict=True):
            if pt <= threshold:
                weight_rejected += w.sum()
                positive += w.sum() if z > 0 else 0.0
                negative += w.sum() if z < 0 else 0.0
                n_rejected[i] += len(ids)
                rejected_rows.extend((int(s), j, float(z), float(pt)) for s in ids)
        cr[i] = weight_rejected / tested_weight[j] if tested_weight[j] > 0 else 0.0
        if weight_rejected > 0:
            sign[i] = max(positive, negative) / weight_rejected
    timings["within"] = round(time.perf_counter() - clock, 2)

    frame = pd.DataFrame({
        "taxon_id": taxa.astype(int),
        "max_abs_z": observed_max[taxa],
        "p_family_raw": raw,
        "p_family": estimate,
        "p_family_tail_fitted": used,
        "q_family": adjusted,
        "selected": selected,
        "certified_share": cr,
        "certified_sign_agreement": sign,
        "n_specs_rejected": n_rejected,
    })
    frame["certified_robust"] = (frame["selected"] & (frame["certified_share"] >= CERTIFIED_CR)
                                 & (frame["certified_sign_agreement"].fillna(0)
                                    >= CERTIFIED_SIGN))
    rejected = pd.DataFrame(rejected_rows,
                            columns=["spec_id", "taxon", "signed_z", "p_within"])
    timings["total"] = round(time.perf_counter() - started, 2)
    return CalibrationResult(
        n_permutations=int(n_permutations), seed=int(seed), q=float(q),
        discovery=discovery, scheme=scheme, strata=strata_name, taxa=frame,
        rejected=rejected, n_taxa_tested=m, n_selected=r_sel, within_threshold=threshold,
        specs_calibrated=int(sum(len(e["spec_ids"]) for e in fits.values())),
        specs_excluded=excluded, timings=timings, permutations_sha256=labels_sha256,
    )


#: The per-taxon columns a calibrated run adds to the robustness table.
TABLE_COLUMNS = {
    "p_family": "calibrated_p",
    "p_family_tail_fitted": "calibrated_p_tail_fitted",
    "q_family": "calibrated_q",
    "selected": "calibrated_discovery",
    "certified_share": "certified_share",
    "certified_sign_agreement": "certified_sign_agreement",
    "n_specs_rejected": "n_specs_certified",
    "certified_robust": "certified_robust",
}


def attach(summary, result: CalibrationResult):
    """Put a calibration beside the descriptive tier, never in place of it (§27.3
    step 10): the robustness table gains the calibrated columns and the summary keeps
    the whole result. A taxon outside every calibrated family gets no calibrated value
    and is not certified."""
    columns = result.taxa[["taxon_id", *TABLE_COLUMNS]].rename(columns=TABLE_COLUMNS)
    table = summary.table.drop(columns=[c for c in TABLE_COLUMNS.values()
                                        if c in summary.table.columns])
    table = table.merge(columns, on="taxon_id", how="left")
    for column in ("calibrated_discovery", "certified_robust", "calibrated_p_tail_fitted"):
        table[column] = table[column].astype("boolean").fillna(False).astype(bool)
    table["n_specs_certified"] = table["n_specs_certified"].fillna(0).astype(int)
    table["certified_share"] = table["certified_share"].fillna(0.0)
    summary.table = table
    summary.calibration = result
    return summary


ASSUMPTIONS = (
    "Permutation validity requires the samples to be exchangeable between the groups "
    "under the null hypothesis: no batch, site or other structure that the group labels "
    "are confounded with, unless it was given as a stratum.",
    "The per-taxon family test (single-step maxT) relies on subset pivotality, which a "
    "strong compositional shift in other taxa can violate; how far it holds is what "
    "validation V9 measures.",
    "Covariate adjustment is not calibrated: covariate-mode grids, ANCOM-BC, ALDEx2 and "
    "PyDESeq2 are outside the calibrated family.",
)
