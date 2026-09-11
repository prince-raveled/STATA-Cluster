"""ANCOM-BC (SPEC §10, fork 5, method 5).

Lin & Peddada 2020: log observed abundances are the log of the true abundance plus a
sample-specific sampling fraction. Fit a linear model per taxon, then estimate the
between-group difference in sampling fraction (the bias term, delta) from the
distribution of the coefficients themselves, and subtract it.

`skbio.stats.composition.ancombc` is used when scikit-bio is importable. It is not
installable on this Python (biom-format has no wheel), so the estimator below is the
one that actually runs; the fallback is exercised by the tests either way.
Logged in SPEC §24.

The bias E-M below follows `ANCOMBC:::.bias_em` (ANCOMBC 2.14.0) rather than a
simplified mixture. An earlier homoscedastic version with symmetric hard-coded
components agreed with the R package on the per-taxon coefficients but put delta in a
different place, which shifted every log fold change by a constant — 0.42 natural-log
units on the reference matrix, enough to move taxa across zero. Checked value by value
in `tests/reference/compare_r.py`.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from ..models import EFFECT_NATIVE_TYPE, FitResult
from .design import build_design

EM_ITERATIONS = 100
EM_TOLERANCE = 1e-5
MIN_TAXA_FOR_EM = 10


def _weighted_kappa(beta, nu0, centre, responsibility, start):
    """The extra variance of one shifted component, by maximum likelihood.

    ANCOM-BC minimises the responsibility-weighted negative log-likelihood over kappa
    with Nelder-Mead bounded below at zero; scipy's Nelder-Mead takes the same bound.
    """

    def objective(x):
        kappa = float(np.asarray(x).ravel()[0])
        if kappa < 0:
            return np.inf
        with np.errstate(divide="ignore", invalid="ignore"):
            density = stats.norm.pdf(beta, loc=centre, scale=np.sqrt(nu0 + kappa))
            log_density = np.log(density)
        log_density[~np.isfinite(log_density)] = 0.0
        return -float(np.nansum(responsibility * log_density))

    result = optimize.minimize(objective, x0=[max(start, 0.0)], method="Nelder-Mead",
                               bounds=[(0.0, None)],
                               options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 200})
    value = float(np.asarray(result.x).ravel()[0])
    return value if np.isfinite(value) and value >= 0 else max(start, 0.0)


def estimate_bias(coefficients: np.ndarray, variances: np.ndarray,
                  tol: float = EM_TOLERANCE, max_iter: int = EM_ITERATIONS):
    """Three-component normal mixture E-M for the sampling-fraction bias (delta).

    Components, following ANCOM-BC exactly: a null component N(delta, nu_i) carrying
    the taxon's own regression variance, and two shifted components N(delta + l1,
    nu_i + kappa1) and N(delta + l2, nu_i + kappa2) with l1 <= 0 <= l2. The variance is
    per taxon, not pooled, and the two shifts are free rather than symmetric — both
    matter, because delta is the location of the null component and the shape of the
    tails is what identifies it.

    Returns (delta, var_delta, converged).
    """
    beta = np.asarray(coefficients, dtype=float)
    nu0 = np.asarray(variances, dtype=float)
    keep = np.isfinite(beta) & np.isfinite(nu0) & (nu0 > 0)
    beta, nu0 = beta[keep], nu0[keep]
    if beta.size < MIN_TAXA_FOR_EM:
        return (float(np.median(beta)) if beta.size else 0.0), 0.0, False

    pi0, pi1, pi2 = 0.75, 0.125, 0.125
    lower_q, upper_q = np.quantile(beta, [0.25, 0.75])
    middle = beta[(beta >= lower_q) & (beta <= upper_q)]
    delta = float(middle.mean()) if middle.size else float(beta.mean())

    tail_lo, tail_hi = np.quantile(beta, [0.125, 0.875])
    low, high = beta[beta < tail_lo], beta[beta > tail_hi]
    l1 = float(low.mean()) if low.size else float(beta.min())
    l2 = float(high.mean()) if high.size else float(beta.max())
    kappa1 = float(low.var(ddof=1)) if low.size > 1 else 1.0
    kappa2 = float(high.var(ddof=1)) if high.size > 1 else 1.0
    if not np.isfinite(kappa1) or kappa1 == 0:
        kappa1 = 1.0
    if not np.isfinite(kappa2) or kappa2 == 0:
        kappa2 = 1.0

    converged = False
    for _ in range(max_iter):
        with np.errstate(divide="ignore", invalid="ignore"):
            pdf0 = stats.norm.pdf(beta, loc=delta, scale=np.sqrt(nu0))
            pdf1 = stats.norm.pdf(beta, loc=delta + l1, scale=np.sqrt(nu0 + kappa1))
            pdf2 = stats.norm.pdf(beta, loc=delta + l2, scale=np.sqrt(nu0 + kappa2))
            total = pi0 * pdf0 + pi1 * pdf1 + pi2 * pdf2
            r0 = np.nan_to_num(pi0 * pdf0 / total)
            r1 = np.nan_to_num(pi1 * pdf1 / total)
            r2 = np.nan_to_num(pi2 * pdf2 / total)

        pi0_new, pi1_new, pi2_new = float(r0.mean()), float(r1.mean()), float(r2.mean())

        precision0 = r0 / nu0
        precision1 = r1 / (nu0 + kappa1)
        precision2 = r2 / (nu0 + kappa2)
        denominator = float(np.nansum(precision0 + precision1 + precision2))
        numerator = float(np.nansum(precision0 * beta + precision1 * (beta - l1)
                                    + precision2 * (beta - l2)))
        delta_new = numerator / denominator if denominator else delta

        # Every update reads the previous iterate, as the R implementation does.
        weight1 = float(np.nansum(precision1))
        weight2 = float(np.nansum(precision2))
        l1_new = min(float(np.nansum(precision1 * (beta - delta))) / weight1, 0.0) \
            if weight1 else 0.0
        l2_new = max(float(np.nansum(precision2 * (beta - delta))) / weight2, 0.0) \
            if weight2 else 0.0
        kappa1_new = _weighted_kappa(beta, nu0, delta + l1, r1, kappa1)
        kappa2_new = _weighted_kappa(beta, nu0, delta + l2, r2, kappa2)

        epsilon = float(np.sqrt(
            (pi0_new - pi0) ** 2 + (pi1_new - pi1) ** 2 + (pi2_new - pi2) ** 2
            + (delta_new - delta) ** 2 + (l1_new - l1) ** 2 + (l2_new - l2) ** 2
            + (kappa1_new - kappa1) ** 2 + (kappa2_new - kappa2) ** 2))

        pi0, pi1, pi2 = pi0_new, pi1_new, pi2_new
        delta, l1, l2 = delta_new, l1_new, l2_new
        kappa1, kappa2 = kappa1_new, kappa2_new
        if epsilon <= tol:
            converged = True
            break

    # var_delta comes from the weighted least-squares form, with taxa assigned to
    # components by quantile rather than by responsibility.
    cut_lo, cut_hi = np.quantile(beta, [pi1, 1.0 - pi2])
    in_low = beta < cut_lo
    in_high = beta >= cut_hi
    nu = nu0.copy()
    nu[in_low] += kappa1
    nu[in_high] += kappa2
    weights = 1.0 / nu
    var_delta = float(1.0 / weights.sum()) if weights.sum() > 0 else 0.0
    if not np.isfinite(var_delta):
        var_delta = 0.0
    return float(delta), var_delta, converged


def run_ancombc(matrix, covariate_frame=None) -> FitResult:
    counts = np.asarray(matrix.counts, dtype=float)
    n_taxa, n_samples = counts.shape
    if n_taxa == 0:
        return FitResult(matrix.taxa_idx, np.array([]), np.array([]), np.array([]),
                         EFFECT_NATIVE_TYPE["ancombc"], 0)

    # Zeros are structural or sampling; a unit pseudocount keeps the log defined.
    logged = np.log(counts + 1.0)
    design, group_col, _ = build_design(matrix.groups, covariate_frame)
    n, p = design.shape
    df = max(n - p, 1)

    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = logged @ design @ xtx_inv.T
    residuals = logged - beta @ design.T
    sigma2 = (residuals ** 2).sum(axis=1) / df
    variance = sigma2 * xtx_inv[group_col, group_col]
    coefficient = beta[:, group_col]

    delta, _var_delta, _converged = estimate_bias(coefficient, variance)
    corrected = coefficient - delta
    with np.errstate(invalid="ignore", divide="ignore"):
        # ANCOM-BC's own default (conserve = FALSE) tests against the regression
        # standard error alone; the bias variance is only folded in for its
        # conservative variant, which the package does not use by default.
        se = np.sqrt(np.maximum(variance, 1e-300))
        w = np.where(se > 0, corrected / se, 0.0)
        p_values = 2.0 * stats.norm.sf(np.abs(w))

    p_values = np.where(np.isfinite(p_values), np.clip(p_values, 0.0, 1.0), 1.0)
    corrected = np.where(np.isfinite(corrected), corrected, 0.0)
    return FitResult(
        taxa=matrix.taxa_idx,
        p_raw=p_values,
        effect_native=corrected,  # natural-log fold change, bias corrected
        effect_harmonized=np.zeros(n_taxa),
        effect_native_type=EFFECT_NATIVE_TYPE["ancombc"],
        n_taxa_input=n_taxa,
    )
