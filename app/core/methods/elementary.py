"""The four elementary methods (SPEC §10, fork 5, tiers 1-4).

These are Pelto et al.'s "elementary methods" — the ones that turned out to be the most
replicable. All four are computed vectorised across taxa: one call per matrix, not one
per taxon, which is what keeps Quick mode inside the 90-second budget.

Where a closed form is exact, it is used instead of an iterative fit. Both substitutions
are verified against statsmodels in `tests/test_methods.py`:
  * `linear`  with a binary predictor and no covariates == Student's pooled t-test;
  * `logistic` with a binary predictor and no covariates == the 2x2 Wald test.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..models import EFFECT_NATIVE_TYPE, FitResult
from .design import build_design, residualise

MAX_IRLS_ITERATIONS = 40
IRLS_TOLERANCE = 1e-8


def _finish(p_values, effect, method, taxa_idx, n_input):
    p = np.asarray(p_values, dtype=float)
    p = np.where(np.isfinite(p), np.clip(p, 0.0, 1.0), 1.0)
    e = np.asarray(effect, dtype=float)
    e = np.where(np.isfinite(e), e, 0.0)
    return FitResult(
        taxa=taxa_idx,
        p_raw=p,
        effect_native=e,
        effect_harmonized=np.zeros_like(p),  # filled in by the runner (§14)
        effect_native_type=EFFECT_NATIVE_TYPE[method],
        n_taxa_input=n_input,
    )


# --------------------------------------------------------------------------
def run_wilcoxon(matrix, covariate_frame=None) -> FitResult:
    values = matrix.values
    if covariate_frame is not None and covariate_frame.shape[1] > 0:
        design, group_col, _ = build_design(matrix.groups, covariate_frame)
        values = residualise(values, design, group_col)
    a = values[:, matrix.groups == 0]
    b = values[:, matrix.groups == 1]
    with np.errstate(invalid="ignore"):
        result = stats.mannwhitneyu(a, b, axis=1, alternative="two-sided", method="asymptotic")
    n_a, n_b = a.shape[1], b.shape[1]
    # Rank-biserial correlation, signed so that positive = higher in group B.
    effect = 1.0 - 2.0 * np.asarray(result.statistic, dtype=float) / max(1, n_a * n_b)
    return _finish(result.pvalue, effect, "wilcoxon", matrix.taxa_idx, matrix.n_taxa)


def run_ttest(matrix, covariate_frame=None) -> FitResult:
    values = matrix.values
    if covariate_frame is not None and covariate_frame.shape[1] > 0:
        design, group_col, _ = build_design(matrix.groups, covariate_frame)
        values = residualise(values, design, group_col)
    a = values[:, matrix.groups == 0]
    b = values[:, matrix.groups == 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        result = stats.ttest_ind(b, a, axis=1, equal_var=False)
    effect = b.mean(axis=1) - a.mean(axis=1)
    return _finish(result.pvalue, effect, "ttest", matrix.taxa_idx, matrix.n_taxa)


# --------------------------------------------------------------------------
def run_linear(matrix, covariate_frame=None) -> FitResult:
    """OLS of abundance on group (+ covariates), fitted for all taxa at once.

    One shared design matrix means the whole grid of taxa is a single solve:
    beta = (X'X)^-1 X' Y'.
    """
    design, group_col, _ = build_design(matrix.groups, covariate_frame)
    y = matrix.values
    n, p = design.shape
    df = n - p
    if df <= 0:
        return _finish(np.ones(y.shape[0]), np.zeros(y.shape[0]), "linear",
                       matrix.taxa_idx, matrix.n_taxa)

    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = y @ design @ xtx_inv.T  # taxa x p
    residuals = y - beta @ design.T
    sigma2 = (residuals ** 2).sum(axis=1) / df
    se = np.sqrt(np.maximum(sigma2 * xtx_inv[group_col, group_col], 1e-300))
    coefficient = beta[:, group_col]
    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat = np.where(se > 0, coefficient / se, 0.0)
        p_values = 2.0 * stats.t.sf(np.abs(t_stat), df)
    p_values = np.where(se > 0, p_values, 1.0)
    return _finish(p_values, coefficient, "linear", matrix.taxa_idx, matrix.n_taxa)


# --------------------------------------------------------------------------
def _logistic_2x2(presence: np.ndarray, groups: np.ndarray):
    """Exact Wald test for presence ~ group from the 2x2 table.

    With a single binary predictor the logistic MLE is the saturated-table estimate,
    so this is the same answer statsmodels' GLM returns — without 380 iterative fits.
    Haldane-Anscombe (+0.5) is applied only when a cell is empty, where the MLE is
    infinite and any implementation must regularise.
    """
    a_mask = groups == 0
    b_mask = groups == 1
    present_a = presence[:, a_mask].sum(axis=1).astype(float)
    absent_a = a_mask.sum() - present_a
    present_b = presence[:, b_mask].sum(axis=1).astype(float)
    absent_b = b_mask.sum() - present_b

    empty = (present_a == 0) | (absent_a == 0) | (present_b == 0) | (absent_b == 0)
    correction = np.where(empty, 0.5, 0.0)
    pa, na = present_a + correction, absent_a + correction
    pb, nb = present_b + correction, absent_b + correction

    with np.errstate(divide="ignore", invalid="ignore"):
        log_or = np.log((pb * na) / (pa * nb))
        se = np.sqrt(1.0 / pa + 1.0 / na + 1.0 / pb + 1.0 / nb)
        z = np.where(se > 0, log_or / se, 0.0)
        p_values = 2.0 * stats.norm.sf(np.abs(z))

    # No variation in presence/absence: nothing to test.
    constant = (present_a + present_b == 0) | (absent_a + absent_b == 0)
    p_values = np.where(constant | ~np.isfinite(p_values), 1.0, p_values)
    log_or = np.where(constant | ~np.isfinite(log_or), 0.0, log_or)
    return p_values, log_or


def _logistic_irls(presence: np.ndarray, design: np.ndarray, group_col: int):
    """Batched Newton-Raphson: one shared design, one response per taxon."""
    y = presence.astype(float)
    n_taxa, n = y.shape
    p = design.shape[1]
    beta = np.zeros((n_taxa, p), dtype=float)
    ridge = 1e-6 * np.eye(p)

    for _ in range(MAX_IRLS_ITERATIONS):
        eta = np.clip(beta @ design.T, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-9, None)
        # Score and information, batched over taxa.
        score = (y - mu) @ design  # taxa x p
        information = np.einsum("tn,np,nq->tpq", w, design, design) + ridge
        try:
            step = np.linalg.solve(information, score[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:  # pragma: no cover
            break
        beta = beta + step
        if np.nanmax(np.abs(step)) < IRLS_TOLERANCE:
            break

    eta = np.clip(beta @ design.T, -30.0, 30.0)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(mu * (1.0 - mu), 1e-9, None)
    information = np.einsum("tn,np,nq->tpq", w, design, design) + ridge
    covariance = np.linalg.pinv(information)
    variance = covariance[:, group_col, group_col]
    coefficient = beta[:, group_col]
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(np.maximum(variance, 0.0))
        z = np.where(se > 0, coefficient / se, 0.0)
        p_values = 2.0 * stats.norm.sf(np.abs(z))
    constant = (y.sum(axis=1) == 0) | (y.sum(axis=1) == n)
    p_values = np.where(constant | ~np.isfinite(p_values), 1.0, p_values)
    coefficient = np.where(constant | ~np.isfinite(coefficient), 0.0, coefficient)
    return p_values, coefficient


def run_logistic(matrix, covariate_frame=None) -> FitResult:
    """Logistic regression on presence/absence — transform-invariant by construction."""
    presence = matrix.counts > 0
    if covariate_frame is not None and covariate_frame.shape[1] > 0:
        design, group_col, _ = build_design(matrix.groups, covariate_frame)
        if design.shape[1] > 2:
            p_values, coefficient = _logistic_irls(presence, design, group_col)
            return _finish(p_values, coefficient, "logistic", matrix.taxa_idx, matrix.n_taxa)
    p_values, coefficient = _logistic_2x2(presence, matrix.groups)
    return _finish(p_values, coefficient, "logistic", matrix.taxa_idx, matrix.n_taxa)
