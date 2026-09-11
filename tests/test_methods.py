"""DA methods — SPEC §10 fork 5, and the §14 harmonisation sanity check.

The elementary methods are vectorised closed forms rather than per-taxon iterative
fits. These tests are what licenses that: each one is checked against the statsmodels
model SPEC §10 names.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from app.core.effects import harmonized_effect, run_pseudocount
from app.core.fdr import adjust
from app.core.methods import available_methods, run_method
from app.core.methods.design import build_design, residualise
from app.core.preprocess import MatrixBuilder


@pytest.fixture(scope="module")
def matrix(dataset):
    return MatrixBuilder(dataset).build("none", None, "input", 0.10, "raw")


# --------------------------------------------------------------------------
# Vectorised closed forms vs the statsmodels models §10 names
# --------------------------------------------------------------------------
def test_linear_matches_statsmodels_ols(matrix):
    import statsmodels.api as sm

    fit = run_method("linear", matrix)
    design, _, _ = build_design(matrix.groups, None)
    for i in range(0, matrix.n_taxa, 7):
        reference = sm.OLS(matrix.values[i], design).fit()
        assert np.isclose(fit.effect_native[i], reference.params[1], rtol=1e-9, atol=1e-12)
        assert np.isclose(fit.p_raw[i], reference.pvalues[1], rtol=1e-7, atol=1e-12)


def test_linear_with_covariates_matches_statsmodels_ols(dataset):
    import statsmodels.api as sm

    builder = MatrixBuilder(dataset)
    matrix = builder.build("none", None, "input", 0.10, "clr")
    frame = dataset.metadata.iloc[matrix.sample_idx][["age", "sex"]]
    fit = run_method("linear", matrix, frame)
    design, group_col, _ = build_design(matrix.groups, frame)
    for i in range(0, matrix.n_taxa, 11):
        reference = sm.OLS(matrix.values[i], design).fit()
        assert np.isclose(fit.effect_native[i], reference.params[group_col], rtol=1e-9)
        assert np.isclose(fit.p_raw[i], reference.pvalues[group_col], rtol=1e-6)


def test_logistic_matches_statsmodels_glm(matrix):
    import statsmodels.api as sm

    fit = run_method("logistic", matrix)
    design, _, _ = build_design(matrix.groups, None)
    presence = (matrix.counts > 0).astype(float)
    checked = 0
    for i in range(matrix.n_taxa):
        y = presence[i]
        # Skip separated tables: the MLE is infinite there and every implementation
        # must regularise, so there is no common reference to compare against.
        table = [
            y[matrix.groups == 0].sum(), (1 - y)[matrix.groups == 0].sum(),
            y[matrix.groups == 1].sum(), (1 - y)[matrix.groups == 1].sum(),
        ]
        if min(table) == 0:
            continue
        reference = sm.GLM(y, design, family=sm.families.Binomial()).fit()
        assert np.isclose(fit.effect_native[i], reference.params[1], rtol=1e-6, atol=1e-9)
        assert np.isclose(fit.p_raw[i], reference.pvalues[1], rtol=1e-5, atol=1e-9)
        checked += 1
    assert checked >= 5, "no unseparated taxa were available to compare"


def test_logistic_with_covariates_matches_statsmodels_glm(dataset):
    import statsmodels.api as sm

    matrix = MatrixBuilder(dataset).build("none", None, "input", 0.20, "raw")
    frame = dataset.metadata.iloc[matrix.sample_idx][["age"]]
    fit = run_method("logistic", matrix, frame)
    design, group_col, _ = build_design(matrix.groups, frame)
    presence = (matrix.counts > 0).astype(float)
    checked = 0
    separated = 0
    for i in range(matrix.n_taxa):
        y = presence[i]
        # Any empty cell of the 2x2 separates the group coefficient: its MLE runs to
        # infinity, and every implementation regularises it differently. Both still
        # report p ~ 1, which is the part that matters — asserted below.
        cells = [
            y[matrix.groups == 0].sum(), (1 - y)[matrix.groups == 0].sum(),
            y[matrix.groups == 1].sum(), (1 - y)[matrix.groups == 1].sum(),
        ]
        try:
            reference = sm.GLM(y, design, family=sm.families.Binomial()).fit()
        except Exception:
            continue
        if min(cells) == 0:
            separated += 1
            assert fit.p_raw[i] > 0.5 and reference.pvalues[group_col] > 0.5
            continue
        assert np.isclose(fit.effect_native[i], reference.params[group_col],
                          rtol=1e-4, atol=1e-6)
        assert np.isclose(fit.p_raw[i], reference.pvalues[group_col], rtol=1e-3, atol=1e-6)
        checked += 1
    assert checked >= 3


def test_wilcoxon_matches_scipy_per_taxon(matrix):
    fit = run_method("wilcoxon", matrix)
    a = matrix.values[:, matrix.groups == 0]
    b = matrix.values[:, matrix.groups == 1]
    for i in range(0, matrix.n_taxa, 9):
        reference = stats.mannwhitneyu(a[i], b[i], alternative="two-sided",
                                       method="asymptotic")
        assert np.isclose(fit.p_raw[i], reference.pvalue)


def test_ttest_matches_scipy_per_taxon(matrix):
    fit = run_method("ttest", matrix)
    a = matrix.values[:, matrix.groups == 0]
    b = matrix.values[:, matrix.groups == 1]
    for i in range(0, matrix.n_taxa, 9):
        reference = stats.ttest_ind(b[i], a[i], equal_var=False)
        assert np.isclose(fit.p_raw[i], reference.pvalue)
        assert np.isclose(fit.effect_native[i], b[i].mean() - a[i].mean())


# --------------------------------------------------------------------------
# Shared contracts
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["wilcoxon", "ttest", "logistic", "linear",
                                    "ancombc", "aldex2"])
def test_every_method_returns_finite_bounded_p_values(matrix, method):
    fit = run_method(method, matrix)
    assert fit.p_raw.shape == (matrix.n_taxa,)
    assert np.all(np.isfinite(fit.p_raw))
    assert np.all((fit.p_raw >= 0) & (fit.p_raw <= 1))
    assert np.all(np.isfinite(fit.effect_native))


def test_logistic_is_invariant_to_the_transform(dataset):
    """Presence/absence cannot depend on the transform — the premise of the §11 rule."""
    builder = MatrixBuilder(dataset)
    raw = run_method("logistic", builder.build("none", None, "input", 0.10, "raw"))
    clr = run_method("logistic", builder.build("none", None, "input", 0.10, "clr"))
    np.testing.assert_allclose(raw.p_raw, clr.p_raw)


def test_methods_direction_agrees_with_the_harmonised_effect(matrix):
    """A method that says 'up in B' must not disagree in sign with the effect estimator."""
    effect = harmonized_effect(matrix.rel, matrix.groups)
    strong = np.argsort(np.abs(effect))[::-1][:15]
    for method in ("ttest", "linear", "wilcoxon"):
        fit = run_method(method, matrix)
        agree = np.sign(fit.effect_native[strong]) == np.sign(effect[strong])
        assert agree.mean() >= 0.8, method


def test_residualisation_removes_covariate_signal(dataset):
    matrix = MatrixBuilder(dataset).build("none", None, "input", 0.10, "clr")
    frame = dataset.metadata.iloc[matrix.sample_idx][["age"]]
    design, group_col, _ = build_design(matrix.groups, frame)
    residual = residualise(matrix.values, design, group_col)
    age = design[:, 2]
    for i in range(0, matrix.n_taxa, 13):
        assert abs(np.corrcoef(residual[i], age)[0, 1]) < 1e-8


# --------------------------------------------------------------------------
# SPEC §14 sanity check: PyDESeq2's own log2FC vs the harmonised estimator
# --------------------------------------------------------------------------
@pytest.mark.skipif(available_methods()["pydeseq2"] != "", reason="pydeseq2 not installed")
def test_harmonised_effect_correlates_with_pydeseq2_native(dataset):
    """SPEC §14: 'for PyDESeq2, effect_native and effect_harmonized should correlate
    >0.9. If they do not, the harmonisation is wrong.'"""
    builder = MatrixBuilder(dataset)
    matrix = builder.build("none", None, "input", 0.10, "raw")
    fit = run_method("pydeseq2", matrix)
    harmonised = harmonized_effect(matrix.rel, matrix.groups,
                                   run_pseudocount(builder.base_counts))
    keep = np.isfinite(fit.effect_native) & np.isfinite(harmonised)
    correlation = np.corrcoef(fit.effect_native[keep], harmonised[keep])[0, 1]
    assert correlation > 0.9, f"harmonisation is wrong: r={correlation:.3f}"


def test_aldex2_is_reproducible(matrix):
    from app.core.methods.aldex2 import run_aldex2

    a = run_aldex2(matrix, n_instances=16, seed=5)
    b = run_aldex2(matrix, n_instances=16, seed=5)
    np.testing.assert_allclose(a.p_raw, b.p_raw)


def test_ancombc_bias_estimate_shifts_the_coefficients():
    """A uniform sampling-fraction shift must be absorbed, not reported as signal."""
    from app.core.methods.ancombc import estimate_bias

    rng = np.random.default_rng(1)
    coefficients = np.concatenate([
        rng.normal(0.8, 0.05, 200),   # the null component, offset by the bias
        rng.normal(2.6, 0.2, 15),     # genuinely up
        rng.normal(-1.2, 0.2, 15),    # genuinely down
    ])
    delta, var_delta, _ = estimate_bias(coefficients, np.full(coefficients.size, 0.01))
    assert 0.6 < delta < 1.0
    assert var_delta >= 0


def test_fdr_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(8)
    p = np.clip(rng.beta(0.4, 6.0, size=400), 1e-12, 1.0)
    np.testing.assert_allclose(adjust(p, "bh"), multipletests(p, method="fdr_bh")[1],
                               rtol=1e-10)
    np.testing.assert_allclose(adjust(p, "by"), multipletests(p, method="fdr_by")[1],
                               rtol=1e-10)


def test_by_is_more_conservative_than_bh():
    rng = np.random.default_rng(9)
    p = np.clip(rng.beta(0.4, 6.0, size=300), 1e-12, 1.0)
    assert np.all(adjust(p, "by") >= adjust(p, "bh") - 1e-12)
