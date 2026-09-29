"""Calibrated multiverse inference — v3 plan §27 (app/core/inference.py).

Acceptance criteria of §27.6 that can be checked without data:
  * the vectorised statistics equal the scalar implementations to 1e-10, on the
    observed labels and on permuted ones;
  * the permutation engine's family p-values equal a brute-force loop over
    permutations;
  * with calibration off, v2 outputs are unchanged (test_v3_weighting's golden test).
Null uniformity over 200 simulated datasets is tests/reference/calibration_null.py.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.core import inference
from app.core.methods import run_method
from app.core.preprocess import MatrixBuilder
from app.core.robustness import compute_robustness
from app.core.runner import run_multiverse

HERE = os.path.dirname(os.path.abspath(__file__))


def _matrix(values, counts=None):
    """The fields of a PreparedMatrix the statistics read."""
    counts = values if counts is None else counts
    n_taxa = values.shape[0]
    return SimpleNamespace(values=values, counts=counts, taxa_idx=np.arange(n_taxa),
                           n_taxa=n_taxa)


def _scalar(method, values, groups, counts=None):
    matrix = SimpleNamespace(values=values, counts=values if counts is None else counts,
                             groups=np.asarray(groups), taxa_idx=np.arange(values.shape[0]),
                             n_taxa=values.shape[0], rel=None)
    return run_method(method, matrix)


# --- 3. vectorised statistics equal the scalar methods --------------------------------
@pytest.mark.parametrize("method", ["wilcoxon", "ttest", "linear", "logistic"])
@pytest.mark.parametrize("kind", ["continuous", "counts_with_ties"])
def test_vectorised_statistics_equal_the_scalar_methods(method, kind):
    rng = np.random.default_rng(7)
    n_taxa, n = 40, 36
    if kind == "continuous":
        values = rng.normal(size=(n_taxa, n)) * rng.uniform(0.5, 3, (n_taxa, 1)) + 5
    else:
        values = rng.negative_binomial(2, 0.3, size=(n_taxa, n)).astype(float)
        values[:5] *= rng.random((5, n)) < 0.4          # sparse rows, many zeros
    groups = np.array([0] * 17 + [1] * 19, dtype=np.int8)
    labels = inference.permutation_matrix(groups, 25, seed=3)
    p, effect = inference.statistics(method, _matrix(values), labels)
    for row in (0, 1, 7, 25):
        scalar = _scalar(method, values, labels[row])
        assert np.allclose(p[row], scalar.p_raw, rtol=0, atol=1e-10), (method, row)
        assert np.allclose(effect[row], scalar.effect_native, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("method", ["wilcoxon", "ttest", "linear", "logistic"])
def test_streamed_z_equals_the_p_value_route_and_skipping_is_exact(method):
    """abs_z_rows gives |z| = Φ⁻¹(1 − p/2) of statistics()'s p; with a bound, cells it
    skips are ones whose |z| could not have exceeded the bound."""
    rng = np.random.default_rng(21)
    values = rng.negative_binomial(3, 0.2, size=(30, 40)).astype(float)
    labels = inference.permutation_matrix(np.repeat([0, 1], 20).astype(np.int8), 60, 5)
    matrix = _matrix(values)
    p, effect = inference.statistics(method, matrix, labels)
    full, signed = inference.abs_z_rows(method, matrix, labels)
    assert np.allclose(full, inference.abs_z(p), rtol=1e-9, atol=1e-12)
    assert np.allclose(signed, np.sign(effect[0]) * full[0])
    bound = np.quantile(full[1:], 0.7, axis=0)[None, :].repeat(60, axis=0)
    pruned, _ = inference.abs_z_rows(method, matrix, labels, bound=bound)
    assert np.array_equal(np.maximum(bound, pruned[1:]), np.maximum(bound, full[1:]))
    assert np.array_equal(pruned[0], full[0])


def test_score_test_for_depth_adjusted_presence_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(11)
    n = 90
    depth = rng.lognormal(9, 0.5, n)
    groups = rng.integers(0, 2, n).astype(np.int8)
    eta = -1 + 0.8 * (np.log(depth) - 9) + 0.4 * groups[None, :] + rng.normal(0, 1, (12, 1))
    presence = rng.random((12, n)) < 1 / (1 + np.exp(-eta))
    p, _ = inference.logistic_score(presence, groups[None, :], np.log(depth))
    design = sm.add_constant(np.log(depth))
    for taxon in range(12):
        y = presence[taxon].astype(float)
        if y.all() or not y.any():
            continue
        null = sm.GLM(y, design, family=sm.families.Binomial()).fit(tol=1e-12)
        expected = null.score_test(exog_extra=groups[:, None].astype(float)).pvalue[0]
        assert p[0, taxon] == pytest.approx(expected, abs=1e-8)


# --- 2. permutations ---------------------------------------------------------------------
def test_permutation_matrix_is_seeded_and_keeps_group_sizes():
    groups = np.array([0] * 10 + [1] * 14, dtype=np.int8)
    labels = inference.permutation_matrix(groups, 50, seed=1)
    assert (labels[0] == groups).all()
    assert (labels.sum(axis=1) == 14).all()
    assert (labels == inference.permutation_matrix(groups, 50, seed=1)).all()
    assert not (labels == inference.permutation_matrix(groups, 50, seed=2)).all()
    strata = np.array(["x"] * 12 + ["y"] * 12)
    stratified = inference.permutation_matrix(groups, 50, seed=1, strata=strata)
    for level in ("x", "y"):
        block = stratified[:, strata == level]
        assert (block.sum(axis=1) == groups[strata == level].sum()).all()


# --- 5-6. tails and discovery ------------------------------------------------------------
def test_step_up_matches_statsmodels():
    multipletests = pytest.importorskip("statsmodels.stats.multitest").multipletests
    p = np.random.default_rng(5).uniform(0, 1, 200) ** 3
    for ours, theirs in (("bh", "fdr_bh"), ("by", "fdr_by")):
        selected, adjusted = inference.step_up(p, 0.05, ours)
        reject, expected, _, _ = multipletests(p, 0.05, theirs)
        assert (selected == reject).all()
        assert np.allclose(adjusted, expected)


def test_ebh_rejects_what_its_definition_says():
    p = np.array([1e-9, 1e-8, 0.001, 0.2, 0.5])
    selected, _ = inference.step_up(p, 0.05, "ebh")
    e = 0.5 * p ** -0.5
    order = np.argsort(-e)
    k = max((i + 1 for i in range(5) if e[order][i] >= 5 / (0.05 * (i + 1))), default=0)
    assert selected.sum() == k
    assert selected[order[:k]].all()


def test_tail_fit_estimates_a_known_tail():
    rng = np.random.default_rng(9)
    maxima = np.abs(rng.normal(size=(4000, 30))).max(axis=1)
    x = 4.6
    truth = 1 - (2 * stats.norm.cdf(x) - 1) ** 30
    count = int((maxima >= x).sum())
    assert count < inference.TAIL_MIN_EXCEEDANCES          # the case the tail is for
    raw, estimate, used, tail = inference.exceedance_p(maxima, x)
    assert tail is not None and used[0]
    assert raw[0] == pytest.approx((1 + count) / 4001)
    assert truth / 5 < estimate[0] < truth * 5


# --- the engine ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small(dataset):
    """A small Quick run on the shared synthetic cohort."""
    run = run_multiverse(dataset, mode="quick")
    summary = compute_robustness(run)
    weights = summary.spec_summary.sort_values("spec_id")["weight"].to_numpy()
    return dataset, run, summary, weights


def test_family_p_values_equal_a_brute_force_loop(small):
    """The streaming maximum is exactly the maximum over every specification, per
    permutation: rebuild it one permutation at a time with the scalar methods."""
    dataset, run, _, weights = small
    b = 12
    result = inference.calibrate(run, dataset, weights, n_permutations=b, seed=4)
    labels = inference.permutation_matrix(dataset.groups, b, seed=4)
    builder = MatrixBuilder(dataset)
    fits, _ = inference._fits(run)
    n = len(run.taxa_names)
    brute = np.zeros((b + 1, n))
    libraries = np.log(np.maximum(builder.base_counts.sum(axis=0), 1.0))
    for entry in fits.values():
        matrix = builder.build(*entry["matrix_key"])
        if not matrix.usable:
            continue
        for row in range(b + 1):
            local = replace(matrix, groups=labels[row, matrix.sample_idx])
            if entry["depth"]:
                p, _ = inference.logistic_score(matrix.counts > 0,
                                                labels[row:row + 1, matrix.sample_idx],
                                                libraries[matrix.sample_idx])
                p = p[0]
            else:
                p = run_method(entry["method"], local).p_raw
            z = inference.abs_z(p)
            brute[row, matrix.taxa_idx] = np.maximum(brute[row, matrix.taxa_idx], z)
    frame = result.taxa.set_index("taxon_id")
    for j in frame.index[:25]:
        count = int((brute[1:, j] >= brute[0, j]).sum())
        assert frame.loc[j, "max_abs_z"] == pytest.approx(brute[0, j], abs=1e-9)
        assert frame.loc[j, "p_family_raw"] == pytest.approx((1 + count) / (b + 1))


def test_certified_robust_requires_all_three_conditions(small):
    dataset, run, _, weights = small
    result = inference.calibrate(run, dataset, weights, n_permutations=199, seed=8)
    taxa = result.taxa
    certified = taxa[taxa["certified_robust"]]
    assert (certified["selected"]).all()
    assert (certified["certified_share"] >= inference.CERTIFIED_CR).all()
    assert (certified["certified_sign_agreement"] >= inference.CERTIFIED_SIGN).all()
    assert (taxa.loc[~taxa["selected"], "certified_share"] == 0).all()
    assert result.within_threshold == pytest.approx(
        result.q * result.n_selected / result.n_taxa_tested)
    assert (result.rejected["p_within"] <= result.within_threshold).all()
    assert 0 <= taxa["certified_share"].max() <= 1


def test_certified_taxa_on_the_demo_are_planted_effects():
    """On the IBD demo, every CERTIFIED ROBUST genus is one of the spiked taxa."""
    from app.services import load_demo
    dataset = load_demo("ibd_genus")
    run = run_multiverse(dataset, mode="quick")
    summary = compute_robustness(run)
    weights = summary.spec_summary.sort_values("spec_id")["weight"].to_numpy()
    result = inference.calibrate(run, dataset, weights, n_permutations=499, seed=1)
    with open(os.path.join(os.path.dirname(HERE), "examples", "ibd_genus_truth.json"),
              encoding="utf-8") as handle:
        planted = set(json.load(handle)["differential_taxa"])
    certified = result.taxa.loc[result.taxa["certified_robust"], "taxon_id"]
    names = {run.taxa_names[int(j)] for j in certified}
    assert names, "the demo's planted effects should produce certified taxa"
    assert names <= planted
    assert result.specs_calibrated == run.n_specs
    assert isinstance(result.taxa, pd.DataFrame)
