"""Preprocessing — SPEC §9. The order-of-operations tests are the point of this file."""
from __future__ import annotations

import numpy as np
import pytest

from app.core.preprocess import (
    MatrixBuilder,
    apply_transform,
    closure,
    clr,
    multiplicative_replacement,
    prevalence_filter,
    rarefy,
    relative_scale,
    tmm_factors,
)


# --------------------------------------------------------------------------
# Order of operations (§9)
# --------------------------------------------------------------------------
def _sparse_table():
    """A table with one taxon that is prevalent but too rare to survive subsampling."""
    rng = np.random.default_rng(42)
    n_taxa, n_samples, depth = 30, 12, 10_000
    counts = np.zeros((n_taxa, n_samples), dtype=float)
    counts[0] = depth * 0.5
    counts[1:5] = rng.integers(400, 900, size=(4, n_samples))
    counts[5:] = rng.integers(0, 60, size=(n_taxa - 5, n_samples))
    # The taxon of interest: present in every sample, at 3 reads in 10,000.
    counts[7] = 3.0
    counts[0] = depth - counts[1:].sum(axis=0)
    return counts


def test_rarefy_then_filter_differs_from_filter_then_rarefy():
    """The §9 order is not cosmetic: swapping steps 1 and 3 changes the taxa tested.

    Taxon 7 is present in every sample (prevalence 1.00) at 3 reads in 10,000.
    Rarefy first and it is mostly lost to the subsampling, so the 20% filter drops it.
    Filter first and it passes on its raw prevalence, and stays in the matrix — a taxon
    tested under one order and absent under the other.
    """
    counts = _sparse_table()

    rarefied, _ = rarefy(counts, 100, seed=1)
    correct_mask = prevalence_filter(rarefied, 0.20)

    pre_filter = prevalence_filter(counts, 0.20)
    wrong, _ = rarefy(counts[pre_filter], 100, seed=1)

    assert pre_filter[7], "taxon 7 should survive the filter on raw counts"
    assert not correct_mask[7], "taxon 7 should not survive filtering after rarefaction"
    assert int(correct_mask.sum()) < int(pre_filter.sum())
    assert wrong.shape[0] == int(pre_filter.sum())


def test_rarefy_happens_before_collapse(synthetic):
    """Rarefying after collapsing would subsample a different multinomial.

    Every retained library sums to exactly the rarefaction depth, which holds only if
    the subsampling ran on the input-rank counts and the collapse merely summed them.
    """

    from app.core.parsers.base import AbundanceTable
    from app.core.validation import validate_dataset

    counts = synthetic["table"].counts
    taxa = list(counts.index)
    lineages = {
        t: ["k__Bacteria", "p__Firmicutes", "c__Clostridia", "o__Clostridiales",
            "f__Lachnospiraceae", f"g__G{i % 12:02d}", f"s__sp{i:03d}"]
        for i, t in enumerate(taxa)
    }
    table = AbundanceTable(counts=counts.copy(), lineages=lineages, value_type="counts")
    dataset = validate_dataset(table, synthetic["metadata"], "group")
    builder = MatrixBuilder(dataset)
    assert "genus" in builder.ranks

    matrix = builder.build("1000", 1, "genus", 0.0, "raw")
    np.testing.assert_allclose(matrix.counts.sum(axis=0), 1000)
    assert matrix.n_taxa <= 12  # collapsed to genus after the subsampling


def test_preprocess_is_deterministic_for_a_seed(dataset):
    a = MatrixBuilder(dataset).build("1000", 2, "input", 0.10, "clr")
    b = MatrixBuilder(dataset).build("1000", 2, "input", 0.10, "clr")
    np.testing.assert_allclose(a.values, b.values)
    np.testing.assert_array_equal(a.taxa_idx, b.taxa_idx)


def test_different_seeds_give_different_matrices(dataset):
    a = MatrixBuilder(dataset).build("1000", 1, "input", 0.0, "raw")
    b = MatrixBuilder(dataset).build("1000", 2, "input", 0.0, "raw")
    assert not np.array_equal(a.counts, b.counts)


# --------------------------------------------------------------------------
# Rarefaction (§10 fork 1)
# --------------------------------------------------------------------------
def test_rarefy_hits_the_target_depth_exactly():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 400, size=(50, 10))
    out, keep = rarefy(counts, 500, seed=1)
    assert keep.all() or out.shape[1] == int(keep.sum())
    assert np.all(out.sum(axis=0) == 500)


def test_rarefy_drops_libraries_below_the_depth():
    counts = np.zeros((10, 4), dtype=int)
    counts[:, 0] = 10       # library 100
    counts[:, 1] = 200      # library 2000
    counts[:, 2] = 300      # library 3000
    counts[:, 3] = 5        # library 50
    out, keep = rarefy(counts, 1500, seed=1)
    assert list(keep) == [False, True, True, False]
    assert out.shape[1] == 2


def test_rarefy_never_exceeds_the_observed_count():
    rng = np.random.default_rng(2)
    counts = rng.integers(0, 30, size=(20, 6))
    out, keep = rarefy(counts, 60, seed=3)
    assert np.all(out <= counts[:, keep])


def test_available_depths_drops_impossible_levels(dataset):
    builder = MatrixBuilder(dataset)
    levels, dropped = builder.available_depths()
    names = {name for name, _ in levels}
    assert "none" in names
    for name, seed in levels:
        assert (seed is None) == (name == "none")
    assert len(levels) == 1 + 3 * (len(names) - 1)


# --------------------------------------------------------------------------
# Prevalence filter (§10 fork 2)
# --------------------------------------------------------------------------
def test_prevalence_filter_thresholds():
    counts = np.array([
        [1, 1, 1, 1, 1],   # 100%
        [1, 1, 0, 0, 0],   # 40%
        [1, 0, 0, 0, 0],   # 20%
        [0, 0, 0, 0, 0],   # 0% — never testable
    ], dtype=float)
    assert list(prevalence_filter(counts, 0.0)) == [True, True, True, False]
    assert list(prevalence_filter(counts, 0.20)) == [True, True, True, False]
    assert list(prevalence_filter(counts, 0.50)) == [True, False, False, False]


def test_stricter_filters_are_nested(dataset):
    builder = MatrixBuilder(dataset)
    previous = None
    for threshold in (0.0, 0.05, 0.10, 0.20):
        matrix = builder.build("none", None, "input", threshold, "raw")
        current = set(matrix.taxa_idx.tolist())
        if previous is not None:
            assert current <= previous
        previous = current


# --------------------------------------------------------------------------
# Transforms (§10 fork 3)
# --------------------------------------------------------------------------
def test_tss_columns_sum_to_one(dataset):
    matrix = MatrixBuilder(dataset).build("none", None, "input", 0.0, "tss")
    np.testing.assert_allclose(matrix.values.sum(axis=0), 1.0, atol=1e-12)


def test_clr_columns_sum_to_zero(dataset):
    matrix = MatrixBuilder(dataset).build("none", None, "input", 0.0, "clr")
    np.testing.assert_allclose(matrix.values.sum(axis=0), 0.0, atol=1e-8)


def test_multiplicative_replacement_removes_zeros_and_preserves_ratios():
    composition = closure(np.array([[10.0, 0.0], [20.0, 5.0], [0.0, 5.0]]))
    replaced = multiplicative_replacement(composition)
    assert (replaced > 0).all()
    np.testing.assert_allclose(replaced.sum(axis=0), 1.0)
    # Ratios between originally non-zero parts survive — the property a fixed
    # pseudocount destroys, which is why §10 requires multiplicative replacement.
    np.testing.assert_allclose(
        replaced[1, 0] / replaced[0, 0], composition[1, 0] / composition[0, 0], rtol=1e-9
    )


def test_clr_is_scale_invariant():
    counts = np.array([[5.0, 10.0], [10.0, 20.0], [1.0, 2.0]])
    np.testing.assert_allclose(clr(counts), clr(counts * 7.0), atol=1e-10)


def test_tmm_factors_have_unit_geometric_mean():
    rng = np.random.default_rng(9)
    counts = rng.integers(1, 500, size=(80, 12)).astype(float)
    factors = tmm_factors(counts)
    assert np.isclose(np.exp(np.mean(np.log(factors))), 1.0)
    assert np.all(factors > 0)


def test_raw_transform_is_the_identity(dataset):
    matrix = MatrixBuilder(dataset).build("none", None, "input", 0.0, "raw")
    np.testing.assert_allclose(matrix.values, matrix.counts)


@pytest.mark.parametrize("kind", ["raw", "tss", "tmm"])
def test_relative_scale_agrees_across_positive_transforms(kind):
    """A per-sample scalar cancels under closure, so raw / TSS / TMM share one scale."""
    rng = np.random.default_rng(12)
    counts = rng.integers(0, 300, size=(40, 10)).astype(float)
    reference = relative_scale(apply_transform(counts, "raw"), "raw")
    np.testing.assert_allclose(
        relative_scale(apply_transform(counts, kind), kind), reference, atol=1e-9
    )


def test_relative_scale_for_clr_is_a_valid_composition():
    rng = np.random.default_rng(13)
    counts = rng.integers(0, 300, size=(40, 10)).astype(float)
    rel = relative_scale(apply_transform(counts, "clr"), "clr")
    np.testing.assert_allclose(rel.sum(axis=0), 1.0)
    assert (rel > 0).all()


# --------------------------------------------------------------------------
# Cache behaviour
# --------------------------------------------------------------------------
def test_matrix_cache_reuses_the_rarefaction_layer(dataset):
    builder = MatrixBuilder(dataset)
    for transform in ("raw", "tss", "clr"):
        for prevalence in (0.0, 0.05, 0.10):
            builder.build("1000", 1, "input", prevalence, transform)
    # Nine matrices, one subsampling.
    assert len(builder._rarefy_cache) == 1
    assert len(builder._filter_cache) == 3


def test_matrix_key_groups_specifications(dataset):
    builder = MatrixBuilder(dataset)
    a = builder.build("none", None, "input", 0.05, "clr")
    b = builder.build("none", None, "input", 0.05, "clr")
    assert a.key == b.key


def test_tmm_factors_survives_an_empty_table():
    """A prevalence filter can leave no taxa; TMM must return unit factors, not raise.

    Found by running MicrobiomeHD at OTU level: `nash_chan` reached a grid point where
    the filter emptied the matrix and np.quantile raised IndexError on the empty axis,
    which surfaced as a 500 rather than a pruned specification.
    """
    factors = tmm_factors(np.empty((0, 5), dtype=float))
    assert factors.shape == (5,)
    assert np.allclose(factors, 1.0)


def test_clr_survives_an_empty_table():
    """Same grid point, other transform: no taxa means no geometric mean to centre on."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = clr(np.empty((0, 4), dtype=float))
    assert out.shape == (0, 4)
