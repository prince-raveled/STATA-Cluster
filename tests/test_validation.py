"""Input validation — SPEC §8. Every constraint, and the message it produces."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.parsers.base import AbundanceTable
from app.core.validation import DatasetError, validate_dataset


def build(counts, groups, taxa=None, samples=None, lineages=None, extra=None):
    taxa = taxa or [f"T{i:03d}" for i in range(counts.shape[0])]
    samples = samples or [f"S{j:03d}" for j in range(counts.shape[1])]
    table = AbundanceTable(
        counts=pd.DataFrame(counts, index=taxa, columns=samples),
        lineages=lineages or {},
        value_type="counts",
    )
    metadata = pd.DataFrame({"group": groups}, index=samples)
    if extra:
        for key, value in extra.items():
            metadata[key] = value
    return table, metadata


def test_accepts_a_well_formed_dataset(synthetic):
    dataset = validate_dataset(synthetic["table"], synthetic["metadata"], "group")
    assert dataset.n_samples == 40
    assert dataset.group_sizes == (20, 20)
    assert dataset.group_labels == ("control", "disease")  # sorted; B is the second
    assert dataset.can_rarefy


def test_rejects_too_few_samples():
    counts = np.random.default_rng(0).integers(1, 50, size=(20, 8))
    table, metadata = build(counts, ["a"] * 4 + ["b"] * 4)
    with pytest.raises(DatasetError) as info:
        validate_dataset(table, metadata, "group")
    assert "at least 10" in info.value.message
    assert "noise" in info.value.hint


def test_rejects_too_few_per_group():
    counts = np.random.default_rng(0).integers(1, 50, size=(20, 14))
    table, metadata = build(counts, ["a"] * 11 + ["b"] * 3)
    with pytest.raises(DatasetError, match="at least 5 samples in each group"):
        validate_dataset(table, metadata, "group")


def test_rejects_too_few_taxa():
    counts = np.random.default_rng(0).integers(1, 50, size=(6, 20))
    table, metadata = build(counts, ["a"] * 10 + ["b"] * 10)
    with pytest.raises(DatasetError, match="at least 10"):
        validate_dataset(table, metadata, "group")


def test_rejects_non_binary_grouping():
    counts = np.random.default_rng(0).integers(1, 50, size=(20, 21))
    table, metadata = build(counts, ["a"] * 7 + ["b"] * 7 + ["c"] * 7)
    with pytest.raises(DatasetError) as info:
        validate_dataset(table, metadata, "group")
    assert "3 distinct value" in info.value.message
    assert "two groups" in info.value.hint


def test_rejects_negative_values():
    counts = np.random.default_rng(0).normal(size=(20, 20))
    table, metadata = build(counts, ["a"] * 10 + ["b"] * 10)
    with pytest.raises(DatasetError) as info:
        validate_dataset(table, metadata, "group")
    assert "already been" in info.value.message


def test_rejects_unmatched_sample_ids():
    counts = np.random.default_rng(0).integers(1, 50, size=(20, 20))
    table, metadata = build(counts, ["a"] * 10 + ["b"] * 10)
    metadata.index = [f"X{i}" for i in range(20)]
    with pytest.raises(DatasetError) as info:
        validate_dataset(table, metadata, "group")
    assert "No sample IDs are shared" in info.value.message
    assert "S000" in info.value.hint  # the mismatch is listed, per §8


def test_partial_id_overlap_warns_and_keeps_the_intersection():
    counts = np.random.default_rng(0).integers(1, 50, size=(20, 24))
    table, metadata = build(counts, ["a"] * 12 + ["b"] * 12)
    metadata = metadata.rename(index={"S000": "GHOST", "S001": "GHOST2"})
    dataset = validate_dataset(table, metadata, "group")
    assert dataset.n_samples == 22
    assert any("no metadata" in w for w in dataset.warnings)
    assert any("absent from the abundance table" in w for w in dataset.warnings)


def test_too_many_taxa_collapses_to_genus_when_taxonomy_allows():
    rng = np.random.default_rng(3)
    counts = rng.integers(1, 200, size=(1600, 20))
    taxa = [f"ASV{i:04d}" for i in range(1600)]
    lineages = {
        t: ["k__Bacteria", "p__Firmicutes", "c__Clostridia", "o__Clostridiales",
            "f__Lachnospiraceae", f"g__G{i % 90:02d}", f"s__sp{i}"]
        for i, t in enumerate(taxa)
    }
    table, metadata = build(counts, ["a"] * 10 + ["b"] * 10, taxa=taxa, lineages=lineages)
    dataset = validate_dataset(table, metadata, "group")
    assert dataset.n_taxa == 90
    assert dataset.table.input_rank == "genus"
    assert any("collapsed to genus" in w for w in dataset.warnings)


def test_too_many_taxa_without_taxonomy_is_refused():
    counts = np.random.default_rng(3).integers(1, 200, size=(1600, 20))
    table, metadata = build(counts, ["a"] * 10 + ["b"] * 10)
    with pytest.raises(DatasetError) as info:
        validate_dataset(table, metadata, "group")
    assert "1500 limit" in info.value.message
    assert "taxonomy" in info.value.message


def test_non_integer_values_disable_rarefaction_but_do_not_refuse():
    rng = np.random.default_rng(4)
    counts = rng.random(size=(20, 20))
    counts = counts / counts.sum(axis=0)
    table, metadata = build(counts, ["a"] * 10 + ["b"] * 10)
    dataset = validate_dataset(table, metadata, "group")
    assert dataset.can_rarefy is False
    assert any("cannot be subsampled" in w for w in dataset.warnings)


def test_empty_sample_is_refused():
    counts = np.random.default_rng(5).integers(1, 50, size=(20, 20))
    counts[:, 3] = 0
    table, metadata = build(counts, ["a"] * 10 + ["b"] * 10)
    with pytest.raises(DatasetError, match="total abundance of zero"):
        validate_dataset(table, metadata, "group")


def test_unknown_group_column_lists_the_available_ones():
    counts = np.random.default_rng(6).integers(1, 50, size=(20, 20))
    table, metadata = build(counts, ["a"] * 10 + ["b"] * 10)
    with pytest.raises(DatasetError) as info:
        validate_dataset(table, metadata, "phenotype")
    assert "not in the metadata" in info.value.message
    assert "group" in info.value.hint


def test_covariates_are_discovered_and_validated(synthetic):
    dataset = validate_dataset(synthetic["table"], synthetic["metadata"], "group")
    assert set(dataset.covariate_columns) >= {"age", "sex"}
    with pytest.raises(DatasetError, match="not in the metadata"):
        validate_dataset(synthetic["table"], synthetic["metadata"], "group",
                         covariates=["nonexistent"])


def test_missing_group_values_are_dropped_with_a_warning():
    counts = np.random.default_rng(7).integers(1, 50, size=(20, 22))
    groups = ["a"] * 11 + ["b"] * 11
    table, metadata = build(counts, groups)
    metadata.loc["S000", "group"] = np.nan
    metadata.loc["S021", "group"] = ""
    dataset = validate_dataset(table, metadata, "group")
    assert dataset.n_samples == 20
    assert any("no value for" in w for w in dataset.warnings)
