"""Parsers against real files, written by the reference libraries.

`app/core/parsers/biom.py` reads BIOM v1 and v2 directly because `biom-format` has no
wheel on this Python (SPEC §24.1 B1). Checking that reader against strings we wrote
ourselves would only prove self-consistency, so the fixtures here were produced by
`biom-format` 2.1.17 itself — see `tests/reference/make_format_fixtures.py`.

Every value asserted comes from the reference library's own view of the table, carried
across in `real_table_expected.json`.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from app.core.parsers import parse_abundance
from app.core.parsers.base import lineage_rank_index

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
REQUIRED = ["real_table_v1.biom", "real_table_v2.biom", "real_table.qza",
            "real_table.tsv", "real_table_expected.json"]

pytestmark = pytest.mark.skipif(
    not all((FIXTURES / name).exists() for name in REQUIRED),
    reason="run tests/reference/make_format_fixtures.py under Python 3.10 first",
)


@pytest.fixture(scope="module")
def expected() -> dict:
    return json.loads((FIXTURES / "real_table_expected.json").read_text(encoding="utf-8"))


def _assert_matches_reference(table, expected, *, taxonomy=True):
    """The parsed table must reproduce biom-format's own view of the data."""
    assert table.n_taxa == expected["n_taxa"]
    assert table.n_samples == expected["n_samples"]
    assert table.sample_ids == expected["sample_ids"]
    assert table.taxa == expected["observation_ids"]

    values = table.counts.to_numpy(dtype=float)
    assert float(values.sum()) == pytest.approx(expected["total"])
    assert int((values == 0).sum()) == expected["n_zeros"]
    np.testing.assert_allclose(values.sum(axis=0), expected["column_sums"])
    np.testing.assert_allclose(values.sum(axis=1), expected["row_sums"])
    assert table.is_integer
    assert table.value_type == "counts"

    if taxonomy:
        assert table.has_taxonomy
        first = table.lineages[expected["observation_ids"][0]]
        assert first[:5] == expected["lineage_of_first"]


# --- BIOM v1 (JSON) -------------------------------------------------------
def test_reads_a_real_biom_v1_file(expected):
    table = parse_abundance((FIXTURES / "real_table_v1.biom").read_bytes(),
                            "real_table_v1.biom")
    assert table.source_format == "biom v1"
    _assert_matches_reference(table, expected)


# --- BIOM v2 (HDF5) -------------------------------------------------------
def test_reads_a_real_biom_v2_file(expected):
    """The v2 reader had no test at all before this: it walks a CSR layout by hand."""
    table = parse_abundance((FIXTURES / "real_table_v2.biom").read_bytes(),
                            "real_table_v2.biom")
    assert table.source_format == "biom v2"
    _assert_matches_reference(table, expected)


def test_biom_v2_is_detected_from_content_not_the_extension(expected):
    """An HDF5 magic number must win over a misleading filename."""
    table = parse_abundance((FIXTURES / "real_table_v2.biom").read_bytes(), "mislabelled.tsv")
    assert table.source_format == "biom v2"
    assert table.n_taxa == expected["n_taxa"]


def test_v1_and_v2_readers_agree(expected):
    v1 = parse_abundance((FIXTURES / "real_table_v1.biom").read_bytes(), "a.biom")
    v2 = parse_abundance((FIXTURES / "real_table_v2.biom").read_bytes(), "b.biom")
    np.testing.assert_allclose(
        v1.counts.to_numpy(dtype=float),
        v2.counts.loc[v1.taxa, v1.sample_ids].to_numpy(dtype=float),
    )
    assert v1.lineages == v2.lineages


# --- QIIME 2 artifact ------------------------------------------------------
def test_reads_a_real_qza_artifact(expected):
    """A .qza is a zip of <uuid>/{VERSION, metadata.yaml, data/<payload>}."""
    table = parse_abundance((FIXTURES / "real_table.qza").read_bytes(), "real_table.qza")
    assert "qza" in table.source_format
    _assert_matches_reference(table, expected)


def test_qza_is_detected_from_content_not_the_extension(expected):
    table = parse_abundance((FIXTURES / "real_table.qza").read_bytes(), "no-extension")
    assert "qza" in table.source_format
    assert table.n_taxa == expected["n_taxa"]


# --- the TSV export of the same table --------------------------------------
def test_reads_the_biom_tsv_export(expected):
    """biom's TSV export carries a leading '# Constructed from biom file' comment and a
    trailing taxonomy column — both of which the tabular reader has to handle."""
    table = parse_abundance((FIXTURES / "real_table.tsv").read_bytes(), "real_table.tsv")
    assert table.n_taxa == expected["n_taxa"]
    assert table.n_samples == expected["n_samples"]
    assert table.sample_ids == expected["sample_ids"]
    np.testing.assert_allclose(
        table.counts.to_numpy(dtype=float).sum(axis=0), expected["column_sums"]
    )
    assert table.has_taxonomy


def test_every_reader_produces_the_same_table(expected):
    """Four formats, one table: the values must not depend on how it was delivered."""
    tables = {
        name: parse_abundance((FIXTURES / name).read_bytes(), name)
        for name in ("real_table_v1.biom", "real_table_v2.biom",
                     "real_table.qza", "real_table.tsv")
    }
    reference = tables["real_table_v1.biom"]
    target = reference.counts.to_numpy(dtype=float)
    for name, table in tables.items():
        aligned = table.counts.loc[reference.taxa, reference.sample_ids].to_numpy(dtype=float)
        np.testing.assert_allclose(aligned, target, err_msg=name)


# --- and it runs end to end ------------------------------------------------
def test_a_real_biom_v2_file_runs_a_multiverse(expected):
    """Parsing is not enough: the table has to survive validation and a full run."""
    import pandas as pd

    from app.core.robustness import compute_robustness
    from app.core.runner import run_multiverse
    from app.core.validation import validate_dataset

    table = parse_abundance((FIXTURES / "real_table_v2.biom").read_bytes(), "t.biom")
    metadata = pd.DataFrame(
        {"group": ["control"] * 10 + ["disease"] * 10}, index=table.sample_ids
    )
    metadata.index.name = "sample_id"

    dataset = validate_dataset(table, metadata, "group")
    assert dataset.n_samples == 20
    assert dataset.group_labels == ("control", "disease")

    run = run_multiverse(dataset, mode="quick")
    assert run.grid_report.n_valid > 0
    summary = compute_robustness(run)
    assert len(summary.table) > 0
    assert sum(summary.tier_counts.values()) == len(summary.table)


def test_lineages_survive_the_round_trip(expected):
    """Ranks matter: fork 4 only has two levels if the lineage reaches past genus."""
    table = parse_abundance((FIXTURES / "real_table_v2.biom").read_bytes(), "t.biom")
    depths = [lineage_rank_index(v) for v in table.lineages.values() if v]
    assert depths, "no lineages were read"
    assert max(depths) >= 5, "expected at least genus-level lineages"
    assert table.input_rank in {"family", "genus", "species"}
