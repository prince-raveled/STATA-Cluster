"""Inputs that used to be altered in silence, found by the final end-to-end audit.

Each of these was accepted and changed without a word, or refused with a message about
something else: text in a count cell became zero, an infinite value passed through, a
repeated sample column was renamed and dropped as "no metadata", a repeated feature row
vanished, a gzipped table (which the upload form offers) failed with a tokenizer error,
and a spreadsheet was called a broken QIIME 2 artifact. What the researcher uploaded
must be what is analysed, or they must be told exactly why not.
"""
from __future__ import annotations

import gzip
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from app import services, ui
from app.core import parsers
from app.core.parsers import ParseError, parse_abundance, parse_metadata
from app.core.validation import DatasetError


def _counts(n_taxa=30, n_samples=20, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.integers(1, 400, size=(n_taxa, n_samples)),
                        index=[f"t{i}" for i in range(n_taxa)],
                        columns=[f"s{j}" for j in range(n_samples)])


def _tsv(frame, index_name="taxon") -> bytes:
    frame = frame.copy()
    frame.index.name = index_name
    return frame.to_csv(sep="\t").encode()


def _meta(samples, groups=None) -> bytes:
    groups = groups or (["a"] * (len(samples) // 2) + ["b"] * (len(samples) - len(samples) // 2))
    return _tsv(pd.DataFrame({"group": groups}, index=list(samples)), "id")


def _with_row(frame, name, value):
    out = frame.copy().astype(object)
    out.loc[name] = value
    return out


# --- cells that are not counts ----------------------------------------------------
def test_text_in_a_count_cell_is_refused_and_located():
    with pytest.raises(ParseError) as info:
        parse_abundance(_tsv(_with_row(_counts(), "t4", "abc")), "c.tsv")
    message = str(info.value)
    assert "not numbers" in message and "'abc'" in message and "row 't4'" in message


def test_an_infinite_value_is_refused():
    with pytest.raises(ParseError, match="infinite"):
        parse_abundance(_tsv(_with_row(_counts(), "t6", np.inf)), "c.tsv")


def test_empty_cells_are_read_as_zero_and_the_page_is_told():
    table = parse_abundance(_tsv(_with_row(_counts(), "t5", np.nan)), "c.tsv")
    assert (table.counts.loc["t5"] == 0).all()
    assert any("20 empty cells" in note for note in table.notes)


def test_a_description_column_is_still_set_aside_not_refused():
    frame = _counts().assign(Description="some text")
    table = parse_abundance(_tsv(frame), "c.tsv")
    assert "Description" not in table.counts.columns


# --- repeated identifiers -----------------------------------------------------------
def test_a_repeated_sample_column_is_refused_by_name():
    frame = _counts().set_axis([f"s{j}" for j in range(19)] + ["s0"], axis=1)
    with pytest.raises(ParseError, match="repeated column names: s0"):
        parse_abundance(_tsv(frame), "c.tsv")


def test_a_repeated_sample_row_is_refused_when_samples_are_rows():
    frame = _counts(n_taxa=12, n_samples=30).T
    frame.index = [f"s{j}" for j in range(29)] + ["s0"]
    samples = [f"s{j}" for j in range(29)]
    with pytest.raises(ParseError, match="appears more than once"):
        parse_abundance(_tsv(frame, "sample"), "c.tsv", sample_ids=samples)


def test_repeated_feature_rows_are_added_together_not_dropped():
    frame = _counts()
    frame.index = [f"t{i}" for i in range(29)] + ["t0"]
    table = parse_abundance(_tsv(frame), "c.tsv")
    assert table.n_taxa == 29
    assert np.allclose(table.counts.loc["t0"], frame.iloc[0] + frame.iloc[-1])
    assert table.counts.to_numpy().sum() == frame.to_numpy().sum(), "every read is kept"
    assert any("added together" in note for note in table.notes)


# --- containers -----------------------------------------------------------------------
def test_a_gzipped_table_reads_exactly_as_the_plain_one():
    plain = _tsv(_counts())
    a = parse_abundance(plain, "c.tsv")
    b = parse_abundance(gzip.compress(plain), "c.tsv.gz")
    pd.testing.assert_frame_equal(a.counts, b.counts)


def test_gzipped_metadata_reads_too():
    assert list(parse_metadata(gzip.compress(_meta([f"s{j}" for j in range(20)])), "m.tsv.gz")
                .index[:2]) == ["s0", "s1"]


def test_a_gzip_that_expands_past_the_limit_is_refused(monkeypatch):
    monkeypatch.setattr(parsers, "MAX_DECOMPRESSED_BYTES", 1000)
    with pytest.raises(ParseError, match="decompression limit"):
        parse_abundance(gzip.compress(_tsv(_counts())), "c.tsv.gz")


def test_an_excel_workbook_is_named_as_one():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("xl/workbook.xml", "<workbook/>")
        archive.writestr("[Content_Types].xml", "<Types/>")
    for name in ("table.xlsx", "table.qza.renamed"):
        with pytest.raises(ParseError, match="spreadsheet workbook"):
            parse_abundance(buffer.getvalue(), name)
    with pytest.raises(ParseError, match="spreadsheet workbook"):
        parse_abundance(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 64, "old.xls")
    with pytest.raises(ParseError, match="spreadsheet workbook"):
        parse_metadata(buffer.getvalue(), "metadata.xlsx")


# --- messages that name the file and the problem ----------------------------------------
def test_an_empty_file_is_named():
    with pytest.raises(ParseError, match="'metadata.tsv' contains no data rows"):
        parse_metadata(b"", "metadata.tsv")


def test_a_ragged_row_is_located_in_plain_words():
    raw = b"taxon,s0,s1,s2\nt1,1,2,3\nt2,1,2,3,4,5\n"
    with pytest.raises(ParseError, match=r"Line 3 of 'c.csv' has 6 fields"):
        parse_abundance(raw, "c.csv")


@pytest.mark.parametrize("groups,expected", [
    (["a"] * 20, "has 1 value (a)"),
    (["a"] * 7 + ["b"] * 7 + ["c"] * 6, "has 3 values (a, b, c)"),
])
def test_a_grouping_column_without_two_values_is_named_with_its_values(groups, expected):
    counts = _counts()
    with pytest.raises(DatasetError) as info:
        services.build_dataset(_tsv(counts), "c.tsv", _meta(counts.columns, groups), "m.tsv")
    assert "'group' column" in info.value.message and expected in info.value.message


# --- MetaPhlAn-style tables say what they left out ----------------------------------------
def test_a_metaphlan_table_names_rows_it_could_not_place():
    rows = ["clade_name\tS1\tS2", "UNCLASSIFIED\t5\t6", "k__Bacteria\t95\t94",
            "k__Bacteria|p__Firmicutes\t50\t40"]
    rows += [f"k__Bacteria|p__Firmicutes|c__C|o__O|f__F|g__G{i}|s__G{i}_sp\t{i + 1}\t{i + 2}"
             for i in range(8)]
    table = parse_abundance("\n".join(rows).encode(), "merged_abundance.txt")
    assert table.n_taxa == 8
    notes = " ".join(table.notes)
    assert "Left out 2 rows at higher ranks" in notes
    assert "'UNCLASSIFIED'" in notes and "not a clade" in notes


# --- what the results page says about analyses that produced nothing -----------------------
class _Run:
    def __init__(self, skipped=None):
        if skipped is not None:
            self.skipped = skipped


class _Summary:
    def __init__(self, n):
        self.n_specs_skipped = n


def test_a_failed_method_is_named_with_its_error_not_blamed_on_subsampling():
    run = _Run({"pydeseq2": {"specs": 24, "matrices": 8,
                             "detail": "AssertionError: ThreadingBackend does not accept"}})
    sentences = ui.skipped_reasons(run, _Summary(24))
    assert sentences == [
        "PyDESeq2 could not be fitted on 8 matrices, so its 24 analyses produced no result "
        "(the error was: AssertionError: ThreadingBackend does not accept)."]


def test_matrices_too_small_to_test_are_described_as_such():
    sentences = ui.skipped_reasons(_Run({"unusable": {"specs": 30, "matrices": 10}}), _Summary(30))
    assert "too few taxa, or one group with too few samples" in sentences[0]


def test_a_run_saved_before_causes_were_recorded_keeps_its_old_explanation():
    assert "subsampling" in ui.skipped_reasons(_Run(), _Summary(12))[0]
    assert ui.skipped_reasons(_Run(), _Summary(0)) == []
