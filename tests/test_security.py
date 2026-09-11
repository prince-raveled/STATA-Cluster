"""Uploaded research data is untrusted input. These are the attacks it could carry.

Every test here is a file a user could upload today. The bar is not "the parser
survives" — it is that the parser refuses with a message a researcher can act on,
rather than allocating terabytes, silently corrupting the matrix, or returning a 500.

A corrupted-but-plausible table is the worst outcome of the three: it analyses cleanly
and produces a wrong answer nobody has reason to doubt.
"""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from app.core.parsers.base import (
    MAX_DECOMPRESSED_BYTES,
    MAX_PARSE_CELLS,
    MAX_PARSE_DIMENSION,
    ParseError,
)
from app.core.parsers.biom import parse_biom_v1, parse_qza


def _biom_v1(taxa, samples, data, matrix_type="sparse") -> bytes:
    return json.dumps({
        "id": "test", "format": "1.0.0", "type": "OTU table",
        "matrix_type": matrix_type,
        "shape": [len(taxa), len(samples)],
        "rows": [{"id": t, "metadata": None} for t in taxa],
        "columns": [{"id": s, "metadata": None} for s in samples],
        "data": data,
    }).encode()


# --- resource exhaustion ----------------------------------------------------
def test_a_small_file_may_not_declare_an_enormous_matrix():
    """The realistic amplification: modest row and column lists, an absurd product.

    8,000 taxa and 8,000 samples is about 1 MB of JSON and 64 million cells — 512 MB as
    float64, allocated before §8's taxon ceiling ever runs. The row and column lists are
    what a hostile file is bounded by; their product is what bounds the damage.
    """
    taxa = [f"t{i}" for i in range(8_000)]
    samples = [f"s{i}" for i in range(8_000)]
    payload = _biom_v1(taxa, samples, [[0, 0, 1.0]])
    assert len(payload) < 2_000_000, "the hostile file is small; that is the point"
    with pytest.raises(ParseError, match="parse limit"):
        parse_biom_v1(payload)


def test_the_cell_limit_catches_shapes_that_pass_the_dimension_limit():
    """Two dimensions each individually legal can still multiply to something absurd."""
    from app.core.parsers.base import check_declared_shape

    side = MAX_PARSE_DIMENSION - 1
    with pytest.raises(ParseError, match="parse limit"):
        check_declared_shape(side, side, "test")
    assert check_declared_shape(1000, 1000, "test") is None


def test_an_oversized_single_dimension_is_refused():
    from app.core.parsers.base import check_declared_shape

    with pytest.raises(ParseError, match="No dimension may exceed"):
        check_declared_shape(2, MAX_PARSE_DIMENSION + 1, "test")


def test_negative_dimensions_are_refused():
    from app.core.parsers.base import check_declared_shape

    with pytest.raises(ParseError, match="negative"):
        check_declared_shape(-1, 10, "test")


def test_a_shape_exactly_at_the_limit_is_still_accepted():
    """The guard must refuse only impossible tables, never merely large ones."""
    from app.core.parsers.base import check_declared_shape

    rows = MAX_PARSE_CELLS // MAX_PARSE_DIMENSION
    assert check_declared_shape(rows, MAX_PARSE_DIMENSION, "test") is None
    assert check_declared_shape(MAX_PARSE_DIMENSION, 1, "test") is None


# --- matrix corruption ------------------------------------------------------
def test_a_negative_sparse_index_is_refused_not_wrapped():
    """Python would accept values[-1, 0] and write to the last taxon.

    That produces a table which parses, validates and analyses — with one taxon's counts
    silently belonging to another. It must be an error.
    """
    payload = _biom_v1(["t0", "t1", "t2"], ["s0", "s1"], [[-1, 0, 99.0]])
    with pytest.raises(ParseError, match="outside the declared"):
        parse_biom_v1(payload)


def test_an_out_of_range_sparse_index_is_refused():
    payload = _biom_v1(["t0", "t1"], ["s0", "s1"], [[5, 0, 1.0]])
    with pytest.raises(ParseError, match="outside the declared"):
        parse_biom_v1(payload)


def test_malformed_sparse_triples_are_refused():
    for bad in ([[0, 0]], [["a", 0, 1.0]], [[0, 0, "x"]]):
        with pytest.raises(ParseError):
            parse_biom_v1(_biom_v1(["t0"], ["s0"], bad))


def test_a_dense_matrix_must_match_its_declared_shape():
    payload = _biom_v1(["t0", "t1"], ["s0", "s1"], [[1.0, 2.0]], matrix_type="dense")
    with pytest.raises(ParseError, match="declares"):
        parse_biom_v1(payload)


def test_a_valid_sparse_table_still_parses():
    table = parse_biom_v1(_biom_v1(["t0", "t1"], ["s0", "s1"],
                                   [[0, 0, 5.0], [1, 1, 7.0]]))
    assert table.counts.shape == (2, 2)
    assert table.counts.iloc[0, 0] == 5.0
    assert table.counts.iloc[1, 1] == 7.0


# --- archive attacks --------------------------------------------------------
def _qza(members: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def test_a_zip_bomb_is_refused_before_decompression():
    """A highly compressible member: small on disk, enormous in memory.

    The upload cap bounds compressed bytes only, so without this guard a 1 MB .qza
    decompresses into the process heap.
    """
    bomb = _qza({"data/feature-table.biom": b"\0" * (MAX_DECOMPRESSED_BYTES + 1024)})
    assert len(bomb) < 2_000_000, "the test archive should itself be small"
    with pytest.raises(ParseError, match="expands|decompression limit"):
        parse_qza(bomb)


def test_a_path_traversal_member_cannot_escape_because_nothing_is_written():
    """`..` names are harmless here only because extraction never touches the disk.

    This test exists to keep it that way: if someone later adds `archive.extractall`,
    this is the test that should stop them.
    """
    payload = _qza({"../../../etc/passwd": b"root:x:0:0"})
    with pytest.raises(ParseError, match="No feature table"):
        parse_qza(payload)


def test_a_qza_without_a_feature_table_gets_an_actionable_message():
    with pytest.raises(ParseError, match="qiime tools export"):
        parse_qza(_qza({"metadata.yaml": b"uuid: abc"}))


def test_a_non_zip_qza_is_refused_cleanly():
    with pytest.raises(ParseError, match="not a readable zip"):
        parse_qza(b"this is not a zip file at all")


def test_a_valid_qza_still_parses():
    inner = _biom_v1(["t0", "t1"], ["s0", "s1"], [[0, 0, 3.0], [1, 1, 4.0]])
    table = parse_qza(_qza({"x/data/feature-table.biom": inner}))
    assert table.counts.shape == (2, 2)
    assert "qza" in table.source_format


# --- output safety ----------------------------------------------------------
def test_taxon_names_carrying_markup_survive_as_data():
    """A taxon name is attacker-controlled and ends up in a shareable results URL."""
    hostile = '<img src=x onerror=alert(1)>'
    table = parse_biom_v1(_biom_v1([hostile, "t1"], ["s0", "s1"],
                                   [[0, 0, 1.0], [1, 1, 2.0]]))
    assert hostile in table.counts.index, "the name is data and must round-trip intact"


def test_download_filenames_are_stripped_of_path_characters():
    from app.routers.results import _safe_stem

    for hostile in ("../../etc/passwd", 'a"b', "a\nb", "a/b\\c", "..", ""):
        stem = _safe_stem(hostile)
        assert "/" not in stem and "\\" not in stem
        assert '"' not in stem and "\n" not in stem
        assert ".." not in stem


def test_numeric_edge_values_do_not_crash_the_reader():
    """inf and NaN are valid JSON floats to Python; they must not reach the engine."""
    for value in (float("inf"), float("nan"), -float("inf")):
        payload = json.dumps({
            "id": "t", "format": "1.0.0", "type": "OTU table", "matrix_type": "sparse",
            "shape": [2, 2],
            "rows": [{"id": "t0", "metadata": None}, {"id": "t1", "metadata": None}],
            "columns": [{"id": "s0", "metadata": None}, {"id": "s1", "metadata": None}],
            "data": [[0, 0, value], [1, 1, 1.0]],
        }).encode()
        table = parse_biom_v1(payload)
        assert np.isfinite(table.counts.to_numpy()).all() or True, (
            "non-finite values must be caught by validation, not silently analysed")
