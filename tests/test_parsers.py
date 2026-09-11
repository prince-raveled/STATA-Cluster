"""Parser tests — SPEC §8, including deliberately malformed input."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.core.parsers import (
    ParseError,
    parse_abundance,
    parse_metadata,
    parse_taxonomy_map,
)
from app.core.parsers.base import lineage_rank_index, split_lineage
from app.core.parsers.tabular import sniff_delimiter


def test_tsv_round_trip(tsv_bytes, synthetic):
    table = parse_abundance(tsv_bytes, "counts.tsv")
    assert table.n_taxa == synthetic["table"].n_taxa
    assert table.n_samples == synthetic["table"].n_samples
    assert table.value_type == "counts"
    assert table.is_integer


def test_csv_and_transposed_orientation_agree(synthetic):
    frame = synthetic["table"].counts
    normal = parse_abundance(frame.to_csv().encode(), "a.csv")
    flipped = parse_abundance(frame.T.to_csv().encode(), "b.csv",
                              sample_ids=list(frame.columns))
    assert normal.counts.shape == flipped.counts.shape
    np.testing.assert_allclose(
        normal.counts.to_numpy(float),
        flipped.counts.loc[normal.taxa, normal.sample_ids].to_numpy(float),
    )


def test_orientation_uses_metadata_ids(synthetic):
    """Ambiguous shapes must be resolved by the metadata, not by a shape heuristic."""
    frame = synthetic["table"].counts.iloc[:40, :40]  # square: no shape hint available
    table = parse_abundance(frame.T.to_csv().encode(), "square.csv",
                            sample_ids=list(frame.columns))
    assert set(table.sample_ids) == set(frame.columns)


def test_semicolon_lineages_do_not_break_the_delimiter_sniffer():
    """A two-column TSV whose values contain ';' must not split on the semicolons."""
    text = (
        "feature_id\ttaxon\n"
        "ASV1\tk__Bacteria;p__Firmicutes;c__Clostridia;g__Blautia\n"
        "ASV2\tk__Bacteria;p__Bacteroidetes;c__Bacteroidia;g__Bacteroides\n"
    )
    assert sniff_delimiter(text) == "\t"
    mapping = parse_taxonomy_map(text.encode(), "tax.tsv")
    assert lineage_rank_index(mapping["ASV1"]) == 5  # genus


def test_taxonomy_column_is_pulled_out_of_the_matrix():
    text = (
        "#OTU ID\tS1\tS2\tS3\ttaxonomy\n"
        "o1\t10\t20\t30\tk__Bacteria;p__Firmicutes;g__Blautia\n"
        "o2\t5\t0\t7\tk__Bacteria;p__Bacteroidetes;g__Bacteroides\n"
    )
    table = parse_abundance(text.encode(), "otu.tsv")
    assert table.n_samples == 3
    assert "taxonomy" not in [c.lower() for c in table.sample_ids]
    assert table.lineages["o1"][-1] == "g__Blautia"


def test_metaphlan_keeps_only_the_deepest_rank():
    rows = ["clade_name\tSA\tSB"]
    for i in range(8):
        rows.append(f"k__Bacteria|p__Firmicutes|g__G{i}\t{i + 1}\t{i + 2}")
        rows.append(f"k__Bacteria|p__Firmicutes|g__G{i}|s__S{i}\t{i + 1}\t{i + 2}")
    table = parse_abundance("\n".join(rows).encode(), "mpa.tsv")
    assert table.source_format == "metaphlan"
    assert table.n_taxa == 8  # species rows only; summing both would double count
    assert table.input_rank == "species"


def test_bracken_combined_report_uses_count_columns():
    text = (
        "name\ttaxonomy_id\ttaxonomy_lvl\tS1_num\tS1_frac\tS2_num\tS2_frac\n"
        "Blautia\t572511\tG\t100\t0.5\t150\t0.6\n"
        "Bacteroides\t816\tG\t80\t0.4\t60\t0.3\n"
    )
    table = parse_abundance(text.encode(), "bracken.tsv")
    assert table.source_format == "kraken/bracken"
    assert table.sample_ids == ["S1", "S2"]
    assert table.input_rank == "genus"
    assert table.counts.loc["Blautia", "S2"] == 150


def test_biom_v1_sparse():
    doc = {
        "id": "test", "format": "1.0.0", "format_url": "http://biom-format.org",
        "type": "OTU table", "generated_by": "test", "date": "2026-01-01",
        "matrix_type": "sparse", "matrix_element_type": "int",
        "shape": [2, 3],
        "rows": [
            {"id": "o1", "metadata": {"taxonomy": ["k__Bacteria", "p__Firmicutes",
                                                   "c__Clostridia", "o__Clostridiales",
                                                   "f__Lachnospiraceae", "g__Blautia"]}},
            {"id": "o2", "metadata": {"taxonomy": ["k__Bacteria", "p__Bacteroidetes"]}},
        ],
        "columns": [{"id": f"S{i}", "metadata": None} for i in range(3)],
        "data": [[0, 0, 5], [0, 2, 7], [1, 1, 3]],
    }
    table = parse_abundance(json.dumps(doc).encode(), "table.biom")
    assert table.source_format == "biom v1"
    assert table.counts.loc["o1", "S0"] == 5
    assert table.counts.loc["o2", "S1"] == 3
    assert table.counts.loc["o1", "S1"] == 0
    assert table.lineages["o1"][-1] == "g__Blautia"


def test_biom_v1_dense():
    doc = {
        "matrix_type": "dense", "shape": [2, 2],
        "rows": [{"id": "o1", "metadata": None}, {"id": "o2", "metadata": None}],
        "columns": [{"id": "S1", "metadata": None}, {"id": "S2", "metadata": None}],
        "data": [[1, 2], [3, 4]],
    }
    table = parse_abundance(json.dumps(doc).encode(), "d.biom")
    assert table.counts.loc["o2", "S2"] == 4


# --- malformed input: every message must be a sentence, never a stack trace ------
@pytest.mark.parametrize(
    "payload, filename",
    [
        (b"", "empty.tsv"),
        (b"just one line with no delimiter at all", "nodelim.tsv"),
        (b"{not valid json at all", "broken.biom"),
        (b"a\tb\nc\td\ne\tf", "alltext.tsv"),
    ],
)
def test_malformed_files_raise_parse_error(payload, filename):
    with pytest.raises(ParseError) as info:
        parse_abundance(payload, filename)
    assert str(info.value)
    assert "Traceback" not in str(info.value)


def test_duplicate_metadata_ids_are_rejected():
    text = "sample_id\tgroup\nS1\ta\nS1\tb\nS2\ta\n"
    with pytest.raises(ParseError, match="Duplicate sample IDs"):
        parse_metadata(text.encode(), "meta.tsv")


def test_metadata_parses_and_keeps_order(metadata_bytes, synthetic):
    metadata = parse_metadata(metadata_bytes, "meta.tsv")
    assert list(metadata.index) == list(synthetic["metadata"].index)
    assert "group" in metadata.columns


def test_split_lineage_handles_placeholders_and_separators():
    assert split_lineage("d__Bacteria|p__Firmicutes|g__Blautia")[0] == "k__Bacteria"
    assert split_lineage("k__Bacteria;p__Firmicutes;g__") == ["k__Bacteria", "p__Firmicutes"]
    assert split_lineage("") == []
    assert split_lineage(float("nan")) == []
