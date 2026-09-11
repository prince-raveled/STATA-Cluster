"""Format dispatch: filename hint first, content sniffing second (SPEC §8)."""
from __future__ import annotations

import pandas as pd

from .base import (  # noqa: F401  (re-exported)
    RANK_NAMES,
    AbundanceTable,
    ParseError,
    split_lineage,
)
from .biom import parse_biom, parse_biom_v1, parse_biom_v2, parse_qza
from .tabular import parse_tabular, read_delimited

SUPPORTED_FORMATS = (
    "CSV / TSV abundance matrix",
    "BIOM v1 (JSON) and v2 (HDF5)",
    "QIIME 2 artifact (.qza)",
    "MetaPhlAn merged table",
    "Kraken2 / Bracken combined report",
)


def parse_abundance(raw: bytes, filename: str = "", sample_ids=None) -> AbundanceTable:
    """Parse an uploaded abundance table of any supported format."""
    name = (filename or "").lower()
    if name.endswith(".qza"):
        return parse_qza(raw)
    if name.endswith(".biom"):
        return parse_biom(raw)

    head = bytes(raw[:8]) if isinstance(raw, (bytes, bytearray)) else b""
    if head.startswith(b"\x89HDF"):
        return parse_biom_v2(raw)
    if head.startswith(b"PK\x03\x04"):
        return parse_qza(raw)

    stripped = raw.lstrip()[:1] if isinstance(raw, (bytes, bytearray)) else b""
    if stripped == b"{":
        return parse_biom_v1(raw)

    return parse_tabular(raw, filename, sample_ids)


def parse_metadata(raw: bytes, filename: str = "") -> pd.DataFrame:
    """Sample metadata: IDs in the first column, one row per sample."""
    frame = read_delimited(raw, filename)
    frame.index = [str(i).strip() for i in frame.index]
    frame.index.name = "sample_id"
    if frame.index.duplicated().any():
        dupes = sorted({str(i) for i in frame.index[frame.index.duplicated()]})[:5]
        raise ParseError(
            "Duplicate sample IDs in the metadata: " + ", ".join(dupes) +
            ". Sample IDs must be unique."
        )
    # QIIME 2 metadata files carry a '#q2:types' directive row.
    directives = ("#q2", "#types", "#sampleid")
    drop_rows = [i for i in frame.index if str(i).lower().startswith(directives)]
    if drop_rows:
        frame = frame.drop(index=drop_rows)

    # Columns arrive as text (the reader is deliberately untyped so IDs survive
    # intact). Restore numeric dtypes here: without it a continuous covariate such
    # as read depth or age looks like a per-sample identifier and is silently
    # dropped from the covariate candidates.
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().mean() >= 0.9 and numeric.nunique(dropna=True) > 1:
            frame[column] = numeric
    return frame


def parse_taxonomy_map(raw: bytes, filename: str = "") -> dict:
    """Optional two-column feature-ID -> lineage map."""
    frame = read_delimited(raw, filename)
    if frame.shape[1] == 0:
        raise ParseError("The taxonomy file needs at least two columns: feature ID and lineage.")
    column = frame.columns[0]
    for candidate in frame.columns:
        if str(candidate).strip().lower() in {"taxon", "taxonomy", "lineage", "consensus lineage"}:
            column = candidate
            break
    mapping = {}
    for key, value in frame[column].items():
        parts = split_lineage(value)
        if parts:
            mapping[str(key)] = parts
    if not mapping:
        raise ParseError(
            "No lineages could be parsed from the taxonomy file. Expected strings like "
            "'k__Bacteria;p__Firmicutes;...;g__Blautia'."
        )
    return mapping
