"""Format dispatch: filename hint first, content sniffing second (SPEC §8)."""
from __future__ import annotations

import gzip
import io
import zipfile

import pandas as pd

from .base import (  # noqa: F401  (re-exported)
    MAX_DECOMPRESSED_BYTES,
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


#: Spreadsheet formats people upload by mistake. They are zip (or OLE) containers, so
#: without this they reached the .qza reader and were called broken QIIME 2 artifacts.
SPREADSHEET_SUFFIXES = (".xlsx", ".xlsm", ".xls", ".ods", ".numbers")


def _gunzip(raw, filename: str):
    """Decompress a gzipped upload, which the upload form offers (.gz), within a limit.

    It used to be decoded as text as it stood, and failed with a tokenizer error about
    line 23. The limit is the one .qza archives already have, so a small compressed file
    cannot expand into all the memory the host has.
    """
    if not isinstance(raw, (bytes, bytearray)) or bytes(raw[:2]) != b"\x1f\x8b":
        return raw, filename
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as handle:
            data = handle.read(MAX_DECOMPRESSED_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise ParseError(
            f"'{filename}' looks gzipped but could not be decompressed: {exc}") from exc
    if len(data) > MAX_DECOMPRESSED_BYTES:
        raise ParseError(f"'{filename}' expands past the {MAX_DECOMPRESSED_BYTES / 1e6:,.0f} MB "
                         "decompression limit.")
    name = filename[:-3] if filename.lower().endswith(".gz") else filename
    return data, name


def _refuse_spreadsheet(raw, filename: str) -> None:
    name = (filename or "").lower()
    head = bytes(raw[:8]) if isinstance(raw, (bytes, bytearray)) else b""
    is_ole = head.startswith(b"\xd0\xcf\x11\xe0")                     # legacy .xls
    is_workbook = False
    if head.startswith(b"PK\x03\x04") and not name.endswith(".qza"):
        try:
            members = zipfile.ZipFile(io.BytesIO(raw)).namelist()
            is_workbook = any(m.startswith(("xl/", "content.xml")) for m in members)
        except zipfile.BadZipFile:
            pass
    if name.endswith(SPREADSHEET_SUFFIXES) or is_ole or is_workbook:
        raise ParseError(
            f"'{filename}' is a spreadsheet workbook, which MicroVerse does not read directly. "
            "Save the sheet as CSV or TSV (File > Save As) and upload that.")


def parse_abundance(raw: bytes, filename: str = "", sample_ids=None) -> AbundanceTable:
    """Parse an uploaded abundance table of any supported format."""
    raw, filename = _gunzip(raw, filename)
    _refuse_spreadsheet(raw, filename)
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
    raw, filename = _gunzip(raw, filename)
    _refuse_spreadsheet(raw, filename)
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
    raw, filename = _gunzip(raw, filename)
    _refuse_spreadsheet(raw, filename)
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
