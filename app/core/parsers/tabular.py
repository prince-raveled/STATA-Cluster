"""CSV/TSV, MetaPhlAn and Kraken2/Bracken parsers (SPEC §8)."""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from .base import (
    RANK_NAMES,
    AbundanceTable,
    ParseError,
    detect_value_type,
    lineage_rank_index,
    resolve_orientation,
    split_lineage,
)

#: Columns that carry a lineage rather than an abundance.
TAXONOMY_COLUMNS = {
    "taxonomy",
    "consensus lineage",
    "consensuslineage",
    "lineage",
    "taxon",
    "taxa",
    "clade_name",
    "classification",
}
#: Kraken/Bracken bookkeeping columns that are not samples.
KRAKEN_META_COLUMNS = {
    "name",
    "taxonomy_id",
    "taxid",
    "taxonomy_lvl",
    "lvl_type",
    "rank",
}


def _decode(raw) -> str:
    if isinstance(raw, str):
        return raw
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ParseError("Could not decode the file as text. Is it a binary format such as BIOM v2?")


def sniff_delimiter(text: str) -> str:
    """Pick the delimiter, in priority order.

    Priority matters: lineage strings contain ';' and '|', so a taxonomy file with two
    tab-separated columns must never be split on the semicolons inside its lineages.
    A candidate wins only if it splits every sampled line into the same number of
    fields — which a within-field character will not do.
    """
    head = [ln for ln in text.splitlines()[:50] if ln.strip()]
    if not head:
        raise ParseError("The file is empty.")
    sample = head[:20]

    for delim in ("\t", ",", ";", "|"):
        widths = [ln.count(delim) for ln in sample]
        if max(widths) == 0:
            continue
        consistent = len(set(widths)) == 1
        # Header rows sometimes lack the index column's name; allow a one-field slack.
        # Header rows sometimes lack the index column's name, so allow one field of slack.
        near = (len(widths) > 1 and len(set(widths[1:])) <= 1
                and abs(widths[0] - widths[1]) <= 1)
        if consistent or near:
            return delim

    # Nothing split cleanly; fall back to the most consistent candidate.
    best, best_score = "", -1.0
    for delim in ("\t", ",", ";", "|"):
        widths = [ln.count(delim) for ln in sample]
        if max(widths) == 0:
            continue
        common = max(set(widths), key=widths.count)
        score = common * (widths.count(common) / len(widths))
        if score > best_score:
            best, best_score = delim, score
    if not best:
        raise ParseError(
            "Could not find a column delimiter. Supported: tab, comma, semicolon, pipe."
        )
    return best


def read_delimited(raw, filename: str = "") -> pd.DataFrame:
    """Read a delimited table, tolerating QIIME-style leading comment lines."""
    text = _decode(raw)
    lines = text.splitlines()
    # Drop leading comments, but keep the last one if it is the header row.
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("#"):
            start = i
        else:
            break
    body = lines[start:]
    if body and body[0].startswith("#"):
        body[0] = body[0].lstrip("#").strip()
    cleaned = "\n".join(ln for ln in body if ln.strip())
    if not cleaned:
        raise ParseError("The file contains no data rows.")
    delim = sniff_delimiter(cleaned)
    try:
        frame = pd.read_csv(io.StringIO(cleaned), sep=delim, index_col=0, dtype=str)
    except Exception as exc:  # pragma: no cover - pandas message passthrough
        where = f" '{filename}'" if filename else ""
        raise ParseError(f"Could not read the table{where}: {exc}") from exc
    if frame.shape[1] == 0:
        where = f" in '{filename}'" if filename else ""
        raise ParseError(
            f"Only one column was found{where}. Check the delimiter — the file may use "
            "a different separator than its extension suggests."
        )
    frame.index = [str(i).strip() for i in frame.index]
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def _extract_taxonomy_column(frame: pd.DataFrame):
    """Pull a trailing lineage column out of the matrix, if present."""
    lineages = {}
    drop = [c for c in frame.columns if str(c).strip().lower() in TAXONOMY_COLUMNS]
    for col in drop:
        for taxon, value in frame[col].items():
            parts = split_lineage(value)
            if parts:
                lineages[str(taxon)] = parts
    if drop:
        frame = frame.drop(columns=drop)
    return frame, lineages


def _to_numeric(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    bad_cols = [c for c in numeric.columns if numeric[c].isna().all()]
    if bad_cols:
        numeric = numeric.drop(columns=bad_cols)
    if numeric.shape[1] == 0:
        raise ParseError(
            f"No numeric columns found in {context}. Every column parsed as text — "
            "check that the first row is a header and the first column is the feature ID."
        )
    n_missing = int(numeric.isna().to_numpy().sum())
    if n_missing:
        numeric = numeric.fillna(0.0)
    return numeric.astype(float)


def _is_metaphlan(frame: pd.DataFrame, text_head: str) -> bool:
    if "mpa_v" in text_head.lower() or "metaphlan" in text_head.lower():
        return True
    idx = [str(i) for i in frame.index[:20]]
    hits = sum(1 for i in idx if "|" in i and re.search(r"[kdpcofgs]__", i))
    return len(idx) > 0 and hits / len(idx) > 0.6


def _is_kraken_combined(frame: pd.DataFrame) -> bool:
    cols = {str(c).strip().lower() for c in frame.columns}
    return bool(cols & KRAKEN_META_COLUMNS) and any(
        str(c).endswith("_num") or str(c).endswith("_frac") for c in frame.columns
    )


def parse_metaphlan(frame: pd.DataFrame) -> AbundanceTable:
    """MetaPhlAn merged table: hierarchical rows, one lineage per row.

    Only the deepest rank present is kept — summing nested clades would double count.
    """
    frame, _ = _extract_taxonomy_column(frame)
    numeric = _to_numeric(frame, "the MetaPhlAn table")
    lineages = {str(t): split_lineage(t) for t in numeric.index}
    depths = {t: lineage_rank_index(p) for t, p in lineages.items() if p}
    if not depths:
        raise ParseError("No parsable MetaPhlAn lineages found in the first column.")
    deepest = max(depths.values())
    keep = [t for t, d in depths.items() if d == deepest]
    if len(keep) < 5:  # too few at the deepest rank; fall back to the whole table
        keep = list(numeric.index)
    matrix = numeric.loc[keep]
    value_type = detect_value_type(matrix)
    return AbundanceTable(
        counts=matrix,
        lineages={t: lineages.get(t, []) for t in matrix.index},
        source_format="metaphlan",
        value_type=value_type,
        input_rank=RANK_NAMES[deepest] if 0 <= deepest < len(RANK_NAMES) else "input",
        notes=[f"MetaPhlAn table: kept {len(keep)} rows at the deepest rank present."],
    )


def parse_kraken_combined(frame: pd.DataFrame) -> AbundanceTable:
    """Combined Bracken/Kraken2 table (combine_bracken_outputs.py layout)."""
    frame = frame.copy()
    lower = {str(c): str(c).strip().lower() for c in frame.columns}
    lineages = {}
    rank_col = next(
        (c for c, name in lower.items() if name in {"taxonomy_lvl", "lvl_type", "rank"}),
        None,
    )
    input_rank = "input"
    if rank_col is not None:
        codes = frame[rank_col].astype(str).str.strip().str.upper()
        code_map = {"S": "species", "G": "genus", "F": "family", "O": "order", "P": "phylum"}
        common = codes.mode()
        if len(common):
            input_rank = code_map.get(common.iloc[0], "input")
    drop = [c for c, name in lower.items() if name in KRAKEN_META_COLUMNS]
    counts_cols = [c for c in frame.columns if str(c).endswith("_num")]
    if counts_cols:
        matrix = frame[counts_cols].copy()
        matrix.columns = [str(c)[: -len("_num")] for c in counts_cols]
    else:
        matrix = frame.drop(columns=drop, errors="ignore")
        matrix = matrix[[c for c in matrix.columns if not str(c).endswith("_frac")]]
    numeric = _to_numeric(matrix, "the Kraken/Bracken table")
    return AbundanceTable(
        counts=numeric,
        lineages=lineages,
        source_format="kraken/bracken",
        value_type=detect_value_type(numeric),
        input_rank=input_rank,
        notes=["Kraken/Bracken combined report: read-count columns used."],
    )


def parse_tabular(raw, filename: str = "", sample_ids=None) -> AbundanceTable:
    """Entry point for text tables. Auto-detects layout, orientation and value type."""
    text = _decode(raw)
    head = "\n".join(text.splitlines()[:3])
    frame = read_delimited(text, filename)

    if _is_kraken_combined(frame):
        return parse_kraken_combined(frame)
    if _is_metaphlan(frame, head):
        return parse_metaphlan(frame)

    frame, lineages = _extract_taxonomy_column(frame)
    numeric = _to_numeric(frame, "the abundance table")
    if lineages:
        # A lineage column settles the orientation: its rows are the taxa. Without this
        # a table with more samples than taxa would be transposed by the shape heuristic.
        matrix = numeric
        note = "orientation: taxa x samples (a taxonomy column identified the rows)"
    else:
        matrix, note = resolve_orientation(numeric, sample_ids)
    matrix = matrix.loc[~matrix.index.duplicated(keep="first")]

    if not lineages:
        lineages = {}
        for taxon in matrix.index:
            parts = split_lineage(taxon)
            if len(parts) > 1:
                lineages[str(taxon)] = parts

    depths = [lineage_rank_index(p) for p in lineages.values() if p]
    input_rank = RANK_NAMES[int(np.median(depths))] if depths else "input"
    return AbundanceTable(
        counts=matrix,
        lineages=lineages,
        source_format="csv/tsv",
        value_type=detect_value_type(matrix),
        input_rank=input_rank,
        notes=[note],
    )
