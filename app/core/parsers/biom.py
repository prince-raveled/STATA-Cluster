"""BIOM v1 (JSON) and v2 (HDF5) readers, plus QIIME 2 .qza extraction (SPEC §8).

`biom-format` has no wheel on this Python, so both versions are read directly:
v1 is plain JSON, v2 is a documented HDF5 layout. Logged in SPEC §24.
"""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd

from .base import (
    MAX_DECOMPRESSED_BYTES,
    RANK_NAMES,
    AbundanceTable,
    ParseError,
    check_declared_shape,
    detect_value_type,
    lineage_rank_index,
    split_lineage,
)


def _lineage_from_metadata(meta) -> list:
    if not meta:
        return []
    if isinstance(meta, dict):
        for key in ("taxonomy", "Taxonomy", "lineage", "ConsensusLineage"):
            if key in meta:
                value = meta[key]
                if isinstance(value, (list, tuple)):
                    return split_lineage(";".join(str(v) for v in value))
                return split_lineage(value)
    if isinstance(meta, (list, tuple)):
        return split_lineage(";".join(str(v) for v in meta))
    return split_lineage(meta)


def parse_biom_v1(raw) -> AbundanceTable:
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"This is not valid BIOM v1 JSON: {exc.msg} (line {exc.lineno}).") from exc

    if "rows" not in doc or "columns" not in doc:
        raise ParseError("BIOM v1 file is missing its 'rows' or 'columns' block.")

    taxa = [str(r["id"]) for r in doc["rows"]]
    samples = [str(c["id"]) for c in doc["columns"]]
    lineages = {
        str(r["id"]): _lineage_from_metadata(r.get("metadata")) for r in doc["rows"]
    }
    lineages = {k: v for k, v in lineages.items() if v}

    check_declared_shape(len(taxa), len(samples), "This BIOM v1 file")

    values = np.zeros((len(taxa), len(samples)), dtype=float)
    if doc.get("matrix_type") == "dense":
        dense = np.asarray(doc["data"], dtype=float)
        if dense.shape != values.shape:
            raise ParseError(
                f"BIOM v1 dense matrix is {dense.shape[0]}x{dense.shape[1]} but the "
                f"file declares {len(taxa)} taxa and {len(samples)} samples.")
        values = dense
    else:
        # Indices come from the file, so they are untrusted. Python would accept a
        # negative index and silently write to the wrong cell — a corrupted table that
        # still analyses cleanly, which is worse than an error.
        for entry in doc["data"]:
            try:
                row, col, val = int(entry[0]), int(entry[1]), float(entry[2])
            except (TypeError, ValueError, IndexError) as exc:
                raise ParseError(
                    "BIOM v1 sparse data must be [row, column, value] triples.") from exc
            if not (0 <= row < len(taxa)) or not (0 <= col < len(samples)):
                raise ParseError(
                    f"BIOM v1 sparse entry [{row}, {col}] is outside the declared "
                    f"{len(taxa)}x{len(samples)} matrix.")
            values[row, col] = val

    matrix = pd.DataFrame(values, index=taxa, columns=samples)
    depths = [lineage_rank_index(p) for p in lineages.values() if p]
    return AbundanceTable(
        counts=matrix,
        lineages=lineages,
        source_format="biom v1",
        value_type=detect_value_type(matrix),
        input_rank=RANK_NAMES[int(np.median(depths))] if depths else "input",
        notes=[f"BIOM v1 ({doc.get('matrix_type', 'sparse')}) read directly."],
    )


def parse_biom_v2(raw) -> AbundanceTable:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ParseError(
            "BIOM v2 (HDF5) support needs the 'h5py' package. Convert to BIOM v1 JSON "
            "or TSV, or install h5py."
        ) from exc

    buffer = io.BytesIO(raw) if isinstance(raw, (bytes, bytearray)) else raw
    try:
        handle = h5py.File(buffer, "r")
    except OSError as exc:
        raise ParseError(f"Could not open this file as BIOM v2 (HDF5): {exc}") from exc

    with handle as f:
        try:
            taxa = [x.decode() if isinstance(x, bytes) else str(x) for x in f["observation/ids"][:]]
            samples = [x.decode() if isinstance(x, bytes) else str(x) for x in f["sample/ids"][:]]
            data = f["observation/matrix/data"][:]
            indices = f["observation/matrix/indices"][:]
            indptr = f["observation/matrix/indptr"][:]
        except KeyError as exc:
            raise ParseError(f"BIOM v2 file is missing the {exc} dataset.") from exc

        check_declared_shape(len(taxa), len(samples), "This BIOM v2 file")
        if len(indptr) != len(taxa) + 1:
            raise ParseError(
                f"BIOM v2 index pointer has {len(indptr)} entries; a CSR matrix over "
                f"{len(taxa)} taxa needs {len(taxa) + 1}.")

        values = np.zeros((len(taxa), len(samples)), dtype=float)
        for i in range(len(taxa)):
            lo, hi = int(indptr[i]), int(indptr[i + 1])
            if not 0 <= lo <= hi <= len(data):
                raise ParseError(
                    f"BIOM v2 index pointer is not monotone at taxon {i} "
                    f"({lo}..{hi} of {len(data)} values).")
            columns = indices[lo:hi].astype(int)
            if columns.size and (columns.min() < 0 or columns.max() >= len(samples)):
                raise ParseError(
                    f"BIOM v2 column index out of range at taxon {i}: the file "
                    f"declares {len(samples)} samples.")
            values[i, columns] = data[lo:hi]

        lineages = {}
        tax_path = "observation/metadata/taxonomy"
        if tax_path in f:
            raw_tax = f[tax_path][:]
            for taxon, row in zip(taxa, raw_tax, strict=False):
                tokens = [
                    x.decode() if isinstance(x, bytes) else str(x)
                    for x in np.atleast_1d(row)
                ]
                parts = split_lineage(";".join(tokens))
                if parts:
                    lineages[taxon] = parts

    matrix = pd.DataFrame(values, index=taxa, columns=samples)
    depths = [lineage_rank_index(p) for p in lineages.values() if p]
    return AbundanceTable(
        counts=matrix,
        lineages=lineages,
        source_format="biom v2",
        value_type=detect_value_type(matrix),
        input_rank=RANK_NAMES[int(np.median(depths))] if depths else "input",
        notes=["BIOM v2 (HDF5) read directly."],
    )


def parse_biom(raw) -> AbundanceTable:
    head = bytes(raw[:8]) if isinstance(raw, (bytes, bytearray)) else b""
    if head.startswith(b"\x89HDF"):
        return parse_biom_v2(raw)
    return parse_biom_v1(raw)


def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    """Read one archive member, refusing anything that expands past the limit.

    The per-archive check uses the declared sizes in the central directory, which a
    crafted file can understate. This reads with an explicit cap so the actual number
    of decompressed bytes is bounded too.
    """
    with archive.open(name) as handle:
        payload = handle.read(MAX_DECOMPRESSED_BYTES + 1)
    if len(payload) > MAX_DECOMPRESSED_BYTES:
        raise ParseError(
            f"'{name}' inside this .qza expands past the "
            f"{MAX_DECOMPRESSED_BYTES / 1e6:,.0f} MB decompression limit.")
    return payload


def parse_qza(raw) -> AbundanceTable:
    """QIIME 2 artifact: a zip whose payload contains a BIOM table."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise ParseError("This .qza file is not a readable zip archive.") from exc

    with archive:
        # A .qza is a zip, so the upload cap bounds the compressed size only. Read the
        # declared expanded size from the central directory and refuse before calling
        # read(), which would otherwise decompress a small file into gigabytes.
        expanded = sum(max(info.file_size, 0) for info in archive.infolist())
        if expanded > MAX_DECOMPRESSED_BYTES:
            raise ParseError(
                f"This .qza expands to {expanded / 1e6:,.0f} MB, above the "
                f"{MAX_DECOMPRESSED_BYTES / 1e6:,.0f} MB limit for a decompressed "
                f"upload. Export the feature table and upload it directly.")
        names = archive.namelist()
        biom_names = [n for n in names if n.endswith(".biom")]
        if not biom_names:
            tsv_names = [n for n in names if n.endswith((".tsv", ".txt", ".csv"))]
            if not tsv_names:
                raise ParseError(
                    "No feature table found inside the .qza. Export it with "
                    "'qiime tools export' and upload the BIOM or TSV instead."
                )
            from .tabular import parse_tabular

            table = parse_tabular(_read_member(archive, tsv_names[0]), tsv_names[0])
            table.source_format = "qza (tsv payload)"
            return table
        table = parse_biom(_read_member(archive, biom_names[0]))
        table.source_format = f"qza ({table.source_format})"
        return table
