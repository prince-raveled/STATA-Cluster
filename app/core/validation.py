"""Input validation — SPEC §8 hard constraints.

Every rejection is a `DatasetError` carrying a sentence a researcher can act on.
Nothing here is allowed to surface a stack trace.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .parsers.base import AbundanceTable

MIN_SAMPLES = 10
MIN_PER_GROUP = 5
MIN_TAXA = 10
MAX_TAXA = 1500

#: Field separator for the fingerprint hashes — a byte that cannot occur in a
#: taxon or sample name, so two different tables cannot collide by concatenation.
SEPARATOR = bytes([0])


class DatasetError(ValueError):
    """A refusal the user can fix. `hint` is shown under the message."""

    status_code = 422

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint


class NotFoundError(DatasetError):
    """An unknown or expired token, or a resource that does not exist."""

    status_code = 404


class RunNotReadyError(DatasetError):
    """The token is valid; the run behind it has not produced results yet.

    Distinct from both 404 (nothing there) and 422 (the request was wrong). Nothing is
    wrong with the request — a results URL is meant to be shared, and sharing it a few
    seconds early must not look like a rejection.
    """

    status_code = 409


@dataclass
class Dataset:
    """A validated, aligned run input."""

    table: AbundanceTable
    metadata: pd.DataFrame
    group_column: str
    group_labels: tuple  # (reference A, comparison B)
    groups: np.ndarray  # int8, 0 = A, 1 = B, aligned to table columns
    covariate_columns: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    can_rarefy: bool = True
    collapsed_from: str = ""

    @property
    def n_samples(self) -> int:
        return self.table.n_samples

    @property
    def n_taxa(self) -> int:
        return self.table.n_taxa

    @property
    def group_sizes(self) -> tuple:
        return int((self.groups == 0).sum()), int((self.groups == 1).sum())

    @property
    def max_library(self) -> float:
        return float(self.table.library_sizes.max())

    @property
    def min_library(self) -> float:
        return float(self.table.library_sizes.min())

    def fingerprint(self) -> dict:
        """Content hashes of exactly what was analysed, after alignment and validation.

        Not a hash of the uploaded file: two different uploads can produce the same
        analysis (column order, an unused metadata column), and the same upload can
        produce different analyses (a different group column). What has to be
        reproducible is the matrix and the labels the engine actually saw, so that is
        what is hashed. SHA-256 over a canonical byte serialisation.
        """
        import hashlib

        counts = self.table.counts
        digest = hashlib.sha256()
        digest.update(",".join(map(str, counts.index)).encode())
        digest.update(SEPARATOR)
        digest.update(",".join(map(str, counts.columns)).encode())
        digest.update(SEPARATOR)
        digest.update(np.ascontiguousarray(counts.to_numpy(dtype=np.float64)).tobytes())

        labels = hashlib.sha256()
        labels.update(",".join(map(str, counts.columns)).encode())
        labels.update(SEPARATOR)
        labels.update(self.groups.astype(np.int8).tobytes())

        covariates = hashlib.sha256()
        if self.covariate_columns:
            frame = self.metadata.loc[list(counts.columns), list(self.covariate_columns)]
            covariates.update(frame.to_csv().encode())

        return {
            "abundance_sha256": digest.hexdigest(),
            "group_labels_sha256": labels.hexdigest(),
            "covariates_sha256": covariates.hexdigest() if self.covariate_columns else None,
            "algorithm": "sha256 over taxa, samples and the float64 count matrix",
        }

    def summary(self) -> dict:
        base = self.table.summary()
        n_a, n_b = self.group_sizes
        base.update(
            {
                "fingerprint": self.fingerprint(),
                "group_column": self.group_column,
                "group_a": self.group_labels[0],
                "group_b": self.group_labels[1],
                "n_group_a": n_a,
                "n_group_b": n_b,
                "can_rarefy": self.can_rarefy,
                "covariates_available": list(self.covariate_columns),
                "warnings": list(self.warnings),
                "collapsed_from": self.collapsed_from,
            }
        )
        return base


def _candidate_covariates(metadata: pd.DataFrame, group_column: str) -> list:
    """Columns usable as covariates: not the grouping variable, not constant, not an ID."""
    out = []
    for column in metadata.columns:
        if column == group_column:
            continue
        series = metadata[column]
        n_unique = series.nunique(dropna=True)
        if n_unique < 2:
            continue
        # A column with a distinct value for nearly every sample is an identifier.
        if n_unique > max(2, 0.9 * len(series)) and not pd.api.types.is_numeric_dtype(series):
            continue
        if series.isna().mean() > 0.2:
            continue
        out.append(str(column))
    return out


def infer_group_column(metadata: pd.DataFrame) -> str:
    """Pick the grouping column: an explicit `group`, else the best binary column."""
    named = {"group", "condition", "status", "disease", "phenotype"}
    for name in metadata.columns:
        if str(name).strip().lower() in named and metadata[name].dropna().nunique() == 2:
            return str(name)
    binary = [c for c in metadata.columns if metadata[c].dropna().nunique() == 2]
    if not binary:
        raise DatasetError(
            "No binary grouping column found in the metadata.",
            "MicroVerse compares exactly two groups. Add a column with two values "
            "(for example 'case' and 'control') and name it 'group'.",
        )
    return str(binary[0])


def validate_dataset(
    table: AbundanceTable,
    metadata: pd.DataFrame,
    group_column: str = "",
    covariates=None,
    force_collapse: bool = True,
) -> Dataset:
    """Apply every §8 constraint and return an aligned Dataset."""
    warnings: list = []

    if table.n_taxa == 0 or table.n_samples == 0:
        raise DatasetError(
            "The abundance table parsed to an empty matrix.",
            "Check that the first row is a header of sample IDs and the first column "
            "holds feature IDs.",
        )

    # --- negative values: already transformed -----------------------------
    if table.has_negative:
        raise DatasetError(
            "The abundance table contains negative values, so it has already been "
            "transformed (CLR, log-ratio or z-scored).",
            "MicroVerse must apply the transformations itself — varying them is one "
            "of the choices it explores. Upload raw counts or relative abundances.",
        )

    # --- metadata alignment -----------------------------------------------
    if group_column and group_column not in metadata.columns:
        raise DatasetError(
            f"Grouping column '{group_column}' is not in the metadata.",
            "Available columns: " + ", ".join(str(c) for c in metadata.columns[:20]),
        )
    group_column = group_column or infer_group_column(metadata)

    table_ids = [str(c) for c in table.counts.columns]
    meta_ids = [str(i) for i in metadata.index]
    shared = [s for s in table_ids if s in set(meta_ids)]

    if not shared:
        raise DatasetError(
            "No sample IDs are shared between the abundance table and the metadata.",
            "Table IDs look like: " + ", ".join(table_ids[:4]) +
            " — metadata IDs look like: " + ", ".join(meta_ids[:4]) +
            ". They must match exactly, including case.",
        )

    only_table = [s for s in table_ids if s not in set(meta_ids)]
    only_meta = [s for s in meta_ids if s not in set(table_ids)]
    if only_table:
        warnings.append(
            f"{len(only_table)} sample(s) in the abundance table have no metadata and were "
            f"dropped: {', '.join(only_table[:5])}" + (" ..." if len(only_table) > 5 else "")
        )
    if only_meta:
        warnings.append(
            f"{len(only_meta)} sample(s) in the metadata are absent from the abundance table "
            f"and were dropped: {', '.join(only_meta[:5])}" + (" ..." if len(only_meta) > 5 else "")
        )

    metadata = metadata.loc[shared]
    group_series = metadata[group_column]
    missing_group = group_series.isna() | (group_series.astype(str).str.strip() == "")
    if missing_group.any():
        dropped = [s for s, m in zip(shared, missing_group, strict=True) if m]
        warnings.append(
            f"{len(dropped)} sample(s) had no value for '{group_column}' and were dropped."
        )
        shared = [s for s, m in zip(shared, missing_group, strict=True) if not m]
        metadata = metadata.loc[shared]
        group_series = metadata[group_column]

    # --- exactly two groups ------------------------------------------------
    levels = [str(v) for v in pd.unique(group_series.astype(str).str.strip())]
    if len(levels) != 2:
        raise DatasetError(
            f"The grouping column '{group_column}' has {len(levels)} distinct value(s): "
            + ", ".join(levels[:8]) + ("..." if len(levels) > 8 else "") + ".",
            "MicroVerse v1 compares exactly two groups (SPEC §21). Subset your metadata to "
            "two levels, or pick a different column.",
        )
    levels = sorted(levels)

    aligned = table.align_to(shared)
    values = group_series.astype(str).str.strip().to_numpy()
    groups = np.where(values == levels[1], 1, 0).astype(np.int8)

    # --- sample-count constraints -----------------------------------------
    n_a, n_b = int((groups == 0).sum()), int((groups == 1).sum())
    if aligned.n_samples < MIN_SAMPLES:
        raise DatasetError(
            f"Only {aligned.n_samples} samples matched between the table and the metadata; "
            f"MicroVerse needs at least {MIN_SAMPLES}.",
            "Below this the multiverse is noise: nearly every specification is "
            "underpowered, so the distribution of answers reflects sampling variability "
            "rather than analytical choice.",
        )
    if min(n_a, n_b) < MIN_PER_GROUP:
        raise DatasetError(
            f"Group sizes are {levels[0]}={n_a} and {levels[1]}={n_b}; MicroVerse needs at "
            f"least {MIN_PER_GROUP} samples in each group.",
            "Below this the multiverse is noise rather than a measurement of analytical "
            "degrees of freedom.",
        )

    # --- taxon-count constraints ------------------------------------------
    if aligned.n_taxa < MIN_TAXA:
        raise DatasetError(
            f"The table has {aligned.n_taxa} taxa; MicroVerse needs at least {MIN_TAXA}.",
            "This is not a community profile. Multiple-testing behaviour — which is half of "
            "what the multiverse measures — is meaningless at this size.",
        )

    collapsed_from = ""
    if aligned.n_taxa > MAX_TAXA:
        if force_collapse and aligned.can_collapse_to_genus:
            before = aligned.n_taxa
            collapsed = aligned.collapse_to_genus(aligned.counts)
            # The collapsed index *is* a ';'-joined lineage truncated at genus, so the
            # new lineage map is read straight back off it.
            aligned = AbundanceTable(
                counts=collapsed,
                lineages={
                    str(t): [p for p in str(t).split(";") if p] for t in collapsed.index
                },
                source_format=aligned.source_format,
                value_type=aligned.value_type,
                input_rank="genus",
                notes=list(aligned.notes),
            )
            collapsed_from = f"{before} features"
            warnings.append(
                f"{before} taxa exceeded the {MAX_TAXA}-taxon limit, so the table was collapsed "
                f"to genus ({aligned.n_taxa} genera). Fork 4 (rank) therefore has one level."
            )
            if aligned.n_taxa > MAX_TAXA:
                raise DatasetError(
                    f"Even after collapsing to genus the table has {aligned.n_taxa} features, "
                    f"above the {MAX_TAXA} limit.",
                    "The grid is quadratic in taxa for the multiple-testing step. Pre-filter "
                    "to the taxa you intend to test.",
                )
        else:
            raise DatasetError(
                f"The table has {aligned.n_taxa} taxa, above the {MAX_TAXA} limit, and no "
                "taxonomy was supplied so it cannot be collapsed to genus.",
                "Collapse to genus yourself, or upload a taxonomy file mapping feature IDs "
                "to lineages.",
            )

    # --- value-type constraints -------------------------------------------
    can_rarefy = True
    if not aligned.is_integer:
        can_rarefy = False
        warnings.append(
            "Values are not integers, so this table cannot be subsampled. Fork 1 is reduced "
            "to its 'none' level and the grid shrinks accordingly — rarefaction variance is "
            "not measured for this dataset."
        )
    if aligned.value_type == "relative":
        warnings.append(
            "Values look like relative abundances. Count-based methods (PyDESeq2, ANCOM-BC) "
            "and TMM are unavailable; specifications requiring them are pruned."
        )

    libs = aligned.library_sizes
    if libs.min() <= 0:
        empty = [s for s, v in zip(aligned.sample_ids, libs, strict=True) if v <= 0]
        raise DatasetError(
            f"{len(empty)} sample(s) have a total abundance of zero: {', '.join(empty[:5])}.",
            "Remove empty samples before uploading.",
        )
    if not aligned.can_collapse_to_genus:
        if aligned.has_taxonomy:
            warnings.append(
                "The table is already at genus level or coarser, so the taxonomic-rank "
                "choice has a single level and the grid shrinks accordingly."
            )
        else:
            warnings.append(
                "No taxonomy was detected, so the taxonomic-rank choice has a single "
                "level. Supply lineages to vary it."
            )

    covariate_columns = _candidate_covariates(metadata, group_column)
    if covariates:
        missing = [c for c in covariates if c not in metadata.columns]
        if missing:
            raise DatasetError(
                "Requested covariate column(s) not in the metadata: " + ", ".join(missing) + ".",
                "Available: " + ", ".join(covariate_columns[:20]),
            )
        covariate_columns = list(covariates)

    return Dataset(
        table=aligned,
        metadata=metadata,
        group_column=str(group_column),
        group_labels=(levels[0], levels[1]),
        groups=groups,
        covariate_columns=covariate_columns,
        warnings=warnings,
        can_rarefy=can_rarefy,
        collapsed_from=collapsed_from,
    )
