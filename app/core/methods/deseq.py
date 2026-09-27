"""PyDESeq2 (SPEC §10, fork 5, method 7).

The only method here backed by an external package. It is optional at import time so
Quick mode never depends on it; when it is missing the specifications that need it are
pruned with a reason the user sees.
"""
from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pandas as pd

from ..models import EFFECT_NATIVE_TYPE, FitResult


def available() -> bool:
    try:
        import pydeseq2  # noqa: F401
    except Exception:
        return False
    return True


@contextlib.contextmanager
def _one_process(*_args, **_kwargs):
    """What PyDESeq2's parallel sections run under here: joblib's sequential backend.

    PyDESeq2 wraps each fit in `parallel_backend("loky", inner_max_num_threads=1)`. Only
    loky accepts that argument, and loky needs multiprocessing, which a Vercel function
    does not have (no /dev/shm). joblib then substitutes its threading backend, which
    refuses the argument, so every PyDESeq2 fit in production raised and Full mode lost
    those specifications. MicroVerse runs PyDESeq2 on one CPU, so nothing here was ever
    parallel: sequential is what `n_cpus=1` already meant, and the numbers are the same.
    """
    from joblib import parallel_backend

    with parallel_backend("sequential"):
        yield


def _run_in_one_process() -> None:
    from pydeseq2 import default_inference

    default_inference.parallel_backend = _one_process


#: DESeq2's size-factor estimator for tables in which every feature has zeros, which is
#: every microbiome table. PyDESeq2's default is median-of-ratios; it cannot take a
#: geometric mean over a zero, so on these tables it silently switches to its iterative
#: fit, and that fit can collapse. On Nearing et al.'s edd_singh (203 samples, library
#: sizes 2,006-10,538) it returned size factors from 1e-10 to 3.7e3 and every one of
#: 1,318 ASVs came out significant; R's DESeq2 as the paper ran it found 80. poscounts
#: is what that paper, phyloseq and DESeq2's own documentation use for such data, and it
#: is closed-form, so there is nothing to converge (SPEC §24.9).
SIZE_FACTORS = "poscounts"


def run_pydeseq2(matrix, covariate_frame=None, n_cpus: int = 1) -> FitResult:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    _run_in_one_process()

    counts = np.rint(np.asarray(matrix.counts, dtype=float)).astype(int)
    n_taxa = counts.shape[0]
    if n_taxa == 0:
        return FitResult(matrix.taxa_idx, np.array([]), np.array([]), np.array([]),
                         EFFECT_NATIVE_TYPE["pydeseq2"], 0)

    # A prevalence filter can leave a sample with no reads in this matrix. It says
    # nothing about any taxon, and poscounts cannot give it a size factor, so it sits
    # out this one fit rather than making the whole fit fail.
    has_reads = counts.sum(axis=0) > 0
    groups = np.asarray(matrix.groups)
    if not has_reads.all():
        counts, groups = counts[:, has_reads], groups[has_reads]
        if covariate_frame is not None:
            covariate_frame = covariate_frame.iloc[np.flatnonzero(has_reads)]

    gene_ids = [f"t{i}" for i in range(n_taxa)]
    sample_ids = [f"s{j}" for j in range(counts.shape[1])]
    count_frame = pd.DataFrame(counts.T, index=sample_ids, columns=gene_ids)
    metadata = pd.DataFrame(
        {"group": np.where(groups == 1, "B", "A")}, index=sample_ids
    )
    factors = ["group"]
    if covariate_frame is not None and covariate_frame.shape[1] > 0:
        for column in covariate_frame.columns:
            name = "cov_" + "".join(ch if ch.isalnum() else "_" for ch in str(column))
            series = pd.to_numeric(covariate_frame[column], errors="coerce")
            if series.notna().mean() > 0.9 and series.nunique(dropna=True) > 2:
                metadata[name] = series.fillna(series.mean()).to_numpy()
            else:
                metadata[name] = covariate_frame[column].astype(str).to_numpy()
            factors.append(name)

    design = "~" + " + ".join(factors)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            dds = DeseqDataSet(counts=count_frame, metadata=metadata, design=design,
                               size_factors_fit_type=SIZE_FACTORS,
                               refit_cooks=True, quiet=True, n_cpus=n_cpus)
        except TypeError:  # older API took design_factors
            dds = DeseqDataSet(counts=count_frame, metadata=metadata,
                               design_factors=factors, refit_cooks=True, quiet=True)
            dds.size_factors_fit_type = SIZE_FACTORS
        dds.deseq2()
        stats_result = DeseqStats(dds, contrast=["group", "B", "A"], quiet=True)
        stats_result.summary()

    frame = stats_result.results_df.reindex(gene_ids)
    p_values = frame["pvalue"].to_numpy(dtype=float)
    effect = frame["log2FoldChange"].to_numpy(dtype=float)
    p_values = np.where(np.isfinite(p_values), np.clip(p_values, 0.0, 1.0), 1.0)
    effect = np.where(np.isfinite(effect), effect, 0.0)

    return FitResult(
        taxa=matrix.taxa_idx,
        p_raw=p_values,
        effect_native=effect,
        effect_harmonized=np.zeros(n_taxa),
        effect_native_type=EFFECT_NATIVE_TYPE["pydeseq2"],
        n_taxa_input=n_taxa,
    )
