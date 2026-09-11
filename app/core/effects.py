"""Effect-size harmonisation — SPEC §14.

Seven methods return seven incompatible statistics; a U statistic, a log odds ratio
and a DESeq2 log2 fold change cannot share a y-axis. So significance comes from the
method and **effect size comes from here** — one estimator, computed identically on
the same preprocessed matrix for every specification.
"""
from __future__ import annotations

import numpy as np


def run_pseudocount(counts: np.ndarray) -> float:
    """Half the smallest non-zero proportion of the *input* table (SPEC §14).

    Fixed once per run, deliberately. Recomputing it per matrix would make the
    pseudocount a function of rarefaction depth and of the CLR zero replacement, so
    the "single estimator, computed identically for every specification" of §14 would
    silently become seven different estimators — and rarefaction would then dominate
    the §17 effect-size attribution as an artefact rather than a finding.
    """
    values = np.asarray(counts, dtype=float)
    totals = values.sum(axis=0)
    totals = np.where(totals <= 0, 1.0, totals)
    proportions = values / totals[None, :]
    positive = proportions[proportions > 0]
    return float(positive.min() / 2.0) if positive.size else 1e-12


def harmonized_effect(rel: np.ndarray, groups: np.ndarray, eps: float = None) -> np.ndarray:
    """log2 fold change of mean relative abundance, group B vs group A.

    `rel` is the matrix on the proportion scale (see `preprocess.relative_scale`),
    taxa x samples. `groups` is 0 for A, 1 for B. Returns one value per taxon.

    `eps` should come from `run_pseudocount` so it is constant across the whole
    multiverse; it falls back to the matrix-local value when called standalone.
    """
    values = np.asarray(rel, dtype=float)
    if values.size == 0:
        return np.zeros(values.shape[0], dtype=float)
    if eps is None:
        positive = values[values > 0]
        eps = (positive.min() / 2.0) if positive.size else 1e-12
    mean_a = values[:, groups == 0].mean(axis=1)
    mean_b = values[:, groups == 1].mean(axis=1)
    return np.log2((mean_b + eps) / (mean_a + eps))


def prevalence_by_group(counts: np.ndarray, groups: np.ndarray):
    """Fraction of samples in which each taxon is observed, per group."""
    presence = counts > 0
    return presence[:, groups == 0].mean(axis=1), presence[:, groups == 1].mean(axis=1)
