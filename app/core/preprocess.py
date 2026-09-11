"""Preprocessing — SPEC §9, in the one correct order.

    raw counts -> 1. RAREFY -> 2. COLLAPSE -> 3. PREVALENCE FILTER -> 4. TRANSFORM

Rarefying after collapsing, or filtering before rarefying, gives different answers.
`preprocess()` is the only place the order is expressed, and `MatrixBuilder` caches
each layer so 416 matrices cost 13 rarefactions and 26 collapses.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .validation import Dataset

RAREFACTION_DEPTHS = {"min": None, "1000": 1000, "5000": 5000, "10000": 10000}
RARE_SEEDS = (1, 2, 3)
PREVALENCE_LEVELS = (0.0, 0.05, 0.10, 0.20)
TRANSFORMS = ("tss", "clr", "raw", "tmm")


@dataclass
class PreparedMatrix:
    """One preprocessed matrix plus everything a DA method needs from it."""

    key: tuple
    values: np.ndarray  # taxa x samples, on the transform's scale
    counts: np.ndarray  # taxa x samples, integer-ish counts after filtering
    rel: np.ndarray  # taxa x samples, closed to 1 — the harmonised-effect scale
    taxa_idx: np.ndarray  # int32 indices into the run's global taxon list
    groups: np.ndarray  # int8, aligned to columns
    sample_idx: np.ndarray  # int32 indices into the dataset's samples
    transform: str
    n_samples_dropped: int = 0
    notes: list = field(default_factory=list)

    @property
    def n_taxa(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[1])

    @property
    def group_sizes(self) -> tuple:
        return int((self.groups == 0).sum()), int((self.groups == 1).sum())

    @property
    def usable(self) -> bool:
        n_a, n_b = self.group_sizes
        return self.n_taxa >= 2 and n_a >= 3 and n_b >= 3


# --------------------------------------------------------------------------
# 1. Rarefy
# --------------------------------------------------------------------------
def rarefy(counts: np.ndarray, depth: int, seed: int):
    """Subsample each library to `depth` reads without replacement.

    Returns (rarefied counts, kept-sample mask). Samples below `depth` are dropped,
    which is what rarefying actually does — the alternative silently keeps a sample
    at a different depth than every other.
    """
    integer = np.rint(counts).astype(np.int64)
    libraries = integer.sum(axis=0)
    keep = libraries >= depth
    if not keep.any():
        return integer[:, :0], keep
    rng = np.random.default_rng(seed)
    kept = integer[:, keep]
    out = np.empty_like(kept)
    for j in range(kept.shape[1]):
        out[:, j] = rng.multivariate_hypergeometric(kept[:, j], depth)
    return out, keep


# --------------------------------------------------------------------------
# 3. Prevalence filter
# --------------------------------------------------------------------------
def prevalence_filter(counts: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean keep-mask for taxa present in at least `threshold` of samples."""
    if counts.shape[1] == 0:
        return np.zeros(counts.shape[0], dtype=bool)
    prevalence = (counts > 0).mean(axis=1)
    # A taxon seen in no sample cannot be tested under any threshold, including 0%.
    return (prevalence >= threshold) & (prevalence > 0)


# --------------------------------------------------------------------------
# 4. Transform
# --------------------------------------------------------------------------
def multiplicative_replacement(closed: np.ndarray, delta: float = 0.0) -> np.ndarray:
    """Replace zeros in a composition with a small value, rescaling the rest.

    This is the `skbio.stats.composition.multi_replace` algorithm (SPEC §10 fork 3):
    zeros become `delta`, non-zeros shrink by (1 - n_zeros * delta) so each column
    still sums to 1. Default delta is (1 / n_taxa)^2, scikit-bio's default.

    A fixed pseudocount would instead add a constant to counts, which changes the
    ratios between observed taxa — the thing CLR exists to preserve.
    """
    out = np.array(closed, dtype=float, copy=True)
    n_taxa = out.shape[0]
    if n_taxa == 0:
        return out
    if delta <= 0:
        delta = (1.0 / n_taxa) ** 2
    zeros = out <= 0
    n_zeros = zeros.sum(axis=0)
    scale = 1.0 - n_zeros * delta
    scale = np.where(scale <= 0, 1e-12, scale)
    out = out * scale[None, :]
    out[zeros] = delta
    totals = out.sum(axis=0)
    totals[totals <= 0] = 1.0
    return out / totals[None, :]


def closure(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=0)
    totals = np.where(totals <= 0, 1.0, totals)
    return matrix / totals[None, :]


def clr(counts: np.ndarray) -> np.ndarray:
    """Centred log-ratio, with multiplicative zero replacement."""
    values = np.asarray(counts, dtype=float)
    if values.shape[0] == 0:
        # No taxa left to take a geometric mean over — see tmm_factors for why this
        # happens. Returning the empty matrix keeps the caller on the normal path.
        return values
    replaced = multiplicative_replacement(closure(values))
    logged = np.log(replaced)
    return logged - logged.mean(axis=0, keepdims=True)


def tmm_factors(counts: np.ndarray, log_ratio_trim: float = 0.3, sum_trim: float = 0.05,
                a_cutoff: float = -1e10):
    """edgeR's trimmed mean of M-values normalisation factors.

    This follows edgeR's `.calcFactorTMM` (Robinson & Oshlack 2010) step for step,
    including the detail that matters most on sparse data: **the trim is by rank, not
    by quantile value**. With microbiome counts, M-values are full of ties, and
    trimming on the quantile *value* removes every tied observation at the boundary
    instead of the intended 30%. Cross-validation against `conorm` — which trims by
    quantile — diverged by up to 31% on sparse tables for exactly this reason; see
    `tests/reference/compare_tmm.py`.

    Reference sample is the one whose upper-quartile count/library-size is closest to
    the mean, as in `calcNormFactors`.
    """
    values = np.asarray(counts, dtype=float)
    if values.shape[0] == 0:
        # A prevalence filter can empty the table at some grid points. edgeR's answer
        # when there is nothing to estimate from is a factor of exactly 1, which is
        # what every other dead end in this function returns; np.quantile below would
        # raise on the empty axis instead.
        return np.ones(values.shape[1], dtype=float)
    libraries = values.sum(axis=0)
    libraries = np.where(libraries <= 0, 1.0, libraries)
    upper = np.quantile(values, 0.75, axis=0) / libraries
    reference = int(np.argmin(np.abs(upper - upper.mean())))

    ref_counts = values[:, reference]
    ref_lib = libraries[reference]
    factors = np.ones(values.shape[1], dtype=float)

    for j in range(values.shape[1]):
        obs = values[:, j]
        obs_lib = libraries[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            o = obs / obs_lib
            r = ref_counts / ref_lib
            log_ratio = np.log2(o / r)                       # M
            absolute = 0.5 * (np.log2(o) + np.log2(r))       # A
            # Estimated asymptotic variance (edgeR eq. 4).
            variance = (obs_lib - obs) / (obs_lib * obs) + (
                ref_lib - ref_counts
            ) / (ref_lib * ref_counts)

        finite = (np.isfinite(log_ratio) & np.isfinite(absolute) & np.isfinite(variance)
                  & (absolute > a_cutoff) & (variance > 0))
        log_ratio, absolute, variance = log_ratio[finite], absolute[finite], variance[finite]
        n = log_ratio.size
        if n == 0 or np.max(np.abs(log_ratio)) < 1e-6:
            continue  # edgeR returns a factor of exactly 1 here

        # Rank-based trim, matching edgeR: ranks are 1-based with ties averaged.
        low_m = np.floor(n * log_ratio_trim) + 1
        high_m = n + 1 - low_m
        low_a = np.floor(n * sum_trim) + 1
        high_a = n + 1 - low_a
        rank_m = stats.rankdata(log_ratio)
        rank_a = stats.rankdata(absolute)
        keep = ((rank_m >= low_m) & (rank_m <= high_m)
                & (rank_a >= low_a) & (rank_a <= high_a))
        if not keep.any():
            continue

        weights = 1.0 / variance[keep]
        total = weights.sum()
        if total <= 0 or not np.isfinite(total):
            continue
        factors[j] = 2.0 ** (np.sum(log_ratio[keep] * weights) / total)

    # edgeR centres factors so their geometric mean is 1.
    factors = np.where(np.isfinite(factors) & (factors > 0), factors, 1.0)
    return factors / np.exp(np.mean(np.log(factors)))


def apply_transform(counts: np.ndarray, kind: str) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    if kind == "raw":
        return values
    if kind == "tss":
        return closure(values)
    if kind == "clr":
        return clr(values)
    if kind == "tmm":
        factors = tmm_factors(values)
        libraries = values.sum(axis=0)
        libraries = np.where(libraries <= 0, 1.0, libraries)
        effective = factors * libraries
        target = np.exp(np.mean(np.log(np.where(effective > 0, effective, 1.0))))
        return values / effective[None, :] * target
    raise ValueError(f"Unknown transform: {kind}")


def relative_scale(transformed: np.ndarray, kind: str) -> np.ndarray:
    """Map a transformed matrix back to proportions for the harmonised effect (§14).

    SPEC §14 forces TSS internally. For raw / TSS / TMM that is a direct closure and
    all three give identical proportions (a per-sample scalar cancels). CLR has no
    column sum to divide by, so it is exponentiated first — exp(CLR) is proportional
    to the zero-replaced composition, which is exactly what the test saw.
    """
    if kind == "clr":
        return closure(np.exp(np.asarray(transformed, dtype=float)))
    return closure(np.clip(np.asarray(transformed, dtype=float), 0.0, None))


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------
class MatrixBuilder:
    """Builds and caches preprocessed matrices in the §9 order.

    Layer 1 (rarefy) is cached per (depth, seed); layer 2 (collapse) per
    (rarefaction, rank); layer 3 (filter) per (rarefaction, rank, prevalence).
    Only layer 4 is recomputed per transform, so the full 416-matrix grid costs
    13 subsamplings rather than 416.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.base_counts = dataset.table.counts.to_numpy(dtype=float)
        self.base_taxa = dataset.table.taxa
        self.groups = dataset.groups
        self.max_library = float(self.base_counts.sum(axis=0).max())
        self.min_library = float(self.base_counts.sum(axis=0).min())

        # Global taxon vocabulary: input-rank taxa plus their genus labels.
        self._genus_of = None
        self._genus_taxa: list = []
        if dataset.table.can_collapse_to_genus:
            labels = [dataset.table.genus_label(t) for t in self.base_taxa]
            unique = list(dict.fromkeys(labels))
            index = {name: i for i, name in enumerate(unique)}
            self._genus_of = np.array([index[label] for label in labels], dtype=np.int32)
            self._genus_taxa = unique

        self.taxa_by_rank = {"input": list(self.base_taxa), "genus": self._genus_taxa}
        self._rarefy_cache: dict = {}
        self._collapse_cache: dict = {}
        self._filter_cache: dict = {}

    # -- vocabulary ------------------------------------------------------
    @property
    def ranks(self) -> tuple:
        return ("input", "genus") if self._genus_of is not None else ("input",)

    def taxa_names(self, rank: str) -> list:
        return self.taxa_by_rank["genus"] if rank == "genus" else self.taxa_by_rank["input"]

    def available_depths(self):
        """Fork 1 levels this dataset supports, with dropped ones reported (§10)."""
        levels = [("none", None)]
        dropped = []
        if not self.dataset.can_rarefy:
            return levels, ["Non-integer values: every subsampling level was dropped."]
        for name, depth in RAREFACTION_DEPTHS.items():
            target = int(self.min_library) if depth is None else depth
            if target > self.max_library:
                dropped.append(
                    f"Depth {name} ({target:,}) exceeds the largest library "
                    f"({int(self.max_library):,}) and was dropped."
                )
                continue
            if target < 100:
                dropped.append(f"Depth {name} ({target:,}) is below 100 reads and was dropped.")
                continue
            for seed in RARE_SEEDS:
                levels.append((name, seed))
        return levels, dropped

    # -- layer 1 ---------------------------------------------------------
    def _rarefied(self, rarefaction: str, seed):
        key = (rarefaction, seed)
        if key in self._rarefy_cache:
            return self._rarefy_cache[key]
        if rarefaction == "none":
            result = (self.base_counts, np.ones(self.base_counts.shape[1], dtype=bool))
        else:
            depth = RAREFACTION_DEPTHS[rarefaction]
            target = int(self.min_library) if depth is None else int(depth)
            counts, keep = rarefy(self.base_counts, target, int(seed))
            result = (counts.astype(float), keep)
        self._rarefy_cache[key] = result
        return result

    # -- layer 2 ---------------------------------------------------------
    def _collapsed(self, rarefaction: str, seed, rank: str):
        key = (rarefaction, seed, rank)
        if key in self._collapse_cache:
            return self._collapse_cache[key]
        counts, keep = self._rarefied(rarefaction, seed)
        if rank == "genus" and self._genus_of is not None:
            n_groups = len(self._genus_taxa)
            collapsed = np.zeros((n_groups, counts.shape[1]), dtype=float)
            np.add.at(collapsed, self._genus_of, counts)
            result = (collapsed, keep)
        else:
            result = (counts, keep)
        self._collapse_cache[key] = result
        return result

    # -- layer 3 ---------------------------------------------------------
    def _filtered(self, rarefaction: str, seed, rank: str, prevalence: float):
        key = (rarefaction, seed, rank, prevalence)
        if key in self._filter_cache:
            return self._filter_cache[key]
        counts, keep = self._collapsed(rarefaction, seed, rank)
        mask = prevalence_filter(counts, prevalence)
        result = (counts[mask], np.flatnonzero(mask).astype(np.int32), keep)
        self._filter_cache[key] = result
        return result

    # -- layer 4 + assembly ----------------------------------------------
    def build(self, rarefaction: str, seed, rank: str, prevalence: float, transform: str):
        """Run the §9 pipeline for one matrix key."""
        counts, taxa_idx, sample_keep = self._filtered(rarefaction, seed, rank, prevalence)
        sample_idx = np.flatnonzero(sample_keep).astype(np.int32)
        groups = self.groups[sample_keep]
        values = apply_transform(counts, transform)
        return PreparedMatrix(
            key=(rarefaction, seed, rank, prevalence, transform),
            values=values,
            counts=counts,
            rel=relative_scale(values, transform),
            taxa_idx=taxa_idx,
            groups=groups,
            sample_idx=sample_idx,
            transform=transform,
            n_samples_dropped=int((~sample_keep).sum()),
        )

    def clear_layer_caches(self):
        """Drop everything but the rarefaction layer; used between prevalence blocks."""
        self._collapse_cache.clear()
        self._filter_cache.clear()


def preprocess(
    dataset: Dataset,
    rarefaction: str,
    rare_seed,
    rank: str,
    prev_filter: float,
    transform: str,
) -> PreparedMatrix:
    """Single-shot preprocessing in the §9 order. Prefer `MatrixBuilder` in a run."""
    return MatrixBuilder(dataset).build(rarefaction, rare_seed, rank, prev_filter, transform)


def matrix_to_frame(matrix: PreparedMatrix, taxa_names: list) -> pd.DataFrame:
    """Debug/export helper: a labelled DataFrame for one prepared matrix."""
    return pd.DataFrame(
        matrix.values,
        index=[taxa_names[i] for i in matrix.taxa_idx],
        columns=[f"s{j}" for j in matrix.sample_idx],
    )
