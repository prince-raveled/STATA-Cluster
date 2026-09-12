"""Grid enumeration and mode selection — SPEC §7, §10, §12.

Three modes, never one crossed grid. Quick is the default and the mode with the best
evidence behind it (Pelto et al. 2025: elementary methods are the most replicable).
"""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np

from .models import (
    ELEMENTARY_METHODS,
    SOPHISTICATED_METHODS,
    GridReport,
    Specification,
)
from .preprocess import PREVALENCE_LEVELS, TRANSFORMS, MatrixBuilder
from .validity import Capabilities, invalid_reason

#: Fork 6 — applied post-hoc to stored p-values, so all three cost essentially nothing.
FDR_SETTINGS = (("bh", 0.05), ("bh", 0.10), ("by", 0.05))

#: Mode C reference sub-grid (§12): 3 rarefaction x 2 rank x 2 transform = 12 matrices.
COVARIATE_REFERENCE_RAREFACTION = ("none", "min", "10000")
COVARIATE_REFERENCE_TRANSFORMS = ("raw", "clr")
COVARIATE_REFERENCE_PREVALENCE = 0.10
MAX_COVARIATE_SUBSETS = 64
MAX_COVARIATES_PER_MODEL = 20  # Tierney's limit
FULL_MODE_MATRIX_FRACTION = 0.10


def capabilities_for(dataset) -> Capabilities:
    return Capabilities(
        integer_counts=dataset.table.is_integer,
        can_rarefy=dataset.can_rarefy,
        can_collapse_to_genus=dataset.table.can_collapse_to_genus,
    )


def covariate_subsets(columns, seed: int = 1):
    """Fork 7 (§10): all 2^k subsets when k <= 6, else 64 stratified random subsets."""
    columns = list(columns)[:MAX_COVARIATES_PER_MODEL]
    k = len(columns)
    if k == 0:
        return [()]
    if k <= 6:
        subsets = []
        for size in range(k + 1):
            subsets.extend(tuple(c) for c in itertools.combinations(columns, size))
        return subsets

    rng = np.random.default_rng(seed)
    chosen = {(), tuple(columns)}
    per_size = max(1, (MAX_COVARIATE_SUBSETS - 2) // max(1, k - 1))
    for size in range(1, k):
        for _ in range(per_size * 4):
            if len(chosen) >= MAX_COVARIATE_SUBSETS:
                break
            pick = tuple(sorted(rng.choice(columns, size=size, replace=False)))
            chosen.add(pick)
            if sum(1 for c in chosen if len(c) == size) >= per_size:
                break
        if len(chosen) >= MAX_COVARIATE_SUBSETS:
            break
    return sorted(chosen, key=lambda c: (len(c), c))[:MAX_COVARIATE_SUBSETS]


def _matrix_keys(rarefactions, ranks, prevalences, transforms):
    return list(itertools.product(rarefactions, ranks, prevalences, transforms))


def _stratified_matrix_sample(keys, fraction: float, n_total: int = 0, seed: int = 7):
    """Sample matrix keys for a sophisticated method, stratified by rarefaction (§12).

    §12 budgets ~10% of the *whole* matrix grid per method. Taking 10% of each method's
    own compatible pool would be far too few: ANCOM-BC and PyDESeq2 are restricted by
    §11 to raw counts without rarefaction, so their pool is a handful of matrices and
    10% of it is one. The target is therefore 10% of the full grid, capped by what the
    method can actually accept — which means small pools are taken whole.
    """
    if not keys:
        return []
    rng = np.random.default_rng(seed)
    by_stratum: dict = {}
    for key in keys:
        by_stratum.setdefault(key[0][0], []).append(key)

    target = max(1, int(round(max(n_total, len(keys)) * fraction)))
    target = min(target, len(keys))
    per_stratum = max(1, -(-target // max(1, len(by_stratum))))  # ceil division
    picked = []
    for stratum in sorted(by_stratum):
        pool = by_stratum[stratum]
        take = min(len(pool), per_stratum)
        idx = rng.choice(len(pool), size=take, replace=False)
        picked.extend(pool[i] for i in sorted(idx))
    return picked


def enumerate_grid(
    builder: MatrixBuilder,
    mode: str = "quick",
    capabilities: Capabilities = None,
    fixed_covariates=(),
    covariate_columns=(),
    declared: Specification = None,
):
    """Return (valid specifications, GridReport) for the requested mode."""
    capabilities = capabilities or Capabilities()
    depths, dropped_notes = builder.available_depths()
    ranks = list(builder.ranks)
    prevalences = list(PREVALENCE_LEVELS)
    transforms = list(TRANSFORMS)

    notes = list(dropped_notes)
    if len(ranks) == 1:
        notes.append(
            "Input is already at genus level (or carries no taxonomy), so the "
            "taxonomic-rank choice has one level and the grid shrinks accordingly."
        )

    enumerated: list = []

    if mode in ("quick", "full"):
        matrix_keys = _matrix_keys(depths, ranks, prevalences, transforms)
        for (rare, seed), rank, prev, transform in matrix_keys:
            for method in ELEMENTARY_METHODS:
                for fdr_method, threshold in FDR_SETTINGS:
                    enumerated.append(
                        Specification(
                            rarefaction=rare,
                            rare_seed=seed,
                            rank=rank,
                            prev_filter=prev,
                            transform=transform,
                            method=method,
                            fdr_method=fdr_method,
                            fdr_threshold=threshold,
                            covariates=tuple(fixed_covariates),
                        )
                    )

        if mode == "full":
            for method in SOPHISTICATED_METHODS:
                pool = [
                    key
                    for key in matrix_keys
                    if not invalid_reason(
                        Specification(
                            key[0][0], key[0][1], key[1], key[2], key[3], method, "bh", 0.05,
                            tuple(fixed_covariates),
                        ),
                        capabilities,
                    )
                ]
                sample = _stratified_matrix_sample(
                    pool, FULL_MODE_MATRIX_FRACTION, n_total=len(matrix_keys)
                )
                if pool:
                    notes.append(
                        f"{method}: {len(sample)} of {len(pool)} compatible matrices sampled "
                        f"({len(sample) / len(pool):.0%}, stratified by rarefaction depth)."
                    )
                for (rare, seed), rank, prev, transform in sample:
                    for fdr_method, threshold in FDR_SETTINGS:
                        enumerated.append(
                            Specification(
                                rarefaction=rare,
                                rare_seed=seed,
                                rank=rank,
                                prev_filter=prev,
                                transform=transform,
                                method=method,
                                fdr_method=fdr_method,
                                fdr_threshold=threshold,
                                covariates=tuple(fixed_covariates),
                            )
                        )

    elif mode == "covariate":
        reference_depths = [
            (name, seed)
            for name, seed in depths
            if name in COVARIATE_REFERENCE_RAREFACTION and (seed is None or seed == 1)
        ]
        if not reference_depths:
            reference_depths = depths[:1]
        subsets = covariate_subsets(covariate_columns)
        notes.append(
            f"Covariate mode: {len(subsets)} covariate subsets over a reference sub-grid of "
            f"{len(reference_depths)} rarefaction x {len(ranks)} rank x "
            f"{len(COVARIATE_REFERENCE_TRANSFORMS)} transform matrices."
        )
        for (rare, seed), rank, transform in itertools.product(
            reference_depths, ranks, COVARIATE_REFERENCE_TRANSFORMS
        ):
            for method in ELEMENTARY_METHODS:
                for subset in subsets:
                    for fdr_method, threshold in FDR_SETTINGS:
                        enumerated.append(
                            Specification(
                                rarefaction=rare,
                                rare_seed=seed,
                                rank=rank,
                                prev_filter=COVARIATE_REFERENCE_PREVALENCE,
                                transform=transform,
                                method=method,
                                fdr_method=fdr_method,
                                fdr_threshold=threshold,
                                covariates=tuple(subset),
                            )
                        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    valid: list = []
    pruned: Counter = Counter()
    for spec in enumerated:
        reason = invalid_reason(spec, capabilities)
        if reason:
            pruned[reason] += 1
        else:
            valid.append(spec)

    if (declared is not None and not invalid_reason(declared, capabilities)
            and declared not in set(valid)):
        valid.append(declared)
        notes.append(
            "Your declared pipeline was not on the grid as enumerated and was added so it "
            "can be located in the distribution."
        )

    report = GridReport(
        mode=mode,
        n_enumerated=len(enumerated),
        n_valid=len(valid),
        n_pruned=len(enumerated) - len(valid),
        n_matrices=len({s.matrix_key for s in valid}),
        n_fits=len({s.fit_key for s in valid}),
        pruned_reasons=dict(pruned.most_common()),
        notes=notes,
    )
    return valid, report
