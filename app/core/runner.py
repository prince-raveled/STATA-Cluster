"""Run orchestration — SPEC §12-§15.

One run walks the §9 pipeline in cache order (rarefy -> collapse -> filter -> transform),
fits every method the grid asks of each matrix, and stores the result in long form:
one row per (specification, taxon). FDR is post-hoc, so a fit is shared by its three
FDR settings and computed once.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import fdr as fdr_module
from .effects import harmonized_effect, run_pseudocount
from .grid import capabilities_for, enumerate_grid
from .methods import available_methods, run_method
from .models import GridReport, Specification
from .preprocess import MatrixBuilder
from .validation import Dataset


@dataclass
class RunResult:
    """Everything the results pages, exports and attribution need."""

    mode: str
    dataset_summary: dict
    grid_report: GridReport
    specs: list
    taxa_names: list
    taxa_display: list
    taxa_rank: list
    long: pd.DataFrame  # spec_id, taxon, p_raw, p_adj, significant, effect_h, effect_n
    specs_frame: pd.DataFrame
    declared_spec_id: int = -1
    runtime_seconds: float = 0.0
    method_status: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    group_labels: tuple = ("A", "B")
    #: Why specifications produced no result, so the page can say so rather than guess:
    #: {"unusable": {...}} for matrices too small to test, {method: {...}} for a method
    #: that raised. Each entry counts specifications and matrices and keeps one error.
    skipped: dict = field(default_factory=dict)

    @property
    def n_specs(self) -> int:
        return len(self.specs)

    def specs_by_rank(self) -> dict:
        counts: dict = defaultdict(int)
        for spec in self.specs:
            counts[spec.rank] += 1
        return dict(counts)


def _covariate_frame(dataset: Dataset, columns, sample_idx):
    if not columns:
        return None
    return dataset.metadata.iloc[sample_idx][list(columns)]


def _method_needs(specs):
    """fit_key -> (matrix_key, method, covariates, [spec indices])."""
    fits: dict = {}
    for i, spec in enumerate(specs):
        key = spec.fit_key
        if key not in fits:
            fits[key] = {
                "matrix_key": spec.matrix_key,
                "method": spec.method,
                "covariates": spec.covariates,
                "spec_ids": [],
            }
        fits[key]["spec_ids"].append(i)
    return fits


def run_multiverse(
    dataset: Dataset,
    mode: str = "quick",
    fixed_covariates=(),
    covariate_columns=(),
    declared: Specification = None,
    progress=None,
    aldex_instances: int = 64,
) -> RunResult:
    """Execute one multiverse. `progress(fraction, message)` is called as it goes."""
    started = time.perf_counter()

    def report(fraction: float, message: str):
        if progress is not None:
            progress(min(max(fraction, 0.0), 1.0), message)

    report(0.02, "Enumerating the specification grid")
    builder = MatrixBuilder(dataset)
    # One pseudocount for the whole run, so the harmonised effect really is a single
    # estimator applied identically to every specification (SPEC §14).
    pseudocount = run_pseudocount(builder.base_counts)
    capabilities = capabilities_for(dataset)
    status = available_methods()

    specs, grid_report = enumerate_grid(
        builder,
        mode=mode,
        capabilities=capabilities,
        fixed_covariates=fixed_covariates,
        covariate_columns=covariate_columns,
        declared=declared,
    )
    # Drop specifications whose method cannot run in this environment.
    unavailable = {name for name, why in status.items() if why}
    if unavailable:
        kept = [s for s in specs if s.method not in unavailable]
        if len(kept) != len(specs):
            for name in sorted(unavailable):
                grid_report.pruned_reasons[status[name]] = len(specs) - len(kept)
            grid_report.n_valid = len(kept)
            grid_report.n_pruned = grid_report.n_enumerated - len(kept)
            specs = kept

    if not specs:
        raise ValueError(
            "No valid specifications remain for this dataset. Check the pruning reasons: "
            + "; ".join(grid_report.pruned_reasons)
        )

    declared_id = specs.index(declared) if declared in set(specs) else -1

    # Global taxon vocabulary, one namespace per rank (§15 denominators depend on it).
    taxa_names: list = []
    taxa_rank: list = []
    offsets: dict = {}
    for rank in builder.ranks:
        offsets[rank] = len(taxa_names)
        names = builder.taxa_names(rank)
        taxa_names.extend(names)
        taxa_rank.extend([rank] * len(names))
    taxa_display = [dataset.table.display_name(n) for n in taxa_names]

    fits = _method_needs(specs)
    grid_report.n_fits = len(fits)
    fits_by_matrix: dict = defaultdict(list)
    for key, info in fits.items():
        fits_by_matrix[info["matrix_key"]].append((key, info))

    matrix_keys = sorted(fits_by_matrix, key=lambda k: (str(k[0]), k[1] or 0, k[2], k[3], k[4]))
    total = len(matrix_keys)

    spec_ids_out: list = []
    taxa_out: list = []
    p_raw_out: list = []
    p_adj_out: list = []
    sig_out: list = []
    eff_h_out: list = []
    eff_n_out: list = []
    failures: list = []
    skipped: dict = {}

    def skip(reason: str, n_specs: int, detail: str = "") -> None:
        entry = skipped.setdefault(reason, {"specs": 0, "matrices": 0, "detail": detail})
        entry["specs"] += n_specs
        entry["matrices"] += 1

    for position, matrix_key in enumerate(matrix_keys):
        rarefaction, seed, rank, prevalence, transform = matrix_key
        matrix = builder.build(rarefaction, seed, rank, prevalence, transform)
        if not matrix.usable:
            failures.append(
                f"Matrix {matrix_key} left {matrix.n_taxa} taxa and "
                f"{matrix.n_samples} samples; skipped."
            )
            skip("unusable", sum(len(info["spec_ids"]) for _, info in fits_by_matrix[matrix_key]))
            continue

        global_idx = matrix.taxa_idx + offsets[rank]
        effect_h = harmonized_effect(matrix.rel, matrix.groups, pseudocount).astype(np.float32)

        for _, info in fits_by_matrix[matrix_key]:
            method = info["method"]
            covariates = info["covariates"]
            frame = _covariate_frame(dataset, covariates, matrix.sample_idx)
            try:
                fit = run_method(method, matrix, frame, aldex_instances=aldex_instances)
            except Exception as exc:  # a single method failing must not kill the run
                failures.append(f"{method} failed on matrix {matrix_key}: {exc}")
                skip(method, len(info["spec_ids"]), f"{type(exc).__name__}: {exc}"[:300])
                continue

            adjusted_cache: dict = {}
            for spec_id in info["spec_ids"]:
                spec = specs[spec_id]
                cache_key = (spec.fdr_method,)
                if cache_key not in adjusted_cache:
                    adjusted_cache[cache_key] = fdr_module.adjust(fit.p_raw, spec.fdr_method)
                adjusted = adjusted_cache[cache_key]
                n = fit.p_raw.size
                spec_ids_out.append(np.full(n, spec_id, dtype=np.int32))
                taxa_out.append(global_idx.astype(np.int32))
                p_raw_out.append(fit.p_raw.astype(np.float32))
                p_adj_out.append(adjusted.astype(np.float32))
                sig_out.append(adjusted <= spec.fdr_threshold)
                eff_h_out.append(effect_h)
                eff_n_out.append(fit.effect_native.astype(np.float32))

        if position % max(1, total // 40) == 0:
            report(
                0.05 + 0.85 * position / max(1, total),
                f"Fitting specifications: matrix {position + 1} of {total}",
            )

    if not spec_ids_out:
        raise ValueError(
            "Every specification failed. The most common reason is that the prevalence "
            "filter removed all taxa — check the sparsity of your table."
        )

    report(0.92, "Assembling results")
    long = pd.DataFrame(
        {
            "spec_id": np.concatenate(spec_ids_out),
            "taxon": np.concatenate(taxa_out),
            "p_raw": np.concatenate(p_raw_out),
            "p_adj": np.concatenate(p_adj_out),
            "significant": np.concatenate(sig_out),
            "effect_h": np.concatenate(eff_h_out),
            "effect_n": np.concatenate(eff_n_out),
        }
    )

    specs_frame = pd.DataFrame([s.as_row() for s in specs])
    specs_frame.insert(0, "spec_id", np.arange(len(specs), dtype=np.int32))

    runtime = time.perf_counter() - started
    report(0.95, "Computing robustness")

    return RunResult(
        mode=mode,
        dataset_summary=dataset.summary(),
        grid_report=grid_report,
        specs=specs,
        taxa_names=taxa_names,
        taxa_display=taxa_display,
        taxa_rank=taxa_rank,
        long=long,
        specs_frame=specs_frame,
        declared_spec_id=declared_id,
        runtime_seconds=runtime,
        method_status=status,
        failures=failures,
        group_labels=dataset.group_labels,
        skipped=skipped,
    )
