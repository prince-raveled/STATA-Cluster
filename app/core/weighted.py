"""Weighted robustness metrics — v3 plan §26.2, the part that reads a run's results.

`weights.py` says how much each specification counts; this applies that to a run.
For every taxon the weights are renormalised over the specifications that actually
tested it — the conditional probability of a specification given that the taxon was
tested, which is the weighted form of SPEC §15's rule that denominators are
specifications tested, never specifications run.

    frac_significant_w = sum_s w_s 1[significant] / sum_s w_s     (s tested the taxon)
    sign_consistency_w = max(sum w_s 1[effect > 0], sum w_s 1[effect < 0]) / sum_s w_s

Tier thresholds are unchanged (SPEC §16.2); only the votes they count are weighted.
Under `uniform` every weight is exactly 1, so the sums are integer counts and the
results are v2's to the last bit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .effects import PSEUDOCOUNT_MULTIPLIERS
from .weights import SCHEMES, all_scheme_weights, composition


def _unit_weights(scheme: str, weights: np.ndarray) -> np.ndarray:
    """Uniform is counted with weight 1 each so its arithmetic is exactly v2's."""
    return np.ones_like(weights) if scheme == "uniform" else weights


def per_taxon_metrics(long: pd.DataFrame, spec_weights: np.ndarray, n_taxa: int) -> dict:
    """Weighted share significant, sign consistency, share nominal and n_eff per taxon.

    Arrays of length n_taxa, NaN for a taxon no specification tested.
    """
    taxon = long["taxon"].to_numpy()
    w = spec_weights[long["spec_id"].to_numpy()]
    effect = long["effect_h"].to_numpy(dtype=float)
    total = np.bincount(taxon, weights=w, minlength=n_taxa)
    with np.errstate(invalid="ignore", divide="ignore"):
        significant = np.bincount(
            taxon, weights=w * long["significant"].to_numpy(), minlength=n_taxa) / total
        nominal = np.bincount(
            taxon, weights=w * (long["p_raw"].to_numpy() < 0.05), minlength=n_taxa) / total
        positive = np.bincount(taxon, weights=w * (effect > 0), minlength=n_taxa)
        negative = np.bincount(taxon, weights=w * (effect < 0), minlength=n_taxa)
        sign = np.maximum(positive, negative) / total
        n_eff = total ** 2 / np.bincount(taxon, weights=w ** 2, minlength=n_taxa)
    missing = total <= 0
    for array in (significant, nominal, sign, n_eff):
        array[missing] = np.nan
    return {"frac_significant": significant, "frac_nominal": nominal,
            "sign_consistency": sign, "n_eff": n_eff}


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, groups: np.ndarray,
                       n_groups: int, quantiles=(0.25, 0.5, 0.75)) -> np.ndarray:
    """Weighted quantiles of `values` within each group, shape (n_groups, len(q)).

    Each observation sits at the midpoint of its cumulative weight and the quantile is
    interpolated between those points, so equal weights give the familiar mid-rank
    definition and no single heavy specification is skipped over.
    """
    out = np.full((n_groups, len(quantiles)), np.nan)
    if values.size == 0:
        return out
    order = np.lexsort((values, groups))
    v, w, g = values[order], weights[order], groups[order]
    bounds = np.flatnonzero(np.diff(g)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [g.size]])
    for start, end in zip(starts, ends, strict=True):
        ww = w[start:end]
        total = ww.sum()
        if total <= 0:
            continue
        position = (np.cumsum(ww) - ww / 2.0) / total
        out[g[start]] = np.interp(quantiles, position, v[start:end])
    return out


def _sensitivity_rows(run, long: pd.DataFrame):
    """The ×0.1 and ×10 harmonised effect for every row of `long`, or None."""
    sensitivity = getattr(run, "effect_sensitivity", None)
    spec_matrix = getattr(run, "spec_matrix", None)
    if sensitivity is None or spec_matrix is None:
        return None
    matrices = spec_matrix[long["spec_id"].to_numpy()]
    taxa = long["taxon"].to_numpy()
    valid = matrices >= 0
    rows = np.full((len(long), len(PSEUDOCOUNT_MULTIPLIERS)), np.nan)
    rows[valid] = sensitivity[matrices[valid], taxa[valid], :]
    return rows


def weighting_table(run, primary: str = "decision_tree", custom: dict = None):
    """Per-taxon weighted columns for every scheme, plus the grid composition.

    Returns (frame indexed by taxon id, {scheme: spec weights}, {scheme: composition}).
    """
    long = run.long
    n_taxa = len(run.taxa_names)
    schemes = all_scheme_weights(run.specs, custom)
    if primary not in schemes:
        raise ValueError(f"Unknown weighting scheme: {primary}")

    columns: dict = {}
    for scheme, weights in schemes.items():
        metrics = per_taxon_metrics(long, _unit_weights(scheme, weights), n_taxa)
        for name, array in metrics.items():
            columns[f"{name}_{scheme}"] = array

    taxa = long["taxon"].to_numpy()
    primary_w = schemes[primary][long["spec_id"].to_numpy()]
    effects = weighted_quantiles(long["effect_h"].to_numpy(dtype=float), primary_w, taxa,
                                 n_taxa)
    columns["iqr_low_w"], columns["median_effect_w"], columns["iqr_high_w"] = effects.T

    sens = _sensitivity_rows(run, long)
    if sens is not None:
        for k, multiplier in enumerate(PSEUDOCOUNT_MULTIPLIERS):
            present = np.isfinite(sens[:, k])
            median = weighted_quantiles(sens[present, k], primary_w[present], taxa[present],
                                        n_taxa, quantiles=(0.5,))[:, 0]
            columns[f"median_effect_pc_x{multiplier:g}"] = median

    frame = pd.DataFrame(columns, index=pd.RangeIndex(n_taxa, name="taxon_id"))
    comps = {scheme: composition(run.specs, weights, scheme)
             for scheme, weights in schemes.items()}
    return frame, schemes, comps


def pseudocount_flip(median: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """True where the direction at x0.1 or x10 differs from the direction at x1."""
    reference = np.sign(median)
    with np.errstate(invalid="ignore"):
        flipped = ((np.sign(low) != reference) & np.isfinite(low)) | (
            (np.sign(high) != reference) & np.isfinite(high))
    return flipped & (reference != 0) & np.isfinite(median)


BUILT_IN = SCHEMES
