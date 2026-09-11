"""Design-matrix construction shared by the regression-based methods."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_design(groups: np.ndarray, covariate_frame: pd.DataFrame = None):
    """Return (X, group_column_index, column_names).

    Column 0 is the intercept, column 1 the group indicator (B = 1), then covariates.
    Categorical covariates are one-hot encoded with the first level dropped; numeric
    covariates are standardised so the Newton step is well conditioned.
    """
    n = len(groups)
    columns = [np.ones(n, dtype=float), np.asarray(groups, dtype=float)]
    names = ["intercept", "group"]

    if covariate_frame is not None and covariate_frame.shape[1] > 0:
        for column in covariate_frame.columns:
            series = covariate_frame[column]
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().mean() > 0.9 and numeric.nunique(dropna=True) > 2:
                values = numeric.to_numpy(dtype=float)
                values = np.where(np.isnan(values), np.nanmean(values), values)
                sd = values.std()
                columns.append((values - values.mean()) / (sd if sd > 0 else 1.0))
                names.append(str(column))
            else:
                levels = sorted(str(v) for v in series.dropna().unique())
                for level in levels[1:]:
                    columns.append((series.astype(str) == level).to_numpy(dtype=float))
                    names.append(f"{column}={level}")

    design = np.column_stack(columns)
    # Drop columns that are constant or collinear with an earlier column.
    keep = [0, 1]
    for j in range(2, design.shape[1]):
        column = design[:, j]
        if np.allclose(column, column[0]):
            continue
        previous = design[:, keep]
        try:
            residual = column - previous @ np.linalg.lstsq(previous, column, rcond=None)[0]
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
        if np.sqrt((residual ** 2).mean()) > 1e-8:
            keep.append(j)
    design = design[:, keep]
    names = [names[j] for j in keep]
    return design, 1, names


def residualise(values: np.ndarray, design: np.ndarray, group_col: int = 1) -> np.ndarray:
    """Remove covariate effects from each taxon, keeping intercept and group.

    Used so Wilcoxon and Welch — which have no covariate form — can still take part in
    covariate mode. Documented as a deviation in SPEC §24.
    """
    if design.shape[1] <= 2:
        return values
    nuisance = np.delete(design, [0, group_col], axis=1)
    if nuisance.shape[1] == 0:
        return values
    nuisance = np.column_stack([np.ones(nuisance.shape[0]), nuisance])
    beta, *_ = np.linalg.lstsq(nuisance, values.T, rcond=None)
    fitted = nuisance @ beta
    return values - fitted.T + values.mean(axis=1, keepdims=True)
