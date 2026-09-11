"""Choice attribution — SPEC §17. Which fork drives the variance?

Two separate decompositions, because FDR affects significance but not effect size:
  (a) effect size    ~ rarefaction + rank + prev_filter + transform + (1|taxon)
  (b) significance   ~ the same, plus method, fdr_method and fdr_threshold

METHODOLOGY AND ITS LIMITS — read before quoting a number from here.

The question "which analytical choice moves the answer most?" is answered in the
multimodel literature by decomposing the variance of the estimates across specifications
into contributions from the modelling choices themselves. Young & Holsteen (2017) give
the canonical treatment; Muñoz & Young (2018) scale it; Steegen et al. (2016) and
Simonsohn et al. (2020) frame the multiverse and specification-curve designs this
decomposition serves. Type II sums of squares are used because validity pruning (§11)
makes the fork grid unbalanced, and Type II is the order-independent choice for main
effects under imbalance (Langsrud 2003).

Three properties of this estimate that the interface must not hide:

  1. **The shares are shares of explained variance, not of total variance.** A fork
     with 60% has 60% of what the fork model explains. `explained_variance` reports how
     much that is — the within-taxon R². When it is small, every share is a large slice
     of a small thing, and the decomposition is describing noise.
  2. **Main effects only.** Interactions between forks are not modelled, so variance
     that lives in, say, rarefaction x transform is left in the residual and attributed
     to nothing.
  3. **There is no external reference implementation for this.** The DA methods are
     checked against edgeR, ALDEx2 and ANCOM-BC; nothing comparable exists for fork
     attribution. Its evidence grade is therefore never better than "internally
     consistent" — three independent estimators agreeing with each other — and the
     product says so.

Three estimators run, always, and all three are reported:
  * `mixedlm`     — §17's named estimator: taxon random intercept, Type II SS on the
                    marginal fit. Used when it converges and is not a boundary fit.
  * `within`      — taxon fixed effects by within-transformation (exact demeaning),
                    then Type II SS. No convergence step, so it cannot fail.
  * `group_means` — §17's mandatory fallback: between-level variance of group means.
Agreement between them is the only validity evidence available, and it is surfaced.

References
----------
Young, C. & Holsteen, K. (2017). Model uncertainty and robustness: a computational
    framework for multimodel analysis. Sociological Methods & Research 46(1), 3-40.
Muñoz, J. & Young, C. (2018). We ran 9 billion regressions: eliminating false positives
    through computational model robustness. Sociological Methodology 48(1), 1-33.
Steegen, S., Tuerlinckx, F., Gelman, A. & Vanpaemel, W. (2016). Increasing transparency
    through a multiverse analysis. Perspectives on Psychological Science 11(5), 702-712.
Simonsohn, U., Simmons, J. P. & Nelson, L. D. (2020). Specification curve analysis.
    Nature Human Behaviour 4, 1208-1214.
Langsrud, Ø. (2003). ANOVA for unbalanced data: use Type II instead of Type III sums of
    squares. Statistics and Computing 13, 163-167.
Nakagawa, S. & Schielzeth, H. (2013). A general and simple method for obtaining R^2 from
    generalized linear mixed-effects models. Methods in Ecology and Evolution 4, 133-142.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import FORK_LABELS

# SPEC §17(a) lists `method` among the effect-size predictors, but §14 defines the
# harmonised effect as a single estimator computed independently of the method — so
# `method` explains exactly 0% of it by construction, and a 0% bar reads as a bug
# rather than as a design property. It is excluded here and explained in the UI.
# Logged in SPEC §24.
EFFECT_FORKS = ["rarefaction", "rank", "prev_filter", "transform"]
SIGNIFICANCE_FORKS = ["rarefaction", "rank", "prev_filter", "transform", "method",
                      "fdr_method", "fdr_threshold"]

EFFECT_EXCLUSION_NOTE = (
    "The statistical test is left out of this chart on purpose: the effect size is "
    "computed the same way whichever test ran, so the test explains none of its "
    "variation. It appears in the significance chart, where it does the work."
)

MAX_TAXA_FOR_MIXEDLM = 120
MAX_ROWS_FOR_MIXEDLM = 60_000
MIXEDLM_TIME_BUDGET = 45.0  # seconds; §17 says do not depend on it converging

#: A random-intercept variance at or below this, relative to the residual variance, is a
#: boundary ("singular") fit: statsmodels reports convergence, but the taxon grouping has
#: collapsed and the fit is not the model that was requested.
SINGULAR_VARIANCE_RATIO = 1e-6

#: Uncertainty is a cluster bootstrap over taxa. Kept cheap enough to run on every
#: analysis: the within estimator is a pair of least-squares solves per fork.
BOOTSTRAP_DRAWS = 200
MAX_TAXA_FOR_BOOTSTRAP = 60
MAX_ROWS_FOR_BOOTSTRAP = 20_000

#: Below this within-taxon R², the forks explain so little that ranking them is
#: describing residual noise, and the run is graded exploratory whatever else holds.
MIN_EXPLAINED_VARIANCE = 0.10

#: Below this rank agreement between the three estimators, they are not telling the same
#: story and no single ranking should be presented as the answer.
MIN_ESTIMATOR_AGREEMENT = 0.60

EVIDENCE_GRADES = {
    "consistent": (
        "Internally consistent — three independent estimators rank the forks the same "
        "way and the fork model explains a usable share of the within-taxon variance. "
        "There is no external reference implementation for fork attribution, so this is "
        "the strongest grade available to it."
    ),
    "exploratory": (
        "Exploratory — treat the ranking as a hypothesis, not a result. Either the "
        "estimators disagree, or the model explains too little of the variance for the "
        "ranking to be meaningful, or the mixed model did not fit."
    ),
}


@dataclass
class Attribution:
    """One decomposition: fork -> percentage of the variance the forks explain."""

    target: str
    shares: dict
    estimator: str
    converged: bool
    note: str = ""
    fallback_shares: dict = field(default_factory=dict)
    within_shares: dict = field(default_factory=dict)
    mixedlm_shares: dict = field(default_factory=dict)
    explained_variance: float = float("nan")
    intervals: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    evidence: str = "exploratory"
    warnings: list = field(default_factory=list)

    @property
    def ranked(self) -> list:
        return sorted(self.shares.items(), key=lambda kv: kv[1], reverse=True)

    @property
    def evidence_blurb(self) -> str:
        return EVIDENCE_GRADES.get(self.evidence, EVIDENCE_GRADES["exploratory"])

    def interval(self, fork: str) -> tuple:
        return self.intervals.get(fork, (float("nan"), float("nan")))

    def sentence(self, dataset_word: str = "your dataset") -> str:
        """One line — hedged in proportion to the evidence behind it."""
        parts = [
            f"{FORK_LABELS.get(fork, fork).lower()} accounts for {share:.0f}% of the "
            f"variance the forks explain"
            if i == 0 else f"{FORK_LABELS.get(fork, fork).lower()} {share:.0f}%"
            for i, (fork, share) in enumerate(self.ranked)
            if share >= 0.5
        ]
        if not parts:
            return "No fork explains a measurable share of the variance."
        lead = "For " if self.evidence == "consistent" else "Exploratory — for "
        tail = ""
        if np.isfinite(self.explained_variance):
            tail = (f" The forks explain {self.explained_variance:.0%} of the "
                    f"within-taxon variance in total.")
        return lead + dataset_word + ", " + "; ".join(parts) + "." + tail

    def agreement(self) -> float:
        """Rank agreement between the estimator used and §17's mandatory fallback."""
        return _rank_agreement(self.shares, self.fallback_shares)

    def estimator_agreement(self) -> float:
        """Lowest pairwise rank agreement across every estimator that produced shares."""
        available = [s for s in (self.mixedlm_shares, self.within_shares,
                                 self.fallback_shares) if s]
        if len(available) < 2:
            return float("nan")
        pairs = [
            _rank_agreement(available[i], available[j])
            for i in range(len(available)) for j in range(i + 1, len(available))
        ]
        finite = [p for p in pairs if np.isfinite(p)]
        return float(min(finite)) if finite else float("nan")


def _rank_agreement(left: dict, right: dict) -> float:
    if not left or not right:
        return float("nan")
    forks = [f for f in left if f in right]
    if len(forks) < 3:
        return float("nan")
    a = pd.Series({f: left[f] for f in forks}).rank()
    b = pd.Series({f: right[f] for f in forks}).rank()
    return float(a.corr(b, method="spearman"))


# --- estimator 3: the mandatory fallback (§17) ------------------------------
def variance_of_group_means(df: pd.DataFrame, fork_col: str, value_col: str) -> float:
    """Between-level variance for one fork, averaged within taxon (SPEC §17 fallback).

    Simple, fast, never fails — which is exactly why it is mandatory.
    """
    if df[fork_col].nunique() < 2:
        return 0.0
    per_taxon = df.groupby(["taxon", fork_col], observed=True)[value_col].mean()
    variance = per_taxon.groupby("taxon", observed=True).var()
    value = float(variance.mean())
    return value if np.isfinite(value) else 0.0


def _normalise(raw: dict) -> dict:
    total = sum(v for v in raw.values() if np.isfinite(v) and v > 0)
    if total <= 0:
        return dict.fromkeys(raw, 0.0)
    return {k: 100.0 * max(v, 0.0) / total for k, v in raw.items()}


def fallback_attribution(frame: pd.DataFrame, forks: list, value_col: str) -> dict:
    return _normalise(
        {fork: variance_of_group_means(frame, fork, value_col) for fork in forks})


# --- shared machinery -------------------------------------------------------
def _subsample(frame: pd.DataFrame, seed: int = 3, max_taxa: int = MAX_TAXA_FOR_MIXEDLM,
               max_rows: int = MAX_ROWS_FOR_MIXEDLM) -> pd.DataFrame:
    """Keep every specification for a random subset of taxa, so within-taxon structure holds."""
    taxa = frame["taxon"].unique()
    rng = np.random.default_rng(seed)
    if len(taxa) > max_taxa:
        taxa = rng.choice(taxa, size=max_taxa, replace=False)
        frame = frame[frame["taxon"].isin(taxa)]
    if len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=seed)
    return frame


def _design(frame: pd.DataFrame, forks: list):
    """Dummy-coded fixed effects plus a map from fork -> its columns, for Type II SS."""
    blocks = []
    columns_by_fork: dict = {}
    for fork in forks:
        series = frame[fork].astype(str)
        if series.nunique() < 2:
            columns_by_fork[fork] = []
            continue
        dummies = pd.get_dummies(series, prefix=fork, drop_first=True, dtype=float)
        columns_by_fork[fork] = list(dummies.columns)
        blocks.append(dummies)
    if not blocks:
        return None, columns_by_fork
    design = pd.concat(blocks, axis=1)
    design.insert(0, "intercept", 1.0)
    return design, columns_by_fork


def _type2_sums_of_squares(values: np.ndarray, y: np.ndarray, column_index: dict,
                           n_columns: int) -> tuple:
    """SS_f = the increase in residual SS when fork f's columns are dropped.

    Returns (per-fork SS, residual SS of the full model, total SS of y).
    """
    beta, *_ = np.linalg.lstsq(values, y, rcond=None)
    full_rss = float(((y - values @ beta) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    out = {}
    for fork, columns in column_index.items():
        if not columns:
            out[fork] = 0.0
            continue
        keep = [c for c in range(n_columns) if c not in set(columns)]
        reduced = values[:, keep]
        beta_r, *_ = np.linalg.lstsq(reduced, y, rcond=None)
        rss = float(((y - reduced @ beta_r) ** 2).sum())
        out[fork] = max(rss - full_rss, 0.0)
    return out, full_rss, total


def _column_index(design: pd.DataFrame, columns_by_fork: dict) -> dict:
    position = {name: i for i, name in enumerate(design.columns)}
    return {fork: [position[c] for c in columns] for fork, columns in columns_by_fork.items()}


def _demean_within(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each taxon's mean — the exact fixed-effects within transformation."""
    frame = pd.DataFrame({"y": y, "g": groups})
    return (frame["y"] - frame.groupby("g")["y"].transform("mean")).to_numpy()


# --- estimator 2: taxon fixed effects, exact -------------------------------
def within_attribution(frame: pd.DataFrame, forks: list, value_col: str):
    """Type II SS after demeaning within taxon. No optimiser, so it cannot fail to fit.

    Returns (shares, explained_variance, diagnostics). `explained_variance` is the
    within-taxon R²: the share of the variance *remaining after* between-taxon
    differences are removed that the fork model accounts for.
    """
    sample = _subsample(frame)
    design, columns_by_fork = _design(sample, forks)
    if design is None:
        return {}, float("nan"), {"reason": "no fork varies in this run"}

    y = sample[value_col].to_numpy(dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 50:
        return {}, float("nan"), {"reason": "too few finite observations"}

    values = design.to_numpy(dtype=float)[finite]
    groups = sample["taxon"].to_numpy()[finite]
    centred = _demean_within(y[finite], groups)

    index = _column_index(design, columns_by_fork)
    raw, rss, total = _type2_sums_of_squares(values, centred, index, values.shape[1])
    explained = float(1.0 - rss / total) if total > 0 else float("nan")
    return _normalise(raw), explained, {
        "n_rows": int(finite.sum()),
        "n_taxa": int(pd.unique(groups).size),
        "residual_ss": rss,
        "total_ss": total,
    }


# --- estimator 1: SPEC §17's named MixedLM ---------------------------------
def mixedlm_attribution(frame: pd.DataFrame, forks: list, value_col: str):
    """MixedLM with taxon as a random intercept; Type II SS on the marginal fit.

    Returns (shares, converged, note, diagnostics). Never raises — §17 requires the run
    to survive MixedLM failing to converge, which it does often enough to plan for.
    A boundary fit (random-intercept variance collapsed to zero) is treated as a
    failure, not a success: statsmodels reports convergence for it, but the model that
    converged is not the model that was asked for.
    """
    started = time.perf_counter()
    diagnostics: dict = {"singular": False, "boundary_checked": True}
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM
    except Exception as exc:  # pragma: no cover
        return {}, False, f"statsmodels unavailable ({exc})", diagnostics

    sample = _subsample(frame)
    design, columns_by_fork = _design(sample, forks)
    if design is None:
        return {}, False, "no fork varies in this run", diagnostics

    y = sample[value_col].to_numpy(dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 50:
        return {}, False, "too few finite observations", diagnostics
    values = design.to_numpy(dtype=float)[finite]
    y = y[finite]
    groups = sample["taxon"].to_numpy()[finite]
    diagnostics["n_rows"] = int(y.size)
    diagnostics["n_taxa"] = int(pd.unique(groups).size)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MixedLM(y, values, groups=groups)
            fit = model.fit(method="lbfgs", maxiter=100)
        converged = bool(getattr(fit, "converged", False))
    except Exception as exc:
        return {}, False, f"MixedLM raised {type(exc).__name__}: {exc}", diagnostics

    diagnostics["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    if time.perf_counter() - started > MIXEDLM_TIME_BUDGET:
        converged = False
        diagnostics["time_budget_exceeded"] = True

    # Boundary check: a random-intercept variance of ~0 relative to the residual means
    # the taxon grouping has collapsed and this is not the requested model.
    try:
        group_variance = float(np.ravel(np.asarray(fit.cov_re))[0])
        residual = float(fit.scale)
        diagnostics["random_intercept_variance"] = group_variance
        diagnostics["residual_variance"] = residual
        if residual > 0 and group_variance / residual <= SINGULAR_VARIANCE_RATIO:
            diagnostics["singular"] = True
            converged = False
    except Exception:
        diagnostics["boundary_checked"] = False

    # Residualise the random intercept out, then take Type II SS on the fixed effects.
    try:
        random_effects = fit.random_effects
        intercepts = np.array(
            [float(np.ravel(random_effects.get(g, [0.0]))[0]) for g in groups],
            dtype=float)
    except Exception:
        intercepts = np.zeros_like(y)
        converged = False
        diagnostics["random_effects_unavailable"] = True

    index = _column_index(design, columns_by_fork)
    raw, _, _ = _type2_sums_of_squares(values, y - intercepts, index, values.shape[1])

    if diagnostics.get("singular"):
        note = ("the model collapsed: it found no taxon-to-taxon variation to work "
                "with, so this fit is not usable")
    elif converged:
        note = "the model fitted"
    else:
        note = "the model did not fit"
    return _normalise(raw), converged, note, diagnostics


# --- uncertainty ------------------------------------------------------------
def bootstrap_intervals(frame: pd.DataFrame, forks: list, value_col: str,
                        draws: int = BOOTSTRAP_DRAWS, seed: int = 5) -> dict:
    """Cluster bootstrap over taxa, using the within estimator.

    Taxa are the independent unit: every specification is evaluated on every taxon, so
    resampling rows would treat one taxon's 3,000 correlated results as 3,000
    observations. The within estimator is used because it has no optimiser and so
    returns a value for every draw; MixedLM would fail on some and bias the interval
    toward the draws that happened to converge.
    """
    sample = _subsample(frame, seed=seed, max_taxa=MAX_TAXA_FOR_BOOTSTRAP,
                        max_rows=MAX_ROWS_FOR_BOOTSTRAP)
    taxa = pd.unique(sample["taxon"])
    if len(taxa) < 5:
        return {}
    # Not dict(groupby): pandas exposes `.keys` as the grouping key (a string), so
    # dict() finds it, calls it, and raises "'str' object is not callable".
    by_taxon = {  # noqa: C416
        taxon: part for taxon, part in sample.groupby("taxon", observed=True)}

    rng = np.random.default_rng(seed)
    collected: dict = {fork: [] for fork in forks}
    for _ in range(draws):
        picked = rng.choice(taxa, size=len(taxa), replace=True)
        # Relabel so a taxon drawn twice counts as two clusters, not one.
        parts = []
        for i, t in enumerate(picked):
            block = by_taxon[t].copy()
            block["taxon"] = f"{t}__{i}"
            parts.append(block)
        drawn = pd.concat(parts, ignore_index=True)
        shares, _, _ = within_attribution(drawn, forks, value_col)
        if not shares:
            continue
        for fork in forks:
            collected[fork].append(shares.get(fork, 0.0))

    intervals = {}
    for fork, values in collected.items():
        if len(values) >= max(20, draws // 10):
            intervals[fork] = (float(np.percentile(values, 2.5)),
                               float(np.percentile(values, 97.5)))
    return intervals


def rarefaction_seed_share(run) -> dict:
    """How much of rarefaction's variance is the random draw rather than the depth?

    SPEC §10 makes each rarefaction seed its own specification precisely so this is
    measurable: variance that survives fixing the depth is Monte-Carlo noise from the
    subsampling draw, and a result that moves with it is not a result.
    """
    specs = run.specs_frame.copy()
    subsampled = specs[specs["rarefaction"] != "none"]
    if subsampled.empty or subsampled["rare_seed"].nunique() < 2:
        return {}
    frame = run.long.merge(
        subsampled[["spec_id", "rarefaction", "rare_seed"]], on="spec_id", how="inner"
    )
    if frame.empty:
        return {}
    frame["depth"] = frame["rarefaction"].astype(str)
    frame["seed"] = frame["rare_seed"].astype(str)

    # Within-depth (seed-driven) vs between-depth variance of the per-taxon mean effect.
    per_cell = frame.groupby(["taxon", "depth", "seed"], observed=True)["effect_h"].mean()
    within = per_cell.groupby(["taxon", "depth"], observed=True).var().groupby("taxon").mean()
    between = per_cell.groupby(["taxon", "depth"], observed=True).mean() \
        .groupby("taxon", observed=True).var()

    def _mean(values) -> float:
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else float("nan")

    within_mean = _mean(within.to_numpy(dtype=float))
    between_mean = _mean(between.to_numpy(dtype=float))
    total = within_mean + between_mean
    if not np.isfinite(total) or total <= 0:
        return {}
    return {
        "seed_share": 100.0 * within_mean / total,
        "depth_share": 100.0 * between_mean / total,
    }


def _grade(converged: bool, explained: float, agreement: float) -> tuple:
    """Evidence grade plus the specific reasons it is not higher."""
    problems = []
    if not converged:
        problems.append("the mixed model of §17 did not produce a usable fit")
    if not np.isfinite(explained) or explained < MIN_EXPLAINED_VARIANCE:
        shown = f"{explained:.0%}" if np.isfinite(explained) else "undefined"
        problems.append(
            f"the forks explain only {shown} of the within-taxon variance, below the "
            f"{MIN_EXPLAINED_VARIANCE:.0%} needed for the ranking to describe signal")
    if not np.isfinite(agreement) or agreement < MIN_ESTIMATOR_AGREEMENT:
        shown = f"{agreement:.2f}" if np.isfinite(agreement) else "undefined"
        problems.append(
            f"the three estimators rank the forks differently (agreement {shown}, "
            f"below {MIN_ESTIMATOR_AGREEMENT:.2f})")
    return ("consistent" if not problems else "exploratory"), problems


def attribute(run, bootstrap: bool = True) -> dict:
    """Both decompositions of SPEC §17, each with three estimators and an evidence grade."""
    long = run.long
    specs = run.specs_frame.copy()
    specs["rarefaction"] = [s.rarefaction_label for s in run.specs]
    frame = long.merge(specs, on="spec_id", how="left")
    frame["prev_filter"] = frame["prev_filter"].map(lambda v: f"{float(v):.0%}")
    frame["fdr_threshold"] = frame["fdr_threshold"].map(lambda v: f"{float(v):g}")
    frame["significant"] = frame["significant"].astype(float)

    results = {}
    for target, forks, value_col in (
        ("effect", EFFECT_FORKS, "effect_h"),
        ("significance", SIGNIFICANCE_FORKS, "significant"),
    ):
        active = [f for f in forks if frame[f].nunique() > 1]
        fallback = fallback_attribution(frame, active, value_col)
        within, explained, within_diag = within_attribution(frame, active, value_col)
        mixed, converged, note, diagnostics = mixedlm_attribution(frame, active, value_col)

        if converged and mixed:
            chosen = mixed
            estimator = ("Mixed-effects model, each taxon its own group "
                         "(the specification's named method)")
        elif within:
            chosen = within
            estimator = ("Per-taxon adjusted comparison — used because the "
                         "mixed-effects model did not fit")
        else:
            chosen = fallback
            estimator = "Simple variance comparison (backup method)"

        probe = Attribution(target=target, shares=chosen, estimator=estimator,
                            converged=converged, fallback_shares=fallback,
                            within_shares=within, mixedlm_shares=mixed)
        agreement = probe.estimator_agreement()
        evidence, problems = _grade(converged, explained, agreement)

        if target == "significance":
            note = (note or "") + (
                " — significance is modelled directly as a yes/no outcome, which "
                "is an approximation; it is recorded in the manifest."
            )
        else:
            note = (note or "") + " — " + EFFECT_EXCLUSION_NOTE

        diagnostics.update({
            "within": within_diag,
            "estimator_agreement": agreement,
            "explained_variance": explained,
            "n_forks": len(active),
            "levels_per_fork": {f: int(frame[f].nunique()) for f in active},
        })

        intervals = (bootstrap_intervals(frame, active, value_col)
                     if bootstrap and active else {})

        results[target] = Attribution(
            target=target,
            shares=chosen,
            estimator=estimator,
            converged=converged,
            note=note,
            fallback_shares=fallback,
            within_shares=within,
            mixedlm_shares=mixed,
            explained_variance=explained,
            intervals=intervals,
            diagnostics=diagnostics,
            evidence=evidence,
            warnings=problems,
        )
    results["rarefaction_draw"] = rarefaction_seed_share(run)
    return results
