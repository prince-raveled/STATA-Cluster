"""Cross-validate the in-tree compositional code against scikit-bio.

SPEC §20 asks for `scikit-bio >= 0.7.1`, which has no wheel on the Python the server
runs (§24.1 B1), so `multiplicative_replacement`, `clr` and `run_ancombc` are
implemented in-tree. This script is the evidence that those implementations agree with
the reference.

It runs on a *separate* Python 3.10 environment where scikit-bio does install:

    py -3.10 -m venv .venv310
    .venv310/Scripts/pip install "scikit-bio>=0.7.1"
    .venv310/Scripts/python tests/reference/compare_scikit_bio.py

Exit code 0 means every comparison met its stated tolerance.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from skbio.stats import composition as skbio_comp  # noqa: E402

from app.core.methods.ancombc import run_ancombc  # noqa: E402
from app.core.preprocess import closure as our_closure  # noqa: E402
from app.core.preprocess import clr as our_clr  # noqa: E402
from app.core.preprocess import multiplicative_replacement  # noqa: E402

FAILURES: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def simulate(n_taxa=90, n_per_group=25, n_differential=12, seed=7):
    """Counts with a known differential set and a deliberate sampling-fraction shift.

    The depth shift is the point: ANCOM-BC exists to correct for exactly that, so a
    reimplementation that ignores it would diverge here.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_group
    groups = np.array([0] * n_per_group + [1] * n_per_group)

    base = np.sort(rng.lognormal(0.0, 1.4, n_taxa))[::-1]
    base /= base.sum()
    differential = rng.choice(n_taxa // 2, size=n_differential, replace=False)
    effect = np.zeros(n_taxa)
    magnitudes = rng.uniform(1.0, 2.2, n_differential)
    effect[differential] = magnitudes * rng.choice([-1, 1], n_differential)
    mean_b = base * 2.0 ** effect
    mean_b /= mean_b.sum()

    # Group B is sequenced ~1.8x deeper: a pure sampling-fraction difference.
    libraries = np.rint(rng.lognormal(9.4, 0.35, n) * np.where(groups == 1, 1.8, 1.0)).astype(int)
    counts = np.zeros((n_taxa, n), dtype=int)
    for j in range(n):
        composition = rng.dirichlet(np.maximum((mean_b if groups[j] else base) * 500.0, 1e-4))
        counts[:, j] = rng.multinomial(libraries[j], composition)
    keep = (counts > 0).sum(axis=1) >= 5           # both tools need observable taxa
    # Filtering renumbers the taxa, so map the spiked indices into the new frame.
    renumber = np.full(n_taxa, -1, dtype=int)
    renumber[keep] = np.arange(int(keep.sum()))
    spiked = {int(renumber[d]) for d in differential if keep[d]}
    return counts[keep], groups, spiked


class _Matrix:
    """The minimal PreparedMatrix surface `run_ancombc` uses."""

    def __init__(self, counts, groups):
        self.counts = counts.astype(float)
        self.groups = groups
        self.taxa_idx = np.arange(counts.shape[0], dtype=np.int32)
        self.n_taxa = counts.shape[0]


# ---------------------------------------------------------------------------
def compare_multi_replace() -> None:
    print("\nmulti_replace — multiplicative zero replacement (SPEC §10 fork 3)")
    rng = np.random.default_rng(3)
    counts = rng.integers(0, 40, size=(60, 12)).astype(float)
    counts[rng.random(counts.shape) < 0.35] = 0.0

    ours = multiplicative_replacement(our_closure(counts))
    # scikit-bio takes samples as rows and uses delta = (1/n_features)^2 by default.
    theirs = skbio_comp.multi_replace(skbio_comp.closure(counts.T)).T

    max_abs = float(np.max(np.abs(ours - theirs)))
    check("values match scikit-bio", max_abs < 1e-12, f"max |difference| = {max_abs:.3e}")
    check("columns remain compositions", np.allclose(ours.sum(axis=0), 1.0))
    check("no zeros remain", bool((ours > 0).all()))


def compare_clr() -> None:
    print("\nclr — centred log-ratio")
    rng = np.random.default_rng(11)
    counts = rng.integers(0, 60, size=(70, 16)).astype(float)
    counts[rng.random(counts.shape) < 0.3] = 0.0

    ours = our_clr(counts)
    replaced = skbio_comp.multi_replace(skbio_comp.closure(counts.T))
    theirs = skbio_comp.clr(replaced).T

    max_abs = float(np.max(np.abs(ours - theirs)))
    check("values match scikit-bio", max_abs < 1e-10, f"max |difference| = {max_abs:.3e}")
    check("columns sum to zero", np.allclose(ours.sum(axis=0), 0.0, atol=1e-9))


def compare_ancombc() -> None:
    print("\nancombc — bias-corrected log fold change (SPEC §10 fork 5, method 5)")
    counts, groups, differential = simulate()
    n_taxa = counts.shape[0]

    ours = run_ancombc(_Matrix(counts, groups))

    # scikit-bio wants samples as rows, strictly positive values, and a formula.
    table = pd.DataFrame(
        counts.T + 1.0,
        index=[f"s{j}" for j in range(counts.shape[1])],
        columns=[f"t{i}" for i in range(n_taxa)],
    )
    metadata = pd.DataFrame({"group": np.where(groups == 1, "B", "A")}, index=table.index)
    reference = skbio_comp.ancombc(table, metadata, "group", p_adjust="BH")

    # A (FeatureID, Covariate) MultiIndex; keep the group contrast rows.
    covariates = reference.index.get_level_values("Covariate")
    contrast = [c for c in covariates.unique() if "group" in str(c).lower()]
    assert contrast, f"no group contrast in {list(covariates.unique())}"
    group_rows = reference.xs(contrast[0], level="Covariate").reindex(table.columns)

    # scikit-bio 0.7.3 names this column "Log2(FC)", but the values are on the
    # NATURAL-log scale. Verified on a controlled 4x spike where the true natural-log
    # fold change is 1.314 and the true log2 fold change is 1.896: scikit-bio returns
    # 1.3826 and so do we. The label is theirs; the scale is what SPEC §10 asks for
    # ("natural-log fold change"), so no conversion is applied here.
    their_lfc = group_rows["Log2(FC)"].to_numpy(dtype=float)
    our_lfc = ours.effect_native
    their_p = group_rows["pvalue"].to_numpy(dtype=float)

    finite = np.isfinite(their_lfc) & np.isfinite(our_lfc)
    r = float(np.corrcoef(our_lfc[finite], their_lfc[finite])[0, 1])
    check("log fold changes correlate with scikit-bio", r > 0.95, f"r = {r:.4f}")

    slope = float(np.polyfit(their_lfc[finite], our_lfc[finite], 1)[0])
    check("on the same scale (slope ~ 1)", 0.95 < slope < 1.05, f"slope = {slope:.4f}")
    max_abs = float(np.max(np.abs(our_lfc[finite] - their_lfc[finite])))
    check("values match scikit-bio", max_abs < 0.05, f"max |difference| = {max_abs:.4f}")

    sign_agreement = float((np.sign(our_lfc[finite]) == np.sign(their_lfc[finite])).mean())
    check("directions agree", sign_agreement > 0.9, f"{sign_agreement:.1%} of taxa")

    # p-values: the ranking is what a multiverse consumes, so compare that.
    ok_p = np.isfinite(their_p) & np.isfinite(ours.p_raw)
    rank_r = float(
        pd.Series(-np.log10(np.clip(ours.p_raw[ok_p], 1e-300, 1))).corr(
            pd.Series(-np.log10(np.clip(their_p[ok_p], 1e-300, 1))), method="spearman"
        )
    )
    check("p-value ranking agrees with scikit-bio", rank_r > 0.85, f"Spearman = {rank_r:.4f}")

    both = (ours.p_raw[ok_p] < 0.05) & (their_p[ok_p] < 0.05)
    either = (ours.p_raw[ok_p] < 0.05) | (their_p[ok_p] < 0.05)
    jaccard = float(both.sum() / max(1, either.sum()))
    check("nominally significant sets overlap", jaccard > 0.7,
          f"Jaccard = {jaccard:.2f} ({int(both.sum())} shared of {int(either.sum())})")

    # Both must recover the taxa that were actually spiked.
    truth = np.zeros(n_taxa, dtype=bool)
    truth[sorted(differential)] = True
    for label, values in (("ours", our_lfc), ("scikit-bio", their_lfc)):
        ranked = set(np.argsort(np.abs(values))[::-1][: len(differential)])
        recall = len(ranked & set(np.flatnonzero(truth))) / max(1, truth.sum())
        check(f"{label} recovers the spiked taxa", recall >= 0.6, f"recall = {recall:.0%}")

    # The bias correction is the point of the method: without it, a 1.8x depth
    # difference would push every taxon's coefficient in the same direction.
    from app.core.methods.design import build_design

    logged = np.log(counts.astype(float) + 1.0)
    design, group_col, _ = build_design(groups, None)
    xtx_inv = np.linalg.pinv(design.T @ design)
    uncorrected = (logged @ design @ xtx_inv.T)[:, group_col]
    check(
        "bias correction removes the depth shift",
        abs(float(np.median(our_lfc))) < abs(float(np.median(uncorrected))),
        f"median |lfc| {abs(np.median(our_lfc)):.3f} corrected vs "
        f"{abs(np.median(uncorrected)):.3f} uncorrected",
    )


if __name__ == "__main__":
    import skbio

    print(f"scikit-bio {skbio.__version__} on Python {sys.version.split()[0]}")
    compare_multi_replace()
    compare_clr()
    compare_ancombc()
    print("\n" + ("FAILED: " + ", ".join(FAILURES) if FAILURES else "All comparisons passed."))
    sys.exit(1 if FAILURES else 0)
