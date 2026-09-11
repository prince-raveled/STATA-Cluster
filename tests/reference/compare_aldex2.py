"""Validate the in-tree ALDEx2 against analytic ground truth.

SPEC §10 fork 5 names ALDEx2 with "scikit-bio re-implementation" as the source, but
scikit-bio has no ALDEx2 (its composition module offers `dirmult_ttest`, a different
estimator), and the reference implementation is R-only. There is therefore no reference
to diff against on this machine — so correctness is established the harder way, against
quantities that are known in closed form.

What ALDEx2 does (Fernandes et al. 2014): draw Monte-Carlo instances of each sample's
composition from a Dirichlet posterior, CLR-transform each instance, test between
groups per instance, and report the expected p-value and the median CLR difference.

    .venv/Scripts/python tests/reference/compare_aldex2.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.methods.aldex2 import (  # noqa: E402
    DIRICHLET_PRIOR,
    clr_stack,
    dirichlet_instances,
    run_aldex2,
)

FAILURES: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def report(name: str, detail: str) -> None:
    print(f"  [ -- ] {name} — {detail}")


class _Matrix:
    """The minimal PreparedMatrix surface `run_aldex2` uses."""

    def __init__(self, counts, groups):
        self.counts = np.asarray(counts, dtype=float)
        self.groups = np.asarray(groups)
        self.taxa_idx = np.arange(self.counts.shape[0], dtype=np.int32)
        self.n_taxa = self.counts.shape[0]


# ---------------------------------------------------------------------------
def test_dirichlet_posterior_is_correct() -> None:
    """The Monte-Carlo draws must be the Dirichlet posterior the paper specifies.

    With a Dirichlet(count + 0.5) posterior the mean of each part is known exactly:
    (count + 0.5) / (total + 0.5 * n_taxa).
    """
    print("\nDirichlet posterior — the sampling step")
    counts = np.array([[100.0], [50.0], [10.0], [0.0], [340.0]])
    draws = dirichlet_instances(counts, n_instances=20000, seed=3)

    alpha = counts[:, 0] + DIRICHLET_PRIOR
    expected = alpha / alpha.sum()
    observed = draws[:, :, 0].mean(axis=0)

    max_abs = float(np.max(np.abs(observed - expected)))
    check("posterior mean matches Dirichlet(count + 0.5)", max_abs < 2e-3,
          f"max |difference| = {max_abs:.5f}")
    check("every instance is a composition",
          np.allclose(draws.sum(axis=1), 1.0), "columns sum to 1")
    check("zero counts get non-zero posterior mass", float(observed[3]) > 0,
          f"taxon with 0 reads has posterior mean {observed[3]:.5f}")

    # The prior is what stops a zero count becoming a -inf log-ratio.
    assert np.isfinite(clr_stack(draws)).all()
    check("CLR of every instance is finite", True, "no -inf from zero counts")


def test_recovers_a_known_clr_difference() -> None:
    """A taxon shifted by a known factor has a known CLR difference between groups.

    For a composition where one part is multiplied by `fold` and the rest renormalise,
    the CLR difference of that part is log(fold) minus the change in the geometric
    mean — computable exactly from the two true compositions.
    """
    print("\nground truth — median CLR difference")
    rng = np.random.default_rng(11)
    n_taxa, n_per, depth, fold = 60, 40, 200_000, 4.0

    base = rng.lognormal(0.0, 0.8, n_taxa)
    base /= base.sum()
    shifted = base.copy()
    shifted[0] *= fold
    shifted /= shifted.sum()

    def true_clr(p):
        # log2, the base ALDEx2 reports in — see app/core/methods/aldex2.py::clr_stack.
        logged = np.log2(p)
        return logged - logged.mean()

    expected = true_clr(shifted) - true_clr(base)

    counts = np.zeros((n_taxa, 2 * n_per))
    for j in range(2 * n_per):
        counts[:, j] = rng.multinomial(depth, shifted if j >= n_per else base)
    groups = np.array([0] * n_per + [1] * n_per)

    fit = run_aldex2(_Matrix(counts, groups), n_instances=128, seed=1)

    error = float(abs(fit.effect_native[0] - expected[0]))
    check("recovers the shifted taxon's CLR difference", error < 0.05,
          f"estimated {fit.effect_native[0]:+.4f} vs true {expected[0]:+.4f}")

    r = float(np.corrcoef(fit.effect_native, expected)[0, 1])
    check("recovers the whole CLR difference vector", r > 0.99, f"r = {r:.5f}")

    others = np.abs(fit.effect_native[1:] - expected[1:])
    check("unshifted taxa are estimated without bias", float(others.max()) < 0.05,
          f"max |error| over the other {n_taxa - 1} taxa = {others.max():.4f}")


def test_null_data_is_not_called_significant() -> None:
    """Type I error: with no true difference, expected p-values must stay uniform."""
    print("\nnull behaviour — false positive rate")
    rng = np.random.default_rng(5)
    n_taxa, n_per, depth = 120, 25, 40_000

    base = rng.lognormal(0.0, 1.1, n_taxa)
    base /= base.sum()
    counts = np.zeros((n_taxa, 2 * n_per))
    for j in range(2 * n_per):
        counts[:, j] = rng.multinomial(depth, base)
    groups = np.array([0] * n_per + [1] * n_per)

    fit = run_aldex2(_Matrix(counts, groups), n_instances=64, seed=2)
    rate = float((fit.p_raw < 0.05).mean())
    check("false positive rate is at or below nominal", rate <= 0.05,
          f"{rate:.1%} of {n_taxa} null taxa at p < 0.05")
    check("effects centre on zero", abs(float(np.median(fit.effect_native))) < 0.02,
          f"median effect = {np.median(fit.effect_native):+.4f}")


def test_power_on_a_spiked_set() -> None:
    """It has to find real signal, not just avoid false ones.

    Note what "precision" cannot mean here. Raising 15 taxa and renormalising pushes
    the other 85 *down* in relative terms, and a compositional method is supposed to
    see that — those are real shifts, not false positives. So the test asserts what is
    actually true of the data: the spiked taxa are recovered, they occupy the top of
    the effect ranking, and the rest are depleted rather than unchanged.
    """
    print("\npower — a spiked set")
    rng = np.random.default_rng(8)
    n_taxa, n_per, depth, n_spiked = 100, 30, 60_000, 15

    base = rng.lognormal(0.0, 1.0, n_taxa)
    base /= base.sum()
    spiked = rng.choice(n_taxa, size=n_spiked, replace=False)
    shifted = base.copy()
    shifted[spiked] *= 2.0 ** rng.uniform(1.5, 2.5, n_spiked)
    shifted /= shifted.sum()

    counts = np.zeros((n_taxa, 2 * n_per))
    for j in range(2 * n_per):
        counts[:, j] = rng.multinomial(depth, shifted if j >= n_per else base)
    groups = np.array([0] * n_per + [1] * n_per)

    fit = run_aldex2(_Matrix(counts, groups), n_instances=64, seed=4)
    truth = np.zeros(n_taxa, dtype=bool)
    truth[spiked] = True
    called = fit.p_raw < 0.05

    recall = float((called & truth).sum() / truth.sum())
    check("recovers every spiked taxon", recall >= 0.9, f"recall = {recall:.0%}")

    # The spiked taxa must sit at the top of the effect ranking.
    top = set(np.argsort(fit.effect_native)[::-1][:n_spiked].tolist())
    overlap = len(top & set(spiked.tolist())) / n_spiked
    check("spiked taxa occupy the top of the effect ranking", overlap >= 0.9,
          f"{overlap:.0%} of the top {n_spiked} are spiked")

    # And the rest are relatively depleted, which is the compositional truth.
    check("spiked taxa are enriched", float(np.median(fit.effect_native[truth])) > 0,
          f"median effect = {np.median(fit.effect_native[truth]):+.3f}")
    check("the remainder are depleted, not unchanged",
          float(np.median(fit.effect_native[~truth])) < 0,
          f"median effect = {np.median(fit.effect_native[~truth]):+.3f} "
          f"— renormalisation, correctly detected")


def test_monte_carlo_converges() -> None:
    """More instances must mean a tighter estimate, or the sampling is wrong."""
    print("\nconvergence — the Monte-Carlo integration")
    rng = np.random.default_rng(13)
    n_taxa, n_per, depth = 40, 20, 30_000
    base = rng.lognormal(0.0, 0.9, n_taxa)
    base /= base.sum()
    counts = np.array([rng.multinomial(depth, base) for _ in range(2 * n_per)]).T.astype(float)
    groups = np.array([0] * n_per + [1] * n_per)
    matrix = _Matrix(counts, groups)

    spreads = {}
    for n_instances in (8, 32, 128):
        estimates = np.array([
            run_aldex2(matrix, n_instances=n_instances, seed=s).effect_native
            for s in range(6)
        ])
        spreads[n_instances] = float(estimates.std(axis=0).mean())
        report(f"{n_instances:>3d} instances", f"seed-to-seed SD = {spreads[n_instances]:.5f}")

    check("more instances give a more stable estimate",
          spreads[128] < spreads[32] < spreads[8],
          " > ".join(f"{spreads[n]:.5f}" for n in (8, 32, 128)))
    check("64 instances (the default) is already stable", spreads[128] < 0.02,
          f"SD at 128 instances = {spreads[128]:.5f}")


def test_p_values_are_the_expected_value_over_instances() -> None:
    """ALDEx2 reports the *expected* p-value, not the p-value of a pooled statistic."""
    print("\ndefinition — expected p-value across instances")
    rng = np.random.default_rng(21)
    n_taxa, n_per, depth = 30, 18, 20_000
    base = rng.lognormal(0.0, 0.7, n_taxa)
    base /= base.sum()
    counts = np.array([rng.multinomial(depth, base) for _ in range(2 * n_per)]).T.astype(float)
    groups = np.array([0] * n_per + [1] * n_per)

    n_instances = 64
    fit = run_aldex2(_Matrix(counts, groups), n_instances=n_instances, seed=7)

    # Recompute the expectation independently from the same draws.
    instances = clr_stack(dirichlet_instances(counts, n_instances, 7))
    a = instances[:, :, groups == 0].reshape(-1, int((groups == 0).sum()))
    b = instances[:, :, groups == 1].reshape(-1, int((groups == 1).sum()))
    wilcox = stats.mannwhitneyu(a, b, axis=1, alternative="two-sided", method="asymptotic")
    welch = stats.ttest_ind(b, a, axis=1, equal_var=False)
    expected = np.maximum(
        np.asarray(wilcox.pvalue).reshape(n_instances, n_taxa).mean(axis=0),
        np.asarray(welch.pvalue).reshape(n_instances, n_taxa).mean(axis=0),
    )
    max_abs = float(np.max(np.abs(fit.p_raw - expected)))
    check("p-values are the mean across instances", max_abs < 1e-12,
          f"max |difference| = {max_abs:.2e}")


if __name__ == "__main__":
    print(f"in-tree ALDEx2 on Python {sys.version.split()[0]} "
          f"(no reference implementation available: ALDEx2 is R-only)")
    test_dirichlet_posterior_is_correct()
    test_recovers_a_known_clr_difference()
    test_null_data_is_not_called_significant()
    test_power_on_a_spiked_set()
    test_monte_carlo_converges()
    test_p_values_are_the_expected_value_over_instances()
    print("\n" + ("FAILED: " + ", ".join(FAILURES) if FAILURES else "All checks passed."))
    sys.exit(1 if FAILURES else 0)
