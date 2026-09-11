"""Diff the in-tree TMM and ALDEx2 against the actual R packages.

SPEC §10 names edgeR's TMM and ALDEx2. Both are R-only, so until now they were checked
against analytic ground truth and against `conorm` — good evidence, but not the
reference. This closes that: R runs `edgeR::calcNormFactors(method="TMM")` and
`ALDEx2::aldex()` on the same matrices, and the results are compared value by value.

    .venv/Scripts/python tests/reference/compare_r.py

Needs R with edgeR and ALDEx2:

    Rscript -e "BiocManager::install(c('edgeR','ALDEx2'))"

Set `MICROVERSE_RSCRIPT` if Rscript is not on PATH, and `MICROVERSE_R_LIB` if the
packages live outside the default library.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.methods.aldex2 import run_aldex2  # noqa: E402
from app.core.methods.ancombc import run_ancombc  # noqa: E402
from app.core.preprocess import tmm_factors  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FAILURES: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def report(name: str, detail: str) -> None:
    print(f"  [ -- ] {name} — {detail}")


def find_rscript() -> str | None:
    explicit = os.environ.get("MICROVERSE_RSCRIPT")
    if explicit and os.path.exists(explicit):
        return explicit
    found = shutil.which("Rscript")
    if found:
        return found
    for candidate in (
        r"D:\Rlocal\R-4.6.1\bin\x64\Rscript.exe",
        r"C:\Program Files\R\R-4.6.1\bin\x64\Rscript.exe",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


class _Matrix:
    """The minimal PreparedMatrix surface the methods use."""

    def __init__(self, counts, groups):
        self.counts = np.asarray(counts, dtype=float)
        self.groups = np.asarray(groups)
        self.taxa_idx = np.arange(self.counts.shape[0], dtype=np.int32)
        self.n_taxa = self.counts.shape[0]


def make_counts(n_taxa=180, n_per_group=25, n_differential=30, seed=23,
                depth_mu=8.5, detect_k=10.0):
    """A table with real signal, composition bias and the sparsity of 16S data."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_group
    base = rng.lognormal(4.5, 1.3, n_taxa)

    shifted = base.copy()
    spiked = rng.choice(n_taxa, size=n_differential, replace=False)
    shifted[spiked] *= 2.0 ** rng.uniform(1.5, 3.0, n_differential)

    counts = np.zeros((n_taxa, n), dtype=int)
    for j in range(n):
        mean = shifted if j >= n_per_group else base
        p = mean / mean.sum()
        depth = int(rng.lognormal(depth_mu, 0.3))
        counts[:, j] = rng.multinomial(depth, p)

    # Detection is abundance-dependent, as it is in 16S: rare taxa drop out, abundant
    # ones do not. A flat dropout mask would erase the very signal being compared, and
    # then both tools would agree on noise.
    counts = counts * (rng.random(counts.shape) < counts / (counts + detect_k))
    keep = (counts > 0).sum(axis=1) >= 3          # ALDEx2 needs observable features
    counts = counts[keep]
    counts[:, counts.sum(axis=0) == 0] += 1        # no empty library
    groups = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    return counts.astype(float), groups


def run_r(counts: np.ndarray, groups: np.ndarray, rscript: str, mc_samples: int = 128):
    workdir = tempfile.mkdtemp(prefix="microverse-r-")
    taxa = [f"t{i:04d}" for i in range(counts.shape[0])]
    samples = [f"s{j:03d}" for j in range(counts.shape[1])]

    pd.DataFrame(counts.astype(int), index=taxa, columns=samples).to_csv(
        os.path.join(workdir, "counts.csv"))
    pd.DataFrame({"sample_id": samples, "group": groups}).to_csv(
        os.path.join(workdir, "groups.csv"), index=False)

    completed = subprocess.run(
        [rscript, os.path.join(HERE, "r_reference.R"),
         os.path.join(workdir, "counts.csv"),
         os.path.join(workdir, "groups.csv"),
         workdir, str(mc_samples)],
        capture_output=True, text=True, check=False,
    )
    for line in completed.stdout.splitlines():
        if line.strip():
            print(f"    R| {line.strip()}")
    if completed.returncode != 0:
        print(completed.stderr[-2000:])
        raise RuntimeError(f"Rscript failed with {completed.returncode}")

    tmm = pd.read_csv(os.path.join(workdir, "edger_tmm.csv"))
    aldex = pd.read_csv(os.path.join(workdir, "aldex2.csv"))
    ancombc_path = os.path.join(workdir, "ancombc.csv")
    ancombc = pd.read_csv(ancombc_path) if os.path.exists(ancombc_path) else None
    return tmm, aldex, ancombc, taxa


# ---------------------------------------------------------------------------
def compare_tmm(counts, r_tmm) -> None:
    print("\nedgeR::calcNormFactors(method='TMM') vs app.core.preprocess.tmm_factors")
    ours = tmm_factors(counts)
    theirs = r_tmm["norm_factor"].to_numpy(dtype=float)

    max_abs = float(np.max(np.abs(ours - theirs)))
    max_rel = float(np.max(np.abs(ours - theirs) / theirs))
    r = float(np.corrcoef(ours, theirs)[0, 1])

    check("factors match edgeR", max_rel < 0.01,
          f"max relative difference = {max_rel:.4%}, max absolute = {max_abs:.6f}")
    check("perfectly correlated with edgeR", r > 0.999, f"r = {r:.6f}")
    check("unit geometric mean, as edgeR centres them",
          np.isclose(np.exp(np.mean(np.log(ours))), 1.0),
          f"gm = {np.exp(np.mean(np.log(ours))):.6f}")
    report("factor range", f"ours {ours.min():.4f}–{ours.max():.4f}, "
                           f"edgeR {theirs.min():.4f}–{theirs.max():.4f}")


def compare_aldex2(counts, groups, r_aldex, taxa) -> None:
    print()
    print("ALDEx2::aldex() vs app.core.methods.aldex2.run_aldex2")
    binary = (groups == "B").astype(np.int8)
    matrix = _Matrix(counts, binary)
    fit = run_aldex2(matrix, n_instances=128, seed=1)

    reference = r_aldex.set_index("taxon").reindex(taxa)
    their_effect = reference["diff_btw"].to_numpy(dtype=float)
    their_welch = reference["we_ep"].to_numpy(dtype=float)
    their_wilcox = reference["wi_ep"].to_numpy(dtype=float)
    # We report the more conservative of the two expected p-values.
    their_p = np.maximum(their_welch, their_wilcox)

    finite = np.isfinite(their_effect) & np.isfinite(fit.effect_native)
    r_effect = float(np.corrcoef(fit.effect_native[finite], their_effect[finite])[0, 1])
    check("median CLR difference correlates with ALDEx2", r_effect > 0.98,
          f"r = {r_effect:.5f}")

    slope = float(np.polyfit(their_effect[finite], fit.effect_native[finite], 1)[0])
    check("on the same scale as ALDEx2", 0.9 < slope < 1.1, f"slope = {slope:.4f}")

    # diff.btw is a Monte-Carlo estimate in both tools, so a fixed tolerance would be
    # arbitrary — and at 128 instances a tight one is tighter than the estimator's own
    # noise. Calibrate against ourselves: rerun with other seeds and measure how far
    # the same estimator moves. Agreeing with ALDEx2 at least as well as we agree with
    # ourselves is the actual claim being made.
    others = [run_aldex2(matrix, n_instances=128, seed=s).effect_native
              for s in (2, 3, 4)]
    noise = max(float(np.max(np.abs(fit.effect_native[finite] - o[finite])))
                for o in others)
    max_abs = float(np.max(np.abs(fit.effect_native[finite] - their_effect[finite])))
    mean_abs = float(np.mean(np.abs(fit.effect_native[finite] - their_effect[finite])))
    check("effect sizes agree as closely as the estimator agrees with itself",
          max_abs <= 1.25 * noise,
          f"max |difference| = {max_abs:.4f} vs our own seed-to-seed spread "
          f"{noise:.4f} (both 128 instances)")
    own_mean = max(float(np.mean(np.abs(fit.effect_native[finite] - o[finite])))
                   for o in others)
    check("typical effect-size difference is within our own seed-to-seed spread",
          mean_abs <= 1.1 * own_mean,
          f"mean |difference| = {mean_abs:.4f} vs our own spread {own_mean:.4f}")

    ok_p = np.isfinite(their_p) & np.isfinite(fit.p_raw)
    rank = float(pd.Series(fit.p_raw[ok_p]).corr(pd.Series(their_p[ok_p]), method="spearman"))
    check("p-value ranking agrees with ALDEx2", rank > 0.95, f"Spearman = {rank:.5f}")

    both = (fit.p_raw[ok_p] < 0.05) & (their_p[ok_p] < 0.05)
    either = (fit.p_raw[ok_p] < 0.05) | (their_p[ok_p] < 0.05)
    jaccard = float(both.sum() / max(1, either.sum()))
    check("significant sets overlap", jaccard > 0.8,
          f"Jaccard = {jaccard:.2f} ({int(both.sum())} shared of {int(either.sum())})")

    bh_both = (reference["we_eBH"].to_numpy(dtype=float) < 0.1)
    report("ALDEx2 calls at BH<0.1", f"{int(np.nansum(bh_both))} taxa")


def compare_ancombc(counts, groups, r_ancombc, taxa) -> None:
    print()
    print("ANCOMBC::ancombc() vs app.core.methods.ancombc.run_ancombc")
    if r_ancombc is None:
        report("skipped", "ANCOMBC is not installed in the R library")
        return
    binary = (groups == "B").astype(np.int8)
    fit = run_ancombc(_Matrix(counts, binary))

    reference = r_ancombc.set_index("taxon").reindex(taxa)
    their_lfc = reference["lfc"].to_numpy(dtype=float)
    their_p = reference["p_val"].to_numpy(dtype=float)

    ok = np.isfinite(their_lfc) & np.isfinite(fit.effect_native)
    r_lfc = float(np.corrcoef(fit.effect_native[ok], their_lfc[ok])[0, 1])
    check("log fold changes correlate with ANCOM-BC", r_lfc > 0.95, f"r = {r_lfc:.5f}")

    slope = float(np.polyfit(their_lfc[ok], fit.effect_native[ok], 1)[0])
    check("on the same scale as ANCOM-BC (both natural log)", 0.85 < slope < 1.15,
          f"slope = {slope:.4f}")

    agree = float((np.sign(fit.effect_native[ok]) == np.sign(their_lfc[ok])).mean())
    check("directions agree", agree > 0.95, f"{agree:.1%} of taxa share a sign")

    # The only thing that can differ once the coefficients match is delta, and it
    # shifts every taxon by the same amount — so watch the offset directly rather than
    # letting it hide inside a correlation.
    offset = float(np.mean(fit.effect_native[ok] - their_lfc[ok]))
    spread = float(np.std(fit.effect_native[ok] - their_lfc[ok], ddof=1))
    check("the bias term lands in the same place", abs(offset) < 0.05,
          f"delta differs by {offset:+.4f} natural-log units "
          f"({np.expm1(abs(offset)):.1%} on the fold-change scale); the difference is "
          f"a constant to within {spread:.1e}, and comes from nloptr's and scipy's "
          f"Nelder-Mead stopping on kappa at different points")

    ok_p = np.isfinite(their_p) & np.isfinite(fit.p_raw)
    rank = float(pd.Series(fit.p_raw[ok_p]).corr(pd.Series(their_p[ok_p]),
                                                 method="spearman"))
    check("p-value ranking agrees with ANCOM-BC", rank > 0.90, f"Spearman = {rank:.5f}")

    both = (fit.p_raw[ok_p] < 0.05) & (their_p[ok_p] < 0.05)
    either = (fit.p_raw[ok_p] < 0.05) | (their_p[ok_p] < 0.05)
    jaccard = float(both.sum() / max(1, either.sum()))
    check("significant sets overlap", jaccard > 0.7,
          f"Jaccard = {jaccard:.2f} ({int(both.sum())} shared of {int(either.sum())})")


def main() -> int:
    rscript = find_rscript()
    if rscript is None:
        print("SKIPPED — no Rscript found. Install R and:")
        print('  Rscript -e "install.packages(\'BiocManager\'); '
              "BiocManager::install(c('edgeR','ALDEx2'))\"")
        print("Then set MICROVERSE_RSCRIPT if it is not on PATH.")
        return 0

    print(f"Rscript: {rscript}")
    counts, groups = make_counts()
    print(f"matrix: {counts.shape[0]} taxa x {counts.shape[1]} samples, "
          f"{(counts == 0).mean():.0%} zeros")

    r_tmm, r_aldex, r_ancombc, taxa = run_r(counts, groups, rscript)
    compare_tmm(counts, r_tmm)
    compare_aldex2(counts, groups, r_aldex, taxa)
    compare_ancombc(counts, groups, r_ancombc, taxa)

    print("\n" + ("FAILED: " + ", ".join(FAILURES) if FAILURES else "All comparisons passed."))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
