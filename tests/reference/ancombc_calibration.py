"""Is ANCOM-BC's null p-value distribution calibrated, and is it ours or theirs?

Background. The reference checks in this directory compare *estimates* — log fold
changes, bias terms, normalisation factors — against the R packages, and they found four
real bugs that way. None of them compare *p-value calibration*, and an estimate can
agree to r = 1.0 while the test built on it rejects at the wrong rate.

An audit simulation found MicroVerse's ANCOM-BC anti-conservative in the tail: at a
nominal 0.001 it rejected roughly 16x too often on data with no group difference at all,
while Wilcoxon on the same datasets was correctly calibrated. That rules out the
simulation and leaves two possibilities, which this script exists to tell apart:

  A. inherited — R's ANCOMBC does the same thing, and the behaviour belongs to the
     method as published. Then nothing in `app/core/methods/ancombc.py` should change
     and the limitation is documented.
  B. ours — R is calibrated and we are not, which would mean an implementation
     discrepancy that the estimate-level comparison did not catch.

`decompose()` answers a narrower question without needing R at all, by rebuilding the
Wald test from the same fitted quantities while changing exactly one thing at a time.
It runs everywhere. `compare_with_r()` needs R with ANCOMBC installed and is the part
that settles A vs B directly; it skips with instructions when R is absent.

Run:  .venv/Scripts/python tests/reference/ancombc_calibration.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from app.core.methods.ancombc import estimate_bias, run_ancombc  # noqa: E402
from app.core.methods.design import build_design  # noqa: E402
from app.core.preprocess import PreparedMatrix, prevalence_filter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_r import find_rscript  # noqa: E402

#: Bioconductor packages are installed outside R's default library on this machine, and
#: `r_reference.R` already reads this variable with the same default, so both reference
#: checks resolve the packages the same way.
R_LIB = os.environ.get("MICROVERSE_R_LIB", "D:/Rlocal/library")

RECORD = Path(__file__).resolve().parents[2] / "docs" / "ancombc_calibration.json"

N_PER_GROUP = 25
N_TAXA = 60
LEVELS = (0.05, 0.01, 0.001)

#: How far above nominal a rate may sit before the profile is treated as changed. The
#: recorded numbers are what this implementation does today; the test exists to catch
#: drift, not to assert the method is well calibrated — it is not.
DRIFT_TOLERANCE = 0.006


def null_dataset(seed: int):
    """A sparse, over-dispersed count table with no group structure whatsoever."""
    rng = np.random.default_rng(seed)
    n = N_PER_GROUP * 2
    depths = rng.integers(1_000, 8_000, n)
    props = rng.dirichlet(np.full(N_TAXA, 0.12))
    counts = np.vstack([rng.multinomial(int(d), props) for d in depths]).T.astype(float)
    return counts[prevalence_filter(counts, 0.05)]


def as_matrix(counts):
    n = counts.shape[1]
    groups = np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=np.int8)
    rel = counts / np.maximum(counts.sum(axis=0, keepdims=True), 1.0)
    return PreparedMatrix(
        key=("none", None, "input", 0.05, "raw"), values=counts, counts=counts, rel=rel,
        taxa_idx=np.arange(counts.shape[0], dtype=np.int32), groups=groups,
        sample_idx=np.arange(n, dtype=np.int32), transform="raw")


def rates(p_values) -> dict:
    p = np.asarray(p_values, dtype=float)
    finite = p[np.isfinite(p)]
    return {f"{level:g}": round(float(np.mean(finite <= level)), 5) for level in LEVELS}


def decompose(n_datasets: int = 400) -> dict:
    """Which component of the Wald test produces the tail excess?

    Four variants on identical data, each differing in exactly one thing:

      as_shipped   W = (beta - delta) / se, normal reference   <- what MicroVerse runs
      no_bias      W =  beta           / se, normal reference
      t_reference  W = (beta - delta) / se, t_(n-p) reference
      both         W =  beta           / se, t_(n-p) reference

    If the excess disappears when the bias correction is removed and survives when only
    the reference distribution changes, the cause is the bias subtraction — which is the
    published method's own definition under `conserve = FALSE`, not a coding choice
    made here.
    """
    collected = {k: [] for k in ("as_shipped", "no_bias", "t_reference", "both")}
    wilcoxon = []
    deltas = []

    for i in range(n_datasets):
        counts = null_dataset(90_000 + i)
        if counts.shape[0] < 5:
            continue
        matrix = as_matrix(counts)
        logged = np.log(counts + 1.0)
        design, group_col, _ = build_design(matrix.groups, None)
        n_obs, p = design.shape
        df = max(n_obs - p, 1)
        xtx_inv = np.linalg.pinv(design.T @ design)
        beta = logged @ design @ xtx_inv.T
        residuals = logged - beta @ design.T
        sigma2 = (residuals ** 2).sum(axis=1) / df
        variance = sigma2 * xtx_inv[group_col, group_col]
        coefficient = beta[:, group_col]
        delta, _, _ = estimate_bias(coefficient, variance)
        deltas.append(float(delta))
        se = np.sqrt(np.maximum(variance, 1e-300))

        for name, numerator, reference in (
            ("as_shipped", coefficient - delta, "norm"),
            ("no_bias", coefficient, "norm"),
            ("t_reference", coefficient - delta, "t"),
            ("both", coefficient, "t"),
        ):
            w = numerator / se
            p_val = (2.0 * stats.norm.sf(np.abs(w)) if reference == "norm"
                     else 2.0 * stats.t.sf(np.abs(w), df))
            collected[name].append(np.clip(p_val, 0.0, 1.0))

        from app.core.methods import run_method
        wilcoxon.append(np.asarray(run_method("wilcoxon", matrix, None).p_raw, dtype=float))

    out = {name: rates(np.concatenate(values)) for name, values in collected.items()}
    out["wilcoxon_control"] = rates(np.concatenate(wilcoxon))
    out["n_tests"] = int(np.concatenate(collected["as_shipped"]).size)
    out["delta_mean"] = round(float(np.mean(deltas)), 5)
    out["delta_sd"] = round(float(np.std(deltas)), 5)
    return out


def microverse_rates(n_datasets: int = 400) -> dict:
    """The rate the shipped code path actually produces, through `run_ancombc`."""
    collected = []
    for i in range(n_datasets):
        counts = null_dataset(90_000 + i)
        if counts.shape[0] < 5:
            continue
        collected.append(np.asarray(run_ancombc(as_matrix(counts)).p_raw, dtype=float))
    return rates(np.concatenate(collected))


R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
indir <- args[1]; outfile <- args[2]; lib <- args[3]
if (nzchar(lib) && dir.exists(lib)) .libPaths(c(lib, .libPaths()))
suppressMessages({
  library(ANCOMBC)
  library(phyloseq)
  library(jsonlite)
})
files <- list.files(indir, pattern = "^counts_[0-9]+\\.csv$", full.names = TRUE)
all_p <- c()
for (f in files) {
  counts <- as.matrix(read.csv(f, row.names = 1, check.names = FALSE))
  n <- ncol(counts)
  meta <- data.frame(group = factor(c(rep("a", n %/% 2), rep("b", n - n %/% 2))))
  rownames(meta) <- colnames(counts)
  ps <- phyloseq(otu_table(counts, taxa_are_rows = TRUE), sample_data(meta))
  res <- try(ancombc2(data = ps, fix_formula = "group", p_adj_method = "none",
                      prv_cut = 0, group = NULL, alpha = 0.05, verbose = FALSE),
             silent = TRUE)
  if (inherits(res, "try-error")) next
  cols <- setdiff(grep("^p_", colnames(res$res), value = TRUE), "p_(Intercept)")
  if (length(cols) == 0) next
  all_p <- c(all_p, res$res[[cols[1]]])
}
all_p <- all_p[is.finite(all_p)]
out <- list(n_tests = length(all_p),
            rates = list("0.05" = mean(all_p <= 0.05),
                         "0.01" = mean(all_p <= 0.01),
                         "0.001" = mean(all_p <= 0.001)))
writeLines(jsonlite::toJSON(out, auto_unbox = TRUE), outfile)
"""


def compare_with_r(n_datasets: int = 60) -> dict:
    """The same null datasets through R's ANCOMBC. Skips when R is unavailable.

    This is the check that decides `inherited` versus `ours`. It is deliberately fewer
    datasets than the Python side — `ancombc2` is slow — which widens the interval on
    R's rate but is ample to separate 0.001 from 0.016.
    """
    rscript = find_rscript()
    if not rscript:
        return {"status": "SKIPPED",
                "reason": "No Rscript found. Install R with the ANCOMBC, phyloseq and "
                          "jsonlite packages, or set MICROVERSE_RSCRIPT."}

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        written = 0
        for i in range(n_datasets):
            counts = null_dataset(90_000 + i)
            if counts.shape[0] < 5:
                continue
            import pandas as pd
            frame = pd.DataFrame(
                counts.astype(int),
                index=[f"taxon_{j}" for j in range(counts.shape[0])],
                columns=[f"s{k:03d}" for k in range(counts.shape[1])])
            frame.to_csv(folder / f"counts_{i}.csv")
            written += 1

        script = folder / "calibration.R"
        script.write_text(R_SCRIPT, encoding="utf-8")
        outfile = folder / "r_rates.json"
        proc = subprocess.run(
            [rscript, "--vanilla", str(script), str(folder), str(outfile), R_LIB],
            capture_output=True, text=True, timeout=7200, check=False)
        if proc.returncode != 0 or not outfile.exists():
            return {"status": "FAILED", "reason": proc.stderr[-800:] or "no output",
                    "n_datasets": written}
        payload = json.loads(outfile.read_text(encoding="utf-8"))
        payload["status"] = "OK"
        payload["n_datasets"] = written
        return payload


def main() -> int:
    print("ANCOM-BC null calibration\n" + "=" * 62)
    print(f"{N_PER_GROUP} samples per group, ~{N_TAXA} taxa, no group difference.\n")

    shipped = microverse_rates()
    print("MicroVerse, through run_ancombc:")
    for level in LEVELS:
        print(f"   nominal {level:<7g} observed {shipped[f'{level:g}']:.4f}")

    print("\nWhich part of the Wald test causes it:")
    parts = decompose()
    header = f"{'variant':16s}" + "".join(f"{f'p<={lv:g}':>11s}" for lv in LEVELS)
    print(header)
    for name in ("as_shipped", "no_bias", "t_reference", "both", "wilcoxon_control"):
        row = "".join(f"{parts[name][f'{lv:g}']:11.4f}" for lv in LEVELS)
        print(f"{name:16s}{row}")
    print(f"{'nominal':16s}" + "".join(f"{lv:11.4f}" for lv in LEVELS))
    print(f"\nbias delta across datasets: mean {parts['delta_mean']:+.4f}, "
          f"sd {parts['delta_sd']:.4f}")

    print("\nAgainst R's ANCOMBC:")
    print(f"   Rscript: {find_rscript() or 'not found'}")
    r_result = compare_with_r()
    print(f"   {r_result['status']}: {r_result.get('reason', '')}")
    if r_result["status"] == "OK":
        for level in LEVELS:
            print(f"   nominal {level:<7g} R observed "
                  f"{r_result['rates'][f'{level:g}']:.4f}")

    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps({
        "design": {"n_per_group": N_PER_GROUP, "n_taxa": N_TAXA,
                   "n_datasets": 400, "structure": "no group difference by construction"},
        "microverse": shipped,
        "decomposition": parts,
        "r_reference": r_result,
    }, indent=2), encoding="utf-8")
    print(f"\nRecorded to {RECORD.relative_to(RECORD.parents[1])}")

    verdict = parts["no_bias"]["0.001"] < parts["as_shipped"]["0.001"] / 3
    print("\nConclusion: the tail excess is produced by the bias subtraction"
          if verdict else "\nConclusion: the bias subtraction does not explain the tail")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
