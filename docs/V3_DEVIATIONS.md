# v3 deviation log

Every place the implementation departs from the letter of `docs/MICROVERSE_V3_PLAN.md`,
in the style of v2's SPEC §24.1: what was found, what was done, and where. Entries are
grouped by phase and never deleted; a reversed decision gets a new entry.

## Phase 0 — weighting, pruning R8–R9, signed z (plan §26)

### A. Scientific choices the plan left open

**A1 — R9 cites McMurdie & Holmes 2014 and Weiss et al. 2017, not the MaAsLin 3
workflow.** Plan §26.3 justifies R9 with "the OMA MaAsLin 3 workflow adjusts for total
reads". That source could not be verified, so it is not cited. The register cites the two
papers that establish that library size confounds detection: McMurdie & Holmes (2014) and
Weiss et al. (2017). `app/core/validity.py::DEFENSIBILITY`.

**A2 — A declared pipeline outside the grid is explained, under both rule sets.** v2
dropped a declared pipeline that its own rules pruned without a word, and the reader
looked for a percentile that was never going to appear. It now gets a note naming the
rule, under v2 and v3 alike. This does not change any v2 golden output (the demos declare
no pipeline). `app/core/grid.py::enumerate_grid`.

**A3 — V8's primary comparison is decision_tree against uniform, both under rule set
v3.** The plan's success criterion ("decision_tree AUC non-inferior to uniform") does not
say which rules the uniform labels use. Both are v3 so the difference is the weighting
alone; the v2-rules baseline is reported beside them. `tests/reference/weighting_validation.py`.

**A4 — V8's non-inferiority test is the lower limit of a 95% paired cluster-bootstrap
interval.** The plan gives the margin (0.03) but not the interval. The difference
AUC(decision_tree) − AUC(uniform) is bootstrapped by resampling cohorts, both AUCs on the
same draw; non-inferiority holds if the lower 2.5% limit exceeds −0.03. Fixed before any
data were seen (V8 has not been run).

**A5 — V8 scores each labelling against its own discovery direction.** A taxon's weighted
median effect can differ in sign from its unweighted one, and the label is attached to
the weighted one, so "replicated in the same direction" uses each labelling's own
direction.

**A6 — V8's Kendall tau is computed two ways.** The plan names the metric but not what it
is computed on. τ-b is reported on the tier order NOT DETECTED < UNSTABLE < FRAGILE <
CONDITIONAL < ROBUST (INSUFFICIENT dropped) and on `frac_significant`, for every pair of
labellings. Descriptive only; no criterion attaches to it.

**A7 — V8's held-out half stays on rule set v2.** Its `reference` replication definition
is one pre-specified analysis, unrarefied raw counts with Wilcoxon — exactly what R8
prunes. Running the held-out half under v3 would change the outcome along with the labels
predicting it.

**A8 — Weight stability covers the built-in schemes only.** `weight_stable` compares
`uniform`, `flat_tree` and `decision_tree`. `practice` joins it when the practice review
exists (plan §28); a user's `custom` scheme is reported but does not define stability.

### B. Records, hosting and reproducibility

**B1 — The v2 golden record is per platform, and the Linux set was regenerated from
8d1cd9f.** The golden hashes committed in a45e243 were written on Windows. On Linux the
v2 engine at 8d1cd9f, given byte-identical demo inputs, does not write the same bytes.
Of its 18 outputs, 6 (three `verdict.txt`, three `grid.json`) are identical as written;
8 CSV files match once their LF line ends are converted to CRLF, because pandas ends CSV
lines with the platform's separator; and 4 differ in content: the three
`results_long.csv` tables and the covariate-mode `robustness.csv`. The last was stored
on Windows, so it can be read: one row differs, Genus_102's `frac_nominal` (0.16518
against 0.16369, i.e. 3 of its 2,016 p-values on the other side of 0.05), which is
last-bit floating-point difference in the covariate models' linear algebra. The long
tables were stored on Windows only as hashes, so where they differ cannot be inspected.
`sha256.json` therefore holds one set per `sys.platform`, each written by 8d1cd9f on that
platform: the Windows set is the original; the Linux set was written from a git worktree
at 8d1cd9f with `tests/fixtures/make_v2_golden.py` copied in and its `ruleset=` argument
removed; the readable copies are the Linux output. The test compares against its own
platform's set and skips, saying why, on a platform with none. With rule set v2, all 18
v3 outputs on Linux are byte-identical to 8d1cd9f's on Linux.

**B1a — ...and per Python version.** Once Phases 0-1 reached `vercel-migration`
(951ce21), CI's Python 3.14 job passed this test against the Linux set while its 3.12 and
3.13 jobs failed, with the same pinned packages; the image build, also Python 3.12, failed
at the same test (`-x` stopped there, 633 passed). So bytes depend on the Python version
as well as the platform. The sets are now keyed by platform and Python minor version: the
Linux set as `linux-py3.14`, the version CI reproduces it under, and the Windows set as
`win32-py3.14`, the version it was written with. Where no set exists for the environment —
CI's 3.12 and 3.13 jobs and the image, for now — the test skips and names the missing
environment rather than failing. The claim the test makes is unchanged: under rule set v2,
v3 reproduces the v2 engine exactly in an environment where the v2 engine was recorded.
Recording `linux-py3.12` and `linux-py3.13` from 8d1cd9f would restore the check there.

**B1b — CI checks the claim against the v2 engine itself, not recorded hashes.** At
1c51084 the CI Python 3.14 job (3.14.7) failed the same test that passed there at
951ce21: `results_long.csv` differed for all three demos while every output compared
before it matched. Nothing in the code under test had changed, so the runner's
environment had. Recorded hashes cannot be checked on a runner that changes underneath
them. `tests/reference/compare_v2_engine.py` now runs the v2 engine (8d1cd9f, from a
temporary git worktree) and this tree with rule set v2 in one environment, on the same
demo files, and compares all 18 outputs byte for byte; CI runs it as its own job
(`v2-reproduction`, Python 3.14, full history). On Windows (Python 3.14.6) all 18 are
identical. The Linux hash set was removed, so the pytest check skips on Linux and names
the script; the Windows set stays for local runs.

**B2 — The weighting note is linked in the repository, at the deployed commit.** `docs/`
is not deployed (`vercel.json` excludes it, and deployment settings are not changed
here), so /about links to `docs/v3_weighting_note.md` on GitHub at
`VERCEL_GIT_COMMIT_SHA` when it is set and at the default branch otherwise.
`app/config.py::source_url`.

**B3 — The V8 record is copied into `app/core/records/`.** For the same reason the site
cannot read `docs/weighting_validation.json`. The script writes both, only on the full
pre-registered design (smaller runs write under the git-ignored `data/`), and
`tests/test_v3_outputs.py` fails if the two copies differ. v2's convention — numbers
typed into `app/core/evidence.py` and checked by a drift test — is not used for V8,
because every V8 number shown must be written by the script.

**B4 — The V8 margin is defined once, in `app/core/evidence.py::V8_PREREGISTRATION`.**
The script reads it from there, so the criterion the Evidence page states and the one
the script tests cannot differ.

**B5 — `run_all.py` finds the interpreter on Linux.** It hard-coded
`.venv/Scripts/python.exe`; it now uses `.venv/bin/python` where that is the layout.

### C. Presentation

**C1 — Replication evidence is keyed by labelling.** v2's rates were measured on labels
assigned with one vote per specification under the v2 rules. A v3 run's tiers are a
different labelling, so no page, export or API response shows v2's rates beside them:
the tier cards say "replication not yet measured for these labels", the evidence panel
is graded "Internally verified", the taxon page and glossary say the weighted labels are
untested, and the manifest records which labellings have been measured. A run stored
before v3 still shows v2's rates, which were measured on exactly its labels.
`app/core/evidence.py::replication_by_labelling`.

**C2 — The Method page's pruning table is the defensibility register.** It replaces the
two-column list of v2 rules with every rule (R1–R9) and every fork, each with its reason,
type and sources. The old `app/ui.py::pruning_rules` is removed.

**C3 — Rule numbers appear beside pruning reasons on the results page.** `rule_id` maps
each engine reason to its register id, so "R8" on a run links to its entry.

**C4 — The glossary's tier entry reads its rates from the evidence record.** They were
typed text; they are now formatted from `TIER_REPLICATION`, which is checked against
`docs/tier_validation.json`, and qualified as the first version's labels.

**C5 — Kish (1965) and Weiss et al. (2017) join the reference list.** The methods
paragraph of a v3 run names Kish's effective sample size and cites Weiss et al. for R8
and R9; neither is in the plan's reference list. Kish is a book with no DOI; Weiss et al.
was already cited in the register by a45e243. Neither is marked [verify] in the plan.

### D. Tests that assumed the v2 rules

The full suite at a45e243 had 8 failures on Linux: the three golden tests (B1) and five
tests written for v2's behaviour.

- `test_validity.py::test_transform_agnostic_methods_accept_every_transform` states
  SPEC §11 literally (Wilcoxon, Welch and linear accept every transform). That is the v2
  rule set, so it now passes `ruleset="v2"`; a new v3 test checks that R8 removes exactly
  unrarefied raw counts.
- `test_validity.py::test_capabilities_prune_impossible_specifications` is about dataset
  capabilities, not rules. Its default specification (unrarefied raw Wilcoxon) is pruned
  by R8 before any capability is checked, so it now uses TSS; the raw-count case is
  checked under v2.
- `test_robustness.py::test_denominators_use_n_specs_tested_not_total` compared a v3
  run's `frac_significant`, now a weighted share, with an unweighted mean. It now checks
  the weighted share over exactly the specifications that tested the taxon, and the
  one-vote share in `frac_significant_unweighted`; a copy on a v2 run keeps the original
  check.

### E. Not done in Phase 0, and why

**E1 — V8 has not been run.** Its data are Pelto et al.'s curated cohorts, fetched from
Zenodo (record 15047338). In the build environment for this phase the network policy
refused zenodo.org, so the archive could not be downloaded. Nothing was substituted: the
script, its registration in `run_all.py`, the record format, the site's rendering of it
and the tests of its statistics are in place, and the site says V8 has not been run.
R 4.3.3 is installed for the export step once the archive is available.

## Phase 1 — calibrated multiverse inference (plan §27)

### A. Scientific choices the plan left open

**A1 — Depth-adjusted presence/absence uses the score test in the calibrated family.**
Plan §27.3 step 3 says the R9 logistic model is "handled by a residualised score". The
uncalibrated grid reports a Wald test from an iterative fit, which would need one fit per
permutation. The calibrated engine therefore uses the score test for group in
presence ~ 1 + group + log depth, whose null model does not involve the labels and is
fitted once; the same statistic is used for the observed and the permuted labels, as
validity requires. Its observed p-value differs from the Wald p-value the grid shows for
those specifications. It equals statsmodels' GLM score test (`tests/test_inference.py`).
The 1e-10 equality of plan §27.6 is held for the four closed-form tests (Wilcoxon, Welch,
linear, 2×2 logistic).

**A2 — Specifications differing only in their FDR setting are one hypothesis.** They share
one fit and one p-value, so they are one member of the family for the maxT and are
rejected together; their weights still add up in the certified share. Per-specification
FDR is what the calibrated procedure replaces.

**A3 — The within-family threshold is q·R/m whichever discovery procedure is chosen.**
Plan §27.3 step 7 states it for BH selection; BY and e-BH selections use the same level.

**A4 — e-BH uses the p-to-e calibrator e = κ p^(κ−1) with κ = 1/2**, fixed before any
data were seen. The plan names p-to-e calibration but not the calibrator.

**A5 — The generalised Pareto tail is accepted by a Cramér–von Mises test.** The number of
exceedances starts at 250 (or B/4) and falls by 10 until the fit is not rejected at 0.05,
down to 10; with no acceptable fit the raw permutation p-value is kept. Parameters are
estimated from the same exceedances the test uses, so the test is approximate. A
tail-fitted p-value below 1e-12 is reported as 1e-12: an extrapolation that far is not a
measurement, and no threshold in use (q/m > 1e-6) can depend on it.

**A6 — |z| is taken directly for the normal-based tests, and skipped where it cannot
matter for the t-based ones.** For Wilcoxon and the logistic tests |z| is the normal
deviate itself, which equals Φ⁻¹(1 − p/2) up to rounding, and is computed the same way
for observed and permuted labels. For Welch and the linear model, Student's t has heavier
tails than the normal for every df > 0, so |z| ≤ |t|; a permuted cell whose |t| does not
exceed the running maximum cannot raise it and is not converted. The maxima are exactly
those of full evaluation (tested), and calibration on the IBD demo took 20 s rather than
61 s for B = 2,000 with two BLAS threads in the build container.

**A7 — Calibration covers the Quick grid only.** Plan §27.4 lists ALDEx2 as calibrated on
the web as well; a calibrated run is a Quick run, which has no ALDEx2, so ALDEx2's
vectorised calibration is deferred. Covariate-mode grids are not calibrated (plan §27.3,
v3.1). Specifications outside the calibrated family are counted in the manifest.

**A8 — The running weighted significance share is not streamed.** Plan §27.3 step 4 keeps
it beside the running maximum; nothing in steps 5–10 uses it, so it is not computed.

### B. Interface and records

**B1 — "calibrated" is a job type, not a fourth grid.** It is chosen by a checkbox under
Quick on the configure page (or mode `calibrated` in the API), recorded as the job's mode,
and runs the Quick grid (`config.ENGINE_MODE`). The progress page shows one more stage.

**B2 — The permutation matrix is recorded, not stored.** The manifest gives the number of
permutations, the seed, the generator and a SHA-256 of the (B + 1) × n label matrix,
which is enough to regenerate it exactly and prove it was the same one. The plan says to
store it; regenerating from the seed is equivalent and keeps the bundle small.

**B3 — Calibration is graded "Internally verified" on the Evidence page.** Its tests show
it is computed as specified; nothing has yet measured its error rate or power.

**B4 — One test pinned the list of modes.** `test_web.py::test_api_info_declares_the_policy`
asserted that `/api/info` offers exactly Quick, Full and Covariate; it now expects the
calibrated job type too.

### C. Not done in Phase 1, and why

**C1 — V9 has not been run.** The plan stops Phase 1 before it, and it is to be run where
its data are: Zenodo, which holds the Pelto cohorts, is not reachable from the build
environment. Its script is written (section D below); until its record exists, every
page, the methods paragraph and the manifest describe CERTIFIED ROBUST as not yet
validated on real data (`app/core/evidence.py::certified_status`).

**C2 — The null-uniformity check and the compute benchmark have not been recorded.** Both
scripts exist and are registered in `run_all.py`: `tests/reference/calibration_null.py`
(200 simulated null tables, KS test of the family p-values against Uniform(0, 1), the
§27.6 criterion) and `tests/reference/calibration_benchmark.py` (the §27.4 budget,
written into `docs/benchmark.json`). This phase was limited to code and tests, so neither
record was written; the 20 s above is a development measurement, not a record.

### D. V9's script (`tests/reference/v9_calibration.py`)

Written to plan §33 and not run. Choices the plan leaves open, fixed before any data:

**D1 — Datasets.** Pelto: the 25 cohorts of the v2 tier validation (at least 40 samples in
the smaller group), control samples only. Nearing: every dataset of the paper's filtered
run with at most 1,500 features (the web limit) and at least 20 samples in its reference
group, taken as the grouping level with more samples, because Nearing's groupings are not
all case/control and a mock comparison needs one homogeneous group. The plan's "24
Nearing datasets" is the expected count, not a filter; the record gives the actual one.

**D2 — Metrics.** FDR is the mean over replicates of V/max(R, 1) among BH discoveries.
"Within-family FWER" is the share of non-planted taxa whose family test rejects at 0.05:
the family test is single-step maxT over the taxon's specifications, so that is exactly
"some specification of this taxon rejected". Power is the mean share of planted taxa
discovered; CERTIFIED ROBUST precision is planted taxa among all certified, pooled.
Criteria are judged on point estimates; 95% intervals are reported beside them (bootstrap
over datasets for (a) and (b), over replicates for (c)).

**D3 — Part (a).** 4 random halves of each dataset's reference samples. Uniformity is tested
with Kolmogorov–Smirnov on one family p-value per split (the most prevalent taxon's), so
the values are independent; KS p above 0.01 passes, as in §27.6.

**D4 — Part (b).** 2 replicates per dataset and scenario. Planting multiplies proportions
in one half and redraws every sample's counts as a multinomial at its own library size
(both halves redrawn). "standard": up to 10 taxa, present in at least 25% of samples and
outside the 5 most abundant, fold change 2 or 4 up or down. "dominant_shift": the same
plus the 3 most abundant taxa at fold change 4. Only "standard" carries the FDR criterion;
"dominant_shift" is reported, as the plan says.

**D5 — Part (c) uses a Dirichlet-multinomial simulator.** The plan asks for MIDASim-style
simulations; the MIDASim port (§31.1, `app/core/sim.py`) is Phase 5 and does not exist
yet. The grid is n per group 20, 40, 80; sparsity low or high (Dirichlet concentration
500 or 50); fold change 1.5, 2, 4; 10 planted taxa; 10 replicates per cell. The record
names the simulator, and (c) is to be rerun with MIDASim once it exists.

**D6 — B = 2,000 permutations**, the web setting being validated; the plan's 20,000 is
for offline Atlas runs.

**D7 — Resumable.** Replicates are cached in `data/v9_calibration_rows.jsonl` with the code
commit, so a long run can be interrupted; a changed commit starts again.
