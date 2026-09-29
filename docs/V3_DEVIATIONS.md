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
