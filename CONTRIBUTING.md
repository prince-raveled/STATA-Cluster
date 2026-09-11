# MicroVerse — contributor guide

Multiverse analysis of microbiome differential abundance, behind a URL.
`MICROVERSE_SPEC_v2_idea2.md` is the **frozen** specification. Read it before changing anything.

## Scope guard (SPEC §21) — applies to every change

- **Two-group comparisons only.** No multi-group, no continuous outcomes, no ordinal. v2 work.
- **No longitudinal or paired designs.** No repeated measures.
- No read processing, no DADA2, no denoising, no taxonomic classification. Input starts at the abundance table.
- No alpha/beta diversity as standalone features.
- No contamination screening (that's Bioflow).
- No user accounts, no login, no saved projects.
- **No "best specification" export.** Ever (§18).
- Do not add a DA method beyond the seven in §10 without asking.

## Non-negotiables

- **Pipeline order (§9):** rarefy → collapse → prevalence filter → transform → test → FDR.
  Enforced in `app/core/preprocess.py::preprocess()`. Never reorder.
- **Rarefaction seeds are separate specifications, not averages** (§10).
- **Effect size is harmonised** (§14): significance comes from the method, effect size always
  from `harmonized_effect()` — log2 fold change of mean relative abundance. Never plot a native
  statistic on the specification curve.
- **Denominators (§15):** `frac_significant` and `sign_consistency` divide by `n_specs_tested`,
  never `n_specs_total`.
- Tier evaluation order is ROBUST → CONDITIONAL → FRAGILE → UNSTABLE → INSUFFICIENT,
  **first match wins**, with INSUFFICIENT short-circuiting first (§16.2).

## Layout

    app/core/models.py       Specification, TaxonResult (§13)
    app/core/parsers/        csv/tsv, biom v1+v2, qza, metaphlan, kraken/bracken
    app/core/validation.py   §8 hard constraints — clear messages, never a stack trace
    app/core/preprocess.py   §9 order + cached matrix builder
    app/core/validity.py     INCOMPATIBLE + RULES (§11)
    app/core/grid.py         enumeration + mode selection (§12)
    app/core/methods/        7 DA methods, vectorised across taxa
    app/core/effects.py      harmonized_effect (§14)
    app/core/fdr.py          BH / BY post-hoc
    app/core/robustness.py   tiers, metrics (§16.2)
    app/core/attribution.py  MixedLM + mandatory fallback (§17)
    app/core/runner.py       orchestration, caching, joblib
    app/core/report.py       methods paragraph, exports, ZIP

## Verification gates

`tests/test_worked_example.py` encodes SPEC §19. If it fails, the engine is wrong —
fix the engine, not the test.

`tests/reference/` holds the checks pytest cannot run in one environment: comparisons
against **the R packages themselves** (edgeR, ALDEx2, ANCOMBC), against scikit-bio and
conorm and analytic ground truth, real BIOM/QZA files written by `biom-format`, an
axe-core accessibility sweep, a concurrency test, the §23.5 compute benchmark, and the
§23 reproductions of Tierney et al. and Pelto et al. Run them with
`tests/reference/run_all.py`; the record is SPEC §24.2.

Five things to know before touching the methods. Each is a bug that was actually
shipped, found by diffing against the reference implementation, and is easy to
reintroduce because none of them change a correlation or a ranking:

- `tmm_factors` follows edgeR's `.calcFactorTMM` including the **rank-based** trim.
  Do not "simplify" it to a quantile trim — that is the bug the conorm comparison found.
- **ALDEx2 works in log2.** `clr_stack` uses `np.log2`, so `diff.btw` is a log2 fold
  change, and the effect type is `clr_diff_log2`. Switching it to natural log leaves
  every p-value identical and makes every effect size 31% too small.
- **ALDEx2's effect is the median of randomly paired between-group differences**, over
  pooled samples *and* Monte-Carlo instances — not the difference of the two group
  medians. The two are not the same on skewed data.
- **`estimate_bias` follows `ANCOMBC:::.bias_em` exactly**: heteroscedastic (each taxon
  brings its own regression variance), asymmetric free shifts `l1 <= 0 <= l2`, and
  `kappa1/kappa2` re-optimised each iteration. A simplified symmetric mixture still
  correlates at r = 1.0 with the reference but displaces delta, which shifts **every**
  log fold change by a constant.
- **ANCOM-BC's standard error is the regression standard error alone.** The bias
  variance is added only under the package's non-default `conserve = TRUE`.
- `run_ancombc` returns **natural-log** fold change. scikit-bio's column is named
  `Log2(FC)` but holds the same natural-log values; that label is theirs and is wrong.

## Interface conventions

Editorial, not dashboard. Paper ground, ink type, serif display, monospaced tabular
numerals, sharp rectangles. The indigo `--accent` is brand only — never a data colour;
the tier palette (`--robust` … `--undetected`) carries all data semantics. Plot colours
are read from the stylesheet via `MicroVerseCurve.theme()`, so never hard-code a colour
in JavaScript.

Anything derived from an uploaded file — taxon names above all — is escaped before it
reaches the DOM. A results URL is shareable, so an unescaped label is a stored-XSS
vector, not a self-inflicted one.

## Load and admission control

`app/limits.py` bounds concurrent runs and rate-limits the endpoints that start work.
Both are calibrated from `tests/reference/load_test.py`, not guessed. If you change
`MAX_CONCURRENT_RUNS`, re-run that test — the point of the bound is that runs stop
contending, and a bound above the core count does nothing.

Never rate-limit reads. A results URL is meant to be shared and must keep working.

## Commands

    .venv/Scripts/python -m pytest -q
    .venv/Scripts/python -m ruff check .
    .venv/Scripts/python -m uvicorn app.main:app --reload
