# MicroVerse

**A web server for multiverse analysis of microbiome differential abundance: run every
defensible analytical pipeline, report the distribution of answers.**

A microbiome differential abundance result depends on at least six analytical choices,
none of which has field consensus. Researchers make one set of choices, report one
p-value, and never show that a different set gives a different answer.

Tierney et al. (2022) fitted 6,035,110 models across 15 cohorts and found that one taxon
in three flipped the sign of its association, that over 90% of published T1D/T2D findings
were nonrobust, and that **84% of published associations could be recovered as
significant by picking the right model.**

MicroVerse enumerates that space of choices, executes it, and reports what happens.

> **Report the distribution, not your favourite point in it.**
> There is no "export best specification" button, and there never will be.

Implements `MICROVERSE_SPEC_v2_idea2.md` (frozen). Deviations are logged in the spec's
§24 and cross-referenced from the code.

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt
.venv/Scripts/python examples/make_examples.py
.venv/Scripts/python -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>, click a demo dataset, and run it. API docs at `/api/docs`.

With Docker:

```bash
docker build -t microverse . && docker run -p 8000:8000 microverse
```

## What it does

| Fork | Levels | Why it is contested |
|---|---|---|
| 1. Rarefaction depth | `none`, `min`, `1000`, `5000`, `10000` × 3 seeds | McMurdie & Holmes 2014 vs Cameron 2022 vs Schloss 2024 — twelve years, no consensus |
| 2. Prevalence filter | 0%, 5%, 10%, 20% | No standard for how rare is too rare |
| 3. Transformation | TSS, CLR, raw, TMM | Compositional school vs count-model advocates |
| 4. Taxonomic rank | genus, input rank | Wang 2023: resolution vs power |
| 5. DA method | Wilcoxon, Welch's t, logistic, linear, ANCOM-BC, ALDEx2, PyDESeq2 | Nearing 2022: 14 methods, 38 datasets, different answers |
| 6. FDR | BH@0.05, BH@0.10, BY@0.05 | Published studies use 0.05, 0.07, 0.15 inconsistently |
| 7. Covariates | every subset (covariate mode) | Tierney 2022: BMI, age, depth and gender changed size *and direction* |

Each rarefaction seed is its own specification, never averaged — averaging p-values
across subsampling draws is statistically wrong, and keeping them separate makes the
draw's own contribution measurable. On the demo data, **43% of the variance attributed to
rarefaction comes from the random draw rather than the choice of depth.**

### Three modes, never one crossed grid

Fully crossing all seven forks is over 500,000 specifications, which is not achievable in
web-request time. So the modes are separated:

| Mode | Varies | Typical size | Measured runtime |
|---|---|---|---|
| **Quick** (default) | forks 1–4, four elementary methods, FDR | ~1,600–3,200 valid | **~1 s** |
| **Full** | Quick + ANCOM-BC, ALDEx2, PyDESeq2 on a stratified matrix sample | ~1,700–3,400 valid | ~19 s |
| **Covariate** | fork 7 over every covariate subset, on a reference sub-grid | ~500–2,300 valid | ~5 s |

Runtimes are model fitting for the bundled demo datasets on 4 cores. The spec's budget
is 90 s for Quick and 10 minutes for Full. A complete run also apportions the variation
between the choices and writes the exports: roughly 30 s more on a laptop, and on the
public instance (one vCPU) a Quick run takes one to two minutes end to end, most of it
that attribution step. Each run records its own phase timings in the job summary.

### What comes back

- **A verdict sentence** — how many taxa are ROBUST / CONDITIONAL / FRAGILE / UNSTABLE.
- **Per-taxon robustness table** — `frac_significant`, `sign_consistency`, `frac_tested`,
  median log2FC and IQR, with a tier. Sortable, filterable, exportable.
- **Specification curve** — Simonsohn's two-panel plot per taxon: effect size for every
  specification sorted ascending, over a dot matrix of which fork level was active.
- **Choice attribution** — variance decomposed across forks, so you know which decision
  you have to justify in your methods section.
- **Locate my result** — declare the pipeline you would have reported, and MicroVerse
  tells you which percentile of significance it sits at.
- **A methods paragraph** with citations, and the full specification-level results.

### Robustness tiers

Each taxon gets exactly one tier from two numbers, counted only over the specifications
that actually tested it: the fraction in which it was significant, and the fraction
agreeing on the direction of its effect. The first rule that matches wins
(`assign_tier` in `app/core/robustness.py`):

| Tier | Rule |
|---|---|
| INSUFFICIENT | tested in fewer than 10 specifications, or its measurements are undefined — not tiered |
| ROBUST | significant in ≥ 80%, direction agreement ≥ 95% |
| CONDITIONAL | significant in 30–80%, direction agreement ≥ 95% (also: ≥ 30% with agreement 80–95%) |
| FRAGILE | significant in more than 0% and under 30%, direction agreement ≥ 80% |
| UNSTABLE | direction agreement below 80% |
| NOT DETECTED | never significant |

A tier measures how much the conclusion depends on analytical choice, not whether an
effect is real or large. How often each tier replicated on held-out data is on the
`/validation` page and in SPEC §24.6.

## Design constraints that are not negotiable

**Pipeline order (§9).** `rarefy → collapse → prevalence filter → transform → test → FDR`.
Rarefying after collapsing, or filtering before rarefying, gives different answers.
Enforced in one place: `app/core/preprocess.py`.

**Effect-size harmonisation (§14).** Seven methods return seven incompatible statistics,
which cannot share a y-axis. So significance comes from the method's own p-value, and
effect size is computed separately and identically for every specification as the log2
fold change of mean relative abundance. The pseudocount is fixed once per run — see §24.

**Denominators (§15).** `frac_significant` and `sign_consistency` divide by the number of
specifications in which the taxon actually survived the prevalence filter, never by the
run total. A taxon testable in only part of the grid is reported as `frac_tested` rather
than hidden in a denominator.

**Anti-cherry-picking (§18).** No "best specification" export, anywhere. Exports always
contain the full distribution — even the curve's point thinning takes an even stride so
the shape survives. If your declared pipeline is above the 75th percentile of
significance, the interface says so.

## Deployment

The public instance, <https://stata-cluster.vercel.app>, runs on Vercel's free plan from
the `vercel-migration` branch. FastAPI is a Python function; datasets and results live
in private Vercel Blob storage; a run is published to Vercel Queues and executed by the
subscriber in `app/queue_worker.py`; job records live in PostgreSQL. A small JavaScript
service in `blob/` signs the two things the Python SDK cannot: a browser's upload token,
so a table goes straight to storage, and a short-lived download URL for the exports too
large for a function's 4.5 MB response. Configuration, and how a release reaches
production, are in [deploy/README.md](deploy/README.md).

Everywhere else the defaults are a disk and a long-lived process — local storage,
inline runs, SQLite — which is what local development and the Docker image use. The
image is built and exercised on every push by `.github/workflows/ci.yml`, which starts
the container, runs a complete analysis through it and downloads the bundle.

```bash
fly launch --no-deploy && fly volumes create microverse_data --size 10 && fly deploy
```

`fly.toml` and `render.yaml` are ready; sizing guidance and every environment variable
are in [deploy/README.md](deploy/README.md). Two vCPU and 2 GB is the floor — a Quick
run at the §8 taxon ceiling needs about 25 s of CPU and ~300 MB.

MicroVerse bounds its own load: an in-process queue caps simultaneous runs
(`MICROVERSE_MAX_CONCURRENT`, default `min(4, cpus)`) with a bounded backlog, and a
per-client rate limit covers uploading and starting runs. Reading results is never
limited, so a shared link keeps working. `/healthz` reports queue depth and which
storage, job and database backends the process chose. On Vercel the same
`MICROVERSE_MAX_CONCURRENT` is the queue consumer's concurrency, which is global.

Both exist because of a measurement: unbounded, six simultaneous runs took 63 s each
against ~10 s alone. Bounded, the same six take 45 s and latency is predictable.

## Scope (§21)

Two-group comparisons only. No multi-group, continuous or ordinal outcomes; no
longitudinal or paired designs; no read processing, denoising or taxonomic
classification — input starts at the abundance table. No accounts, no login.

## Input formats

CSV/TSV, BIOM v1 (JSON) and v2 (HDF5), QIIME 2 `.qza`, MetaPhlAn merged tables,
Kraken2/Bracken combined reports. Orientation, value type and taxonomic rank are
detected, not assumed. Every rejection is a sentence explaining what to fix.

Hard constraints: ≥10 samples and ≥5 per group; ≥10 taxa; exactly two group levels;
≤1,500 taxa (collapsed to genus automatically when a taxonomy allows it); no negative
values.

## Layout

```
app/core/    the engine: parsers, preprocess, grid, validity, methods,
             effects, fdr, robustness, attribution, runner, report
app/routers/ upload, job, results, api
app/storage.py, db.py, jobs.py, queue_worker.py, uploads.py
             where bytes live, the job table, how a run starts, the queue
             consumer, direct-upload tickets -- each with a local and a Vercel side
app/templates, app/static   Jinja2 + HTMX + Alpine + Plotly + Tabulator, no build step
blob/        the JavaScript Blob signing service (Vercel only) and its tests
tests/       the pytest suite, including the SPEC §19 verification gate
examples/    the generator for three simulated cohorts with known spiked taxa
deploy/      Dockerfile support, lock generation, deployment notes
```

## Interface

An editorial research interface rather than a dashboard: a paper ground with ink type,
one indigo accent reserved for the brand, and the tier palette kept strictly separate
from it so a coloured row never reads as branding. Display type is a serif; labels and
every numeral are monospaced and tabular, so columns of results align down the page.
Geometry is rules and rectangles, not rounded cards.

There is no build step — Jinja2 templates, one stylesheet, HTMX for job polling, Alpine
for two form toggles, Plotly for the curve and Tabulator for the table. Plot colours are
read from the stylesheet at render time, so the specification curve and the page cannot
drift apart.

## Code quality

```bash
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m pytest -q
```

Ruff is configured in `pyproject.toml` at `--select F,E,W,B,SIM,UP,C4,RET,ARG,I` and the
tree is clean. The post-build audit and what it found — including a stored-XSS vector in
the results table, a response-header injection in downloads, and a denominator bug for
specifications that produce no rows — is written up in SPEC §24.1 E.

## Reviewing this

If you have been asked to assess this independently, start with
[REVIEWING.md](REVIEWING.md) rather than here. It names the five files that contain
every scientific decision, states where the risk actually is, and lists what has and
has not been validated. **No external review has taken place.**

## Validation

`pytest` checks MicroVerse against itself and against statsmodels. A second set of
checks compares it to *other people's* implementations and to analytic ground truth:

```bash
.venv/Scripts/python tests/reference/run_all.py
```

Anything unavailable is reported as SKIPPED with the reason, never silently passed.
The full record is SPEC §24.2. In summary:

| Component | Checked against | Result |
|---|---|---|
| multiplicative replacement | `skbio.stats.composition.multi_replace` | identical to 4.9e-17 |
| CLR | `skbio.stats.composition.clr` | identical to 1.8e-15 |
| TMM | **`edgeR::calcNormFactors` 4.10.1, in R** | **identical** — max relative difference 0.0000% |
| ALDEx2 | **`ALDEx2::aldex` 1.44.0, in R** | effect r = 0.998; p-value ranking Spearman 0.976; significant sets Jaccard 0.96 |
| ANCOM-BC | **`ANCOMBC::ancombc` 2.14.0, in R** | log fold change r = 1.00000; p-value ranking Spearman 0.994; Jaccard 0.93 |
| ANCOM-BC | `skbio.stats.composition.ancombc` | r = 1.0000, agrees to 0.007 |
| TMM, ALDEx2 | analytic ground truth | recovers known factors and a known CLR difference; 0% type I error |
| BIOM v1, BIOM v2, `.qza`, TSV | files written by `biom-format` 2.1.17 | exact agreement, all four readers |
| Accessibility | axe-core 4.10 (WCAG 2.1 AA) | zero violations on every page |
| Scale (SPEC §23.5) | 60 → 1,500 taxa | 25 s at the §8 ceiling, 3x inside the O6 budget |

These checks keep finding real defects. **TMM was trimming by quantile value where
edgeR trims by rank** — with tied M-values, common in microbiome counts, that discards
every tied observation at the boundary. **BIOM v2 and `.qza` had no tests at all** before
fixtures were generated with the reference library. And profiling the benchmark showed
the results bundle was **re-gzipping ~2.4M rows the run had already written**, doubling
the cost of every large job.

Installing R and diffing against the packages §10 actually names found four more, two of
which nothing else could have caught, because both shift results by a *constant* and so
are invisible to correlations and rank tests (SPEC §24.1 G):

- **ALDEx2 works in log2**; our CLR was natural log, so every ALDEx2 effect size was a
  factor of ln 2 too small. Regression slope against the package: 0.687. ln 2 = 0.693.
- **ALDEx2's `diff.btw` is the median of randomly paired between-group differences**, not
  the difference of the two group medians. On skewed sparse data those differ by up to
  3.0 CLR units.
- **ANCOM-BC's bias E-M is heteroscedastic with free asymmetric components.** Ours was a
  homoscedastic approximation, which put the bias term 0.417 natural-log units away — a
  1.5x fold change applied to every taxon, enough to move 44 of them across zero.
- **ANCOM-BC does not add the bias variance to its standard error** by default. Ours
  always did, inflating every standard error by ~5%.

One finding is about somebody else's code: **scikit-bio 0.7.3 labels its ANCOM-BC output
`Log2(FC)` but returns natural log.** Verified on a controlled 4x spike (true natural-log
1.314, true log2 1.896, both tools return 1.383).

## Real published data — and a finding

SPEC §23 validation 2 asks for the extension nobody has measured: turn on forks 1–5 and
quantify the *additional* instability. Run it with:

```bash
.venv/Scripts/python tests/reference/real_data_study.py
```

Seventeen published 16S case-control cohorts from [MicrobiomeHD](https://zenodo.org/records/1146764)
(Duvallet et al. 2017), downloaded on demand. **19,728 specifications and 2.9M
taxon-level results in 39 seconds** at genus, or **37,704 specifications and 56.3M
taxon-level results in 83 seconds** at OTU level (`--otu`), on one laptop — against
Tierney et al.'s 6,035,110 models on an HPC cluster over a single fork.

| Quantity | Genus (pooled, 95% CI) | OTU (pooled, 95% CI) | Tierney et al. |
|---|---|---|---|
| Taxa flipping association sign | **19%** (11–32) | **36%** (26–46) | 1 in 3 |
| Fraction of specifications nominally significant | 33% (detected) | 19% (detected) | 38% |
| Fraction FDR significant | 20% (detected) | 11% (detected) | 16% |
| Detected taxa reaching ROBUST | 4% | **0%** | — |

Between a fifth and a half of taxa change the *direction* of their association on
analytical choices alone, with no covariates varied. Tierney reached a similar number
varying covariates only. Different forks, different data, same place.

Figures are pooled — one taxon one vote — with a bootstrap over cohorts, not an average
of per-cohort percentages: cohorts here range from 40 taxa to 800, and averaging their
percentages inflated the genus sign-flip figure from 19% to 26%.

**Rank is not a neutral choice.** Running at OTU rather than collapsed genus roughly
doubles the sign-flipping and takes the fraction reaching ROBUST to zero. Collapsing to
genus — usually done for convenience — was stabilising the answers.

**Which fork drives significance — a claim this README used to make, and has withdrawn.**
An earlier version reported "rarefaction leads in 7 of 17 cohorts, DA method in 1" as a
finding. A sensitivity analysis added specifically to check it
(`tests/reference/attribution_sensitivity.py`) found the fork model explains a **median
1% of the within-taxon variance**, and that three independent estimators pick the same
leading fork in only **8 of 17** cohorts for significance (11 of 17 for effect size).
Only 3 of 17 cohorts meet both thresholds for a ranking worth reporting. The shares are
percentages of explained variance, and there is almost nothing to explain — so the
ranking was a decomposition of noise, and it is gone.

Fork attribution is now graded **exploratory** on every run, with the R² and the
estimator agreement shown next to the bars, because the question is still the right one
even though this answer is not yet trustworthy.

**What survives that check:** within rarefaction, **30% of the variance at genus (40% at
OTU) is the random subsampling draw, not the choice of depth** — visible only because
each seed is its own specification rather than averaged away. That is a direct ratio of
within-depth to between-depth variance on a balanced design: no model, no R², nothing
for estimators to disagree about.

Caveats, and what has since been done about each, are in SPEC §24.3.

Running on real data immediately exposed a parser defect that would have made
MicroVerse **unusable on MicrobiomeHD entirely**: every lineage there ends in a de novo
OTU id (`d__denovo5393`), and `d__` was aliased to `k__` for MetaPhlAn's leading domain
marker — so every table read as kingdom-rank and was refused as uncollapsible.

### What is still not validated

- **The Docker image is built in CI, not here.** `.github/workflows/ci.yml` builds it,
  starts it, runs a whole analysis through the container and downloads the bundle — so
  a green badge means the container works. It cannot be built on this machine: Docker
  Desktop needs administrator rights and so does `wsl --install`, and WSL is not
  installed. What is checked locally is the layer that actually fails — every
  dependency resolves to a Linux wheel for Python 3.12, and CI fails if
  `requirements.lock.txt` is stale.
- **The O6 budget is not verified on the live host.** The public instance runs on
  Vercel's free plan; `tests/reference/load_test.py` accepts `MICROVERSE_URL` but has not
  been run against it, which is what SPEC §20 asks for. Fly.io and Render are
  configured (`fly.toml`, `render.yaml`) but not deployed.
- **No browser testing outside Chromium.** The one risky CSS feature (`:has()`) now has
  a scripted fallback, so selection state shows even where it is unsupported.
- **ANCOM-BC's bias term still differs from R's by 0.020 natural-log units** — a
  Nelder-Mead stopping-rule difference between `nloptr` and `scipy`. Asserted on
  directly rather than left to drift.

## Tests

```bash
.venv/Scripts/python -m pytest -q
npm ci --prefix blob && npm test --prefix blob      # the Blob signing service
```

`tests/test_worked_example.py` encodes SPEC §19 — the tiers the engine must reproduce.
If it fails, the engine is wrong, not the test. It also checks that every taxon called
ROBUST on simulated data is a true positive.

The vectorised closed forms used for speed are verified against the statsmodels models
the spec names: `linear` against `sm.OLS`, `logistic` against `sm.GLM(Binomial)`, and the
harmonised effect against PyDESeq2's own log2 fold change (§14's stated sanity check,
r > 0.9).

## Validation against published results (§23)

All five of the spec's validation experiments now run.

| # | Experiment | Status |
|---|---|---|
| 1 | **Reproduce Tierney et al.** on their cohorts, Mode C | **done** — 28.7% sign-flipping over adjustment sets (theirs: 1 in 3) and 97.9% of T1D/T2D associations not ROBUST (theirs: >90%), over 8 curatedMetagenomicData contrasts. SPEC §24.4 |
| 2 | **Extend past their grid** with forks 1–5 | **done** — above, and SPEC §24.3 |
| 3 | **Reproduce Pelto et al.** — elementary methods give tighter specification distributions | **done** on their own 60 curated cohorts and their own split halves — and the answer is *partly*: true of the prevalence-filter fork, reversed on the rarefaction fork, and null on split-half replicability once detection is equalised. SPEC §24.5 |
| 4 | **Simulated ground truth** | `tests/test_worked_example.py` and `examples/make_examples.py` — spiked taxa land ROBUST, nulls do not |
| 5 | **Compute benchmark** | `tests/reference/benchmark.py`, curve in `docs/benchmark.json` |

Reproducing 1 and 3 needs R, which is now installed alongside Bioconductor:

```bash
# validation 1 — Tierney's cohorts, from curatedMetagenomicData
Rscript tests/reference/tierney_export.R data/cmd/export
.venv/Scripts/python tests/reference/tierney_study.py

# validation 3 — Pelto's own curated cohorts, from Zenodo
.venv/Scripts/python tests/reference/pelto_fetch.py
Rscript tests/reference/pelto_export.R data/pelto/data_171023.rds data/pelto/export
.venv/Scripts/python tests/reference/pelto_study.py
```

Both are registered in `tests/reference/run_all.py` and report SKIPPED with install
instructions when R or the cached cohorts are absent.

**Validation 3 exposed something about how replicability gets measured.** On Pelto's own
split halves, a call-conditioned replication rate rates the elementary methods far
higher (0.774 vs 0.436) — but at these half-sizes the median elementary method calls
*nothing* in a half, so it is scored only on the pairs where it happened to find
something, while PyDESeq2 calls 26 taxa and is scored on nearly all of them. Compared at
equal detection — the overlap of each method's top 20 taxa between halves, scored on
every pair — the two families are indistinguishable: **0.326 vs 0.321, p = 0.39**.

**Validation 1 exposed something about the design worth knowing.** SPEC §14 harmonises
the effect size on purpose: significance comes from the method, the effect from one
method-independent estimator computed from the matrix. That estimator never sees the
adjustment set — so the harmonised effect is *identical* across covariate subsets, and
its standard deviation over 32 adjustment sets is exactly 0.0 while the linear model's
own coefficient moves with a standard deviation of 0.65. **MicroVerse's specification
curve cannot show covariate-driven sign instability**; that instability appears in
significance and in the native effect. Tierney vibrated coefficients, so the comparable
quantity is the native one, and on it the engine reproduces their number.

## Licence

MIT.

## What this is not

Not a new statistical method — it orchestrates published methods inside a published
framework. Not truth: a taxon robust to analytical choice is not thereby biologically
real, because confounding, contamination and batch effects survive a multiverse intact.
Not exhaustive: the grid is bounded by what is defensible and computable, and every run
reports how many specifications it enumerated, how many it pruned, and why.
