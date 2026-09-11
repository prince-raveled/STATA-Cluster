# Reviewing MicroVerse

Written for someone asked to assess this independently. It says where the scientific
risk actually is, what has and has not been validated, and which files to read in which
order. It does not try to sell you anything.

**No external review has taken place.** This document exists to make one possible.

---

## 1. What the system claims to do

Take one microbiome abundance table and a two-group label. Run every combination of
seven analytical choices that a competent researcher could defend. Report, per taxon,
the distribution of answers rather than a single p-value — plus which choice is
responsible when the answer moves.

It does **not** claim to tell you whether a taxon is really different between groups. It
claims to tell you how much that conclusion depends on how you asked.

---

## 2. Read these five files first

In this order. Together they are about 1,200 lines and contain every scientific decision.

| File | Why it matters |
|---|---|
| `app/core/preprocess.py` | The pipeline order (rarefy → collapse → filter → transform) is enforced in `MatrixBuilder.build`. Reordering it silently changes every result. Also holds `tmm_factors`, which is the one preprocessing step with a non-obvious algorithm. |
| `app/core/effects.py` | 50 lines, and the most consequential file in the repo. `harmonized_effect` is what the specification curve plots for every method. Read §7 of this document before judging it. |
| `app/core/robustness.py` | `assign_tier` — six tiers from two numbers, first match wins. The denominators are the part that silently corrupts everything if wrong. |
| `app/core/validity.py` | Which fork combinations are refused, and why. Every rule is one line with a stated reason. |
| `app/core/attribution.py` | The weakest part of the system, and the docstring says so at length. |

Then `app/core/runner.py` for orchestration, and `tests/test_worked_example.py`, which
encodes the specification's own worked example and is the gate that fails if the engine
is wrong.

---

## 3. Data flow

```
upload  →  parse  →  validate (§8 floors)  →  readiness (advisory)
                                    ↓
                          enumerate specifications
                                    ↓
                       prune incoherent ones (§11, with reasons)
                                    ↓
              for each matrix:  rarefy → collapse → filter → transform
                                    ↓
              for each method:  fit  →  p-value + native statistic
                                    ↓
                       harmonised effect (from the matrix, not the model)
                                    ↓
                          FDR, per specification
                                    ↓
             aggregate per taxon  →  tier  →  per-choice instability
```

Two things about this diagram carry most of the scientific risk:

- The harmonised effect branches off the **matrix**, not the model. It therefore does
  not depend on the method or on covariate adjustment. See §7.
- FDR is applied **within** each specification, across the taxa that survived that
  specification's prevalence filter. So the multiple-testing burden itself varies
  across the grid — which is a real effect, not an artefact.

---

## 4. The seven choices, and how many levels each has

| Choice | Levels | Notes for a reviewer |
|---|---|---|
| Rarefaction | 13 | No rarefaction, plus 4 depths × **3 random seeds**. Seeds are separate specifications, not averaged — this is deliberate and is what makes the draw's contribution measurable. |
| Taxonomic rank | 1–2 | Only 2 if lineages are supplied. Most public 16S tables give 1, which shrinks the grid. |
| Prevalence filter | 4 | 0%, 5%, 10%, 20%. Changes the multiple-testing burden as well as which taxa are tested. |
| Transformation | 4 | raw, TSS, CLR, TMM. |
| Statistical test | 7 | Four per-taxon, three whole-table. **Quick mode runs only the four per-taxon tests.** |
| Multiple testing | 3 | BH 0.05, BH 0.10, BY 0.05. Applied post-hoc to stored p-values, so nearly free. |
| Covariates | up to 2ᵏ, capped at 64 | Covariate mode only. |

For the `gut_species` demo in Quick mode this is 4,992 enumerated → **3,192 valid** after
1,152 (TMM on rarefied counts) and 648 (logistic on a transform that cannot affect it)
are ruled out. 1,064 model fits. The arithmetic is exact and worth checking.

---

## 5. Where the bodies are buried

Things a reviewer should specifically interrogate, listed because they are easy to miss.

**The specification count is not an evidence count.** 3,192 specifications are 3,192
correlated re-analyses of the same samples. "Significant in 86% of specifications" is a
statement about analytical stability, not a probability, not a p-value, and not 3,192
replications. The effective number of independent analyses is much smaller and is **not
computed anywhere**. If you think the interface implies otherwise at any point, that is a
bug worth reporting.

**Quick mode excludes three of the seven tests.** ANCOM-BC, ALDEx2 and PyDESeq2 never run
in the default mode. Full mode runs them on a *sample* of matrices (e.g. ALDEx2 on 45 of
216, stated in the interface). Nothing is described as exhaustive that is not.

**The harmonised effect is covariate-invariant.** Measured directly, its standard
deviation across 32 adjustment sets is exactly 0.0, while the linear model's own
coefficient moves with SD 0.65. This is a design property of §14, not a bug — but it
means the specification curve *cannot* show covariate-driven instability. That
instability appears in significance and in the native statistic. This is the single most
likely thing for a user to misread.

**Fork attribution is unreliable and labelled so.** A sensitivity analysis
(`tests/reference/attribution_sensitivity.py`) found the fork model explains a median
**1%** of the within-taxon variance, with three estimators agreeing on the leading fork
in only 8 of 17 cohorts. A headline claim previously built on it was withdrawn from both
the specification and the README. If you find the interface presenting it as a finding
anywhere, that is a regression.

**ROBUST rests on few cohorts.** In the tier validation, 49 of 60 ROBUST calls came from
one cohort with an unusually large effect. Dropping it leaves 11 taxa replicating at 73%
rather than 90% — the ordering survives, the rate does not. The confidence interval
(60–93%) reflects this and the interface shows the interval.

**Split-half is not external replication.** Both halves share protocol, population,
batch and sequencing run. The measured replication rates are an upper bound on what
independent collection would give.

---

## 6. Validation status, honestly

Four kinds of evidence, deliberately kept apart in `app/core/evidence.py` and rendered at
`/validation`:

| | Meaning | Where it applies |
|---|---|---|
| **Internal** | Implementation matches the specification. Says nothing about whether the specification is a good idea. | Harmonised effect; tier implementation |
| **Reference** | Agrees with an independent implementation or a published result. | TMM (identical to edgeR), ALDEx2, ANCOM-BC, CLR, parsers, Tierney reproduction |
| **Empirical** | Predicts something on data it never saw. | **Robustness tiers only** |
| **Exploratory** | Correct arithmetic, unvalidated. | Fork attribution |

The empirical result, in full: 25 published cohorts, 75 stratified discovery/validation
splits, 12,564 taxon observations. Tiers assigned on the discovery half, replication
scored blind on the held-out half. ROBUST 90% (CI 60–93), CONDITIONAL 51%, FRAGILE 12%,
UNSTABLE 2%. AUC 0.785. Label-permuted null 0.7%. Two independent definitions of
replication agree (AUC 0.785 and 0.773).

Reference validation found four shipped bugs, two of which shift results by a *constant*
and were therefore invisible to every correlation-based check that existed:
ALDEx2 working in log2 rather than natural log, its `diff.btw` being a median of paired
differences rather than a difference of medians, ANCOM-BC's bias E-M being a
homoscedastic approximation, and its standard error wrongly including the bias variance.

---

## 7. Reproducing everything

```bash
.venv/Scripts/python -m pytest -q                       # 307 tests
.venv/Scripts/python -m ruff check .
.venv/Scripts/python tests/reference/run_all.py         # every external check
.venv/Scripts/python deploy/verify_image.py             # container, minus the daemon
```

`run_all.py` reports SKIPPED with install instructions when a dependency is absent — it
never silently passes. The R comparisons need R with edgeR, ALDEx2 and ANCOMBC; the
cohort studies download public archives into a git-ignored cache.

One check is expected to **fail**, and that is the documented result:
`attribution_sensitivity.py` asks whether fork attribution is stable enough to report and
answers no. The runner marks it RECORDED rather than FAILED, and shouts if it ever starts
passing, because that would mean the documented claim needs revisiting in the other
direction.

---

## 8. Known limitations

Not a list of future work — these are things that are currently true.

- No external code review, no third-party security audit, no independent users.
- The container image has never been built on the development machine (no runtime
  available; installing one needs administrator rights). It builds and runs in CI, and
  `deploy/verify_image.py` checks everything a daemon is not required for.
- No browser testing outside Chromium.
- All performance numbers come from a single laptop and are labelled as such.
- ANCOM-BC's bias term differs from R's by 0.020 natural-log units (~2% on the
  fold-change scale), traced to `nloptr` and `scipy` stopping their optimiser at
  different points. Asserted on so it cannot drift; not eliminated.
- Two-group comparisons only. No longitudinal, paired, multi-group or continuous
  outcomes. No read processing, no taxonomic classification, no diversity metrics.
- Specification dependence is acknowledged but not quantified.

---

## 9. What would change our mind

Concretely, the findings that would falsify parts of this:

- **Tier validation.** If ROBUST's advantage over CONDITIONAL disappears on cohorts
  outside Pelto's collection, or if the ordering fails under a third definition of
  replication, the empirical grade for the tiers should be withdrawn.
- **Harmonised effect.** If a reviewer shows a case where its covariate-invariance
  produces a materially misleading curve that the interface does not warn about, that
  is a design fault, not a documentation gap.
- **Attribution.** If a better-specified model (interactions, a proper binary link,
  more clusters) achieves a usable R² and stable estimator agreement, the withdrawn
  claim could be revisited — but it would need to be re-earned, not restored.

---

## 10. Repository layout

```
app/core/        the engine — statistics, no web framework
app/routers/     HTTP, thin
app/templates/   Jinja2; no build step, no bundler
tests/           307 automated tests
tests/reference/ checks that need R, network, or a second environment
docs/            the JSON records every validation claim is read from
deploy/          Dockerfile support, lock generation, image verification
MICROVERSE_SPEC_v2_idea2.md   the frozen specification, plus §24 — the deviation
                              log, validation record and every experiment write-up
```

`MICROVERSE_SPEC_v2_idea2.md` §24.1 lists every place the implementation departs from the
specification and why. §24.6 is the tier validation. §24.3 contains a withdrawn claim,
left in place with the reason for its withdrawal rather than deleted.
