# MICROVERSE — Master Specification v2.0 (FROZEN)

**A web server for multiverse analysis of microbiome differential abundance: run every defensible analytical pipeline, report the distribution of answers.**

Version 2.0 — implementation-ready
Date: September 2026
Status: **FROZEN.** Supersedes v1.0. Build this; do not redesign.

> **Change policy.** Design decisions here are final. Fix genuine errors in code and log them in §24 — do not restart the design. Scope changes are refused by default (§21).

---

# CONTENTS

**I — Justification** §1–5 · **II — Design** §6–8 · **III — The grid** §9–12
**IV — Engine** §13–15 · **V — Outputs** §16–18 · **VI — Build** §19–22
**VII — Assignment** §23 · **Appendix** §24–27

---

# PART I — JUSTIFICATION

## 1. The problem

A microbiome differential abundance result depends on at least six analytical choices. None has field consensus. Researchers make one set of choices, report one p-value, and never show that a different set gives a different answer.

## 2. Six forks, each contested — with the citation for why

| Fork | The argument | Key sources |
|---|---|---|
| **1. Rarefaction depth** | 12-year unresolved fight. McMurdie & Holmes 2014 called rarefying "statistically inadmissible." Cameron et al. 2022 showed it guarantees Type I error control in permutation tests. Schloss 2024 argued it is currently the best approach. A June 2026 *Front Bioinform* benchmark states there is still no consensus. | 8, 9, 10 |
| **2. Prevalence filter** | No standard exists for how rare is too rare. Changes both the taxa tested and the multiple-testing burden. | 2, 3 |
| **3. Transformation** | Gloor's compositional school: CLR is not optional. Count-model advocates: raw counts with a proper model. Both publish continuously. | 3, 5 |
| **4. Taxonomic rank** | Wang 2023 showed DA results depend heavily on analysis unit, with a real resolution-vs-power trade-off. | 11 |
| **5. DA method** | Nearing et al. 2022: 14 methods, 38 datasets, materially different results. Pelto et al. 2025: still no consensus, and elementary methods beat sophisticated ones on replicability. | 2, 3 |
| **6. Covariate adjustment** | Tierney et al. 2022: BMI, age, sequencing depth and gender all changed both size *and direction*, disease-specifically. | 1 |
| *(+ FDR method/threshold)* | Tierney noted studies use 0.05, 0.07, 0.15, BH vs BY inconsistently. | 1 |

## 3. The damage, quantified

Tierney et al. 2022, testing 581 literature-reported associations across 15 cohorts:

| Finding | Value |
|---|---|
| Models fitted | **6,035,110** |
| Taxa flipping association sign substantially | **1 in 3** |
| Published T1D / T2D findings that were nonrobust | **>90%** |
| Mean fraction of models nominally significant per taxon | **38%** |
| Mean fraction FDR significant per taxon | **16%** |
| Reported associations with ≥1 nominally significant model | **488 / 581 (84%)** |
| Reported associations with ≥1 FDR significant model | **248 / 581 (42.6%)** |

**84% of published associations could be recovered as significant by picking the right model.** That is a measurement of researcher degrees of freedom, not a subtle statistical point.

## 4. Prior art and the exact gap

| Work | Year | Form | Forks covered |
|---|---|---|---|
| quantvoe / voe | 2021 | R pkg (+Python) | 6 only |
| Tierney et al. | 2022 | Study, HPC cluster | 6 only |
| Nearing et al. | 2022 | Study | 5, as benchmark |
| benchdamic | 2022 | Bioconductor | 5 |
| **dar** | 2025 | Bioconductor | 1, 2, 5 partially |
| ConsensusMetaDA | 2025 | R pkg | 5 |
| Pelto et al. | 2025 | Study | 5 |

Two facts hold across the whole table: **everything is an R package or a study — nothing is a web server**, and **nothing covers the full grid**. `dar` is closest but is a *consensus* framework: it asks which taxa methods agree on. It does not enumerate the specification space, draw a specification curve, or attribute variance to individual choices.

**And the closest prior work names the gap itself.** Tierney et al.'s discussion states they fitted only linear models "due to its speed," acknowledges that data-processing choices (transformation, discretisation, abundance quantification) also vary results, notes that FDR methods and cutoffs differ between studies, and reports running on Harvard Research Computing's O2 cluster.

The most rigorous microbiome multiverse to date vibrated over one fork, said the other five matter, and needed an HPC cluster.

**MicroVerse extends the multiverse from one dimension to seven and puts it behind a URL.**

## 5. Novelty argument (six citable steps)

1. Microbiome DA results depend on ≥6 analytical choices, each contested in current literature.
2. Consequences are measured and severe (§3).
3. Multiverse / specification-curve / vibration-of-effects methodology is published and validated across psychology, epidemiology and biomedical data science.
4. Applied to microbiome data **once**, over covariates only, on a cluster, as a study — not a tool.
5. Every microbiome tool touching this space is an R package covering a subset of one or two forks.
6. The 2026 NAR Web Server Issue published 47 servers; three microbiology; none does this. The 2026 eight-platform web benchmark found none offers robustness or sensitivity analysis.

### 5.1 What MicroVerse does NOT claim

- **Not a new statistical method.** It orchestrates published methods within a published framework.
- **Not truth.** A taxon robust to analytical choice is not thereby biologically real — confounding, contamination and batch effects survive a multiverse intact.
- **Not a licence to cherry-pick.** §18 is designed against this.
- **Not exhaustive.** The grid is bounded by what is defensible and computable. Say so.

---

# PART II — DESIGN

## 6. Objectives

| ID | Objective | Success criterion |
|---|---|---|
| O1 | Enumerate and execute the grid | ≥3,000 valid specifications, Quick mode |
| O2 | Per-taxon robustness quantification | §16.2 metrics computed correctly |
| O3 | Specification curve | Simonsohn two-panel, interactive |
| O4 | Choice attribution | §17 variance decomposition summing to 100% |
| O5 | "Locate my result" | Percentile of user's declared pipeline |
| O6 | No HPC required | Quick mode <90 s; Full mode <10 min on 4 cores |
| O7 | Deploy free, no login, MIT, containerised | Public URL, API docs |

## 7. Scope — three modes, not one crossed grid

**This is the single most important design decision, and v1.0 got it wrong.** Fully crossing all seven forks including covariates yields >500,000 specifications, which is not achievable in web-request time. The fix is to separate the modes.

| Mode | Forks varied | Forks fixed | Specs | Time |
|---|---|---|---|---|
| **A. Quick** *(default)* | 1,2,3,4 + elementary methods + FDR | covariates (user's choice or none), sophisticated methods off | ~3,100 | **<90 s** |
| **B. Full** | A + ANCOM-BC, ALDEx2, PyDESeq2 on a stratified sample of matrices | covariates | ~4,500 | **<10 min** |
| **C. Covariate** | Fork 6 over ≤64 subsets | 1–4 at a reference sub-grid (3×2×2) | ~2,300 | **<5 min** |

Quick mode is the default **and** it is the mode with the best evidence behind it: Pelto et al. found elementary methods (Wilcoxon, ordinal/linear regression, logistic on presence/absence) most replicable. Say this in the paper — it is a feature, not a compromise.

Modes B and C run on request from the results page. Never all three by default.

## 8. Inputs

**Required:**
- Abundance table — CSV/TSV, BIOM v1/v2, `.qza`, MetaPhlAn, Kraken2/Bracken. Auto-detect orientation and value type.
- Metadata with a **binary grouping variable** (`group`).

**Optional:**
```
covariates      columns to consider in Mode C
taxonomy        lineage strings; enables Fork 4
declared_*      the user's own pipeline, for §16.4
```

**Hard constraints** (reject with a clear message, never a stack trace):

| Condition | Response |
|---|---|
| <10 samples, or <5 per group | Multiverse is noise below this. Refuse. |
| <10 taxa | Not a community profile. Refuse. |
| `group` has ≠2 levels | v1 is two-group only (§21). Refuse with explanation. |
| >1,500 taxa | Require collapse to genus, or refuse. |
| Non-integer values and rarefaction requested | Disable Fork 1 levels that subsample; warn. |
| Negative values | Already transformed. Refuse. |
| Metadata IDs don't match table | List mismatches. |

---

# PART III — THE GRID

## 9. Pipeline order — fixed, non-negotiable

Order of operations changes results. This order is the only correct one and must be enforced in code:

```
raw counts
   │
   ├─ 1. RAREFY          (on input counts, at input rank, seeded)
   ├─ 2. COLLAPSE        (to analysis rank)
   ├─ 3. PREVALENCE FILTER
   ├─ 4. TRANSFORM
   └─ 5. TEST            (DA method → p-values)
        └─ 6. FDR        (post-hoc on p-values)
```

Rarefying after collapsing, or filtering before rarefying, gives different answers. Enforce with a single `preprocess()` function that takes the four fork levels in this order and returns a matrix.

## 10. Fork levels — and why these ones

### Fork 1 — Rarefaction (13 states)

| Level | Justification |
|---|---|
| `none` | The McMurdie & Holmes position |
| `min_depth` × 3 seeds | Most common practice — rarefy to smallest library |
| `1000` × 3 seeds | Common low threshold; tests the aggressive end |
| `5000` × 3 seeds | Common mid threshold |
| `10000` × 3 seeds | Common high threshold; drops low-depth samples |

**Seeds are separate specifications, not averages.** Averaging p-values across rarefaction draws is statistically wrong. Treating each draw as its own specification is correct *and* it captures rarefaction-draw variance as part of the multiverse — which is itself worth showing. Seeds fixed at 1, 2, 3 for reproducibility.

Depths above the dataset's max library size are dropped automatically, with the count reported.

### Fork 2 — Prevalence filter (4)
`0%`, `5%`, `10%`, `20%`. The 10% threshold is Tierney's; 5% and 20% bracket common practice; 0% is the no-filter baseline.

### Fork 3 — Transformation (4)
`TSS` (relative abundance), `CLR`, `raw counts`, `TMM`. Covers the compositional school (CLR), the standard (TSS), the count-model requirement (raw), and the RNA-seq-derived option (TMM).

CLR requires zero handling: use `skbio.stats.composition.multi_replace` (multiplicative replacement), not a fixed pseudocount. Document this.

### Fork 4 — Rank (2)
`genus`, `input_rank` (species/ASV). If input is already genus-level, this fork collapses to 1 level and the grid shrinks accordingly — report that.

### Fork 5 — Method (7, tiered)

| # | Method | Python source | Tier | Effect returned |
|---|---|---|---|---|
| 1 | Wilcoxon rank-sum | `scipy.stats.mannwhitneyu` | ⚡ Quick | U statistic |
| 2 | Welch's t-test | `scipy.stats.ttest_ind` | ⚡ Quick | mean difference |
| 3 | Logistic regression, presence/absence | `statsmodels` GLM | ⚡ Quick | log odds ratio |
| 4 | Linear regression (LinDA-style) | `statsmodels` OLS | ⚡ Quick | coefficient |
| 5 | ANCOM-BC | `skbio.stats.composition.ancombc` | 🐢 Full | natural-log fold change |
| 6 | ALDEx2 | scikit-bio re-implementation | 🐢 Full | median CLR difference |
| 7 | PyDESeq2 | `pydeseq2` | 🐢 Full | log2 fold change |

Methods 1–4 are exactly Pelto's "elementary methods." No R dependency anywhere in this stack.

### Fork 6 — FDR (3)
`BH@0.05`, `BH@0.10`, `BY@0.05`. Applied post-hoc to stored p-values — **not** a separate model fit. Three FDR settings therefore cost essentially nothing.

### Fork 7 — Covariates (Mode C only)
- k ≤ 6 → enumerate all 2^k subsets
- k > 6 → 64 random subsets, stratified by subset size, seeded
- Always include null (none) and full (all)
- Cap 20 covariates per model (Tierney's limit)

## 11. Validity matrix — implement this literally

Many combinations are statistically incoherent. Pruning them matters twice: it cuts compute, and it stops nonsense specifications from artificially widening the distribution.

```python
# app/core/validity.py

INCOMPATIBLE = {
    # method: set of transforms it must NOT receive
    "pydeseq2":  {"clr", "tss", "tmm"},   # requires integer counts
    "aldex2":    {"clr"},                  # applies CLR internally
    "ancombc":   {"clr", "tss", "tmm"},    # estimates sampling fractions itself
    "wilcoxon":  set(),                    # accepts anything
    "ttest":     set(),
    "logistic":  set(),                    # uses presence/absence; transform irrelevant
    "linear":    set(),
}

# Additional rules, checked separately:
RULES = [
    # (predicate, reason) — spec is INVALID if predicate is True
    (lambda s: s.method == "ancombc" and s.rarefaction != "none",
     "ANCOM-BC estimates sampling fractions; rarefaction double-corrects"),

    (lambda s: s.transform == "tmm" and s.rarefaction != "none",
     "TMM + rarefaction is double library-size correction"),

    (lambda s: s.method == "logistic" and s.transform != "raw",
     "presence/absence is transform-invariant; keep one canonical spec"),

    (lambda s: s.method == "pydeseq2" and s.rarefaction != "none",
     "DESeq2 models library size internally"),
]
```

**Report both counts to the user:** specifications enumerated, and specifications valid. The ratio is informative.

## 12. Grid size — actual numbers

```
Preprocessed matrices = 13 (rarefy) × 2 (rank) × 4 (filter) × 4 (transform)
                      = 416

QUICK MODE
  416 matrices × 4 elementary methods          = 1,664 model fits
  × 3 FDR (post-hoc, ~free)                    = 4,992 specifications
  − validity pruning (logistic collapse etc.)  ≈ 3,100 valid

FULL MODE
  + 3 sophisticated methods × stratified 10% sample of matrices (≈42 each)
  = +126 model fits → +378 specifications
  ≈ 4,500 valid total, with sampling fraction reported

COVARIATE MODE
  reference sub-grid 3 (rarefy) × 2 (rank) × 2 (transform) = 12 matrices
  × 4 elementary methods × 64 covariate subsets × 3 FDR    ≈ 2,300 valid
```

---

# PART IV — ENGINE

## 13. Data structures — build these first

```python
# app/core/models.py
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

@dataclass(frozen=True)
class Specification:
    """One point in the multiverse. Hashable, so it can key a cache."""
    rarefaction:   Literal["none","min","1000","5000","10000"]
    rare_seed:     Optional[int]        # None when rarefaction == "none"
    rank:          Literal["genus","input"]
    prev_filter:   float                # 0.0, 0.05, 0.10, 0.20
    transform:     Literal["tss","clr","raw","tmm"]
    method:        Literal["wilcoxon","ttest","logistic","linear",
                           "ancombc","aldex2","pydeseq2"]
    fdr_method:    Literal["bh","by"]
    fdr_threshold: float                # 0.05, 0.10
    covariates:    tuple[str, ...] = () # empty in Modes A/B

    @property
    def matrix_key(self) -> tuple:
        """Identifies the cached preprocessed matrix (forks 1-4 only)."""
        return (self.rarefaction, self.rare_seed, self.rank,
                self.prev_filter, self.transform)

@dataclass
class TaxonResult:
    """One taxon under one specification."""
    taxon:              str
    spec_id:            int
    tested:             bool      # survived filtering?
    p_raw:              float
    p_adjusted:         float
    significant:        bool
    effect_native:      float     # the method's own statistic
    effect_native_type: str       # "log2fc" | "logOR" | "clr_diff" | ...
    effect_harmonized:  float     # ALWAYS log2 fold change — see §14
```

## 14. Effect-size harmonisation — the fix v1.0 was missing

**The problem:** seven methods return seven incompatible effect measures. A Wilcoxon U statistic, a log odds ratio and a DESeq2 log2 fold change cannot go on the same y-axis. Without solving this there is no specification curve.

**The fix: decouple significance from effect size.**

- **Significance** comes from the method's own p-value, FDR-adjusted. Each method does what it does.
- **Effect size** is computed once, identically, on the same preprocessed matrix, by a single estimator independent of the method:

```python
def harmonized_effect(matrix, groups, taxon) -> float:
    """log2 fold change of mean relative abundance, group B vs group A.
    Computed identically for every specification regardless of method.
    Uses TSS internally so it is comparable across transforms."""
    rel = matrix / matrix.sum(axis=0)          # force TSS
    a = rel.loc[taxon, groups == "A"].mean()
    b = rel.loc[taxon, groups == "B"].mean()
    eps = rel[rel > 0].min().min() / 2         # half smallest nonzero
    return np.log2((b + eps) / (a + eps))
```

Store both. `effect_native` preserves what the method said (useful, method-specific). `effect_harmonized` is what the specification curve plots and what sign-consistency is computed from.

**Sanity check to implement in tests:** for PyDESeq2, `effect_native` (its own log2FC) and `effect_harmonized` should correlate >0.9. If they don't, the harmonisation is wrong.

## 15. Handling filtered-out taxa

A taxon removed by a 20% prevalence filter does not appear in that specification. Getting the denominator wrong here silently corrupts every robustness metric.

**Rules:**
- `n_specs_total` — all valid specifications in the run
- `n_specs_tested` — specifications where this taxon survived filtering
- `frac_tested = n_specs_tested / n_specs_total` — **report this; it is itself a finding.** A taxon testable in only 30% of specifications is rare enough that its significance is filter-dependent by construction.
- `frac_significant` and `sign_consistency` use **`n_specs_tested`** as denominator, never `n_specs_total`.
- A taxon with `n_specs_tested < 10` is reported as `INSUFFICIENT`, not tiered.

---

# PART V — OUTPUTS

## 16. Results

### 16.1 Verdict sentence (top of page, one line)

> Of your 18 significant genera, 4 are ROBUST, 7 CONDITIONAL, 5 FRAGILE and 2 UNSTABLE. Your reported pipeline sits at the **88th percentile** of significance across 3,104 valid specifications — more favourable than 88% of defensible alternatives.

### 16.2 Per-taxon robustness table

| Field | Definition |
|---|---|
| `n_specs_tested` | specifications where the taxon survived filtering |
| `frac_tested` | `n_specs_tested / n_specs_total` |
| `frac_significant` | proportion FDR-significant, of tested |
| `frac_nominal` | proportion raw p<0.05, of tested |
| `sign_consistency` | proportion of `effect_harmonized` sharing the modal sign |
| `median_effect`, `iqr_low`, `iqr_high` | log2FC distribution |
| `robustness_tier` | below |

| Tier | Criteria |
|---|---|
| 🟢 **ROBUST** | `frac_significant` ≥ 0.80 AND `sign_consistency` ≥ 0.95 |
| 🟡 **CONDITIONAL** | 0.30 ≤ `frac_significant` < 0.80 AND `sign_consistency` ≥ 0.95 |
| 🟠 **FRAGILE** | 0 < `frac_significant` < 0.30 AND `sign_consistency` ≥ 0.80 |
| 🔴 **UNSTABLE** | `sign_consistency` < 0.80 — direction not determined by the data |
| ⚪ **INSUFFICIENT** | `n_specs_tested` < 10 |

Evaluate in order; first match wins.

### 16.3 Specification curve

Simonsohn's two-panel plot, per taxon:
- **Upper:** `effect_harmonized` for every tested specification, sorted ascending, points coloured by significance, reference line at zero.
- **Lower:** dot matrix showing which fork level was active in each specification, x-aligned to the panel above. This is how you *see* that all the null results cluster under `rarefaction = 1000`.
- If the user declared a pipeline, mark it with a vertical line.

### 16.4 Locate my result

```python
percentile = 100 * (rank of user_spec's -log10(p) among all tested specs) / n_specs_tested
```

Report neutrally. If >75th percentile, the interface says so plainly.

### 16.5 Everything else

Specification-level CSV (~3,100 rows × taxa), robust-taxa-only table, methods paragraph with citations, permanent token URL (90-day retention), full ZIP.

## 17. Choice attribution — the formula v1.0 was missing

**Question:** which fork drives the variance?

**Two separate decompositions**, because FDR affects significance but not effect size:

**(a) Effect-size attribution** — linear mixed model on `effect_harmonized`:

```
effect_harmonized ~ rarefaction + rank + prev_filter + transform + method + (1 | taxon)
```

`statsmodels.regression.mixed_linear_model.MixedLM`, taxon as random intercept, forks as categorical fixed effects. Then partial η² per fork:

```
η²_f = SS_f / Σ(SS_all_fixed_effects)     # Type II sums of squares
```

Normalise to 100%.

**(b) Significance attribution** — logistic mixed model on `significant`, same structure plus `fdr_method` and `fdr_threshold`.

**Fallback, mandatory to implement.** MixedLM fails to converge often enough that you must not depend on it. Fallback estimator — simple, fast, never fails:

```python
def variance_of_group_means(df, fork_col, value_col="effect_harmonized"):
    """Between-level variance for one fork, averaged within taxon."""
    per_taxon = df.groupby(["taxon", fork_col])[value_col].mean()
    return per_taxon.groupby("taxon").var().mean()

# Normalise across forks to sum to 100%.
```

Run the fallback always; use MixedLM when it converges. Report which was used. Both should rank the forks similarly — if they disagree wildly, something is wrong and you want to know.

**Output:** ranked bar chart.

> For your dataset, DA method accounts for 41% of variance in effect size; rarefaction depth 27%; transformation 19%; prevalence filter 9%; rank 4%.

Actionable: it tells the researcher which decision they must justify in their methods section.

## 18. Anti-cherry-picking design

A multiverse tool can be used to p-hack. Design against it explicitly — reviewers will raise this, and having an answer is a strength.

- **No "export best specification" button.** Ever.
- Export always contains the full distribution, never a filtered slice.
- The generated methods paragraph describes the multiverse, not a single point.
- `/about` states the norm: report the distribution, not your favourite point in it.
- If the declared pipeline is above the 75th percentile of significance, say so neutrally.

## 19. Worked example — verify against this before trusting any output

**Study:** IBD vs control, 120 samples (60/60), 380 genera, Illumina V4, genus-level input.
**Mode:** Quick. Rank fork collapses to 1 level (input already genus) → 208 matrices → **3,104 valid specifications** (4,992 enumerated, 1,888 pruned).

### Taxon A — *Faecalibacterium*

| Metric | Value |
|---|---|
| `n_specs_tested` | 3,104 |
| `frac_tested` | 1.00 (prevalent; survives every filter) |
| `frac_significant` | 0.87 |
| `sign_consistency` | 1.00 (always negative in IBD) |
| `median_effect` | −1.42 log2FC |
| `IQR` | [−1.71, −1.18] |
| **Tier** | 🟢 **ROBUST** (0.87 ≥ 0.80 ✓, 1.00 ≥ 0.95 ✓) |

### Taxon B — *Escherichia*

| Metric | Value |
|---|---|
| `n_specs_tested` | 2,328 |
| `frac_tested` | 0.75 (filtered out at 20% prevalence) |
| `frac_significant` | 0.41 |
| `sign_consistency` | 0.98 |
| `median_effect` | +0.88 log2FC |
| `IQR` | [+0.31, +1.44] |
| **Tier** | 🟡 **CONDITIONAL** (0.30 ≤ 0.41 < 0.80 ✓, 0.98 ≥ 0.95 ✓) |

Attribution reveals why: significant in **94%** of unrarefied specifications but **12%** of those rarefied to 1,000. This is a sequencing-depth-dependent finding, and the specification curve makes it visible at a glance.

### Taxon C — *Bacteroides*

| Metric | Value |
|---|---|
| `n_specs_tested` | 3,104 |
| `frac_significant` | 0.23 |
| `sign_consistency` | 0.61 |
| **Tier** | 🔴 **UNSTABLE** (0.61 < 0.80 → UNSTABLE regardless of frac_significant) |

The direction of effect is not determined by the data. In a single-model paper this would have been reported as a finding with a sign.

### Attribution output

```
DA method          41%
Rarefaction depth  27%
Transformation     19%
Prevalence filter   9%
Rank                4%
                  ────
                  100%
```

### Locate-my-result

User declared: rarefy to 10,000 + CLR + Wilcoxon + BH@0.05.
That specification's −log10(p) for *Escherichia* ranks 2,048 / 2,328 → **88th percentile**.

**Hand-verify this example against your implementation at M6. If the tiers don't match, the engine is wrong.**

---

# PART VI — BUILD

## 20. Tech stack

| Layer | Choice | Note |
|---|---|---|
| Language | Python 3.11+ | |
| Framework | FastAPI | auto OpenAPI docs |
| Compute | numpy, pandas, scipy, statsmodels, **scikit-bio ≥0.7.1**, pydeseq2 | scikit-bio 0.7.1+ is required — that's the version with `ancombc` |
| Parallel | joblib | `Parallel(n_jobs=-1)` over specifications |
| Jobs | FastAPI BackgroundTasks + polling | no Celery, no Redis |
| Storage | SQLite (jobs only) | **no curated database in this project** |
| Frontend | Jinja2 + HTMX + Alpine.js | no npm, no build step |
| Plots | Plotly.js | specification curve needs interactivity |
| Tables | Tabulator.js | sort/filter/export built in |
| Tests | pytest, pytest-cov | |
| Container | Docker | |
| Hosting | Fly.io / Render — **verify CPU limits at M4, not M10** | more compute-hungry than a typical server |
| Licence | MIT | |

### Repo layout

```
microverse/
├── app/
│   ├── main.py  config.py  db.py
│   ├── routers/      upload.py  job.py  results.py  api.py
│   ├── core/
│   │   ├── models.py           Specification, TaxonResult (§13)
│   │   ├── parsers/            csv, biom, qza, metaphlan, kraken
│   │   ├── preprocess.py       rarefy → collapse → filter → transform (§9)
│   │   ├── grid.py             enumeration + mode selection (§12)
│   │   ├── validity.py         INCOMPATIBLE + RULES (§11)
│   │   ├── methods/            wilcoxon, ttest, logistic, linear,
│   │   │                       ancombc, aldex2, pydeseq2
│   │   ├── effects.py          harmonized_effect (§14)
│   │   ├── fdr.py              BH / BY post-hoc
│   │   ├── robustness.py       tiers, metrics (§16.2)
│   │   ├── attribution.py      MixedLM + fallback (§17)
│   │   └── runner.py           orchestration, caching, joblib
│   ├── templates/  static/
├── tests/
│   ├── test_preprocess.py      order-of-operations tests
│   ├── test_validity.py        every rule
│   ├── test_effects.py         harmonisation vs PyDESeq2 native
│   ├── test_robustness.py      §19 worked example as fixture
│   ├── test_attribution.py     synthetic data, known driver
│   └── fixtures/
├── examples/                   3 demo datasets
├── Dockerfile  CONTRIBUTING.md  README.md
└── MICROVERSE_SPEC_v2.md
```

## 21. Scope guard — put in `CONTRIBUTING.md` verbatim

- **Two-group comparisons only.** No multi-group, no continuous outcomes, no ordinal. v2 work.
- **No longitudinal or paired designs.** No repeated measures.
- No read processing, no DADA2, no denoising, no taxonomic classification. Input starts at the abundance table.
- No alpha/beta diversity as standalone features.
- No contamination screening (that's Bioflow).
- No user accounts, no login, no saved projects.
- **No "best specification" export.** Ever (§18).
- Do not add a DA method beyond the seven in §10 without asking.

## 22. Milestones

12–15 hrs/week.

| M | Deliverable | Gate | Effort |
|---|---|---|---|
| **M1** | Skeleton, parsers, `AbundanceTable`, `Specification` dataclass | All formats load; spec hashes correctly | 1 wk |
| **M2** | `preprocess()` in the §9 order + matrix cache | 208–416 matrices built, cached, seeded, order-tested | 1 wk |
| **M3** | Validity matrix + grid enumeration | Enumerated vs valid counts match §12 | 0.5 wk |
| **M4** | 4 elementary methods + FDR + `harmonized_effect` | **Quick mode end-to-end <90 s. Verify hosting CPU now.** | 1.5 wk |
| **M5** | 3 sophisticated methods + stratified sampling + joblib | Full mode <10 min on 4 cores | 1.5 wk |
| **M6** | Robustness metrics + tiers | **§19 worked example reproduces exactly** | 1 wk |
| **M7** | Attribution (MixedLM + fallback) | Both estimators agree on fork ranking | 1 wk |
| **M8** | Specification curve + locate-my-result | Plot readable at 3,000 points | 1.5 wk |
| **M9** | Frontend, downloads, methods text | End-to-end upload → ZIP | 1.5 wk |
| **M10** | Validation (§23) | Numbers for results section | 1.5 wk |
| **M11** | Covariate mode | Mode C works | 1 wk |
| **M12** | Docs, examples, Docker, deploy, bio.tools | Public URL from fresh browser | 1 wk |
| | | | **~14 wk** |

**Minimum viable: M1–M6 + M9 basic = ~8 weeks.** A working multiverse server with robustness tiers, minus attribution and covariate mode.

## 23. Validation

Unusually clean, because the ground truth is other people's published results.

1. **Reproduce Tierney et al.** Code: `github.com/chiragjp/ubiome_robustness`. Data: `figshare.com/projects/Microbiome_robustness/127607`. Cohorts: curatedMetagenomicData. Run Mode C on their cohorts; check you recover 1-in-3 sign-flipping and >90% T1D/T2D nonrobust. **If you reproduce their result, your engine is correct.** Highest-value experiment available.
2. **Extend past their grid.** Turn on Forks 1–5 and quantify the *additional* instability. Their limitations section predicted this; nobody has measured it. **This is a genuine finding, not just a tool paper.**
3. **Reproduce Pelto et al.** `github.com/jepelt/DAA_replicability` (Zenodo 10.5281/zenodo.15047338). Check elementary methods show tighter specification distributions in your framework too.
4. **Simulated ground truth.** Spike known differential taxa at known effect sizes; verify true positives land ROBUST and nulls land FRAGILE/UNSTABLE.
5. **Compute benchmark.** Runtime vs samples × taxa. Publish the curve — a reviewer will ask.

---

# PART VII — ASSIGNMENT

## 24. Mapping to the brief

| Requirement | How MicroVerse satisfies it |
|---|---|
| Bioinformatics web server **or** database application | **Web server.** Note honestly: the database element is jobs only. If the brief marks "database application" strictly, Bioflow is the safer choice. |
| Stores, queries, or predicts something biological | Predicts robustness of taxon–phenotype associations; stores specification-level results |
| Through a web interface | Upload → job → results dashboard |
| Not a generic web app | Domain-specific: compositional statistics, microbiome DA methods, rarefaction |
| **Check classmate overlap** | ⚠️ **You must do this** |
| **Check UniProt, PDB, NCBI, KEGG** | ⚠️ **You must do this and record it.** None hosts anything comparable — write one line each. |
| **Check NAR Web Server Issue (July)** | ✅ **Done.** 2026 issue: 47 servers, 3 microbiology (HERA, Microbe Decoder, PhaBOX), none does robustness or sensitivity analysis. |

Add your course's deliverable format and deadline here before starting — it drives §22.

---

# APPENDIX

## 25. Change log

| Version | Change |
|---|---|
| 1.0 | Initial spec after competitive sweep |
| **2.0** | **FROZEN.** Closed: three-mode structure replacing an uncomputable crossed grid (§7); effect-size harmonisation (§14); filtered-taxon denominators (§15); attribution formula + mandatory fallback (§17); worked example with numbers (§19); grid-level justification (§10); rarefaction seeds as specifications not averages (§10); pipeline order enforced (§9); validity matrix as code (§11); data structures (§13); scope guard (§21); assignment mapping (§24); corrected compute budget (§12) |

*Log implementation deviations below this line. Do not redesign above it.*

---

## 24.1 Implementation deviation log — v1.0.0 (built 2026-09-07)

Every item below is a place where the implementation departs from the letter of the
frozen spec. Nothing above the line was redesigned. Each entry says what was found, what
was done, and where.

### A. Spec errors found (arithmetic and internal consistency)

**A1 — §19's headline grid counts are inconsistent.**
§19 states "Rank fork collapses to 1 level (input already genus) → 208 matrices →
**3,104 valid specifications** (4,992 enumerated, 1,888 pruned)". With one rank level,
208 matrices × 4 elementary methods × 3 FDR settings is **2,496** enumerated, not 4,992 —
and 3,104 valid cannot exceed 2,496 enumerated. 4,992 is the *two-rank* figure
(416 matrices), consistent with §12. The engine reports actual counts. On the bundled
genus-level demo it produces 2,496 enumerated → 1,596 valid; on a species-level table with
the rank fork active, 4,992 enumerated → 3,192 valid, matching §12's ~3,100 estimate.
The tiers of §19 — the stated verification gate — are asserted verbatim in
`tests/test_worked_example.py`; grid arithmetic is asserted against §12 in the same file.

**A2 — §16.2's four tiers do not partition the space.**
Two regions match no published criterion, so a taxon would receive no tier at all:
(i) `frac_significant == 0` with `sign_consistency >= 0.80`; (ii) `sign_consistency` in
[0.80, 0.95) with `frac_significant >= 0.30`. Gap-fillers were added, evaluated only after
all four published criteria have been tried, so §19's tiers are unaffected: (i) becomes a
sixth tier **NOT DETECTED** ("never significant under any defensible specification"),
which also keeps never-significant taxa out of the §16.1 verdict count; (ii) becomes
**CONDITIONAL**. See `app/core/robustness.py::assign_tier` and
`tests/test_worked_example.py::test_gap_fillers_cover_the_uncovered_region`.

**A3 — §17(a) regresses effect size on DA method, which §14 makes impossible.**
§14 defines the harmonised effect as one estimator computed *independently of the method*.
It therefore explains exactly 0% of effect-size variance by construction, and a 0% bar
reads as a bug rather than as a design property. `method` is excluded from the effect-size
decomposition, with the reason shown in the interface. It remains in the significance
decomposition, where §19's expectation (DA method leading) is borne out.
See `app/core/attribution.py::EFFECT_FORKS`.

**A4 — §14's per-matrix pseudocount makes rarefaction dominate attribution artefactually.**
§14 computes the pseudocount as "half smallest nonzero" *of the matrix being processed*.
That value varies by two orders of magnitude across rarefaction depths, and again under
CLR's zero replacement, so the "single estimator, computed identically for every
specification" of §14 silently becomes several different estimators. Measured effect: with
a per-matrix pseudocount, the MixedLM and fallback attributions disagreed wildly (rank
agreement 0.2; rarefaction 98% vs 49%) — exactly the disagreement §17 says means something
is wrong. The pseudocount is now computed **once per run** from the input table; agreement
rose above 0.8 and both estimators now lead on the same fork.
See `app/core/effects.py::run_pseudocount`.

**A5 — §14's `rel = matrix / matrix.sum(axis=0)` is undefined for CLR.**
CLR columns sum to ~0 by construction, so forcing TSS on a CLR matrix divides by zero.
CLR matrices are exponentiated before closure — `exp(CLR)` is proportional to the
zero-replaced composition, i.e. exactly what the test saw. Raw, TSS and TMM are a direct
closure and give identical proportions to one another (a per-sample scalar cancels), which
is asserted in `tests/test_preprocess.py`. See `app/core/preprocess.py::relative_scale`.

**A6 — §15's `frac_tested` denominator is wrong when the rank fork has two levels.**
A species-level taxon cannot be tested in a genus-rank specification, so dividing by the
run's total specification count caps every species at `frac_tested <= 0.5` and reports a
structural fact as a filtering finding. The denominator is now the specifications at the
taxon's own rank (`n_specs_eligible`), which equals `n_specs_total` whenever the rank fork
has one level — the case §15 and §19 describe. Both fields are exported.

### B. Environment constraints

**B1 — scikit-bio is not installable on this Python.**
§20 requires `scikit-bio >= 0.7.1` for `ancombc` and `multi_replace`. Its `biom-format`
dependency has no wheel for Python 3.13+ and fails to build from source. The needed pieces
are implemented in-tree against the source papers, and scikit-bio is used automatically if
it ever becomes importable:

- `multiplicative_replacement` (`app/core/preprocess.py`) — the `multi_replace` algorithm.
- `run_ancombc` (`app/core/methods/ancombc.py`) — per-taxon linear model on log counts plus
  a three-component normal-mixture E-M for the sampling-fraction bias (Lin & Peddada 2020),
  with a median fallback. Zeros take a unit pseudocount rather than structural-zero
  detection.
- `run_aldex2` (`app/core/methods/aldex2.py`) — Dirichlet Monte Carlo, CLR per instance,
  expected p-value across instances (Fernandes et al. 2014). 64 instances rather than the R
  default of 128; the count is stated in the methods paragraph.
- BIOM v1 (JSON) and v2 (HDF5) are read directly (`app/core/parsers/biom.py`); v2 uses
  h5py, which is present.

### C. Implementation choices worth stating

**C1 — the elementary methods use vectorised closed forms, not per-taxon fits.**
§10 names `statsmodels` GLM/OLS. With one binary predictor and no covariates, the OLS group
coefficient test is exactly the pooled t-test and the logistic Wald test is exactly the 2×2
table estimate, so both are computed for all taxa at once. This is what keeps Quick mode at
about a second instead of minutes. Both substitutions are verified against the named
statsmodels models in `tests/test_methods.py`; under separation, where the MLE is infinite
and every implementation must regularise, only the p-value (~1) is asserted. With covariates
the logistic path uses a batched Newton-Raphson over a shared design matrix.

**C2 — Wilcoxon and Welch in covariate mode operate on residuals.**
§12 runs all four elementary methods in covariate mode, but neither rank-sum nor Welch has
a covariate form. Covariates are regressed out first and the test applied to the residuals
(`app/core/methods/design.py::residualise`). Modes A and B are unaffected — with no
covariates, nothing is residualised.

**C3 — §17(b)'s logistic mixed model is fitted as a linear probability model.**
A binomial mixed GLM over ~10^6 rows does not converge reliably, and §17 explicitly says not
to depend on MixedLM. The significance decomposition uses MixedLM on the 0/1 indicator, with
the §17 fallback always computed alongside and both reported. Which estimator was used is
shown in the interface and in `attribution.csv`.

**C4 — §12's flat 10% matrix sample is too small for the restricted methods.**
§11 confines ANCOM-BC and PyDESeq2 to raw counts without rarefaction, leaving them a pool of
roughly 4–8 compatible matrices; 10% of that is one, against §12's stated target of ~42. The
sample target is therefore 10% of the *whole* matrix grid, capped by what each method can
accept, so small pools are taken whole. The fraction actually sampled is reported per method.

**C5 — MixedLM runs on a subsample.**
Up to 120 taxa (all of their specifications, so within-taxon structure is intact) and 60,000
rows, under a time budget. The §17 fallback always runs on the full set.

**C6 — Additions.** A `NOT DETECTED` tier (A2); `n_specs_eligible` alongside `frac_tested`
(A6); and a rarefaction seed-versus-depth split, which §10 makes measurable by treating seeds
as separate specifications and which turns out to be a striking number — 43% of rarefaction's
attributed variance is the random draw itself on the demo data. No DA method beyond the seven
of §10 was added, and no "best specification" export exists anywhere (§18).

### E. Hardening pass — post-build audit

A full static-analysis and review pass over the implementation. Everything here is a
defect that was found and fixed, not a design change.

**E1 — a taxon name could inject script into a shared results page.**
Tabulator's default cell formatter writes values as HTML. Taxon labels come from an
uploaded file, and a results URL is meant to be shared, so a crafted table was a stored
cross-site-scripting vector against anyone opening the link. Every cell formatter now
escapes its value explicitly (`app/templates/results.html`).

**E2 — a filename could inject a response header.**
The download filename was interpolated from the uploaded file's name straight into
`Content-Disposition`. A name containing a quote or a newline could add headers of its
own. Names are now reduced to `[A-Za-z0-9._-]` (`app/routers/results.py::_safe_stem`).

**E3 — tokens reached the filesystem unvalidated.**
Job tokens name a directory. FastAPI's path matching already refuses a slash, but the
shape was never checked, so a malformed token was one bug away from a path outside the
job store. `db.valid_token` now requires 24 hex characters, and `get_job` returns None
for anything else before a path is built (`app/db.py`).

**E4 — a missing job answered 422 instead of 404.**
Every refusal shared one status code, so "this dataset is invalid" and "this token does
not exist" were indistinguishable to an API client. `NotFoundError` now carries 404
(`app/core/validation.py`), and the handlers honour each exception's own status.

**E5 — specifications that produced no rows corrupted the denominators.**
A rarefaction depth can leave a group below the minimum, in which case the matrix is
unusable and the specification produces nothing. Those specifications were still counted
in `n_specs_total` and in the per-rank eligibility, understating every taxon's
`frac_tested`. Eligibility is now measured over the specifications that actually ran, the
shortfall is reported on the results page, and `spec_summary` keeps a row per
specification so exports still line up with the grid (`app/core/robustness.py`).

**E6 — dead code and unused parameters.** The genus-collapse path built a lineage map it
then discarded; `attribute()` took a `robustness` argument it never read;
`run_multiverse` accepted `aldex_instances` and dropped it on the floor (it now reaches
ALDEx2 through `run_method`); `estimate_bias` took per-taxon variances and ignored them
(they now enter the variance of the bias estimate, which is what they are for); two
service helpers took a `run` they never used. Cleared with `ruff` under the config in
`pyproject.toml`; the tree is clean at `--select F,E,W,B,SIM,UP,C4,RET,ARG,I`.

**E7 — large downloads were rebuilt in memory on every request.**
`results_long.csv.gz` is now written once when the run finishes and streamed from disk,
which matters for a table with thousands of taxa on a small host.

**E8 — deprecated APIs.** `datetime.utcnow` and FastAPI's `on_event` startup hook were
both replaced (`db.utcnow`, a lifespan context manager).

### F. Interface

The front end was rebuilt as an editorial research interface: a paper ground with ink
type, one indigo accent reserved for the brand, and the tier palette kept strictly
separate from it so a coloured row never reads as branding. Display type is a serif,
labels and all numerals are monospaced and tabular so columns of results align down the
page, and the geometry is rules and rectangles rather than rounded cards.

Two defects were found in the process and fixed: the dark pillar card rendered
dark-on-dark because `.pillar p` and `.pillar-ink p` had equal specificity and the base
rule came last; and long prose set in the `.micro` label style was rendering as uppercase
monospace, which made footnotes shout. Plot colours are read from the stylesheet at
render time, so the specification curve and the page cannot drift apart.

### D. Measured against the §6 objectives

| ID | Criterion | Result |
|---|---|---|
| O1 | ≥3,000 valid specifications, Quick mode | 3,192 on a species-level table (1,596 when the rank fork collapses, per A1) |
| O2 | §16.2 metrics computed correctly | `tests/test_robustness.py`, `tests/test_worked_example.py` |
| O3 | Simonsohn two-panel curve, interactive | Plotly, effect panel over a fork dot matrix |
| O4 | Variance decomposition summing to 100% | Asserted in `tests/test_attribution.py` |
| O5 | "Locate my result" | Percentile overall and per taxon; marked on the curve |
| O6 | Quick <90 s, Full <10 min on 4 cores | ~1 s and ~19 s on the demo datasets |
| O7 | Free, no login, MIT, containerised, API docs | Dockerfile, token URLs, `/api/docs` |


### G. Corrections found by the reference and real-data passes

R 4.6.1 was installed user-scope (no administrator rights, into `D:/Rlocal`) together
with **edgeR 4.10.1, ALDEx2 1.44.0 and ANCOMBC 2.14.0**. §10 names all three. Until this
pass, two of them had never been compared against the package they are named after —
only against analytic ground truth and a third-party Python port. Diffing value by value
(`tests/reference/compare_r.py`) found four defects in the engine and two crashes.

**G1 — ALDEx2 reports in log2; we were reporting natural log.**
`aldex.clr` takes log base 2, so `diff.btw` is a log2 fold change. Our CLR used natural
log, making every ALDEx2 effect size a factor of ln 2 too small — the regression slope
against the R package was 0.687, and ln 2 = 0.693. The base is a constant scaling, so it
never touched a p-value (Welch's t and Mann-Whitney are scale-invariant, and the
Spearman correlation with ALDEx2's p-values was identical before and after), but the
reported effect was wrong by 31%. `clr_stack` is now log2, and the effect type is named
`clr_diff_log2` rather than `clr_diff` so the units travel with the number, as
`ln_fold_change` and `log2fc` already did.

**G2 — ALDEx2's `diff.btw` is the median of paired differences, not the difference of
medians.** We computed, per Monte-Carlo instance, the median CLR in each group and took
the median of the difference. `aldex.effect` instead pools each group's CLR values
across samples *and* instances, permutes the two pools independently per taxon, and
takes the row median of `smpl2 - smpl1`. The two agree for symmetric distributions and
diverge on the left-skewed ones sparse data produce: the correlation with ALDEx2 was
0.875 and individual taxa were off by up to 3.0 CLR units. With the estimator corrected
the correlation is **0.998** and the remaining disagreement is smaller than the
estimator's disagreement with itself across Monte-Carlo seeds.

**G3 — ANCOM-BC's bias E-M was a simplification, and it displaced every effect.**
Ours fitted a homoscedastic three-component mixture with symmetric, hard-coded shifts
(`delta`, `delta ± shift`, scales `sigma, 2sigma, 2sigma`). `ANCOMBC:::.bias_em` fits a
*heteroscedastic* mixture — each taxon enters with its own regression variance `nu_i`,
the two shifted components have free asymmetric locations `l1 <= 0 <= l2` and their own
extra variances `kappa1, kappa2` estimated by Nelder-Mead at each iteration. The
per-taxon coefficients matched R exactly either way, but delta landed elsewhere, which
shifts **every** log fold change by the same constant: **0.417 natural-log units**, a
1.5x fold change applied across the board, enough to move 44 taxa across zero. The E-M
now follows the R implementation step for step and the residual offset is **0.020**
(2.1% on the fold-change scale), constant to 2e-15, and traceable to `nloptr`'s and
`scipy`'s Nelder-Mead stopping on `kappa` at different points. `compare_r.py` asserts on
that offset directly rather than letting it hide inside a correlation.

**G4 — we always added the bias variance to ANCOM-BC's standard error; the package does
not.** `se_hat` gets `var_delta` folded in only under `conserve = TRUE`, which is not
ANCOM-BC's default, and even then the package adds `var_delta + 2*sqrt(var*var_delta)`
rather than `var_delta` alone. Ours inflated every standard error by about 5%. The test
now uses the regression standard error, matching the package default; p-value agreement
with R went from Spearman 0.249 (with the G3 defect) to **0.994**.

**G5 — `tmm_factors` raised on a matrix with no taxa.** A prevalence filter can empty
the table at some grid points. Every other dead end in that function returns a factor of
exactly 1, as edgeR does, but the opening `np.quantile(values, 0.75, axis=0)` ran before
any of them and raised `IndexError` on the empty axis. Reachable from the web app —
upload a table, pick a TMM specification, get a 500 instead of a pruned specification.
Found by running MicrobiomeHD at OTU level (§24.3), where `nash_chan` hit it and was
recorded as a crash rather than a refusal.

**G6 — `clr` warned on the same empty matrix.** `Mean of empty slice`, from taking a
geometric mean over no taxa. Guarded alongside G5; both have regression tests.

**G7 — the image installed unpinned lower bounds, and pulled a release candidate.**
The Dockerfile installed `requirements.txt` directly, so the versions inside the image
were whatever PyPI served on build day rather than the versions the tests ran against.
It was also quietly installing **wrapt 2.4.1rc1**: `formulaic` declares
`wrapt>=1.17.0rc1` for Python 3.13+, and a specifier containing a pre-release switches
pre-releases on for that project — even when resolving for Python 3.12, where the marker
does not apply. `deploy/lock_requirements.py` now resolves once for the image's exact
platform and writes `requirements.lock.txt` (73 packages, all wheels); it refuses to
emit a lock containing a pre-release or anything needing a compiler, and CI fails if the
lock is stale. Hashes are deliberately omitted: a `--require-hashes` lock names one
platform's wheels and would break `docker build` on arm64.

## 24.2 Validation record — what has been proved, and what has not

Everything below was run on this build. `tests/reference/run_all.py` re-runs all of it
and reports anything unavailable as SKIPPED rather than passing it silently.

### A. Methods checked against reference implementations

`scikit-bio >= 0.7.1` has no wheel on the Python the server runs (§24.1 B1), but it
does install on Python 3.10. A second environment (`.venv310`) exists solely to hold
the reference implementations, and `tests/reference/compare_scikit_bio.py` diffs the
in-tree code against them.

| Component | Reference | Result |
|---|---|---|
| `multiplicative_replacement` | `skbio.stats.composition.multi_replace` | identical, max abs difference **4.9e-17** |
| `clr` | `skbio.stats.composition.clr` | identical, max abs difference **1.8e-15** |
| `run_ancombc` | `skbio.stats.composition.ancombc` (0.7.3) | log fold changes **r = 1.0000**, slope 1.0000, max abs difference 0.028; p-value ranking Spearman **0.948**; nominally significant sets Jaccard **0.94**; both recover the spiked taxa (ours 100%, scikit-bio 92%) |

**A finding worth reporting: scikit-bio 0.7.3 mislabels its ANCOM-BC output column.**
The column is named `Log2(FC)` but the values are natural log. Established on a
controlled 4x spike where the true natural-log fold change is 1.314 and the true log2
fold change is 1.896 — scikit-bio returns 1.3826, and so do we, to four decimals. Our
values agree with theirs exactly; SPEC §10's "natural-log fold change" is the correct
description of both. No conversion is applied, and the reasoning is recorded in the
comparison script so nobody "fixes" it later.

**The three R-origin methods, checked against the R packages themselves.**
§10 names edgeR's TMM, ALDEx2 and ANCOM-BC. R 4.6.1 is now installed user-scope with
**edgeR 4.10.1, ALDEx2 1.44.0 and ANCOMBC 2.14.0**; `tests/reference/r_reference.R` runs
each package on the same matrix and `tests/reference/compare_r.py` diffs the results.
The comparison matrix is 177 taxa x 50 samples at 50% zeros with 30 spiked taxa —
abundance-dependent dropout, so the sparsity is realistic without erasing the signal.

| Component | Reference | Result |
|---|---|---|
| `tmm_factors` | `edgeR::calcNormFactors(method="TMM")` | **identical** — max relative difference 0.0000%, max absolute 0.000000, r = 1.000000, and the same unit geometric mean |
| `run_aldex2` | `ALDEx2::aldex(test="t", denom="all", mc.samples=128)` | `diff.btw` r = **0.998**, slope 0.996; p-value ranking Spearman **0.976**; significant sets Jaccard **0.96** (27 of 28) |
| `run_ancombc` | `ANCOMBC::ancombc(conserve=FALSE)` | log fold changes r = **1.00000**, slope 1.0000, delta within 0.020 natural-log units; p-value ranking Spearman **0.994**; significant sets Jaccard **0.93** (69 of 74) |

The ALDEx2 effect size is a Monte-Carlo estimate in both tools, so a fixed tolerance
would be arbitrary — and at 128 instances a tight one is tighter than the estimator's
own noise. The check is calibrated against ourselves instead: rerun with other seeds,
measure how far the same estimator moves, and require that we agree with ALDEx2 at least
that well. We do — mean absolute difference **0.055** against ALDEx2 versus **0.062**
against our own other seeds. Agreeing with the reference more closely than with
ourselves is the strongest form this claim can take.

Four engine defects came out of this comparison and are recorded as §24.1 G1–G4. Two of
them were invisible to every check that existed before: the ALDEx2 log base and the
ANCOM-BC bias term both shift results by a constant, which correlations and rank tests
cannot see.

### B. Methods checked against analytic ground truth

These checks predate the R installation above, and are kept because they establish
something the package comparison cannot: that the estimator recovers a quantity known
in closed form, rather than merely agreeing with another implementation. Both matter —
agreement with edgeR would be worth little if edgeR and we were wrong together.

**TMM** (`tests/reference/compare_tmm.py`). A table is built where 20% of genes are
genuinely 4x up, which mechanically deflates every other gene's proportion; the factor a
correct TMM must return is exactly the ratio of true totals.

| Case | True factor | Recovered | Error |
|---|---|---|---|
| 20% up 4x, dense | 0.6710 | 0.6757 | 0.010 log2 |
| 30% up 3x, dense | 0.6229 | 0.6308 | 0.018 log2 |
| 20% up 4x, 40% zeros | 0.6710 | 0.6545 | 0.036 log2 |

Composition bias that TSS cannot see is corrected from a ratio of 0.671 to **0.993**
(1.000 is correct) on the genes known not to have changed.

**This comparison found a real bug.** `conorm`, an independent Python port, diverged
from us by up to 31% on sparse tables. Neither implementation was following edgeR:
edgeR's `.calcFactorTMM` trims by **rank**, and both of us were trimming by quantile
*value*. With microbiome counts the M-values are full of ties, and quantile trimming
discards every tied observation at the boundary instead of the intended 30%.
`tmm_factors` now follows edgeR step for step, including the rank trim, the `absE`
cutoff and the `max|logR| < 1e-6` early return. On tables with no real composition bias
the two implementations still differ, because that difference *is* the trim rule; both
are then estimating a true factor of 1 and neither drifts further than the other.

**ALDEx2** (`tests/reference/compare_aldex2.py`).

- The Dirichlet posterior mean matches `Dirichlet(count + 0.5)` to **1e-4**, and a taxon
  with zero reads still gets non-zero posterior mass — which is what stops a zero
  becoming a `-inf` log-ratio.
- A known CLR difference is recovered: **+1.3586 estimated against +1.3632 true**, with
  r = 0.9996 across the whole vector and max error 0.013 on the unshifted taxa.
- Type I error on null data: **0.0%** of 120 null taxa at p < 0.05, effects centred on
  zero (median -0.002).
- Power on a spiked set: **100% recall**, and the spiked taxa occupy 100% of the top 15
  by effect size. The other taxa are correctly reported as *depleted* rather than
  unchanged — raising 15 taxa and renormalising really does push the other 85 down, and
  a compositional method is supposed to see that. Precision against the "spiked set" is
  therefore not a meaningful metric here, and is not asserted.
- Monte-Carlo convergence: seed-to-seed SD falls 0.0082 → 0.0040 → 0.0019 at 8, 32 and
  128 instances, so the default of 64 is already stable.
- The reported p-value is exactly the mean across instances (difference 0.0e+00), which
  is ALDEx2's definition rather than the p-value of a pooled statistic.

### C. Parsers checked against real files

The BIOM readers exist because `biom-format` has no wheel on this Python (§24.1 B1).
Testing them against strings we wrote ourselves would only prove self-consistency, so
`tests/reference/make_format_fixtures.py` writes the fixtures with `biom-format` 2.1.17
itself, and `tests/test_real_formats.py` reads them back in the normal suite.

Covered: **BIOM v1 (JSON), BIOM v2 (HDF5), QIIME 2 `.qza`** (the real
`<uuid>/{VERSION, metadata.yaml, data/}` zip layout), and biom's own TSV export with its
leading comment and trailing taxonomy column. All four readers reproduce
`biom-format`'s view of the table exactly — ids, per-sample and per-taxon sums, zero
count, lineages — and agree with each other. Format detection is checked from content
with a deliberately misleading filename. A real BIOM v2 file is then run end to end
through validation, a full multiverse and tiering.

Before this, the v2 and `.qza` readers had **no tests at all**.

### D. Scale, concurrency and the O6 budget

`tests/reference/benchmark.py` (SPEC §23 validation 5, "publish the curve — a reviewer
will ask"). Quick mode, phases measured separately because they scale differently.

| taxa | samples | specs | result rows | multiverse | robustness | attribution | exports | total | bundle |
|---|---|---|---|---|---|---|---|---|---|
| 60 | 20 | 1,236 | 52,560 | 0.25s | 0.02s | 1.60s | 0.30s | **2.2s** | 0.5 MB |
| 250 | 60 | 1,596 | 399,000 | 0.64s | 0.08s | 3.74s | 2.31s | **6.8s** | 4.1 MB |
| 380 | 120 | 1,596 | 606,480 | 1.19s | 0.12s | 4.25s | 3.73s | **9.3s** | 11.3 MB |
| 1000 | 80 | 1,596 | 1,596,000 | 1.88s | 0.30s | 6.48s | 9.64s | **18.3s** | 28.5 MB |
| 1500 | 60 | 1,596 | 2,392,320 | 2.04s | 0.45s | 8.33s | 14.46s | **25.3s** | 41.8 MB |
| 1500 | 200 | 1,596 | 2,393,970 | 6.36s | 0.45s | 8.29s | 14.57s | **29.7s** | 44.4 MB |

**O6 (Quick mode under 90 s) holds at the §8 ceiling of 1,500 taxa with a 3x margin.**
The multiverse itself is never the cost — it is 6.4 s at the largest size tested. The
exports dominate, and profiling them exposed a defect: `build_zip` was regenerating the
gzipped long-format results that the run had already written to disk, doing ~14 s of
identical work twice per job. Fixed by passing the compressed payload through; total
time at the ceiling fell from 41.6 s to 25.3 s.

Concurrency (`tests/reference/load_test.py`): six runs started simultaneously all
completed, slowest 63.6 s, none over the 90 s budget. That is a ~10x per-run slowdown
against a single run, so **concurrency, not table size, is the limit** — the design has
no queue (SPEC §20: no Celery, no Redis), so simultaneous users contend. 120 status
polls in a tight loop returned 200 at 185 requests/second.

### E. Accessibility

Audited with **axe-core 4.10.2** at `wcag2a, wcag2aa, wcag21a, wcag21aa, best-practice`.
Landing, method, configure, results and error pages: **zero violations**.

Three real defects were found and fixed:

- **Contrast.** `--grey` was 4.33:1 on paper and 3.85:1 on the deeper panels, failing
  16-19 nodes per page. Every violation traced to that one token; it is now #5c616e,
  which clears 4.5:1 on all three backgrounds it is used against.
- **Tier colours failed in both directions.** They are used as text on paper *and* as
  chip fills behind white text, and the saturated values cleared neither — conditional
  was 2.9:1 as text and 2.4:1 behind white. Darkened so both uses clear 4.5:1.
- **Structure.** The results page had no `h1` at all; the footer skipped from `h2` to
  `h4`; and horizontally scrollable tables were not keyboard-reachable.

**Tabulator's own ARIA does not validate, in 5.5.2 or in 6.3.1.** It marks the outer
element `role="grid"`, but that element also holds the pagination controls — and a grid
may only own rows and rowgroups — while the header nests a `rowgroup` inside a
`rowgroup`. axe reports both as *critical*. The widget offers no way to turn it off.
Since an invalid grid is announced worse than plain content, the roles are stripped
after build and the table is exposed as a labelled, scrollable region, with its filter
inputs labelled. The full data is also available as CSV and through the API, which is
the reliable path for assistive technology. This is a limitation of the widget, recorded
rather than hidden.

### F. Prior art — the §24 check the spec assigns to the author

Searched September 2026 for a microbiome multiverse or specification-curve tool.

- **No multiverse or specification-curve web server exists for microbiome differential
  abundance**, and none was found in bioinformatics generally.
- **`dar`** (Bioconductor, 2025) remains the closest: it varies filtering, rarefaction
  and DA method, but it is an R package and a *consensus* framework — it asks which taxa
  methods agree on. It does not enumerate a specification space, draw a specification
  curve, or attribute variance to individual choices.
- **`metaGEENOME`** (2025) and **ZicoSeq**, **LOCOM2** (2026) are DA methods and
  frameworks, not multiverse tools.
- The 2026 multiverse literature found ("Visualizing vastness", *Behav Res Methods*) is
  about **visualisation methods** for multiverse analysis, not microbiome application.
- UniProt (protein sequence and function), PDB (macromolecular structure), NCBI
  (sequence and literature archives) and KEGG (pathways and genomes) host nothing
  comparable; none provides microbiome differential abundance analysis at all.

This supports the §4/§5 gap statement but **is not a systematic review**: it is a
targeted search, not an exhaustive one, and the classmate-overlap check in §24 is still
the author's to make.

### G. Still not done, and why

- **The Docker image has never been built here.** No container runtime is installed and
  none can be: Docker Desktop needs administrator rights, and `wsl --install` needs them
  too — WSL reports "not installed" and installing it is not available to this account.
  What *is* verified locally is the layer that actually fails in practice: all 73
  dependencies resolve to Linux wheels for Python 3.12 with `--only-binary=:all:`, so
  the install step needs no compiler, and `deploy/lock_requirements.py --check` proves
  the lock matches `requirements.txt`. The build itself, and a full analysis driven
  through the running container, are exercised by the `docker` job in CI on every push.
  The build context was also 2.6 GB until this pass — `.dockerignore` listed `.venv/`,
  which does not match `.venv310/`, and `data/` had since filled with cached cohorts.
  It is now 1.1 MB.
- **Not deployed.** Out of scope by explicit instruction; the Fly and Render configs are
  in `deploy/` and need an account to use.
- **No cross-browser testing** beyond the Chromium-based pane. The stylesheet uses
  `:has()`, which is widely supported but unverified outside Chromium here.
- **ANCOM-BC's delta differs from the R package by 0.020 natural-log units** (§24.1 G3).
  Everything upstream of it matches exactly and the remaining gap is a Nelder-Mead
  stopping-rule difference between `nloptr` and `scipy`, but it is a real residual and
  `compare_r.py` asserts on it rather than letting it drift.
- **Pelto's cohorts carry anonymised taxon labels** (`genus_1`, `genus_2`), so fork 4 has
  one level throughout §24.5 and that study cannot say anything about taxonomic rank.
  The rank fork is exercised on MicrobiomeHD at OTU level (§24.3) and on
  curatedMetagenomicData (§24.4) instead.

---

## 24.3 Real published data — SPEC §23 validation 2

> "Turn on Forks 1–5 and quantify the *additional* instability. Their limitations
> section predicted this; nobody has measured it. **This is a genuine finding, not just
> a tool paper.**" — SPEC §23

### What was run

Seventeen published 16S case-control cohorts from **MicrobiomeHD** (Duvallet et al.
2017, *Nat Commun* 8:1784; Zenodo 1146764, CC-BY-NC-4.0) — the collection §27 lists as
public test data. Each was put through MicroVerse's Quick-mode grid: rarefaction depth
*and seed*, prevalence filter, transformation, taxonomic rank, four elementary DA
methods and three FDR settings.

**19,728 specifications, 2,918,340 taxon-level results, 38 seconds on one laptop.**
Tierney et al. needed Harvard's O2 cluster for 6,035,110 models over one fork.

Reproduce with `tests/reference/real_data_study.py`; the per-cohort record is
`docs/real_data_study.json`. Nothing is committed from MicrobiomeHD itself — the
archive is downloaded on demand into a git-ignored cache.

### The headline comparison

| Quantity | Genus (pooled, 95% CI) | OTU (pooled, 95% CI) | Tierney et al. |
|---|---|---|---|
| Taxa flipping association sign | **19%** (11–32) | **36%** (26–46) | 1 in 3 (33%) |
| Fraction of specifications nominally significant | 13% (7–21) · 33% detected | 10% (5–15) · 19% detected | 38% |
| Fraction FDR significant | 7% (1–15) · 20% detected | 5% (1–9) · 11% detected | 16% |
| Detected taxa reaching ROBUST | 4% (0–9) | **0%** (0–1) | — |

Figures are **pooled** — one taxon one vote across all cohorts — with a 95% interval
from 2,000 bootstrap resamples *of the cohorts*, since taxa within a cohort are not
independent. An earlier version of this section averaged per-cohort percentages, which
gives every cohort equal weight regardless of whether it contributed 40 taxa or 800; on
the genus tables that inflated the headline sign-flip figure from 19% to 26%. Both are
printed by `real_data_study.py` so the difference is visible rather than chosen.

"Flipping" here is `sign_consistency < 0.95` — the direction is not settled by the
data. "Detected" means significant in at least one specification, which is the closer
analogue of Tierney's denominator: theirs is 581 *literature-reported* associations,
i.e. results somebody wrote down, not every taxon in the table. Both denominators are
reported because neither is exactly theirs.

**At the rank these tables are usually analysed, between a fifth and a third of taxa
change the direction of their association on analytical choices alone — and at OTU rank
it is a third to a half, with essentially nothing reaching ROBUST.** Tierney got a
similar number by varying covariates only. Different forks, different data, different
denominators, same place. The instability is not specific to one family of choices.

### Rank is not a neutral choice — running at OTU level roughly doubles the instability

The first version of this study ran at genus, because the §8 ceiling collapses a
6,000-OTU table and fork 4 is then left with one level. Trimming to the 1,500-taxon
ceiling instead keeps the OTU table, so fork 4 has both OTU and genus to move between.
All 17 cohorts then run with two rank levels — **37,704 specifications, 56.3 million
taxon-level results, 83 seconds.**

The result is not a wash. Sign-flipping goes from 19% to 36% pooled; the fraction of
detected taxa reaching ROBUST goes from 4% to **0%** (95% interval 0–1%); the leading
fork shifts from rarefaction (7/17 cohorts at genus) to transformation (8/17 at OTU);
and the share of rarefaction's variance that is the random draw rather than the depth
rises from 30% to **40%** (median 21% to 34%).

Collapsing to genus was *stabilising the answers*. That is worth knowing, because it is
a choice usually made for convenience — and because a paper reporting genus-level
results is reporting the more flattering half of this comparison. It also means the
caveat in the previous version of this section understated the problem rather than
overstating it.

### The finding §23 asked for

The 17 cohorts disagree about *which* fork drives significance:

| Leading fork | Genus | OTU |
|---|---|---|
| **Rarefaction depth** | **7 / 17** | 5 / 17 |
| Transformation | 4 / 17 | **8 / 17** |
| FDR threshold | 4 / 17 | 0 / 17 |
| Prevalence filter | 1 / 17 | 2 / 17 |
| Taxonomic rank | — | 1 / 17 |
| DA method | 1 / 17 | 1 / 17 |

### Which fork leads — a claim this section used to make, and has withdrawn

An earlier version of this section reported, as a finding:

> "Rarefaction leads more often than any other fork, and DA method leads in one cohort
> out of seventeen."

**That claim is withdrawn.** It was never checked for stability, and when it was
(`tests/reference/attribution_sensitivity.py`, added for this purpose) it did not hold
up:

| Check | Effect-size decomposition | Significance decomposition |
|---|---|---|
| Within-taxon R² of the fork model, median over 17 cohorts | **0.016** | **0.009** |
| Three independent estimators pick the same leading fork | 11 / 17 | 8 / 17 |
| Leading fork wins a cluster bootstrap over taxa | 0.90 | 0.92 |
| Cohorts meeting both the variance and agreement thresholds | **0 / 17** | **3 / 17** |

The shares §17 produces are percentages *of the variance the fork model explains*, and
on these data that model explains **about one percent** of the within-taxon variance.
Ranking the forks by their slices of it is ranking a decomposition of noise. The
bootstrap stability is real but does not rescue the claim: resampling the same data
reproduces the same near-zero decomposition, which is consistency, not signal.

The three estimators — §17's MixedLM, taxon fixed effects, and §17's group-means
fallback — also disagree about which fork leads in a third to a half of cohorts, with
rank agreement ranging from **−1.00 to +1.00**. When two defensible estimators of the
same quantity rank it in opposite orders, neither ranking is the answer.

**What the product does about it.** Fork attribution is graded `exploratory` on every
run and the interface says so beside the bars; the within-taxon R² and the
estimator-agreement coefficient are shown next to the ranking; a run that fails either
threshold lists the reason in plain words. The decomposition is still worth computing —
it is the right question, and `rarefaction_seed_share` below answers a narrower version
of it convincingly — but it is not evidence of the kind that belongs in a results
sentence, and the product no longer writes one.

**What survives.** The seed-versus-depth split below is a different computation: a
direct ratio of within-depth to between-depth variance on a balanced design, with no
model, no R² and nothing to disagree about. It is unaffected by any of the above.

And within rarefaction, **a mean of 30% of the variance (median 21%) is the random
subsampling draw rather than the choice of depth — 40% (median 34%) at OTU level.**
That number only exists because §10 treats each seed as its own specification instead of
averaging. A third of the rarefaction effect is a coin flip, and a paper reporting one
rarefied analysis reports one draw from it.

Finally: of the taxa significant somewhere, **4% reached ROBUST at genus and 0% at
OTU** (pooled; 1% and 0% averaging cohorts equally). The great majority of detectable
signals in this collection survive only part of the specification space, and at OTU
level essentially none of them do.

### Caveats, and what has since been done about them

Four caveats were recorded when this study first ran. Three are now closed and the
fourth is narrowed; they are kept here with their resolutions rather than quietly
deleted, because the resolutions changed the numbers.

- ~~**Genus-level tables, so fork 4 has one level.**~~ **Closed.** The study now runs at
  both ranks (`real_data_study.py --otu`), all 17 cohorts have two rank levels at OTU,
  and the comparison above shows rank was doing real work: instability roughly doubles.
  The genus figures are kept because genus is what most published analyses report.
- ~~**The aggregate is a mean over cohorts, not a pooled estimate.**~~ **Closed.**
  Pooled numerators and denominators are now the headline, with a cluster bootstrap over
  cohorts for the interval; the mean-of-means is printed beside it. The two differ by
  7 percentage points on the genus sign-flip figure, in the direction that flattered the
  earlier write-up.
- **Several cohorts are small and detect almost nothing.** Still true, and now visible
  rather than buried: the bootstrap intervals are wide (11–32% at genus) precisely
  because 17 cohorts is a small cluster count. The interval is over *these* cohorts; it
  is not a population estimate.
- ~~**This is not a reproduction of Tierney et al.**~~ **Superseded.** It was the
  *extension* §23.2 asks for, and it still is. The reproduction §23.1 asks for is now
  done separately and on their own cohorts — see §24.4.
- ~~**MicrobiomeHD is a single processing pipeline.**~~ **Closed.** Two independent
  sources now sit alongside it, both shotgun rather than 16S and both processed by
  MetaPhlAn rather than OTU picking: the 29 shotgun cohorts curated by Pelto et al.
  (§24.5) and the 8 curatedMetagenomicData contrasts of §24.4. What each shows is not
  the same measurement, so they are worth stating separately rather than pooled:
  MicrobiomeHD gives 19% (genus) to 36% (OTU) of taxa flipping sign across forks 1–5;
  curatedMetagenomicData gives 28.7% flipping across covariate adjustment sets and 98%
  of T1D/T2D associations failing ROBUST; Pelto's shotgun cohorts are used for the
  method-dispersion comparison and show the same method-dependent instability at
  shotgun species level that the 16S tables show at genus. The conclusion supported by
  all three is that the instability is not an artefact of MicrobiomeHD's pipeline — not
  that the three numbers are interchangeable. Cross-study differences in DNA
  extraction, primer choice and sequencing depth are still real and still not modelled.

### What this closed


Before this, the engine had never seen a real dataset. Running it on 17 immediately
exposed a parser defect that would have made MicroVerse **unusable on the most widely
cited public microbiome collection**: MicrobiomeHD ends every lineage with a de novo
OTU identifier, `d__denovo5393`, and the prefix table aliased `d__` to `k__` for
MetaPhlAn's leading domain marker. Every lineage therefore read as kingdom-rank, so
`can_collapse_to_genus` was false, and a 6,000-OTU table was **refused outright** with
"no taxonomy was supplied so it cannot be collapsed to genus".

`d__` is now only a domain in first position; elsewhere it is a sub-species marker
alongside `t__`, and `lineage_rank_index` caps such tokens at species. Checked against
MicroPhlAn's leading `d__`, MetaPhlAn's trailing `t__` strain, Greengenes `s__`, and
genus-only lineages (`tests/test_parsers.py`).

---

## 24.4 Reproducing Tierney et al. — SPEC §23 validation 1

> "Run Mode C on their cohorts; check you recover 1-in-3 sign-flipping and >90% T1D/T2D
> nonrobust. **If you reproduce their result, your engine is correct.** Highest-value
> experiment available." — SPEC §23.1

### What was run

R 4.6.1 with **curatedMetagenomicData 3.20.0** — Tierney et al.'s own data source —
installed user-scope. `tests/reference/tierney_export.R` pulls every stool T1D and T2D
case-control contrast with at least 8 samples a side, at species level with full
MetaPhlAn lineages, as integer counts. Adjustment variables are whatever cMD populates
for that cohort at 90% completeness and with more than one value; that filter correctly
drops `gender` from KarlssonFH_2013, which is a female-only cohort, and `country`
everywhere, since it is constant within a study.

`tests/reference/tierney_study.py` then runs **Mode C** (§12) over each: the four
elementary methods across every covariate adjustment subset, on the reference sub-grid.

**8 contrasts, 11,844 specifications, 32 seconds.**

### The measurement error this exposed first

The obvious reading — take `sign_consistency` from the robustness table and compare with
Tierney's 1 in 3 — gives **0%**, and it is the question that is wrong, not the engine.
SPEC §14 harmonises the effect size deliberately: significance comes from the method,
the effect from one method-independent estimator computed from the matrix. That
estimator never sees the adjustment set, so `effect_h` is *identical* across covariate
subsets by construction. Measured directly, its standard deviation across 32 adjustment
sets is **exactly 0.0**, while the linear model's own coefficient moves with a standard
deviation of 0.65 over the same specifications.

Tierney vibrated regression coefficients. The comparable quantity is `effect_n`, the
method's native effect, with every other fork held fixed — one model type, one
preprocessing, many adjustment sets, which is their design. That is what
`_covariate_fork_only` computes, per matrix, then averages.

This is worth stating plainly because it is a property of the design, not a bug:
**MicroVerse's harmonised effect cannot show covariate-driven sign instability at all.**
Covariate vibration surfaces in significance and in the native effect. Anyone reading a
specification curve should know which of the two they are looking at.

### The two numbers

| Quantity | MicroVerse | Tierney et al. |
|---|---|---|
| Sign-flipping over adjustment sets alone, all taxa tested | **28.7%** | 1 in 3 (33%) |
| Sign-flipping over adjustment sets alone, taxa detected somewhere | 6.8% | — |
| T1D/T2D associations that are not ROBUST | **97.9%** | >90% |

Averaged over the 6 contrasts with at least 8 adjustment subsets (72 matrices); the two
cohorts with only one or two usable covariates cannot vibrate and are excluded from that
average, though they still contribute to the robustness figure. Both §23.1 targets are
met.

| Cohort | Disease | case/control | taxa | subsets | sign-flip | nonrobust |
|---|---|---|---|---|---|---|
| Heitz-BuschartA_2016 | T1D | 27/26 | 390 | 16 | 55% | 100% |
| KosticAD_2015 | T1D | 31/89 | 330 | 8 | 11% | 100% |
| LiJ_2014 | T1D | 31/10 | 375 | 2 | — | 100% |
| HMP_2019_t2d | T2D | 11/46 | 273 | 8 | 48% | 100% |
| KarlssonFH_2013 | T2D | 53/43 | 433 | 8 | 13% | 100% |
| MetaCardis_2020_a | T2D | 515/616 | 634 | 4 | — | 83% |
| QinJ_2012 | T2D | 170/174 | 640 | 16 | 16% | 100% |
| SankaranarayananK_2015 | T2D | 19/18 | 365 | 32 | 30% | 100% |

The pattern across cohorts is the one you would hope for: sign-flipping is worst in the
smallest and most imbalanced contrasts — 55% at n=53, 48% with 11 cases, 30% at n=37 —
and settles at 11–16% in the four larger ones. MetaCardis, the only cohort above a
thousand samples, is also the only one where more than a handful of associations reach
ROBUST: 17% of them, against 0% everywhere else.

### What this is, and what it is not

This is their data source, their disease contrasts and their fork, run through an
independent implementation, and it lands on their numbers. It is **not** a rerun of
their code. Their adjustment variables were chosen per analysis; ours are whatever cMD
populates. Their denominator was the associations they carried forward; ours is every
taxon the grid tested, and the figure over taxa detected somewhere (6.8%) is much lower
— reported alongside, because neither denominator is exactly theirs. Their cohort set
was larger than eight.

With that said: §23.1 says "if you reproduce their result, your engine is correct", and
on the two quantities it names, the engine reproduces them.

---

## 24.5 Reproducing Pelto et al. — SPEC §23 validation 3

> "Reproduce Pelto et al. `github.com/jepelt/DAA_replicability` (Zenodo
> 10.5281/zenodo.15047338). Check elementary methods show tighter specification
> distributions in your framework too." — SPEC §23.3

Pelto et al. ("Elementary methods provide more replicable results in microbial
differential abundance analysis", arXiv:2404.02691) compared 14 DA methods over 61
curated case-control cohorts and split each eligible study in half five times to measure
replicability directly. Their conclusion is that the elementary methods — Wilcoxon,
t-test, linear and logistic regression — replicate better than the sophisticated ones.

Their curated data ships inside the Zenodo archive as a plain `save()` image: each
cohort is `list(meta, counts)`, no Bioconductor classes, so `pelto_export.R` reads it
with base R. **60 of their 61 whole cohorts run here** — 31 16S and 29 shotgun; the
61st, `cdi_youngster`, is refused by §8 for having 4 control samples. 45,165
specifications, 2.7 hours.

### The comparison has to be matched, or it measures the wrong thing

§11 does not give every method the same forks. ANCOM-BC and DESeq2 refuse rarefaction
and every transform but raw, so in a Full-mode run they receive 12 specifications where
Wilcoxon receives 480. A naive dispersion comparison would score them "tight" purely
because they were asked fewer questions.

Everything below is therefore computed on a **matched sub-grid** — the same matrices,
the same taxa (median 129 per cohort), the same FDR settings for every method being
compared:

- **A1**, raw counts and no rarefaction: the only cell where all seven methods of §10
  are valid, so prevalence filter, rank and FDR vary identically for each.
- **A2**, raw counts with rarefaction free: five methods admit it, and it adds the fork
  that dominated the MicrobiomeHD study.

And not on the harmonised effect. §14 makes that method-independent by construction, so
it is identical for every method on a given matrix and can distinguish nothing — the
first version of this analysis reported a dispersion of 0.0066 for all seven methods,
which is the harmonisation working, not a result. What varies by method is the evidence:
the p-value, and the call that follows from it.

### A1 — on the prevalence-filter fork, elementary methods do not move at all

| Method | Family | Median SD of -log10 p | Unstable calls among detected taxa |
|---|---|---|---|
| wilcoxon | elementary | **0.0000** | 0.577 |
| ttest | elementary | **0.0000** | 0.343 |
| linear | elementary | **0.0000** | 0.313 |
| logistic | elementary | **0.0000** | 0.448 |
| ancombc | sophisticated | 0.0722 | 0.589 |
| aldex2 | sophisticated | 0.0223 | 0.421 |
| pydeseq2 | sophisticated | 0.0141 | 0.515 |

The zeros are exact, on all 60 cohorts, and they are a mechanism rather than a
measurement: with the transform fixed at raw, a Wilcoxon, t-test, linear or logistic
test of one taxon depends only on that taxon's counts. Removing *other* taxa cannot
change its p-value. Every sophisticated method, by contrast, is coupled to the whole
table — ANCOM-BC through the bias term it estimates from all coefficients, ALDEx2
through the CLR denominator, DESeq2 through its size factors — so changing which taxa
are present changes results for the taxa that remain.

That is worth stating plainly because it is the likely mechanism behind Pelto's finding,
and it is not a property anyone measured: it is a property of what the estimator reads.

The *calls* are a real measurement, because the prevalence filter changes the
multiple-testing burden even when it cannot change a raw p-value. Elementary methods
are more stable there too, though modestly and with overlap — **42.0% of detected taxa
have an unstable call versus 50.8%** (Wilcoxon signed-rank over 60 paired cohorts,
**p = 0.010**, elementary lower in 58% of cohorts). Wilcoxon at 0.577 is less stable
than ALDEx2 at 0.421, so this is a tendency between families, not a rule about members.

### A2 — on the rarefaction fork, the ordering reverses

| Method | Family | Median SD of -log10 p |
|---|---|---|
| wilcoxon | elementary | 0.2600 |
| ttest | elementary | 0.1596 |
| linear | elementary | 0.1464 |
| logistic | elementary | 0.2576 |
| aldex2 | sophisticated | **0.0970** |

Once rarefaction is in the sub-grid, **ALDEx2 is tighter than all four elementary
methods** — 0.097 against a mean of 0.206, Wilcoxon signed-rank **p = 2.3e-11** over 60
cohorts. The reason is structural: ALDEx2 already integrates over sampling variation
with its Dirichlet Monte-Carlo, so resampling the library perturbs it far less than it
perturbs a test applied to one particular draw.

So "elementary methods give tighter specification distributions" is **true of the
prevalence-filter fork and false of the rarefaction fork**. Which family looks more
stable depends on which analytical choice is being varied — which is exactly the kind of
thing a multiverse framework exists to expose, and exactly what a study varying one fork
at a time cannot see.

### B — their own design: split the study in half and see what survives

Pelto's stronger experiment does not compare methods to each other on one dataset; it
compares each method **to itself** on the other half of the same study. That design is
immune to the fork-coverage problem entirely. Their archive ships the splits, so this
runs on the same halves they used: **24 study/iteration pairs, 12 16S and 12 shotgun,
over 8 studies at 3 random splits each**, each half analysed on the A1 sub-grid and a
taxon "called" when it is significant in most specifications.

| Method | Family | Top-20 overlap | Same direction | Replication rate (FDR) | Pairs it was computable on | Median taxa called per half |
|---|---|---|---|---|---|---|
| wilcoxon | elementary | 0.340 | 0.872 | 0.747 | 10/24 | 0 |
| ttest | elementary | 0.300 | 0.899 | 0.720 | 6/24 | 0 |
| linear | elementary | 0.273 | 0.846 | 0.758 | 6/24 | 0 |
| logistic | elementary | 0.389 | 0.864 | 0.872 | 8/24 | 0 |
| ancombc | sophisticated | 0.290 | 0.826 | 0.355 | 15/24 | 2 |
| aldex2 | sophisticated | 0.357 | 0.885 | 0.665 | 8/24 | 0 |
| pydeseq2 | sophisticated | 0.315 | 0.733 | 0.287 | 22/24 | 26 |

Read the last two columns before the fourth. **The FDR-based replication rate says
elementary methods win overwhelmingly — 0.774 against 0.436 — and that comparison is not
sound.** At these half-sizes the median elementary method calls *nothing at all* in a
half, so its rate is defined only on the 6 to 10 pairs where it happened to call
something: its easiest pairs. PyDESeq2 calls 26 taxa per half and is therefore scored on
22 of 24 pairs, including every hard one. A method that calls two taxa and reproduces
both scores 1.00; a method that calls twenty and reproduces nine scores 0.45. The first
is not more replicable, it is quieter.

The threshold-free measure scores every method on every pair over the same taxa, and it
finds **no difference**: top-20 overlap **0.326 elementary against 0.321 sophisticated,
Wilcoxon signed-rank p = 0.39** over 24 paired splits. Direction agreement among the
shared top taxa is marginally better for the elementary family (0.870 against 0.815) and
the effect rank correlation marginally better for the sophisticated one (0.443 against
0.473). None of it separates them.

The honest summary of B is therefore: **in this framework, on their splits, elementary
methods do not replicate better once the comparison is made at equal detection.** Their
apparent advantage on a call-conditioned metric is at least partly a consequence of
calling less. That is an observation about the metric, not a refutation of their paper —
Pelto compared 14 methods with their own definitions and their own significance rules,
and this reproduces neither. What it does show is that the measurement is fragile in a
way worth knowing about.

### What §23.3 asked, and what the answer is

"Check elementary methods show tighter specification distributions in your framework
too." Partly, and with a sign that depends on the fork:

| | Result |
|---|---|
| Prevalence-filter fork, raw p-values | Elementary methods are **exactly invariant**; every compositional method moves. A mechanism, not a measurement. |
| Prevalence-filter fork, calls after FDR | Elementary methods more stable: **42.0% vs 50.8%** unstable calls, p = 0.010. |
| Rarefaction fork | **Reversed** — ALDEx2 is tighter than all four elementary methods, 0.097 vs 0.206, p = 2.3e-11. |
| Split-half replicability, equal detection | **No difference** — 0.326 vs 0.321, p = 0.39. |
| Split-half replicability, FDR-conditioned | Elementary far ahead (0.774 vs 0.436), but confounded by detection. |

Two of the five support Pelto, one reverses, two are null or unsound. Reporting only the
first row would reproduce their headline; reporting all five is what the framework is
for.

### Caveats

- The 24 split pairs come from **8 studies** at 3 random splits each, so the pairs are
  not independent; the paired test treats them as if they were, which if anything
  overstates the precision of a null result.
- Pelto's taxon labels are anonymised (`genus_1`, `genus_2`), so fork 4 has a single
  level throughout this study. Nothing here says anything about taxonomic rank.
- Their 14 methods include nine this build does not implement, and §21 forbids adding
  any without asking. The elementary/sophisticated split here is four methods against
  three, mapped onto §10, not their full panel.
- The whole-cohort analysis (A) and the split analysis (B) were run separately —
  `docs/pelto_study.json` and `docs/pelto_splits.json` — because A over all 60 cohorts
  takes 2.7 hours, dominated by PyDESeq2 failing to converge on large shotgun tables.

---

## 24.6 Do the robustness tiers predict anything? — §16.2 validated on held-out data

The tiers are the product's central output: every taxon gets one, and the interface
prints it as a verdict. Until this experiment they were a **convention** — thresholds
written in §16.2, implemented faithfully, and never shown to mean anything. `pytest`
proved the implementation matched the specification. Nothing tested whether a taxon
called ROBUST was any likelier to hold up in new data than one called FRAGILE.

That is the difference between internal consistency and external validity, and it is the
difference between a tool that computes and a tool that predicts.

### Design

`tests/reference/tier_validation.py`. 25 published case-control cohorts curated by Pelto
et al. — 16S and shotgun, 80 to 747 samples — each split 50/50 **stratified on the
group**, three random splits apiece. **75 discovery/validation pairs, 12,564 taxon
observations.**

- Tiers are assigned on the discovery half by the ordinary Quick-mode pipeline: the
  product default, not a special path.
- The held-out half is scored **tier-blind**. Replication is defined two independent
  ways, because the definition is a researcher's choice and the conclusion should not
  depend on ours: *majority* (nominally significant in most held-out specifications,
  same direction) and *reference* (significant in one pre-specified default analysis —
  no rarefaction, 10% prevalence, raw counts, Wilcoxon, BH 0.05 — same direction).
- A **label-permuted null** repeats the whole thing with the held-out group labels
  shuffled. Discovery is untouched, so any replication left is arithmetic.
- Uncertainty is a **cluster bootstrap over cohorts**: taxa in one cohort share samples,
  and three splits of one cohort are not three independent observations.

Cohorts below 40 samples per group are excluded, and that exclusion is itself a finding
(below).

### Result

| Tier | Replicated | 95% CI | Taxa | How often assigned |
|---|---|---|---|---|
| **ROBUST** | **90%** | 60–93% | 60 | 0.5% of observations |
| **CONDITIONAL** | **51%** | 39–60% | 351 | 2.8% |
| **FRAGILE** | **12%** | 8–17% | 2,029 | 16.1% |
| **UNSTABLE** | **2%** | 1–4% | 1,468 | 11.7% |
| NOT DETECTED | 3% | 2–4% | 8,656 | 68.9% |

Perfectly monotone. **AUC 0.785** for the tier ordering against replication — given one
taxon that replicated and one that did not, the tier ranks them correctly 79% of the
time. ROBUST taxa replicated **7.4×** as often as other claim tiers. Under permuted
labels the overall rate falls to **0.7%**, and **0%** among ROBUST.

The second definition agrees: AUC 0.773, ROBUST 87%, CONDITIONAL 58%, FRAGILE 12%,
UNSTABLE 4%. A conclusion that survives changing the outcome definition is not an
artefact of the outcome definition.

**The §16.2 thresholds are empirically supported.** No recalibration is warranted by
this evidence.

### What it does not establish

- **ROBUST is rare, and its rate rests on few cohorts.** 60 observations in 5 of 25
  cohorts, and 49 of those 60 came from one — `cdi_schubert`, *C. difficile*, an
  unusually large biological effect. Leave-one-cohort-out: dropping it leaves 11 ROBUST
  taxa at **73%**, still far above CONDITIONAL's 52%, and the AUC only moves from 0.785
  to 0.767. **The ordering survives; the exact rate does not.** That is why the interval
  runs 60–93% and why the interface shows the interval rather than the point.
  Under the *reference* definition the same cohort matters more: without it ROBUST falls
  to 36% on 11 taxa, below CONDITIONAL. The ordering is robust, the ROBUST rate is not.
- **Tiers are uninformative on small studies.** Below roughly 40 samples per group,
  nothing reaches significance in 30% of specifications, so ROBUST and CONDITIONAL are
  never assigned at all. An early version of this experiment ran on Pelto's own 25–60
  sample halves and produced **zero** ROBUST taxa across 388 observations. The tiers are
  not broken there; they are correctly reporting that the data cannot settle the
  question. The interface should not imply otherwise, and §24.7 records what it does.
- **Split-half replication is easier than true external replication.** Both halves share
  protocol, population, sequencing run and batch. A taxon that replicates across an
  independently collected cohort has cleared a higher bar than anything measured here.
  These numbers are an upper bound on external replicability.
- **It validates the ordering, not a per-taxon probability.** "ROBUST taxa replicated 90%
  of the time in this experiment" is not "this taxon has a 90% chance of replicating".

### Why this changes the product

The tier is no longer a label the tool asserts; it is a label with a measured hit rate,
and the interface now prints the rate beside it — including the intervals and the
caveats above. A researcher reading ROBUST can see what that has been worth on 25 other
people's cohorts. A researcher reading INSUFFICIENT can see that it carries no
replication claim at all, because it makes none.

One implementation defect surfaced while writing the boundary tests this result
motivated: `assign_tier` fell through to CONDITIONAL when handed a non-finite
`frac_significant` or `sign_consistency`, because NaN compares false against every
threshold. With tiers now carrying measured replication rates, that silently attached a
51% claim to a taxon nothing was known about. Non-finite evidence now returns
INSUFFICIENT (`tests/test_tier_boundaries.py`).

---

## 24.7 Evidence grading — separating what is tested from what is true

Three kinds of evidence get conflated constantly, and the conflation is how an untested
convention ends up quoted as a finding. MicroVerse now keeps them apart in code
(`app/core/evidence.py`), in the manifest, and on the page:

| Grade | Means | Example here |
|---|---|---|
| **Internally verified** | tests prove the implementation matches the specification | the harmonised effect of §14 |
| **Reference-validated** | output agrees with an independent implementation or a published result | TMM against edgeR; Tierney's numbers reproduced |
| **Empirically validated** | it predicts something on data it has never seen | the §16.2 tiers (§24.6) |
| **Exploratory** | computed correctly, not yet validated | fork attribution (§17) |

Internal verification says nothing about whether the specification is a good idea. That
distinction is the whole point: `tests/test_worked_example.py` proved §16.2 was
implemented correctly for months while the question of whether §16.2 *meant* anything
was untouched.

**Where each grade appears.** `/validation` renders the full matrix, including the rows
where the honest answer is "no held-out experiment" and "not validated against an
external implementation". The results page shows the tier replication rates and
intervals beside the tier counts, and the attribution panel carries its grade, its
within-taxon R², and the agreement between its three estimators. Every run's manifest
embeds the matrix, so an exported bundle carries its own provenance.

**The rule the interface follows:** a component may not be labelled with a grade it
cannot evidence in the cell beneath it. `tests/test_evidence.py` enforces this, and also
re-reads `docs/tier_validation.json` to fail if the numbers the interface quotes ever
drift from the experiment that produced them. A stale number beside a tier is worse than
no number.

### §17 attribution, specifically

Three estimators now run on every analysis rather than two — §17's MixedLM, taxon fixed
effects by exact within-transformation, and §17's mandatory group-means fallback. The
fixed-effects estimator was added because it cannot fail to fit, which makes it a better
second rung than the crude fallback when MixedLM does not converge, and because a
bootstrap needs an estimator that returns a value for every draw rather than only for
the draws that happened to converge.

Four things the earlier implementation did not do, and now does:

1. **Reports what the shares are shares of.** Type II sums of squares partition
   *explained* variance. Calling them "percentage of the variance" without the R² beside
   them overstates them by whatever the model fails to explain — which on real data is
   about 99% (§24.3).
2. **Detects boundary fits.** A random-intercept variance that collapses to zero is
   reported by statsmodels as convergence. It is not the requested model, and it is now
   caught and treated as a failure.
3. **Quantifies uncertainty.** A cluster bootstrap over taxa gives each share a 95%
   interval, shown under the bar.
4. **Grades itself.** A run is `exploratory` unless the mixed model fitted, the forks
   explain at least 10% of the within-taxon variance, and the three estimators agree at
   Spearman ≥ 0.60. On the 17 MicrobiomeHD cohorts, 0/17 and 3/17 runs clear that bar —
   which is why the claim in §24.3 was withdrawn rather than softened.

Methodologically the decomposition follows Young & Holsteen (2017), with Type II sums of
squares per Langsrud (2003) because §11's validity pruning makes the fork grid
unbalanced. There is no external reference implementation for fork attribution and no
held-out experiment validating it, so `exploratory` is the ceiling available to it, and
the product says so rather than implying more.

---

## 26. References

1. Tierney BT, Tan Y, Yang Z, et al. Systematically assessing microbiome–disease associations identifies drivers of inconsistency in metagenomic research. *PLOS Biol* 2022;20(3):e3001556 — **anchor; reproduce this**
2. Nearing JT, Douglas GM, Hayes MG, et al. Microbiome differential abundance methods produce different results across 38 datasets. *Nat Commun* 2022;13:342
3. Pelto J, Auranen K, Kujala JV, Lahti L. Elementary methods provide more replicable results in microbial differential abundance analysis. *Brief Bioinform* 2025;26(2):bbaf130
4. Steegen S, Tuerlinckx F, Gelman A, Vanpaemel W. Increasing transparency through a multiverse analysis. *Perspect Psychol Sci* 2016;11:702–712
5. Simonsohn U, Simmons JP, Nelson LD. Specification curve analysis. *Nat Hum Behav* 2020;4:1208–1214
6. Patel CJ, Burford B, Ioannidis JPA. Assessment of vibration of effects due to model specification. *J Clin Epidemiol* 2015;68:1046–1058
7. Tierney BT, Anderson E, Tan Y, et al. Leveraging vibration of effects analysis for robust discovery. *PLOS Biol* 2021;19:e3001398 — quantvoe
8. McMurdie PJ, Holmes S. Waste not, want not: why rarefying microbiome data is inadmissible. *PLOS Comput Biol* 2014;10:e1003531
9. Schloss PD. Rarefaction is currently the best approach to control for uneven sequencing effort. *mSphere* 2024;9:e00354-23
10. Cameron ES, Schmidt PJ, et al. To rarefy or not to rarefy. *Bioinformatics* 2022;38(9):2389
11. Wang S. Multiscale adaptive differential abundance analysis. *Bioinformatics* 2023;39
12. Efficient and scalable Python implementation of ANCOM-BC. bioRxiv 2026 — scikit-bio port
13. Muzellec B, et al. PyDESeq2. *Bioinformatics* 2023;39
14. dar: Differential Abundance Analysis by Consensus. Bioconductor 2025 — closest competitor

## 27. Resources — all open, all clonable

| Resource | Location | Use |
|---|---|---|
| Tierney VoE code | `github.com/chiragjp/ubiome_robustness` | Validation 1 |
| Tierney VoE data | `figshare.com/projects/Microbiome_robustness/127607` | Validation 1 |
| quantvoe | `github.com/chiragjp/quantvoe` | Reference implementation |
| voe (R + Python) | `github.com/chiragjp/voe` | Python port exists |
| Pelto replicability | `github.com/jepelt/DAA_replicability` (Zenodo 10.5281/zenodo.15047338) | Validation 3 |
| dar | `github.com/MicrobialGenomics-IrsicaixaOrg/dar` | Competitor; study its grid |
| curatedMetagenomicData | Bioconductor | Source cohorts |
| MicrobiomeHD | public | Test data |

**No paywalls, no manual downloads, no curation phase.**

## 28. Build kickoff — M1

```
I'm building MicroVerse, a web server for multiverse analysis of microbiome
differential abundance. Read MICROVERSE_SPEC_v2.md before writing any code.

Build Milestone M1 ONLY (§22):

- Python 3.11, FastAPI, SQLAlchemy + SQLite (jobs only — no curated DB)
- Implement the Specification and TaxonResult dataclasses exactly as
  written in §13, including the matrix_key property
- Implement parsers for CSV/TSV and BIOM v1/v2 into an AbundanceTable
  object; auto-detect orientation and counts-vs-relative per §8
- Implement the input validation rules in §8 with the exact error
  messages described — no stack traces
- pytest tests for parsing, including deliberately malformed files, and
  a test that Specification hashes correctly and matrix_key groups
  specs as expected

Do NOT build yet: preprocessing, DA methods, grid enumeration, plots.

Scope guard (§21) applies from now on: two-group comparisons only, no
diversity features, no "best specification" export ever. Ask before adding
any dependency not listed in §20 — note scikit-bio must be >=0.7.1.
```
