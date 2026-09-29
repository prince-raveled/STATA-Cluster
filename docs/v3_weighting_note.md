# How MicroVerse v3 weights specifications

A one-page rationale for `app/core/weights.py` and `app/core/weighted.py`
(plan §26). Every quantity below is computed on every v3 run and shown on its results
page; nothing here is a measured result about data.

## 1. Why one vote per specification takes a side

v2 counted every valid specification once. The rarefaction fork has 13 levels (no
rarefaction, plus 4 depths × 3 seeds), while "do not rarefy" is one level. On the IBD
demo after rule R8, each branch has 10 valid (transformation, test) pairs, so per FDR
setting there are 12 × 10 = 120 rarefied and 10 unrarefied specifications. One vote each
puts 120/130 ≈ 0.92 of the weight behind rarefying. A taxon significant only after
rarefying can then be ROBUST, while one significant only without rarefying cannot
rise above FRAGILE. Whether to rarefy is a contested choice (type U in the defensibility
register; McMurdie & Holmes 2014, Schloss 2024), so the vote count should not decide it.
Del Giudice & Gangestad (2021) describe how an unbalanced multiverse can bury a
defensible alternative among many variants of another.

## 2. Weights as a walk down a decision tree

An analyst walks the pipeline in the SPEC §9 order and, at each node, chooses among
the options that are still valid given the choices already made:

    rarefy? → depth → seed → rank → prevalence → (transformation, test) → FDR → covariates

For a specification *s* with level *ℓₖ(s)* at node *k*,

    w(s) = ∏ₖ P( ℓₖ(s) | ℓ₁(s), …, ℓₖ₋₁(s) ),

where each conditional probability is taken over the valid children of that node only.

| Scheme | Conditional probabilities |
|---|---|
| `uniform` | w(s) = 1/N for all N valid specifications (v2) |
| `decision_tree` (default) | rarefy yes / no: ½ each when both are valid; every other node: 1/(number of valid children) |
| `flat_tree` | the first two nodes merged into one five-level node (none, min, 1000, 5000, 10000), 1/5 each; then as `decision_tree` |
| `custom` | user-declared probabilities per node, renormalised over the valid children |

Transformation and test form one node because pruning acts on the pair: TMM exists only
without rarefaction, and after R8 raw counts with a rank, t or linear test exist only with
rarefaction. Separate nodes would have structural zeros that no per-fork balancing can
satisfy.

**Properties.** (i) At every node the conditional probabilities sum to 1 over the valid
children, so the weights sum to 1 whatever was pruned — a weighting can never be
infeasible. (ii) Seeds split their depth's weight equally but stay separate
specifications, so the random draw's share of the variation stays measurable.
(iii) On the IBD demo the rarefied share is 120/130 under `uniform`, 4/5 under
`flat_tree` and 1/2 under `decision_tree`, by construction.

## 3. Weighted tiers, conditioned on the taxon being tested

Let *Tⱼ* be the specifications that tested taxon *j* (it survived the prevalence
filter). The weights are renormalised over *Tⱼ*, which is the weighted form of the SPEC
§15 rule that a denominator is "specifications tested", never "specifications run":

    w̃ⱼ(s) = w(s) / Σ_{t ∈ Tⱼ} w(t),                 s ∈ Tⱼ
    frac_significant_w(j) = Σ_{s ∈ Tⱼ} w̃ⱼ(s) · 1[s significant for j]
    sign_consistency_w(j) = max( Σ w̃ⱼ(s)·1[effect > 0], Σ w̃ⱼ(s)·1[effect < 0] )

The tier rules and thresholds of SPEC §16.2 are unchanged; only the votes they count are
weighted. Under `uniform` every weight is exactly 1, so the arithmetic is v2's to the last
bit. The median and interquartile range of the harmonised effect are weighted quantiles:
each specification sits at the midpoint of its cumulative weight and the quantile is
interpolated between those points.

## 4. Effective number of specifications

Kish's effective sample size, applied to specifications:

    n_eff = (Σₛ w(s))² / Σₛ w(s)²

It equals N under `uniform` and falls as the weight concentrates. It is reported for the
whole grid under every scheme, and per taxon over *Tⱼ*. It is a summary of how the weight
is spread, not a count of independent analyses: the specifications are re-analyses of the
same samples under any weighting.

## 5. Weight stability

A taxon is `weight_stable` when its tier is the same under `uniform`, `flat_tree` and
`decision_tree` (and `practice`, once the practice review exists). Otherwise the results
table marks it and the taxon page shows its tier under each scheme.

## 6. Two further axes

**Signed z.** For every specification, z(s) = sign(native effect) · Φ⁻¹(1 − p/2), with p
floored at 10⁻³⁰⁰ so z stays finite; p = 1 gives 0. The harmonised log₂ fold change of
SPEC §14 is computed from the matrix alone and cannot move when only the covariate set
changes; the signed z comes from each method's own test and does. The specification curve
can be drawn on either axis.

**Pseudocount sensitivity.** The harmonised effect uses one pseudocount ε per run
(SPEC §24.1 A4). It is recomputed at 0.1ε and 10ε; a taxon whose median direction differs
at either is flagged.

## 7. What this does not do

A weight is not the probability that a pipeline is correct, and no weighting makes the
specifications independent. Choosing the weighting scheme with the best validation
result would itself be a fork, so `decision_tree` is the default on principle (plan §33,
V8). The replication rates measured for v2's tiers were measured with one vote per
specification under v2's rules, and are not shown beside decision-tree tiers.

## References

- Del Giudice M, Gangestad SW. A traveler's guide to the multiverse: promises, pitfalls,
  and a framework for the evaluation of analytic decisions. *Adv Methods Pract Psychol
  Sci* 2021;4(1). doi:10.1177/2515245920954925
- Kish L. *Survey Sampling.* New York: Wiley; 1965.
- McMurdie PJ, Holmes S. Waste not, want not: why rarefying microbiome data is
  inadmissible. *PLoS Comput Biol* 2014;10:e1003531.
- Schloss PD. Rarefaction is currently the best approach to control for uneven sequencing
  effort in amplicon sequence analyses. *mSphere* 2024;9:e00354-23.
