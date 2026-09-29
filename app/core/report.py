"""Outputs — SPEC §16.3, §16.5, §18.

Everything a user can take away is built here. §18 is a design constraint on this
module specifically: there is no "best specification" export, and every export
contains the whole distribution rather than a filtered slice.
"""
from __future__ import annotations

import gzip
import io
import json
import zipfile

import numpy as np
import pandas as pd

from .effects import signed_z
from .evidence import (
    CROSS_STUDY,
    NEARING_STUDY,
    PUBLISHED_STUDY,
    REPLICATION_BY_LABELLING,
    TIER_CAVEATS,
    TIER_VALIDATION,
    WEIGHTING_VALIDATION,
    labelling_of,
    matrix_rows,
)
from .methods.deseq import SIZE_FACTORS
from .models import FORK_LABELS, METHOD_LABELS, METHOD_SHORT, ordinal
from .robustness import TIERS, locate_declared

CURVE_FORKS = ["rarefaction", "rank", "prev_filter", "transform", "method", "fdr_method",
               "fdr_threshold"]

#: Fixed display order for the dot matrix. Alphabetical would put rarefaction depths in
#: the order 1000, 10000, 5000, which reads as a mistake and hides the depth gradient
#: the panel exists to reveal.
_FIXED_ORDER = {
    "rank": ["input", "genus"],
    "transform": ["RAW", "TSS", "CLR", "TMM"],
    "method": list(METHOD_SHORT.values()),
    "fdr_method": ["BH", "BY"],
}


def _order_levels(fork: str, levels) -> list:
    values = list(levels)
    fixed = _FIXED_ORDER.get(fork)
    if fixed:
        known = [v for v in fixed if v in values]
        return known + sorted(v for v in values if v not in set(fixed))

    if fork == "rarefaction":
        def key(label: str):
            if label == "none":
                return (0, 0.0, 0)
            depth, _, seed = label.partition(" (seed ")
            seed_number = int(seed.rstrip(")")) if seed else 0
            if depth.startswith("min"):
                return (1, 0.0, seed_number)
            try:
                return (2, float(depth), seed_number)
            except ValueError:
                return (3, 0.0, seed_number)
        return sorted(values, key=key)

    if fork in ("prev_filter", "fdr_threshold"):
        def numeric(label: str):
            try:
                return float(str(label).rstrip("%"))
            except ValueError:
                return float("inf")
        return sorted(values, key=numeric)

    return sorted(values)


# --------------------------------------------------------------------------
# §16.3 Specification curve
# --------------------------------------------------------------------------
#: The two y axes the curve can be drawn on (plan §26.4). `effect` is SPEC §14's
#: harmonised log2 fold change; `z` is each specification's signed z, which moves with
#: the covariate set where the harmonised effect cannot.
CURVE_AXES = {
    "effect": "harmonised log2 fold change",
    "z": "signed z (from each test's own p-value and direction)",
}


def specification_curve(run, taxon_id: int, max_points: int = 4000,
                        axis: str = "effect", certified=None) -> dict:
    """Simonsohn's two-panel plot for one taxon.

    Upper panel: every tested specification on the chosen axis — the harmonised effect
    (default) or the signed z — sorted ascending, coloured by significance. Lower panel:
    which fork level was active in each specification, x-aligned to the panel above —
    this is how you *see* that the null results all sit under one rarefaction depth.

    `effect` and `z` are both returned in plotting order whichever axis sorts them, so
    a client can label a point with either. `certified` is the set of specification ids
    a calibrated run rejected for this taxon with error control (plan §27.5); each
    plotted point says whether it is one.
    """
    if axis not in CURVE_AXES:
        raise ValueError(f"Unknown axis '{axis}'. Available: {', '.join(CURVE_AXES)}.")
    rows = run.long[run.long["taxon"] == taxon_id]
    if rows.empty:
        # Same keys as the populated case, all empty. Returning a different shape here
        # meant the renderer read `data.effect` as undefined and threw, so a taxon the
        # prevalence filter removed everywhere broke the plot instead of showing that
        # there was nothing to plot.
        return {
            "taxon": run.taxa_display[taxon_id],
            "taxon_full": run.taxa_names[taxon_id],
            "taxon_id": int(taxon_id),
            "n_specs": 0, "n_plotted": 0, "stride": 1,
            "group_a": run.group_labels[0], "group_b": run.group_labels[1],
            "axis": axis, "axis_label": CURVE_AXES[axis],
            "effect": [], "z": [], "significant": [], "certified": [], "p_adjusted": [],
            "p_raw": [],
            "spec_id": [], "labels": [],
            "forks": {}, "fork_labels": {}, "categories": {},
            "declared_position": -1, "declared": {},
            # None, not NaN. This branch is delivered as JSON, and JSON has no NaN:
            # Starlette serialises with allow_nan=False, so a float("nan") here made
            # the endpoint answer 500 instead of this carefully worded note. null is
            # also the honest value -- 0.0 would claim an effect of zero and a
            # significance rate of zero, when nothing was measured at all.
            "median_effect": None, "frac_significant": None,
            "note": "This taxon was filtered out of every specification, so there is "
                    "nothing to plot.",
        }

    rows = rows.assign(z=signed_z(rows["p_raw"].to_numpy(dtype=float),
                                  rows["effect_n"].to_numpy(dtype=float)))
    sort_key = "effect_h" if axis == "effect" else "z"
    rows = rows.sort_values(sort_key, kind="mergesort").reset_index(drop=True)
    thinned = rows
    stride = 1
    if len(rows) > max_points:
        # Keep the shape of the curve: take an even stride, never a filtered slice (§18).
        stride = int(np.ceil(len(rows) / max_points))
        thinned = rows.iloc[::stride].reset_index(drop=True)

    specs = [run.specs[i] for i in thinned["spec_id"]]
    fork_levels = {fork: [] for fork in CURVE_FORKS}
    for spec in specs:
        levels = spec.fork_levels()
        for fork in CURVE_FORKS:
            fork_levels[fork].append(levels[fork])

    declared_position = -1
    if run.declared_spec_id >= 0:
        matches = np.flatnonzero(thinned["spec_id"].to_numpy() == run.declared_spec_id)
        if matches.size:
            declared_position = int(matches[0])

    # Fork level -> the ordered set of categories, so the dot matrix has stable rows.
    categories = {fork: _order_levels(fork, set(values)) for fork, values in fork_levels.items()}

    return {
        "taxon": run.taxa_display[taxon_id],
        "taxon_full": run.taxa_names[taxon_id],
        "taxon_id": int(taxon_id),
        "n_specs": int(len(rows)),
        "n_plotted": int(len(thinned)),
        "stride": stride,
        "group_a": run.group_labels[0],
        "group_b": run.group_labels[1],
        "axis": axis,
        "axis_label": CURVE_AXES[axis],
        "effect": [float(v) for v in thinned["effect_h"]],
        "z": [float(v) for v in thinned["z"]],
        "significant": [bool(v) for v in thinned["significant"]],
        "certified": ([int(v) in certified for v in thinned["spec_id"]]
                      if certified is not None else []),
        "p_adjusted": [float(v) for v in thinned["p_adj"]],
        "p_raw": [float(v) for v in thinned["p_raw"]],
        "spec_id": [int(v) for v in thinned["spec_id"]],
        "labels": [spec.describe() for spec in specs],
        "forks": fork_levels,
        "fork_labels": {fork: FORK_LABELS[fork] for fork in CURVE_FORKS},
        "categories": categories,
        "declared_position": declared_position,
        "declared": locate_declared(run, taxon_id),
        "median_effect": float(rows["effect_h"].median()),
        "frac_significant": float(rows["significant"].mean()),
    }


# --------------------------------------------------------------------------
# v3 plan §26.2 — how the weight was spread, for the page, the paragraph and the manifest
# --------------------------------------------------------------------------
def weighting_summary(summary):
    """The grid-composition panel's contents, or None for a run summarised as v2.

    One entry per scheme, the primary (the one the tiers use) first: where the weight
    sits, how many effective specifications that is, and how many taxa would be
    labelled differently under it.
    """
    if not getattr(summary, "weighted", False) or not summary.composition:
        return None
    from .weights import SCHEME_LABELS

    table = summary.table
    changes = summary.tier_changes or {}
    order = [summary.scheme] + [s for s in summary.composition if s != summary.scheme]
    schemes = []
    for scheme in order:
        comp = summary.composition[scheme]
        schemes.append({
            "scheme": scheme,
            "label": SCHEME_LABELS.get(scheme, scheme),
            "primary": scheme == summary.scheme,
            "n_specs": int(comp.n_specs),
            "n_effective": float(comp.n_effective),
            "rarefied_share": float(comp.rarefied_share),
            "by_method": {k: float(v) for k, v in comp.by_method.items()},
            "by_transform": {k: float(v) for k, v in comp.by_transform.items()},
            "by_rank": {k: float(v) for k, v in comp.by_rank.items()},
            "by_prevalence": {k: float(v) for k, v in comp.by_prevalence.items()},
            "taxa_labelled_differently": (None if scheme == summary.scheme
                                          else int(changes.get(scheme, 0))),
        })
    unstable = (~table["weight_stable"].astype(bool)) if "weight_stable" in table else None
    flips = table.get("pseudocount_sign_flip")
    return {
        "ruleset": summary.ruleset,
        "scheme": summary.scheme,
        "scheme_label": SCHEME_LABELS.get(summary.scheme, summary.scheme),
        "schemes": schemes,
        "n_weight_unstable": int(unstable.sum()) if unstable is not None else 0,
        "n_pseudocount_sign_flips": int(flips.astype(bool).sum()) if flips is not None
                                    else 0,
    }


def _v3_rule_sentence(run) -> str:
    report = run.grid_report
    from .validity import R8_REASON
    removed = int(report.pruned_reasons.get(R8_REASON, 0))
    adjusted = sum(1 for s in run.specs if getattr(s, "depth_adjusted", False))
    return (
        f"Pruning followed rule set v3, which adds two rules to the v2 validity matrix: "
        f"Wilcoxon, Welch and linear-regression tests on unrarefied raw counts were "
        f"removed ({removed:,} specifications), because nothing in them corrects for "
        f"library size, and presence/absence (logistic) models on unrarefied counts "
        f"included log library size as a covariate ({adjusted:,} specifications), "
        f"because whether a taxon is detected depends on sequencing depth (McMurdie & "
        f"Holmes 2014; Weiss et al. 2017)."
    )


def _v3_weighting_sentence(summary) -> str:
    facts = weighting_summary(summary)
    primary = facts["schemes"][0]
    uniform = next((s for s in facts["schemes"] if s["scheme"] == "uniform"), None)
    if summary.scheme == "decision_tree":
        how = ("by a decision tree over the pipeline order: at every choice the weight "
               "was divided equally among the options still valid given the earlier "
               "choices, except that rarefying and not rarefying received half each, "
               "because that is the contested decision (Del Giudice & Gangestad 2021)")
    elif summary.scheme == "flat_tree":
        how = ("by a decision tree whose first choice gave each rarefaction level an "
               "equal share, dividing the weight equally among valid options thereafter")
    elif summary.scheme == "uniform":
        how = "equally, one vote per specification"
    else:
        how = f"by the declared scheme ({primary['label']})"
    sentence = (
        f"Specifications were weighted {how}. Under these weights "
        f"{primary['rarefied_share']:.0%} of the weight was on rarefied specifications"
    )
    if uniform and summary.scheme != "uniform":
        sentence += f" ({uniform['rarefied_share']:.0%} with one vote per specification)"
    sentence += (
        f", and the effective number of specifications (Kish) was "
        f"{primary['n_effective']:,.0f} of {primary['n_specs']:,}. Each taxon's weighted "
        f"share of significant specifications and its weighted direction agreement were "
        f"computed over the specifications in which it was testable, with the weights "
        f"renormalised over them."
    )
    others = [_SCHEME_PHRASES.get(s["scheme"], s["label"]) for s in facts["schemes"]
              if not s["primary"]]
    if others:
        listed = others[0] if len(others) == 1 else (", ".join(others[:-1]) + " and "
                                                     + others[-1])
        n = facts["n_weight_unstable"]
        sentence += (
            f" {n} {'taxon' if n == 1 else 'taxa'} received a different label when the "
            f"specifications were instead weighted by {listed}."
        )
    return sentence


def _calibration_sentences(calibration) -> str:
    """The error-control guarantee and its assumptions, in plain sentences (§27.5)."""
    procedure = {"bh": "Benjamini-Hochberg", "by": "Benjamini-Yekutieli",
                 "ebh": "e-BH"}[calibration.discovery]
    return (
        f"Error control was calibrated by permutation ({calibration.n_permutations:,} "
        f"permutations of the group labels, seed {calibration.seed}): for each taxon, "
        f"the maximum absolute z over the {calibration.specs_calibrated:,} calibrated "
        f"specifications was compared with its permutation distribution (single-step "
        f"maxT, with a generalised Pareto approximation where fewer than ten permutation "
        f"maxima exceeded the observed value), and taxa were selected by "
        f"{procedure} at {calibration.q:g}; {calibration.n_selected} of "
        f"{calibration.n_taxa_tested} taxa were selected. Within a selected taxon, a "
        f"specification was rejected when its maxT-adjusted p-value was at most "
        f"q·R/m = {calibration.within_threshold:.2g}, and a taxon was called CERTIFIED "
        f"ROBUST when rejected specifications carried at least 80% of its weight with at "
        f"least 95% sign agreement ({calibration.n_certified} taxa). This controls the "
        f"false discovery rate across taxa and the selective error within each selected "
        f"taxon if samples are exchangeable between the groups under the null "
        f"hypothesis, and relies on subset pivotality within each taxon's family, which "
        f"strong compositional shifts in other taxa can violate; covariate adjustment "
        f"was not calibrated."
    )


#: How the methods paragraph names each scheme mid-sentence.
_SCHEME_PHRASES = {
    "uniform": "one vote per specification",
    "flat_tree": "an equal share for each rarefaction level",
    "decision_tree": "the decision tree",
}


# --------------------------------------------------------------------------
# §16.5 Methods paragraph
# --------------------------------------------------------------------------
def methods_paragraph(run, summary, attribution=None) -> str:
    """A paragraph describing the multiverse — never a single point in it (§18)."""
    report = run.grid_report
    dataset = run.dataset_summary
    counts = summary.tier_counts
    methods = sorted({METHOD_LABELS[s.method] for s in run.specs})
    depths = sorted({s.rarefaction for s in run.specs})
    transforms = sorted({s.transform.upper() for s in run.specs})
    filters = sorted({f"{s.prev_filter:.0%}" for s in run.specs})
    ranks = sorted({s.rank for s in run.specs})

    lines = []
    lines.append(
        f"Differential abundance was assessed by multiverse analysis using MicroVerse "
        f"(v1.0), which enumerates the space of defensible analytical pipelines and "
        f"reports the distribution of results rather than a single specification "
        f"(Steegen et al. 2016; Simonsohn et al. 2020; Patel et al. 2015). "
        f"The dataset comprised {dataset['n_samples']} samples "
        f"({dataset['n_group_a']} {dataset['group_a']}, {dataset['n_group_b']} "
        f"{dataset['group_b']}) and {dataset['n_taxa']} taxa."
    )
    lines.append(
        f"In {report.mode} mode, {report.n_enumerated:,} specifications were enumerated "
        f"across the analytical forks and {report.n_valid:,} were retained after pruning "
        f"{report.n_pruned:,} statistically incoherent combinations (for example, "
        f"rarefaction combined with a method that models library size internally). "
        f"Forks varied were: rarefaction depth ({', '.join(depths)}; each random draw "
        f"treated as a separate specification rather than averaged), taxonomic rank "
        f"({', '.join(ranks)}), prevalence filter ({', '.join(filters)}), transformation "
        f"({', '.join(transforms)}), differential abundance method "
        f"({', '.join(methods)}), and multiple-testing correction "
        f"(Benjamini-Hochberg at 0.05 and 0.10, Benjamini-Yekutieli at 0.05)."
    )
    weighted = bool(getattr(summary, "weighted", False))
    if getattr(run, "ruleset", "v2") != "v2":
        lines.append(_v3_rule_sentence(run))
    lines.append(
        "Significance was taken from each method's own p-value after FDR adjustment "
        "within that specification. Effect size was computed separately and identically "
        "for every specification as the log2 fold change of mean relative abundance "
        "between groups, so that results from methods returning incompatible statistics "
        "could be placed on a common axis."
    )
    if any(s.method == "pydeseq2" for s in run.specs):
        lines.append(
            "PyDESeq2 used DESeq2's poscounts size factors, the estimator for tables in "
            "which every feature has zeros, and its raw Wald p-values entered the same "
            "FDR correction as every other method."
        )
    if weighted:
        lines.append(_v3_weighting_sentence(summary))
    share_of = ("of the weight of the specifications" if weighted
                else "of the specifications")
    lines.append(
        f"Taxa were tiered by robustness: {counts['ROBUST']} ROBUST (FDR-significant in "
        f"at least 80% {share_of} in which they were testable, with at least "
        f"95% sign consistency), {counts['CONDITIONAL']} CONDITIONAL (30-80%), "
        f"{counts['FRAGILE']} FRAGILE (under 30%), and {counts['UNSTABLE']} UNSTABLE "
        f"(sign consistency below 80%, i.e. the direction of effect is not determined by "
        f"the data). {counts['NOT DETECTED']} taxa were significant in no specification "
        f"and {counts['INSUFFICIENT']} were testable in fewer than ten."
    )
    calibration = getattr(summary, "calibration", None)
    if calibration is not None:
        lines.append(_calibration_sentences(calibration))
    if attribution and attribution.get("significance"):
        significance = attribution["significance"]
        lines.append(
            "Variance in whether a taxon was called significant was attributed to the "
            "analytical forks by " + significance.estimator.split(",")[0].lower() + ": "
            + "; ".join(
                f"{FORK_LABELS.get(fork, fork).lower()} {share:.0f}%"
                for fork, share in significance.ranked if share >= 0.5
            ) + "."
        )
        draw = attribution.get("rarefaction_draw") or {}
        if draw:
            lines.append(
                f"Of the variance attributable to rarefaction, {draw['seed_share']:.0f}% "
                f"arose from the random subsampling draw itself rather than from the "
                f"choice of depth."
            )
    if run.declared_spec_id >= 0 and np.isfinite(summary.declared_percentile):
        lines.append(
            f"The pipeline we would otherwise have reported "
            f"({run.specs[run.declared_spec_id].describe()}) sits at the "
            f"{ordinal(summary.declared_percentile)} percentile of significance across the "
            f"multiverse."
        )
    lines.append(
        "The full specification-level results are provided as supplementary data. "
        "We report the distribution of results across specifications, not a preferred "
        "point within it."
    )
    return "\n\n".join(lines)


#: Every methodology the engine implements, with its primary source. Grouped so the
#: methods report can cite only what a given run actually used, and so a reader can tell
#: a statistical method's reference from a design reference. Nothing here is a
#: convenience citation: each entry is the paper the implementation follows.
CITATIONS = {
    "design": [
        "Steegen S, Tuerlinckx F, Gelman A, Vanpaemel W. Increasing transparency through "
        "a multiverse analysis. Perspect Psychol Sci 2016;11:702-712.",
        "Simonsohn U, Simmons JP, Nelson LD. Specification curve analysis. "
        "Nat Hum Behav 2020;4:1208-1214.",
        "Patel CJ, Burford B, Ioannidis JPA. Assessment of vibration of effects due to "
        "model specification. J Clin Epidemiol 2015;68:1046-1058.",
        "Tierney BT, Tan Y, Yang Z, et al. Systematically assessing microbiome-disease "
        "associations identifies drivers of inconsistency in metagenomic research. "
        "PLOS Biol 2022;20(3):e3001556.",
        "Nearing JT, Douglas GM, Hayes MG, et al. Microbiome differential abundance "
        "methods produce different results across 38 datasets. Nat Commun 2022;13:342.",
        "Pelto J, Auranen K, Kujala JV, Lahti L. Elementary methods provide more "
        "replicable results in microbial differential abundance analysis. "
        "Brief Bioinform 2025;26(2):bbaf130.",
        "Del Giudice M, Gangestad SW. A traveler's guide to the multiverse: promises, "
        "pitfalls, and a framework for the evaluation of analytic decisions. "
        "Adv Methods Pract Psychol Sci 2021;4(1).",
        "Kish L. Survey Sampling. New York: Wiley; 1965.",
    ],
    "attribution": [
        "Young C, Holsteen K. Model uncertainty and robustness: a computational "
        "framework for multimodel analysis. Sociol Methods Res 2017;46(1):3-40.",
        "Munoz J, Young C. We ran 9 billion regressions: eliminating false positives "
        "through computational model robustness. Sociol Methodol 2018;48(1):1-33.",
        "Langsrud O. ANOVA for unbalanced data: use Type II instead of Type III sums of "
        "squares. Stat Comput 2003;13:163-167.",
        "Nakagawa S, Schielzeth H. A general and simple method for obtaining R2 from "
        "generalized linear mixed-effects models. Methods Ecol Evol 2013;4:133-142.",
    ],
    "methods": [
        "Wilcoxon F. Individual comparisons by ranking methods. "
        "Biometrics Bull 1945;1(6):80-83.",
        "Mann HB, Whitney DR. On a test of whether one of two random variables is "
        "stochastically larger than the other. Ann Math Stat 1947;18(1):50-60.",
        "Welch BL. The generalization of Student's problem when several different "
        "population variances are involved. Biometrika 1947;34(1-2):28-35.",
        "Lin H, Peddada SD. Analysis of compositions of microbiomes with bias "
        "correction. Nat Commun 2020;11:3514.",
        "Fernandes AD, Reid JNS, Macklaim JM, et al. Unifying the analysis of "
        "high-throughput sequencing datasets: characterizing RNA-seq, 16S rRNA gene "
        "sequencing and selective growth experiments by compositional data analysis. "
        "Microbiome 2014;2:15.",
        "Love MI, Huber W, Anders S. Moderated estimation of fold change and dispersion "
        "for RNA-seq data with DESeq2. Genome Biol 2014;15:550.",
        "Muzellec B, Telenczuk M, Cabeli V, Andreux M. PyDESeq2: a python package for "
        "bulk RNA-seq differential expression analysis. Bioinformatics 2023;39(9).",
        "Robinson MD, Oshlack A. A scaling normalization method for differential "
        "expression analysis of RNA-seq data. Genome Biol 2010;11:R25.",
        "Robinson MD, McCarthy DJ, Smyth GK. edgeR: a Bioconductor package for "
        "differential expression analysis of digital gene expression data. "
        "Bioinformatics 2010;26(1):139-140.",
    ],
    "preprocessing": [
        "Aitchison J. The statistical analysis of compositional data. "
        "J R Stat Soc B 1982;44(2):139-177.",
        "Martin-Fernandez JA, Barcelo-Vidal C, Pawlowsky-Glahn V. Dealing with zeros "
        "and missing values in compositional data sets using nonparametric imputation. "
        "Math Geol 2003;35(3):253-278.",
        "McMurdie PJ, Holmes S. Waste not, want not: why rarefying microbiome data is "
        "inadmissible. PLOS Comput Biol 2014;10:e1003531.",
        "Schloss PD. Rarefaction is currently the best approach to control for uneven "
        "sequencing effort. mSphere 2024;9:e00354-23.",
        "Cameron ES, Schmidt PJ, et al. To rarefy or not to rarefy. "
        "Bioinformatics 2022;38(9):2389.",
        "Gloor GB, Macklaim JM, Pawlowsky-Glahn V, Egozcue JJ. Microbiome datasets are "
        "compositional: and this is not optional. Front Microbiol 2017;8:2224.",
        "Weiss S, Xu ZZ, Peddada S, et al. Normalization and microbial differential "
        "abundance strategies depend upon data characteristics. Microbiome 2017;5:27.",
    ],
    "multiple_testing": [
        "Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and "
        "powerful approach to multiple testing. J R Stat Soc B 1995;57(1):289-300.",
        "Benjamini Y, Yekutieli D. The control of the false discovery rate in multiple "
        "testing under dependency. Ann Stat 2001;29(4):1165-1188.",
    ],
    "validation": [
        "Duvallet C, Gibbons SM, Gurry T, Irizarry RA, Alm EJ. Meta-analysis of gut "
        "microbiome studies identifies disease-specific and shared responses. "
        "Nat Commun 2017;8:1784.",
        "Pasolli E, Schiffer L, Manghi P, et al. Accessible, curated metagenomic data "
        "through ExperimentHub. Nat Methods 2017;14:1023-1024.",
        "Ioannidis JPA. Why most published research findings are false. "
        "PLOS Med 2005;2(8):e124.",
    ],
}

CITATION_GROUP_LABELS = {
    "design": "Multiverse and specification-curve design",
    "attribution": "Variance attribution across specifications (§17)",
    "methods": "Differential abundance methods (§10)",
    "preprocessing": "Normalisation, transformation and rarefaction (§9)",
    "multiple_testing": "Multiple testing (§10 fork 6)",
    "validation": "Data sources and validation",
}


def citation_list() -> list:
    """Flat list, in group order — what the Method page renders."""
    return [c for group in CITATIONS.values() for c in group]


def citation_groups() -> list:
    """Grouped citations, for the report and anywhere a heading helps."""
    return [{"key": key, "label": CITATION_GROUP_LABELS[key], "citations": list(items)}
            for key, items in CITATIONS.items()]


# --------------------------------------------------------------------------
# §16.5 Exports
# --------------------------------------------------------------------------
def specifications_csv(run, summary) -> bytes:
    frame = summary.spec_summary.copy()
    frame["rarefaction_label"] = [run.specs[i].rarefaction_label for i in frame["spec_id"]]
    frame["description"] = [run.specs[i].describe() for i in frame["spec_id"]]
    frame["frac_taxa_significant"] = frame["n_significant"] / frame["n_taxa_tested"].clip(lower=1)
    return frame.to_csv(index=False).encode()


def robustness_csv(summary) -> bytes:
    return summary.table.to_csv(index=False).encode()


def long_results_csv_gz(run) -> bytes:
    """Every (specification, taxon) row. The whole distribution, never a slice (§18)."""
    frame = run.long.copy()
    if getattr(run, "ruleset", "v2") != "v2":
        # v3 plan §26.4: the signed z for every row, beside the native statistic it is
        # signed by. A v2 run keeps exactly v2's columns.
        z = signed_z(frame["p_raw"].to_numpy(dtype=float),
                     frame["effect_n"].to_numpy(dtype=float)).astype(np.float32)
        frame.insert(frame.columns.get_loc("effect_n") + 1, "signed_z", z)
    frame["taxon_name"] = [run.taxa_names[i] for i in frame["taxon"]]
    frame = frame.merge(run.specs_frame, on="spec_id", how="left")
    frame = frame.drop(columns=["taxon"]).rename(columns={"effect_h": "effect_harmonized",
                                                          "effect_n": "effect_native"})
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=5) as handle:
        handle.write(frame.to_csv(index=False).encode())
    return buffer.getvalue()


def attribution_csv(attribution) -> bytes:
    rows = []
    for target in ("effect", "significance"):
        result = attribution.get(target)
        if not result:
            continue
        for fork, share in result.ranked:
            rows.append({
                "decomposition": target,
                "fork": fork,
                "fork_label": FORK_LABELS.get(fork, fork),
                "percent_of_variance": round(share, 3),
                "fallback_percent": round(result.fallback_shares.get(fork, float("nan")), 3),
                "estimator": result.estimator,
                "converged": result.converged,
            })
    draw = attribution.get("rarefaction_draw") or {}
    for key, value in draw.items():
        rows.append({"decomposition": "rarefaction_draw", "fork": key,
                     "fork_label": key.replace("_", " "), "percent_of_variance": round(value, 3),
                     "fallback_percent": "", "estimator": "within/between depth variance",
                     "converged": True})
    return pd.DataFrame(rows).to_csv(index=False).encode()


def dependency_versions() -> dict:
    """Versions of every package whose behaviour can change a number in the output.

    Recorded per run rather than looked up later: an analysis re-read in a year has to
    say what produced it, not what happens to be installed at the time it is read.
    """
    import platform
    from importlib.metadata import PackageNotFoundError, version

    packages = ("numpy", "pandas", "scipy", "statsmodels", "scikit-learn",
                "pydeseq2", "scikit-bio", "h5py", "fastapi")
    installed = {}
    for name in packages:
        try:
            installed[name] = version(name)
        except PackageNotFoundError:
            installed[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": installed,
    }


def run_manifest(run, summary, attribution=None) -> dict:
    """Everything needed to say exactly how a result was produced — SPEC §20.

    A researcher reading this a year later should be able to establish what data went
    in, what code ran, what versions of what libraries, which specifications were
    executed, which were pruned and why, and which failed.
    """
    import datetime

    report = run.grid_report
    manifest = {
        "microverse_version": "1.0.0",
        "spec_version": "2.0 (frozen)",
        "generated_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "environment": dependency_versions(),
        "seeds": {
            "rarefaction_seeds": sorted(
                {int(s.rare_seed) for s in run.specs if s.rare_seed is not None}),
            "aldex2_instances": getattr(run, "aldex_instances", None),
            "pydeseq2_size_factors": SIZE_FACTORS,
            "note": "Every stochastic step is seeded. Re-running the same input with "
                    "the same mode reproduces the same numbers; rarefaction seeds are "
                    "specification identity, not nuisance.",
        },
        "mode": run.mode,
        "runtime_seconds": round(run.runtime_seconds, 2),
        "dataset": run.dataset_summary,
        "grid": {
            "enumerated": report.n_enumerated,
            "valid": report.n_valid,
            "pruned": report.n_pruned,
            "pruned_reasons": report.pruned_reasons,
            "matrices": report.n_matrices,
            "model_fits": report.n_fits,
            "notes": report.notes,
        },
        "tiers": summary.tier_counts,
        "declared_specification": (
            run.specs[run.declared_spec_id].as_row() if run.declared_spec_id >= 0 else None
        ),
        "declared_percentile": (
            round(float(summary.declared_percentile), 2)
            if np.isfinite(summary.declared_percentile) else None
        ),
        "method_status": run.method_status,
        "failures": run.failures[:50],
    }
    ruleset, scheme = labelling_of(summary)
    if getattr(summary, "weighted", False):
        # v3 plan §26: which rules pruned the grid, how the specifications were
        # weighted, and where that put the weight. A v2 run's manifest is unchanged.
        from .validity import defensibility_register
        manifest["ruleset"] = ruleset
        manifest["rules_applied"] = defensibility_register()["rulesets"][ruleset]
        manifest["weighting"] = weighting_summary(summary)
    calibration = getattr(summary, "calibration", None)
    if calibration is not None:
        manifest["calibration"] = calibration_manifest(calibration)
    if attribution:
        manifest["attribution"] = {
            target: {
                "estimator": result.estimator,
                "converged": result.converged,
                "evidence": result.evidence,
                "explained_variance": (
                    round(float(result.explained_variance), 4)
                    if np.isfinite(result.explained_variance) else None),
                "estimator_agreement": (
                    round(float(result.estimator_agreement()), 3)
                    if np.isfinite(result.estimator_agreement()) else None),
                "warnings": list(result.warnings),
                "shares": {k: round(v, 3) for k, v in result.shares.items()},
                "shares_are": "percentages of the variance the forks explain, "
                              "not of total variance",
                "confidence_intervals": {
                    k: [round(v[0], 2), round(v[1], 2)]
                    for k, v in result.intervals.items()},
                "mixedlm_shares": {k: round(v, 3) for k, v in result.mixedlm_shares.items()},
                "within_shares": {k: round(v, 3) for k, v in result.within_shares.items()},
                "fallback_shares": {k: round(v, 3) for k, v in result.fallback_shares.items()},
                "diagnostics": _jsonable(result.diagnostics),
            }
            for target, result in attribution.items()
            if hasattr(result, "shares")
        }
        manifest["attribution"]["rarefaction_draw"] = attribution.get("rarefaction_draw", {})
        manifest["attribution"]["methodology"] = {
            "reference": "Young & Holsteen (2017), Sociological Methods & Research "
                         "46(1):3-40; Type II SS per Langsrud (2003)",
            "validation_status": "exploratory — no external reference implementation "
                                 "and no held-out experiment validates fork attribution",
        }

    manifest["validation"] = {
        "matrix": matrix_rows(),
        # Replication measured for the labelling this run's tiers were assigned under,
        # or {} when nobody has measured it; a rate belongs to its labelling (V8).
        "tier_replication": _jsonable({
            tier: {"rate": facts["rate"], "ci": list(facts["ci"]), "n": facts["n"]}
            for tier, facts in REPLICATION_BY_LABELLING.get((ruleset, scheme), {}).items()
        }),
        "tier_replication_labelling": {
            "ruleset": ruleset, "scheme": scheme,
            "measured": (ruleset, scheme) in REPLICATION_BY_LABELLING,
        },
        # A V8 interval can be undefined (a tier seen in fewer than three cohorts), and
        # JSON has no NaN: these two pass through _jsonable, which writes null.
        "tier_replication_by_labelling": _jsonable({
            f"{key[0]}/{key[1]}": {
                tier: {"rate": facts["rate"], "ci": list(facts["ci"]), "n": facts["n"]}
                for tier, facts in rates.items()}
            for key, rates in REPLICATION_BY_LABELLING.items()
        }),
        "weighting_experiment": _jsonable(WEIGHTING_VALIDATION) or {
            "experiment": "V8", "status": "pre-registered, not yet run"},
        "tier_experiment": TIER_VALIDATION,
        "tier_caveats": list(TIER_CAVEATS),
        "published_study": PUBLISHED_STUDY,
        "cross_study_replication": CROSS_STUDY,
        "method_benchmark": NEARING_STUDY,
    }
    return manifest


def calibration_manifest(calibration) -> dict:
    """What a calibrated run did, enough to repeat it (plan §27.3): the permutations
    (count, seed and a hash of the label matrix), the procedures, the counts and the
    assumptions its guarantee rests on."""
    from .inference import ASSUMPTIONS, CERTIFIED_CR, CERTIFIED_SIGN, TAIL_P_FLOOR
    return _jsonable({
        "plan": "docs/MICROVERSE_V3_PLAN.md section 27",
        "permutations": calibration.n_permutations,
        "seed": calibration.seed,
        "permutation_generator": "numpy.random.default_rng(seed).permuted, row 0 the "
                                 "observed labels",
        "permutations_sha256": calibration.permutations_sha256 or None,
        "strata": calibration.strata,
        "family_test": "single-step maxT over the calibrated specifications, generalised "
                       "Pareto tail where fewer than 10 permutation maxima reach the "
                       "observed value",
        "tail_p_floor": TAIL_P_FLOOR,
        "discovery": calibration.discovery,
        "q": calibration.q,
        "weights": calibration.scheme,
        "n_taxa_tested": calibration.n_taxa_tested,
        "n_selected": calibration.n_selected,
        "within_threshold": calibration.within_threshold,
        "certified_robust_rule": {"selected": True, "certified_share_at_least": CERTIFIED_CR,
                                  "sign_agreement_at_least": CERTIFIED_SIGN},
        "n_certified_robust": calibration.n_certified,
        "specs_calibrated": calibration.specs_calibrated,
        "specs_excluded": calibration.specs_excluded,
        "timings": calibration.timings,
        "assumptions": list(ASSUMPTIONS),
        "validation": "V9 (plan section 33) has not been run",
    })


def calibration_csv(run, calibration) -> bytes:
    """Every (specification, taxon) pair rejected with error control, with the
    specification's choices: the certified part of the distribution, in full."""
    frame = calibration.rejected.copy()
    frame["taxon_name"] = [run.taxa_names[i] for i in frame["taxon"]]
    frame = frame.merge(run.specs_frame, on="spec_id", how="left")
    return frame.drop(columns=["taxon"]).to_csv(index=False).encode()


def _jsonable(value):
    """Diagnostics carry numpy scalars; JSON does not."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


ZIP_README = """MicroVerse results bundle
=========================

This archive contains the complete distribution of results across every valid
analytical specification. It deliberately does NOT contain a "best" specification,
and never will (SPEC section 18): the point of a multiverse is the distribution, and
a tool that hands you your favourite point in it is a p-hacking device.

Files
-----
  methods.txt              A methods paragraph describing the multiverse, with citations.
  taxa_robustness.csv      Per-taxon metrics and robustness tier (SPEC section 16.2).
  specifications.csv       One row per specification, with how many taxa it called.
  results_long.csv.gz      Every (specification, taxon) result. The full distribution.
  attribution.csv          Variance attributed to each analytical fork (SPEC section 17).
  manifest.json            Run parameters, grid counts, pruning reasons, failures.

How to report this
------------------
Report the distribution, not a point in it. A taxon that is ROBUST survived nearly
every defensible pipeline; a taxon that is UNSTABLE does not have a determined
direction of effect in your data, whatever a single model said.

A taxon being robust to analytical choice does not make it biologically real.
Confounding, contamination and batch effects survive a multiverse intact.
"""


ZIP_README_V3 = """
This run: rule set v3, weighted specifications
----------------------------------------------
The grid was pruned with rule set v3 (v2's rules plus R8 and R9; the register with
reasons and citations is on the Method page), and the tiers were assigned with every
specification weighted by the scheme named in manifest.json ("weighting"). Columns
that exist only in a weighted run:

  taxa_robustness.csv   frac_significant, frac_nominal and sign_consistency are the
                        weighted shares the tier uses; frac_significant_unweighted is
                        the one-vote count; tier_<scheme> is the tier under each scheme;
                        weight_stable is True when every scheme gives the same tier;
                        n_eff_specs is the taxon's effective number of specifications;
                        median_effect_pc_low / _high and pseudocount_sign_flip are the
                        harmonised effect at 0.1x and 10x the run's pseudocount.
  specifications.csv    weight (the scheme the tiers use) and weight_<scheme>.
  results_long.csv.gz   signed_z: sign(native effect) x z of the test's own p-value.

A weight is not a probability that a pipeline is right. The replication rates measured
for v2's one-vote labels are not evidence about weighted labels; manifest.json says which
labellings have been measured ("tier_replication_by_labelling").
"""


ZIP_README_CALIBRATED = """
This run: calibrated
--------------------
Error control was calibrated by permuting the group labels (manifest.json,
"calibration", gives the number of permutations, the seed, the procedures and the
assumptions). taxa_robustness.csv gains calibrated_p and calibrated_q (the taxon's
family p-value and its FDR-adjusted value across taxa), calibrated_discovery,
certified_share (the weight of specifications rejected with error control),
certified_sign_agreement and certified_robust. calibration_rejections.csv lists every
(specification, taxon) pair rejected, with the specification's choices. The descriptive
tier is kept beside the calibrated result; neither replaces the other.
"""


def build_zip(run, summary, attribution=None, long_csv_gz: bytes = None) -> bytes:
    """Assemble the results bundle.

    `long_csv_gz` lets the caller pass the already-compressed long-format results.
    Gzipping ~2.4M rows takes about 14 seconds at the §8 taxon ceiling, and the run
    writes that file to disk anyway — regenerating it here doubled the cost of every
    large job.
    """
    if long_csv_gz is None:
        long_csv_gz = long_results_csv_gz(run)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.txt", ZIP_README + (
            ZIP_README_V3 if getattr(summary, "weighted", False) else "") + (
            ZIP_README_CALIBRATED if getattr(summary, "calibration", None) is not None
            else ""))
        archive.writestr("methods.txt",
                         methods_paragraph(run, summary, attribution) + "\n\nReferences\n\n"
                         + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(citation_list())))
        archive.writestr("taxa_robustness.csv", robustness_csv(summary))
        archive.writestr("specifications.csv", specifications_csv(run, summary))
        archive.writestr("results_long.csv.gz", long_csv_gz)
        if attribution:
            archive.writestr("attribution.csv", attribution_csv(attribution))
        calibration = getattr(summary, "calibration", None)
        if calibration is not None:
            archive.writestr("calibration_rejections.csv", calibration_csv(run, calibration))
        archive.writestr("manifest.json",
                         json.dumps(run_manifest(run, summary, attribution), indent=2,
                                    default=str))
    return buffer.getvalue()


def tier_legend() -> list:
    return [{"name": name, **meta} for name, meta in TIERS.items()]
