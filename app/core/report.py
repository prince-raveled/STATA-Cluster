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

from .evidence import TIER_CAVEATS, TIER_REPLICATION, TIER_VALIDATION, matrix_rows
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
def specification_curve(run, taxon_id: int, max_points: int = 4000) -> dict:
    """Simonsohn's two-panel plot for one taxon.

    Upper panel: the harmonised effect for every tested specification, sorted
    ascending, coloured by significance. Lower panel: which fork level was active in
    each specification, x-aligned to the panel above — this is how you *see* that the
    null results all sit under one rarefaction depth.
    """
    rows = run.long[run.long["taxon"] == taxon_id]
    if rows.empty:
        return {"taxon": run.taxa_display[taxon_id], "n_specs": 0, "points": []}

    rows = rows.sort_values("effect_h", kind="mergesort").reset_index(drop=True)
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
        "effect": [float(v) for v in thinned["effect_h"]],
        "significant": [bool(v) for v in thinned["significant"]],
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
    lines.append(
        "Significance was taken from each method's own p-value after FDR adjustment "
        "within that specification. Effect size was computed separately and identically "
        "for every specification as the log2 fold change of mean relative abundance "
        "between groups, so that results from methods returning incompatible statistics "
        "could be placed on a common axis."
    )
    lines.append(
        f"Taxa were tiered by robustness: {counts['ROBUST']} ROBUST (FDR-significant in "
        f"at least 80% of the specifications in which they were testable, with at least "
        f"95% sign consistency), {counts['CONDITIONAL']} CONDITIONAL (30-80%), "
        f"{counts['FRAGILE']} FRAGILE (under 30%), and {counts['UNSTABLE']} UNSTABLE "
        f"(sign consistency below 80%, i.e. the direction of effect is not determined by "
        f"the data). {counts['NOT DETECTED']} taxa were significant in no specification "
        f"and {counts['INSUFFICIENT']} were testable in fewer than ten."
    )
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
        "tier_replication": {
            tier: {"rate": facts["rate"], "ci": list(facts["ci"]), "n": facts["n"]}
            for tier, facts in TIER_REPLICATION.items()
        },
        "tier_experiment": TIER_VALIDATION,
        "tier_caveats": list(TIER_CAVEATS),
    }
    return manifest


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
        archive.writestr("README.txt", ZIP_README)
        archive.writestr("methods.txt",
                         methods_paragraph(run, summary, attribution) + "\n\nReferences\n\n"
                         + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(citation_list())))
        archive.writestr("taxa_robustness.csv", robustness_csv(summary))
        archive.writestr("specifications.csv", specifications_csv(run, summary))
        archive.writestr("results_long.csv.gz", long_csv_gz)
        if attribution:
            archive.writestr("attribution.csv", attribution_csv(attribution))
        archive.writestr("manifest.json",
                         json.dumps(run_manifest(run, summary, attribution), indent=2,
                                    default=str))
    return buffer.getvalue()


def tier_legend() -> list:
    return [{"name": name, **meta} for name, meta in TIERS.items()]
