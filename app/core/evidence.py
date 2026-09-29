"""What is actually known about each part of MicroVerse, and how it was established.

Three kinds of evidence get confused constantly, and the confusion is what lets a tool
present an untested convention as a finding. They are kept apart here and the interface
renders them differently:

  INTERNAL    The implementation does what the specification says. `pytest` proves this.
              It says nothing about whether the specification is a good idea.
  REFERENCE   The output agrees with an independent implementation of the same method —
              edgeR, ALDEx2, ANCOM-BC, scikit-bio — or with a published result.
  EMPIRICAL   The thing predicts something on data it has never seen. This is the only
              grade that licenses a claim about the world.
  EXPLORATORY Interesting, computed correctly, and not yet validated as any of the above.

Nothing is marked EMPIRICAL here unless a held-out experiment in `tests/reference/`
produced the number quoted, and the number quoted is the one that experiment produced.
`tests/test_evidence.py` re-reads the experiment's own output and fails if these drift.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

INTERNAL = "internal"
REFERENCE = "reference"
EMPIRICAL = "empirical"
EXPLORATORY = "exploratory"

GRADE_LABEL = {
    INTERNAL: "Internally verified",
    REFERENCE: "Reference-validated",
    EMPIRICAL: "Empirically validated",
    EXPLORATORY: "Exploratory",
}

GRADE_BLURB = {
    INTERNAL: "Tests prove the implementation matches the specification. That is not "
              "evidence that the specification is correct.",
    REFERENCE: "Output agrees with an independent implementation of the same method, or "
               "with a published result.",
    EMPIRICAL: "Validated on data it had never seen — it predicts, not merely computes.",
    EXPLORATORY: "Computed correctly, not yet validated. Treat as a hypothesis.",
}


# ---------------------------------------------------------------------------
# The §16.2 tier validation — tests/reference/tier_validation.py
# ---------------------------------------------------------------------------
#: Provenance of the numbers below. Regenerate with:
#:     .venv/Scripts/python tests/reference/tier_validation.py
TIER_VALIDATION = {
    "experiment": "tests/reference/tier_validation.py",
    "record": "docs/tier_validation.json",
    "design": "stratified 50/50 discovery/validation splits; tiers assigned on the "
              "discovery half, replication scored tier-blind on the held-out half",
    "data": "25 published case-control cohorts curated by Pelto et al. "
            "(arXiv:2404.02691; Zenodo 10.5281/zenodo.15047338), 16S and shotgun",
    "n_cohorts": 25,
    "n_splits": 75,
    "n_observations": 12564,
    "definition": "nominally significant in most held-out specifications, same direction",
    "auc": 0.785,
    "risk_ratio_robust": 7.4,
    "null_rate": 0.007,
    "mode": "quick",
    #: What the labels in that experiment were: the v2 engine's, one vote per
    #: specification under the v2 pruning rules. A v3 run's decision-tree labels are a
    #: different labelling, and these rates are not evidence about them (plan §33 V8).
    "ruleset": "v2",
    "scheme": "uniform",
}

#: Replication rate by tier, with a cluster bootstrap over cohorts. `share` is how often
#: the tier is assigned at all — ROBUST is rare, which is the point of it.
TIER_REPLICATION = {
    "ROBUST": {"rate": 0.90, "ci": (0.60, 0.93), "n": 60, "cohorts": 5, "share": 0.005},
    "CONDITIONAL": {"rate": 0.51, "ci": (0.39, 0.60), "n": 351, "cohorts": 12,
                    "share": 0.028},
    "FRAGILE": {"rate": 0.12, "ci": (0.08, 0.17), "n": 2029, "cohorts": 25,
                "share": 0.161},
    "UNSTABLE": {"rate": 0.02, "ci": (0.01, 0.04), "n": 1468, "cohorts": 25,
                 "share": 0.117},
    "NOT DETECTED": {"rate": 0.03, "ci": (0.02, 0.04), "n": 8656, "cohorts": 23,
                     "share": 0.689},
}

# ---------------------------------------------------------------------------
# A published analysis reproduced, and replication across studies —
# tests/reference/published_findings_study.py (SPEC §24.8)
# ---------------------------------------------------------------------------
#: Provenance of the numbers below. Regenerate with:
#:     .venv/Scripts/python tests/reference/published_findings_study.py
PUBLISHED_STUDY = {
    "experiment": "tests/reference/published_findings_study.py",
    "record": "docs/published_findings_study.json",
    "paper": "Duvallet C et al., Meta-analysis of gut microbiome studies identifies "
             "disease-specific and shared responses, Nat Commun 2017;8:1784",
    "data": "MicrobiomeHD (Zenodo 1146764, CC-BY-NC-4.0): 16S case-control cohorts in "
            "colorectal cancer, C. difficile infection, IBD, HIV and obesity",
    "design": "the paper's pipeline re-implemented from the authors' own code, compared "
              "with the MicroVerse specification that matches it",
    "n_cohorts": 19,
    "n_refused": 1,          # 4 controls, under the 5-per-group minimum: refused, correctly
    "n_samples": 2824,
    "paper_significant": 217,
    "recovered": 214,
    "extra": 9,
    "n_exact": 14,
    "same_matrix_identical": 19,
    "max_p_difference": 3e-8,
    #: The paper's shared-response genera (its Supplementary File 3): (agree, total) on
    #: direction, by the MicroVerse label.
    "shared_response_direction": {
        "ROBUST": (4, 4), "CONDITIONAL": (45, 54), "FRAGILE": (152, 215), "UNSTABLE": (72, 128),
    },
}

#: The same record's external test. A genus the paper's pipeline calls significant in one
#: cohort is checked, with the paper's own pipeline, in every other cohort of the disease.
CROSS_STUDY = {
    "n_cohorts": 19,
    "n_pairs": 3688,
    "n_genera": 181,
    "definition": "significant (q < 0.05) under the paper's pipeline in another cohort of "
                  "the same disease, in the same direction",
    "stable": {"label": "ROBUST or CONDITIONAL", "n": 211, "rate": 0.133,
               "ci": (0.071, 0.196), "direction": 0.644},
    "shaky": {"label": "FRAGILE or UNSTABLE", "n": 176, "rate": 0.136,
              "ci": (0.072, 0.202), "direction": 0.636},
    "difference": -0.004,
    "difference_ci": (-0.072, 0.068),
}

# ---------------------------------------------------------------------------
# The methods against a published benchmark — tests/reference/nearing_study.py
# (SPEC §24.9)
# ---------------------------------------------------------------------------
#: Nearing et al. published, for 38 datasets and 14 methods, how many ASVs each method
#: called significant; five of those methods are MicroVerse's. `same_count` is (datasets
#: where MicroVerse gives the paper's exact count, datasets compared).
NEARING_STUDY = {
    "experiment": "tests/reference/nearing_study.py",
    "record": "docs/nearing_study.json",
    "paper": "Nearing JT et al., Microbiome differential abundance methods produce "
             "different results across 38 datasets, Nat Commun 2022;13:342",
    "data": "the paper's datasets (figshare 14531724, CC BY 4.0) and its published count "
            "of significant ASVs for every dataset and method",
    "n_datasets": 24,
    "n_comparisons": 117,
    "inputs_rebuilt": (117, 117),       # tables with exactly the paper's feature count
    "methods": [
        {"paper": "Wilcoxon (rarefied)", "same_count": (10, 10),
         "note": "Identical: same table, same test, same correction"},
        {"paper": "Welch's t-test (rarefied)", "same_count": (10, 10),
         "note": "Identical"},
        {"paper": "Wilcoxon (CLR)", "same_count": (33, 33),
         "note": "Identical given the paper's CLR (counts + 1); with MicroVerse's own "
                 "zero replacement, 20 of 33"},
        {"paper": "ALDEx2", "same_count": (19, 33),
         "note": "MicroVerse's version is more conservative: it takes the larger of the "
                 "Wilcoxon and Welch expected p-values"},
        {"paper": "DESeq2", "same_count": (11, 31),
         "note": "Counts close but lower: MicroVerse applies one FDR correction to every "
                 "method and skips DESeq2's independent filtering"},
    ],
    "own_clr_same_count": (20, 33),
    #: Datasets where the five methods fall in exactly the paper's order, of those
    #: where more than one method found anything; median rank correlation 1.0.
    "method_order_exact": (14, 19),
    "method_order_median_spearman": 1.0,
    #: Through the live site, on the paper's own rarefied tables (26 September 2026):
    #: Chemerin 533 and 365, edd_singh 340 and 18 — the paper's numbers.
    "website_same_count": (4, 4),
}

#: Stated wherever a ROBUST call is displayed. The rate is real and the ordering survives
#: dropping any single cohort, but the rate itself rests on few cohorts and must not be
#: quoted as a precise probability — and it is a within-study rate.
TIER_CAVEATS = [
    "These rates are replication within a study, on held-out samples of the same cohort. "
    f"Across independent studies of the same disease ({CROSS_STUDY['n_cohorts']} published "
    "cohorts), the labels did not predict which findings replicated: "
    f"{CROSS_STUDY['stable']['rate']:.0%} of ROBUST or CONDITIONAL findings against "
    f"{CROSS_STUDY['shaky']['rate']:.0%} of FRAGILE or UNSTABLE ones. A label says how far "
    "the answer depends on analytical choices in these data, not whether it will hold in "
    "another population.",
    "ROBUST is rare: it was assigned to 0.5% of taxon observations, in 5 of 25 cohorts. "
    "The tier is conservative by construction — it says little, and what it says held "
    "up 9 times in 10.",
    "49 of the 60 ROBUST calls came from one cohort with an unusually large effect "
    "(cdi_schubert, C. difficile). Dropping it leaves 11 ROBUST taxa replicating at "
    "73%, still far above CONDITIONAL — the ordering survives, the exact rate is "
    "uncertain, and the confidence interval reflects that.",
    "Cohorts smaller than about 40 samples per group produce no ROBUST or CONDITIONAL "
    "calls at all, because nothing reaches significance in 30% of specifications. "
    "Tiers are informative about larger studies; on small ones they mostly report that "
    "the data cannot settle the question.",
]


# ---------------------------------------------------------------------------
# Replication by labelling — v3 plan §33, V8
# ---------------------------------------------------------------------------
#: The V8 record, written by tests/reference/weighting_validation.py (never by hand).
#: docs/ is not deployed, so the script copies its record here as well, and
#: tests/test_evidence.py fails if the two differ. Absent until V8 has been run.
WEIGHTING_RECORD = Path(__file__).resolve().parent / "records" / "weighting_validation.json"


def _load_weighting_record() -> dict:
    try:
        with open(WEIGHTING_RECORD, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


WEIGHTING_VALIDATION = _load_weighting_record()

#: V8 as pre-registered in the v3 plan (§33). The script reads its margin from here, so
#: the criterion the site states and the one the script tests cannot drift apart.
V8_PREREGISTRATION = {
    "question": "Does neutral weighting change the labels, and do they keep their "
                "within-study predictive value?",
    "data": "25 published cohorts curated by Pelto et al., 75 stratified 50/50 splits "
            "(the design of the v2 label experiment)",
    "metrics": "held-out replication AUC for each weighting; share of taxa whose label "
               "changes; Kendall tau between weightings",
    "margin": 0.03,
    "success": "the decision-tree labels' AUC is non-inferior to one vote per "
               "specification",
    "if_it_fails": "reported as it falls; the decision tree stays the default, because it "
                   "was chosen on principle and choosing the weighting with the best AUC "
                   "would itself be an analytical choice",
}


def _by_tier(block: dict) -> dict:
    """A V8 labelling's majority-definition rates in TIER_REPLICATION's shape."""
    rows = block["definitions"]["replicated_majority"]["by_tier"]
    total = sum(int(r["n_taxa"]) for r in rows) or 1
    return {
        r["tier"]: {"rate": float(r["replication_rate"]),
                    "ci": (float(r["ci_low"]), float(r["ci_high"])),
                    "n": int(r["n_taxa"]), "cohorts": int(r["n_cohorts"]),
                    "share": int(r["n_taxa"]) / total}
        for r in rows if r["tier"] != "INSUFFICIENT"
    }


def replication_by_labelling() -> dict:
    """(rule set, scheme) -> {tier: facts}, for the labellings that were measured.

    A label's replication rate belongs to the labelling it was measured on. v2's rates
    (tier_validation.py) were measured with one vote per specification under the v2
    rules; a decision-tree label is a different label, and quoting v2's rate beside it
    would claim evidence nobody has collected. Only measured labellings appear here.
    """
    out = {(TIER_VALIDATION["ruleset"], TIER_VALIDATION["scheme"]): TIER_REPLICATION}
    if WEIGHTING_VALIDATION.get("registered_design"):
        for block in WEIGHTING_VALIDATION.get("labellings", {}).values():
            key = (block["ruleset"], block["scheme"])
            # v2's own record stays the source for v2; V8 reruns it only as a check.
            out.setdefault(key, _by_tier(block))
    return out


REPLICATION_BY_LABELLING = replication_by_labelling()


def labelling_of(summary) -> tuple:
    """(rule set, scheme) a results summary's tiers were assigned under. A summary
    stored before v3 carries neither, and is v2's."""
    return (getattr(summary, "ruleset", "v2") or "v2",
            getattr(summary, "scheme", "uniform") or "uniform")


def labelling_measured(ruleset: str = "v2", scheme: str = "uniform") -> bool:
    return (ruleset, scheme) in REPLICATION_BY_LABELLING


def tier_rates(ruleset: str = "v2", scheme: str = "uniform") -> dict:
    """{tier: facts} measured for this labelling, or {} when it has not been measured."""
    return REPLICATION_BY_LABELLING.get((ruleset, scheme), {})


def tier_evidence(tier: str, ruleset: str = "v2", scheme: str = "uniform") -> dict:
    """Held-out replication evidence for one tier of one labelling, or {} if none."""
    return tier_rates(ruleset, scheme).get(tier, {})


def tier_sentence(tier: str, ruleset: str = "v2", scheme: str = "uniform") -> str:
    """One line a researcher can read next to the tier itself — only about the
    labelling it was measured on."""
    facts = tier_evidence(tier, ruleset, scheme)
    if not facts:
        return ""
    low, high = facts["ci"]
    interval = (f"95% CI {low:.0%}–{high:.0%}, " if low == low and high == high else "")
    n_cohorts = (TIER_VALIDATION["n_cohorts"] if (ruleset, scheme) == ("v2", "uniform")
                 else WEIGHTING_VALIDATION.get("n_cohorts"))
    return (f"On held-out samples of the same study, across {n_cohorts} "
            f"published cohorts, {facts['rate']:.0%} of {tier} taxa replicated "
            f"({interval}n = {facts['n']}).")


def labelling_summary(ruleset: str = "v2", scheme: str = "uniform") -> dict:
    """The held-out headline for one labelling, from the record that measured it.

    {"measured": False} when nothing has; the page then says so instead of borrowing
    another labelling's numbers.
    """
    if (ruleset, scheme) == (TIER_VALIDATION["ruleset"], TIER_VALIDATION["scheme"]):
        return {"measured": True, "experiment": "tier validation (v2)",
                "record": TIER_VALIDATION["record"],
                "auc": TIER_VALIDATION["auc"], "null_rate": TIER_VALIDATION["null_rate"],
                "n_cohorts": TIER_VALIDATION["n_cohorts"],
                "n_splits": TIER_VALIDATION["n_splits"],
                "n_observations": TIER_VALIDATION["n_observations"]}
    if (ruleset, scheme) in REPLICATION_BY_LABELLING:
        block = next(b for b in WEIGHTING_VALIDATION["labellings"].values()
                     if (b["ruleset"], b["scheme"]) == (ruleset, scheme))
        definition = block["definitions"]["replicated_majority"]
        return {"measured": True, "experiment": "V8",
                "record": "docs/weighting_validation.json",
                "auc": definition["discrimination"].get("auc"),
                "null_rate": definition.get("null_rate"),
                "n_cohorts": WEIGHTING_VALIDATION["n_cohorts"],
                "n_splits": WEIGHTING_VALIDATION["n_splits"],
                "n_observations": WEIGHTING_VALIDATION["n_observations"]}
    return {"measured": False, "experiment": "V8", "record": None}


def unmeasured_sentence(ruleset: str, scheme: str) -> str:
    """What to say beside a tier whose labelling has no replication measurement."""
    from .weights import SCHEME_LABELS
    return (f"No replication rate has been measured for labels assigned this way "
            f"({SCHEME_LABELS.get(scheme, scheme)}, rule set {ruleset}). The held-out "
            f"test for them (V8) is pre-registered and has not been run yet; the rates "
            f"on the Evidence page were measured for v2's labels, one vote per "
            f"specification, and are not evidence about these.")


# ---------------------------------------------------------------------------
# The validation matrix — SPEC §24.2, rendered in the interface
# ---------------------------------------------------------------------------
@dataclass
class Component:
    name: str
    grade: str
    internal: str = ""
    reference: str = ""
    empirical: str = ""
    note: str = ""
    sources: list = field(default_factory=list)


VALIDATION_MATRIX = [
    Component(
        name="Statistical tests (7)",
        grade=REFERENCE,
        internal="Vectorised closed forms checked against the statsmodels models the "
                 "specification names; 200-test suite.",
        reference="edgeR 4.10.1 TMM identical to 0.0000%; ALDEx2 1.44.0 within its own "
                  "Monte-Carlo noise; ANCOM-BC 2.14.0 log fold changes r = 1.00000; "
                  "scikit-bio for CLR and multiplicative replacement.",
        empirical="",
        note="Four engine defects were found this way, two of which shift results by a "
             "constant and so are invisible to correlations. A fifth, found against the "
             "published benchmark below, was the worst: PyDESeq2's default size factors "
             "collapsed on real tables and made every feature significant.",
        sources=["tests/reference/compare_r.py",
                 "tests/reference/compare_scikit_bio.py"],
    ),
    Component(
        name="Methods on a published benchmark",
        grade=REFERENCE,
        internal="The paper's inputs are rebuilt from its archive and scripts; all 117 "
                 "tables have exactly the paper's feature count.",
        reference="Nearing et al. 2022, 24 of its datasets, 117 comparisons: MicroVerse "
                  "gives the paper's exact count of significant ASVs for rarefied Wilcoxon "
                  "(10 of 10), rarefied t-test (10 of 10) and, given the paper's CLR, "
                  "Wilcoxon on CLR (33 of 33). ALDEx2 matches in 19 of 33 and DESeq2 in 11 "
                  "of 31, and in 14 of 19 datasets the methods fall in exactly the paper's order.",
        empirical="",
        note="ALDEx2 and DESeq2 differ where MicroVerse differs by design: a more "
             "conservative ALDEx2 p-value, and one FDR correction for every method in "
             "place of DESeq2's independent filtering. Through the live site the paper's "
             "own tables gave the paper's numbers, 4 of 4.",
        sources=["tests/reference/nearing_study.py", "docs/nearing_study.json"],
    ),
    Component(
        name="Comparable effect size",
        grade=INTERNAL,
        internal="Checked against PyDESeq2's own log2 fold change (r > 0.9), and "
                 "asserted to be method- and covariate-independent by construction.",
        reference="",
        empirical="",
        note="It is a defined quantity, not an estimator with a reference "
             "implementation. Its covariate-independence is a design property that the "
             "interface has to state, because it means the specification curve cannot "
             "show covariate-driven instability.",
        sources=["tests/test_methods.py", "docs/tierney_study.json"],
    ),
    Component(
        name="Robustness tiers",
        grade=EMPIRICAL,
        internal="tests/test_worked_example.py encodes the specification's own worked "
                 "example; adversarial tests cover every threshold boundary.",
        reference="",
        empirical="Within a study — held-out halves of 25 published cohorts, 75 "
                  "discovery/validation splits, 12,564 taxon observations: ROBUST 90%, "
                  "CONDITIONAL 51%, FRAGILE 12%, UNSTABLE 2%. AUC 0.785, risk ratio "
                  "7.4, label-permuted null 0.7%. Across studies — 19 independent "
                  "cohorts of five diseases: no difference, 13% against 14%.",
        note="The strongest evidence in the system, and bounded: the labels predict "
             "replication in more samples from the same study, not in another "
             "population. Ordering survives dropping any single cohort; the ROBUST "
             "rate itself rests on 5 cohorts. Measured for the labels v2 assigns: one "
             "vote per specification, v2's pruning rules. v3's decision-tree labels "
             "are the next row.",
        sources=["tests/reference/tier_validation.py", "docs/tier_validation.json",
                 "tests/reference/published_findings_study.py"],
    ),
    "WEIGHTED_TIERS",
    Component(
        name="Reproducing a published analysis",
        grade=REFERENCE,
        internal="The paper's pipeline re-implemented from the authors' code: its sample "
                 "and OTU filters, genus collapsing, Kruskal-Wallis and Benjamini-Hochberg.",
        reference="Duvallet et al. 2017 (MicrobiomeHD), 19 cohorts, 2,824 samples: the "
                  "matching specification recovers 214 of the 217 genera the paper's "
                  "pipeline finds significant, exactly in 14 cohorts. The paper's test "
                  "on MicroVerse's own matrix gives identical calls in all 19 (p within "
                  "3e-8).",
        empirical="",
        note="The 3 missed and 9 extra genera sit at q = 0.05 and come from the "
             "relative-abundance denominator — the paper keeps reads with no genus, a "
             "genus table does not — not from the statistics. Three of the cohorts, "
             "uploaded to the live site on 26 September 2026, gave results identical to "
             "the local runs.",
        sources=["tests/reference/published_findings_study.py",
                 "docs/published_findings_study.json"],
    ),
    Component(
        name="Choice attribution",
        grade=EXPLORATORY,
        internal="Three independent estimators run on every analysis and their rank "
                 "agreement is reported; boundary and convergence failures are detected "
                 "and shown rather than absorbed.",
        reference="",
        empirical="",
        note="No external reference implementation exists for fork attribution, and no "
             "held-out experiment validates it. Grounded methodologically in Young & "
             "Holsteen (2017), but the numbers a run produces are exploratory and the "
             "interface labels them so.",
        sources=["app/core/attribution.py"],
    ),
    Component(
        name="Replication evidence",
        grade=EMPIRICAL,
        internal="",
        reference="",
        empirical="Within a study, the tier validation above is the replication "
                  "experiment: discovery and validation halves are disjoint samples, and "
                  "the held-out scoring never sees the discovery tier. Across studies, "
                  "3,688 cohort-pair observations in 19 published cohorts: findings the "
                  "paper's pipeline calls significant replicated in another cohort of the "
                  "same disease 13% of the time whether MicroVerse labelled them ROBUST or "
                  "CONDITIONAL or FRAGILE or UNSTABLE (difference -0.4 points, 95% CI -7 "
                  "to +7).",
        note="Two independent definitions of replication agree within a study (AUC 0.785 "
             "and 0.773). Across studies neither significance nor direction differed by "
             "label: differences between populations and protocols outweigh the "
             "analytical choices MicroVerse varies.",
        sources=["tests/reference/tier_validation.py",
                 "tests/reference/published_findings_study.py"],
    ),
    Component(
        name="Preprocessing",
        grade=REFERENCE,
        internal="Pipeline order enforced in one place and asserted; rarefaction, "
                 "prevalence filter, CLR, TSS and TMM each unit-tested.",
        reference="TMM identical to edgeR's `calcNormFactors`; CLR and multiplicative "
                  "replacement identical to scikit-bio to 1e-15.",
        empirical="",
        sources=["tests/test_preprocess.py", "tests/reference/compare_r.py"],
    ),
    Component(
        name="Covariate adjustment",
        grade=REFERENCE,
        internal="Adjustment-set enumeration and residualisation unit-tested.",
        reference="Reproduces Tierney et al. (2022) on their own cohorts from "
                  "curatedMetagenomicData: 28.7% sign-flipping over adjustment sets "
                  "against their 1-in-3, and 97.9% of T1D/T2D associations not ROBUST "
                  "against their >90%.",
        empirical="",
        sources=["tests/reference/tier_validation.py", "docs/tierney_study.json"],
    ),
    Component(
        name="Parsers and exports",
        grade=REFERENCE,
        internal="Round-trip and malformed-input tests for every reader; every export "
                 "regenerated and re-read in the test suite.",
        reference="BIOM v1, BIOM v2 and .qza fixtures written by biom-format 2.1.17 and "
                  "read back byte-for-byte.",
        empirical="",
        sources=["tests/test_real_formats.py",
                 "tests/reference/make_format_fixtures.py"],
    ),
]


def _weighted_tiers() -> Component:
    """The decision-tree labels (plan §26), graded by whether V8 has been run.

    Every number in the held-out cell is formatted from the V8 record; nothing here is
    typed. Until the record exists the row says the test has not been run.
    """
    record = WEIGHTING_VALIDATION
    internal = ("tests/test_v3_weighting.py: weights match a hand-computed tree, sum to "
                "1 over the valid children of every node, give seeds equal shares, and "
                "reproduce v2 byte for byte under one vote each.")
    sources = ["app/core/weights.py", "tests/test_v3_weighting.py",
               "tests/reference/weighting_validation.py"]
    if not record.get("registered_design"):
        return Component(
            name="Weighted tiers (v3 decision tree)", grade=INTERNAL, internal=internal,
            note="The labels v3 runs report by default. Their held-out test, V8, is "
                 "pre-registered in the v3 plan and has not been run, so no replication "
                 "rate is shown beside them anywhere on this site.",
            sources=sources)
    labellings = record["labellings"]

    def auc_of(label):
        return labellings[label]["definitions"]["replicated_majority"][
            "discrimination"].get("auc", float("nan"))
    result = record["non_inferiority"]["replicated_majority"]
    changed = record["label_changes"]["from_v3_uniform"]["v3_decision_tree"]
    verdict = ("non-inferior" if result["non_inferior"]
               else "non-inferiority not shown")
    return Component(
        name="Weighted tiers (v3 decision tree)", grade=EMPIRICAL, internal=internal,
        empirical=(
            f"V8, pre-registered: {record['n_cohorts']} cohorts, {record['n_splits']} "
            f"splits, {record['n_observations']:,} taxon observations. AUC "
            f"{auc_of('v3_decision_tree'):.3f} with decision-tree weights against "
            f"{auc_of('v3_uniform'):.3f} with one vote each; difference "
            f"{result['difference']:+.3f} (95% CI {result['ci'][0]:+.3f} to "
            f"{result['ci'][1]:+.3f}), margin {result['margin']:+.2f}: {verdict}. "
            f"Weighting changed the label of {changed['share_changed']:.1%} of "
            f"observations."),
        note="Within a study only, like the v2 result above. The default weighting was "
             "chosen on principle before V8 and stays whatever V8 found.",
        sources=[*sources, "docs/weighting_validation.json"])


VALIDATION_MATRIX = [_weighted_tiers() if c == "WEIGHTED_TIERS" else c
                     for c in VALIDATION_MATRIX]


def matrix_rows() -> list:
    """The validation matrix as plain dicts, for templates and the JSON manifest."""
    return [
        {
            "name": c.name,
            "grade": c.grade,
            "grade_label": GRADE_LABEL[c.grade],
            "internal": c.internal,
            "reference": c.reference,
            "empirical": c.empirical,
            "note": c.note,
            "sources": list(c.sources),
        }
        for c in VALIDATION_MATRIX
    ]
