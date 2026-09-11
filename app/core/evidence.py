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

from dataclasses import dataclass, field

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

#: Stated wherever a ROBUST call is displayed. The rate is real and the ordering survives
#: dropping any single cohort, but the rate itself rests on few cohorts and must not be
#: quoted as a precise probability.
TIER_CAVEATS = [
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


def tier_evidence(tier: str) -> dict:
    """Held-out replication evidence for one tier, or {} if it has none."""
    return TIER_REPLICATION.get(tier, {})


def tier_sentence(tier: str) -> str:
    """One line a researcher can read next to the tier itself."""
    facts = TIER_REPLICATION.get(tier)
    if not facts:
        return ""
    low, high = facts["ci"]
    return (f"In held-out validation on {TIER_VALIDATION['n_cohorts']} published "
            f"cohorts, {facts['rate']:.0%} of {tier} taxa replicated "
            f"(95% CI {low:.0%}–{high:.0%}, n = {facts['n']}).")


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
        internal="Vectorised closed forms checked against the statsmodels models §10 "
                 "names; 200-test suite.",
        reference="edgeR 4.10.1 TMM identical to 0.0000%; ALDEx2 1.44.0 within its own "
                  "Monte-Carlo noise; ANCOM-BC 2.14.0 log fold changes r = 1.00000; "
                  "scikit-bio for CLR and multiplicative replacement.",
        empirical="",
        note="Four engine defects were found this way, two of which shift results by a "
             "constant and so are invisible to correlations.",
        sources=["tests/reference/compare_r.py",
                 "tests/reference/compare_scikit_bio.py"],
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
        internal="tests/test_worked_example.py encodes §19; adversarial tests cover "
                 "every threshold boundary.",
        reference="",
        empirical="Held-out replication on 25 published cohorts, 75 discovery/"
                  "validation splits, 12,564 taxon observations: ROBUST 90%, "
                  "CONDITIONAL 51%, FRAGILE 12%, UNSTABLE 2%. AUC 0.785, risk ratio "
                  "7.4, label-permuted null 0.7%.",
        note="The strongest evidence in the system. Ordering survives dropping any "
             "single cohort; the ROBUST rate itself rests on 5 cohorts.",
        sources=["tests/reference/tier_validation.py", "docs/tier_validation.json"],
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
        empirical="The tier validation above is itself the replication experiment: "
                  "discovery and validation halves are disjoint samples, and the "
                  "held-out scoring never sees the discovery tier.",
        note="Two independent definitions of replication agree (AUC 0.785 and 0.773).",
        sources=["tests/reference/tier_validation.py"],
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
