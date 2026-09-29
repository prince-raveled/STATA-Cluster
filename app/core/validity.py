"""Validity matrix — SPEC §11, implemented literally, plus the v3 rules (plan §26.3).

Pruning matters twice: it cuts compute, and it stops incoherent specifications from
artificially widening the distribution of answers. Both counts (enumerated and valid)
are reported to the user.

Two rule sets exist. `v2` is SPEC §11 exactly, rules R1-R7, and is kept so a v2 result
can still be reproduced. `v3` adds R8, which removes a combination with no library-size
correction at all, and R9, which keeps presence/absence on unrarefied counts only when
sequencing depth is modelled. Every rule, and every fork level, carries a reason, a
citation and a Del Giudice & Gangestad (2021) type in `DEFENSIBILITY` below, which
`tests/reference/make_defensibility_register.py` writes to
docs/defensibility_register.json for the /about page.
"""
from __future__ import annotations

from dataclasses import dataclass

RULESETS = ("v3", "v2")
DEFAULT_RULESET = "v3"

INCOMPATIBLE = {
    # method: set of transforms it must NOT receive
    "pydeseq2": {"clr", "tss", "tmm"},  # requires integer counts
    "aldex2": {"clr"},                  # applies CLR internally
    "ancombc": {"clr", "tss", "tmm"},   # estimates sampling fractions itself
    "wilcoxon": set(),                  # accepts anything
    "ttest": set(),
    "logistic": set(),                  # uses presence/absence; transform irrelevant
    "linear": set(),
}

INCOMPATIBLE_REASON = {
    "pydeseq2": "PyDESeq2 requires integer counts",
    "aldex2": "ALDEx2 applies CLR internally",
    "ancombc": "ANCOM-BC estimates sampling fractions itself",
}

#: Methods compared directly on the matrix values, with no library-size model of their
#: own. On unrarefied raw counts they compare reads, not abundance (R8).
DEPTH_NAIVE_METHODS = {"wilcoxon", "ttest", "linear"}

R8_REASON = ("raw counts without rarefaction or normalisation: sequencing depth alone "
             "can make a taxon look differential")

# Additional rules, checked separately:
RULES = [
    # (predicate, reason) — spec is INVALID if predicate is True
    (lambda s: s.method == "ancombc" and s.rarefaction != "none",
     "ANCOM-BC estimates sampling fractions; rarefaction double-corrects"),

    (lambda s: s.transform == "tmm" and s.rarefaction != "none",
     "TMM + rarefaction is double library-size correction"),

    (lambda s: s.method == "logistic" and s.transform != "raw",
     "presence/absence is transform-invariant; keep one canonical "
     "combination"),

    (lambda s: s.method == "pydeseq2" and s.rarefaction != "none",
     "DESeq2 models library size internally"),
]

#: v3 only. R9 is not a removal: `needs_depth_adjustment` marks the specification and
#: the runner adds log library size to the logistic model.
V3_RULES = [
    (lambda s: (s.rarefaction == "none" and s.transform == "raw"
                and s.method in DEPTH_NAIVE_METHODS),
     R8_REASON),
]


@dataclass(frozen=True)
class Capabilities:
    """Dataset-level facts that make otherwise-valid specifications impossible.

    Kept separate from §11 so the published validity matrix stays literal: these
    prunings depend on the upload, not on statistics.
    """

    integer_counts: bool = True
    can_rarefy: bool = True
    can_collapse_to_genus: bool = True

    def reason(self, spec) -> str:
        if not self.can_rarefy and spec.rarefaction != "none":
            return "dataset values are not integers, so subsampling is impossible"
        if not self.integer_counts and spec.method in {"pydeseq2", "ancombc", "aldex2"}:
            return f"{spec.method} needs integer counts; this table holds relative abundances"
        if not self.integer_counts and spec.transform == "tmm":
            return "TMM needs integer counts; this table holds relative abundances"
        if not self.can_collapse_to_genus and spec.rank == "genus":
            return "no taxonomy supplied, so the table cannot be collapsed to genus"
        return ""


def _check_ruleset(ruleset: str) -> None:
    if ruleset not in RULESETS:
        raise ValueError(f"Unknown rule set: {ruleset}")


def invalid_reason(spec, capabilities: Capabilities = None,
                   ruleset: str = DEFAULT_RULESET) -> str:
    """Empty string when the specification is valid; otherwise why it was pruned."""
    _check_ruleset(ruleset)
    forbidden = INCOMPATIBLE.get(spec.method, set())
    if spec.transform in forbidden:
        why = INCOMPATIBLE_REASON.get(spec.method, "transform is incompatible with the method")
        return f"{why}; {spec.transform.upper()} input is not defensible"
    for predicate, reason in RULES:
        if predicate(spec):
            return reason
    if ruleset == "v3":
        for predicate, reason in V3_RULES:
            if predicate(spec):
                return reason
    if capabilities is not None:
        return capabilities.reason(spec)
    return ""


def is_valid(spec, capabilities: Capabilities = None,
             ruleset: str = DEFAULT_RULESET) -> bool:
    return not invalid_reason(spec, capabilities, ruleset)


def needs_depth_adjustment(spec, ruleset: str = DEFAULT_RULESET) -> bool:
    """R9: presence/absence on unrarefied counts must model sequencing depth.

    Rarefied libraries all have the same depth, so the rule concerns only "none".
    """
    _check_ruleset(ruleset)
    return ruleset == "v3" and spec.method == "logistic" and spec.rarefaction == "none"


def apply_rules(spec, ruleset: str = DEFAULT_RULESET):
    """The specification as the rule set runs it: R9 marks it depth-adjusted.

    Used for anything built outside the enumeration — a declared pipeline, a claim to
    audit — so it matches the grid's own copy instead of looking absent from it.
    """
    from dataclasses import replace
    wanted = needs_depth_adjustment(spec, ruleset)
    if bool(getattr(spec, "depth_adjusted", False)) == wanted:
        return spec
    return replace(spec, depth_adjusted=wanted)


# ---------------------------------------------------------------------------
# Defensibility register — plan §26.3
# ---------------------------------------------------------------------------
#: Del Giudice & Gangestad (2021): E — principled equivalence (the options should give
#: the same answer, so varying them measures noise); N — principled non-equivalence
#: (one option is wrong for the question, so it is pruned, not varied); U — uncertainty
#: (the field has not settled which is right, which is what a multiverse is for).
TYPE_LABELS = {
    "E": "principled equivalence",
    "N": "principled non-equivalence",
    "U": "uncertainty",
}

CITATIONS = {
    "mcmurdie2014": "McMurdie PJ, Holmes S. Waste not, want not: why rarefying microbiome "
                    "data is inadmissible. PLoS Comput Biol 2014;10:e1003531.",
    "weiss2017": "Weiss S, et al. Normalization and microbial differential abundance "
                 "strategies depend upon data characteristics. Microbiome 2017;5:27.",
    "schloss2024": "Schloss PD. Rarefaction is currently the best approach to control for "
                   "uneven sequencing effort in amplicon sequence analyses. mSphere "
                   "2024;9:e00354-23.",
    "love2014": "Love MI, Huber W, Anders S. Moderated estimation of fold change and "
                "dispersion for RNA-seq data with DESeq2. Genome Biol 2014;15:550.",
    "fernandes2014": "Fernandes AD, et al. Unifying the analysis of high-throughput "
                     "sequencing datasets: characterizing RNA-seq, 16S rRNA gene "
                     "sequencing and selective growth experiments by compositional data "
                     "analysis. Microbiome 2014;2:15.",
    "lin2020": "Lin H, Peddada SD. Analysis of compositions of microbiomes with bias "
               "correction. Nat Commun 2020;11:3514.",
    "robinson2010": "Robinson MD, Oshlack A. A scaling normalization method for "
                    "differential expression analysis of RNA-seq data. Genome Biol "
                    "2010;11:R25.",
    "gloor2017": "Gloor GB, et al. Microbiome datasets are compositional: and this is not "
                 "optional. Front Microbiol 2017;8:2224.",
    "nearing2022": "Nearing JT, et al. Microbiome differential abundance methods produce "
                   "different results across 38 datasets. Nat Commun 2022;13:342.",
    "pelto2025": "Pelto J, et al. Elementary methods provide more replicable results in "
                 "microbial differential abundance analysis. Brief Bioinform "
                 "2025;26:bbaf130.",
    "benjamini1995": "Benjamini Y, Hochberg Y. Controlling the false discovery rate. "
                     "J R Stat Soc B 1995;57:289-300.",
    "benjamini2001": "Benjamini Y, Yekutieli D. The control of the false discovery rate in "
                     "multiple testing under dependency. Ann Stat 2001;29:1165-1188.",
    "tierney2022": "Tierney BT, et al. Systematically assessing microbiome-disease "
                   "associations identifies drivers of inconsistency in metagenomic "
                   "research. PLoS Biol 2022;20:e3001556.",
    "delgiudice2021": "Del Giudice M, Gangestad SW. A traveler's guide to the multiverse: "
                      "promises, pitfalls, and a framework for the evaluation of analytic "
                      "decisions. Adv Methods Pract Psychol Sci 2021;4(1).",
}

DEFENSIBILITY = {
    "rules": [
        {"id": "R1", "ruleset": "v2", "removes": "PyDESeq2 given CLR, TSS or TMM",
         "reason": "DESeq2 models integer counts; a transformed matrix is not its input",
         "type": "N", "cite": ["love2014"]},
        {"id": "R2", "ruleset": "v2", "removes": "ALDEx2 given CLR",
         "reason": "ALDEx2 applies the CLR itself; a second CLR is not its model",
         "type": "N", "cite": ["fernandes2014"]},
        {"id": "R3", "ruleset": "v2", "removes": "ANCOM-BC given CLR, TSS or TMM",
         "reason": "ANCOM-BC estimates sample-specific sampling fractions from counts",
         "type": "N", "cite": ["lin2020"]},
        {"id": "R4", "ruleset": "v2", "removes": "ANCOM-BC on rarefied counts",
         "reason": "rarefying and ANCOM-BC's bias correction both correct for sampling "
                   "fraction, so together they correct twice",
         "type": "N", "cite": ["lin2020"]},
        {"id": "R5", "ruleset": "v2", "removes": "TMM on rarefied counts",
         "reason": "rarefying and TMM are both library-size corrections",
         "type": "N", "cite": ["robinson2010", "mcmurdie2014"]},
        {"id": "R6", "ruleset": "v2", "removes": "logistic regression on any transform "
                                                 "but raw",
         "reason": "presence/absence is identical under every transform, so the copies "
                   "would count one analysis several times",
         "type": "E", "cite": []},
        {"id": "R7", "ruleset": "v2", "removes": "PyDESeq2 on rarefied counts",
         "reason": "DESeq2 models library size through its size factors",
         "type": "N", "cite": ["love2014", "mcmurdie2014"]},
        {"id": "R8", "ruleset": "v3", "removes": "Wilcoxon, Welch or linear regression on "
                                                 "unrarefied raw counts",
         "reason": "nothing corrects for library size, so a difference in sequencing "
                   "depth between the groups is enough to make a taxon differential",
         "type": "N", "cite": ["mcmurdie2014", "weiss2017"]},
        {"id": "R9", "ruleset": "v3", "removes": "nothing: changes logistic regression on "
                                                 "unrarefied counts",
         "reason": "whether a taxon is detected at all depends on sequencing depth, so "
                   "presence/absence without rarefying is modelled with log library size "
                   "as a covariate",
         "type": "N", "cite": ["mcmurdie2014", "weiss2017"]},
    ],
    "forks": [
        {"fork": "Rarefy or not", "levels": "no; yes",
         "reason": "the field disagrees about whether rarefying is admissible",
         "type": "U", "cite": ["mcmurdie2014", "schloss2024"]},
        {"fork": "Rarefaction depth", "levels": "minimum library, 1,000, 5,000, 10,000",
         "reason": "no depth is agreed; deeper keeps more reads and fewer samples",
         "type": "U", "cite": ["schloss2024"]},
        {"fork": "Rarefaction seed", "levels": "1, 2, 3",
         "reason": "draws from the same subsampling; they should agree, so their spread "
                   "measures the randomness rarefying adds",
         "type": "E", "cite": ["mcmurdie2014"]},
        {"fork": "Taxonomic rank", "levels": "input rank; genus",
         "reason": "the unit of analysis changes what a taxon is",
         "type": "U", "cite": ["nearing2022"]},
        {"fork": "Prevalence filter", "levels": "0%, 5%, 10%, 20%",
         "reason": "no agreed threshold; it trades power against rare-taxon noise",
         "type": "U", "cite": ["nearing2022"]},
        {"fork": "Transformation", "levels": "raw, TSS, CLR, TMM",
         "reason": "compositional and count-scale approaches remain in dispute",
         "type": "U", "cite": ["gloor2017", "weiss2017"]},
        {"fork": "Statistical test", "levels": "Wilcoxon, Welch, linear, logistic; "
                                               "ANCOM-BC, ALDEx2, PyDESeq2 in Full mode",
         "reason": "methods disagree substantially on the same data",
         "type": "U", "cite": ["nearing2022", "pelto2025"]},
        {"fork": "FDR method", "levels": "Benjamini-Hochberg; Benjamini-Yekutieli",
         "reason": "BY is valid under any dependence, BH under positive dependence, which "
                   "microbiome data may or may not satisfy",
         "type": "U", "cite": ["benjamini1995", "benjamini2001"]},
        {"fork": "FDR threshold", "levels": "0.05; 0.10",
         "reason": "the tolerated false discovery rate is a convention, not a fact",
         "type": "U", "cite": ["benjamini1995"]},
        {"fork": "Covariate adjustment", "levels": "every subset of the chosen "
                                                   "covariates (up to 64)",
         "reason": "which confounders to adjust for changes microbiome associations",
         "type": "U", "cite": ["tierney2022"]},
    ],
}


def rule_id(reason: str) -> str:
    """The register id for a pruning reason, or '' for a dataset capability."""
    if reason == R8_REASON:
        return "R8"
    ids = {r: f"R{i + 4}" for i, (_, r) in enumerate(RULES)}
    if reason in ids:
        return ids[reason]
    for number, method in ((1, "pydeseq2"), (2, "aldex2"), (3, "ancombc")):
        if reason.startswith(INCOMPATIBLE_REASON[method]):
            return f"R{number}"
    return ""
