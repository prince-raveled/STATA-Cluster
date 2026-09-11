"""Plain-language definitions for every term the interface uses.

The interface used to label its sections with specification numbers — "§16.2 —
per-taxon robustness". That is precise for anyone holding the specification and
meaningless for everyone else, which is most researchers. Worse, the surrounding prose
assumed the vocabulary too: "the ratio of valid to enumerated tells you how much of the
nominal space was statistically incoherent" is a true sentence that teaches nobody
anything.

Every term below is defined twice: `short` is a single sentence that fits in a tooltip,
`long` explains it properly for the glossary. A term is only listed here if it actually
appears in the interface — a glossary of words nobody reads is furniture.

`spec` keeps the specification reference for the people who do want it, shown as a
small marker rather than as the heading.
"""
from __future__ import annotations

TERMS: dict[str, dict] = {
    # --- the core idea ----------------------------------------------------
    "specification": {
        "term": "specification",
        "short": "One complete analysis — one particular set of analytical choices, "
                 "start to finish.",
        "long": "A specification is one full analysis of your data: a decision about "
                "whether to rarefy and how deep, which taxonomic level to work at, how "
                "aggressively to filter rare taxa, which transformation to apply, which "
                "statistical test to run, and how to correct for multiple testing. "
                "Change any one of those and you have a different specification. "
                "MicroVerse runs thousands of them on the same data.",
        "spec": "13",
    },
    "fork": {
        "term": "analytical choice",
        "short": "A decision point where reasonable researchers disagree — like whether "
                 "to rarefy.",
        "long": "A point in the analysis where there is no field consensus, so two "
                "competent researchers could justifiably do different things. "
                "MicroVerse varies seven of them: rarefaction, taxonomic level, "
                "prevalence filtering, transformation, statistical test, multiple-"
                "testing correction, and which confounders to adjust for. Elsewhere "
                "these are called 'forks' or 'researcher degrees of freedom'.",
        "spec": "10",
    },
    "multiverse": {
        "term": "multiverse analysis",
        "short": "Running every defensible version of an analysis instead of picking one.",
        "long": "Rather than choosing one pipeline and reporting its result, a multiverse "
                "analysis runs all the pipelines you could have defended and reports the "
                "whole distribution of answers. The idea comes from psychology "
                "(Steegen et al. 2016); this applies it to microbiome differential "
                "abundance.",
        "spec": "10",
    },
    "specification_curve": {
        "term": "specification curve",
        "short": "A plot of one taxon's result under every analysis, sorted, showing "
                 "which choice moves it.",
        "long": "The upper panel shows the effect size from every specification for one "
                "taxon, sorted smallest to largest and coloured by whether it was "
                "significant. The lower panel shows which choice was active in each. "
                "When the non-significant points all line up under one row below, that "
                "choice is what is driving your result.",
        "spec": "16.3",
    },

    # --- the analytical choices -------------------------------------------
    "rarefaction": {
        "term": "rarefaction",
        "short": "Randomly subsampling every sample down to the same sequencing depth.",
        "long": "Samples are sequenced to different depths, so a taxon can look more "
                "abundant simply because that sample was sequenced more deeply. "
                "Rarefying randomly discards reads until every sample has the same "
                "total. Whether to do this at all is one of the field's longest-running "
                "arguments. Because it is random, MicroVerse treats each random draw as "
                "its own specification rather than averaging them away — which is how it "
                "can tell you how much of the effect is the coin flip.",
        "spec": "10",
    },
    "prevalence_filter": {
        "term": "prevalence filter",
        "short": "Dropping taxa that appear in too few samples before testing.",
        "long": "A taxon seen in 3 of 200 samples carries almost no information and adds "
                "to the multiple-testing burden. Filtering removes it — but the "
                "threshold is arbitrary, and raising it changes which taxa survive and "
                "therefore how harshly everything else is corrected. MicroVerse tries "
                "0%, 5%, 10% and 20%.",
        "spec": "10",
    },
    "transformation": {
        "term": "transformation",
        "short": "Converting raw counts onto a different scale before testing.",
        "long": "Sequencing gives relative, not absolute, abundances, so raw counts are "
                "usually converted first. MicroVerse tries four: raw counts unchanged, "
                "TSS (proportions), CLR (a log-ratio transform that respects the "
                "compositional nature of the data), and TMM (edgeR's library-size "
                "normalisation).",
        "spec": "10",
    },
    "fdr": {
        "term": "multiple-testing correction",
        "short": "Adjusting p-values because hundreds of taxa are tested at once.",
        "long": "Testing 300 taxa at p < 0.05 yields about 15 false positives by chance. "
                "False-discovery-rate procedures adjust for that. MicroVerse tries three "
                "settings: Benjamini-Hochberg at 0.05 and at 0.10, and the more "
                "conservative Benjamini-Yekutieli at 0.05.",
        "spec": "10",
    },
    "covariate": {
        "term": "covariate",
        "short": "A variable like age or BMI that you adjust for to avoid confounding.",
        "long": "A factor that might explain an apparent difference between your groups. "
                "Adjusting for it is standard, but which ones to adjust for is a "
                "judgement call — and Tierney et al. showed that judgement alone flips "
                "about one association in three. Covariate mode varies every combination.",
        "spec": "12",
    },

    # --- what comes out ----------------------------------------------------
    "taxon": {
        "term": "taxon",
        "short": "One microbial group — a species, genus, or OTU, depending on your table.",
        "long": "One row of your abundance table: whatever level of organism you supplied. "
                "MicroVerse gives every taxon its own verdict.",
        "spec": "",
    },
    "effect_size": {
        "term": "effect size",
        "short": "How big the difference is, as a log2 fold change in mean relative "
                 "abundance.",
        "long": "The size of the difference between your groups, not just whether it is "
                "statistically significant. MicroVerse reports log2 fold change of mean "
                "relative abundance: +1 means twice as abundant in the second group, "
                "-1 means half as abundant, 0 means no difference.",
        "spec": "14",
    },
    "harmonised_effect": {
        "term": "comparable effect size",
        "short": "One effect measure used for every method, so results can share an axis.",
        "long": "The seven statistical tests report seven incompatible numbers — a rank "
                "statistic, a log odds ratio, a fold change — which cannot be plotted "
                "together. So significance comes from whichever test ran, while the "
                "effect size is always recomputed the same way from the same data. "
                "One important consequence: because it is computed from the data rather "
                "than from the model, it does not change when you adjust for covariates. "
                "Each method's own native statistic is kept in the downloads.",
        "spec": "14",
    },
    "significance_rate": {
        "term": "how often significant",
        "short": "The share of analyses that called this taxon significant, out of those "
                 "that tested it.",
        "long": "Of all the specifications where this taxon survived the prevalence "
                "filter and was actually tested, the fraction that found it "
                "significant after multiple-testing correction. Specifications where "
                "the taxon was filtered out are excluded from the denominator — it was "
                "never examined there, so it should not be counted against it.",
        "spec": "15",
    },
    "direction_agreement": {
        "term": "direction agreement",
        "short": "The share of analyses that agreed on which group had more of this taxon.",
        "long": "Whether the analyses agree on the direction of the difference, "
                "regardless of significance. If 99% of specifications say a taxon is "
                "higher in cases, the direction is settled. If it is 60/40, the data "
                "cannot tell you which way the effect goes, and no amount of "
                "significance rescues that.",
        "spec": "16.2",
    },
    "tier": {
        "term": "robustness tier",
        "short": "A verdict summarising how consistently the analyses agreed about a taxon.",
        "long": "One of six labels, from ROBUST (nearly every analysis agrees, and on the "
                "direction) down to NOT DETECTED (no analysis found it). The tiers are "
                "not just a convention: they were tested on 25 published datasets by "
                "assigning them on half the samples and checking replication on the "
                "held-out half. ROBUST taxa replicated 90% of the time, FRAGILE 12%.",
        "spec": "16.2",
    },
    "attribution": {
        "term": "choice attribution",
        "short": "An attempt to say which analytical choice is moving your results most.",
        "long": "A statistical decomposition that tries to apportion the variation in "
                "your results across the seven choices. On real data it explains only a "
                "few percent of the variation and different estimators disagree about "
                "the ranking, so MicroVerse labels it exploratory and shows you the "
                "diagnostics rather than presenting it as a finding.",
        "spec": "17",
    },

    # --- accounting --------------------------------------------------------
    "enumerated": {
        "term": "analyses considered",
        "short": "Every combination of choices, before impossible ones are removed.",
        "long": "The full count of choice combinations before any are ruled out. Some "
                "combinations are statistically incoherent — correcting for library size "
                "twice, for example — so this is always larger than the number that "
                "actually ran.",
        "spec": "12",
    },
    "valid": {
        "term": "analyses run",
        "short": "The combinations that are statistically coherent and actually executed.",
        "long": "What was left after removing combinations that contradict themselves. "
                "MicroVerse always shows both counts and the reason for every removal, "
                "so you can see how much of the nominal space was never real.",
        "spec": "11",
    },
    "pruned": {
        "term": "ruled out",
        "short": "Combinations removed because they contradict themselves statistically.",
        "long": "Some combinations of choices cannot be defended — running TMM "
                "normalisation on already-rarefied counts corrects for sequencing depth "
                "twice; applying a transformation before a presence/absence test changes "
                "nothing. Each removal is listed with its reason and count.",
        "spec": "11",
    },
    "model_fit": {
        "term": "model fits",
        "short": "The actual statistical models run; cheaper than the specification count.",
        "long": "The three multiple-testing settings are applied afterwards to stored "
                "p-values, so they cost almost nothing. The number of models actually "
                "fitted is roughly the specification count divided by three — this is "
                "what the computation really costs.",
        "spec": "12",
    },
}

#: Order for the glossary page: the idea, then the choices, then the outputs.
GLOSSARY_ORDER = [
    ("The idea", ["multiverse", "specification", "fork", "specification_curve"]),
    ("The seven choices", ["rarefaction", "prevalence_filter", "transformation",
                           "fdr", "covariate"]),
    ("Reading your results", ["taxon", "tier", "significance_rate",
                              "direction_agreement", "effect_size",
                              "harmonised_effect", "attribution"]),
    ("Counting the analyses", ["enumerated", "valid", "pruned", "model_fit"]),
]


def term(key: str) -> dict:
    """One definition, or a safe blank if the key is unknown."""
    return TERMS.get(key, {"term": key, "short": "", "long": "", "spec": ""})


def glossary_sections() -> list:
    """The glossary, grouped, for rendering."""
    return [
        {"heading": heading,
         "entries": [dict(TERMS[k], key=k) for k in keys if k in TERMS]}
        for heading, keys in GLOSSARY_ORDER
    ]
